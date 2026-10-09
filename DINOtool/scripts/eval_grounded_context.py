#!/usr/bin/env python3
"""Matched frozen Geometry/BoundedUnion evaluation on remote-sensing data."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from dinotool.contextual_phrase_readout import ContextualPhraseConfig
from dinotool.grounded_context import (
    METHODS, GroundedContextConfig, GroundedContextSegmenter, context_box, context_crop,
)
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.loveda import discover_loveda_samples
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs, serialize_class_specs
from dinotool.tcpr import TCPRConfig
from eval_competitive_evidence_loveda_e1 import (
    _empty_change, _change_summary, _update_change, protocol_classes,
)
from eval_stride_ov_loveda_e1 import (
    Confusion, crop_tile, load_rgb, make_checkpoints, patch_valid_mask, target_ids,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("loveda", "udd5", "oem", "vdd", "potsdam", "vaihingen"),
        required=True,
    )
    for name in ("dinov3-repo", "checkpoint-dir", "data-root", "output-dir", "vocabulary-config"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.12)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--overview-size", type=int, default=512)
    parser.add_argument("--context-extent", type=int, default=1024)
    parser.add_argument("--output-temperature", type=float, default=0.07)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--vdd-ontology", choices=("legacy", "official"), default="legacy",
        help="Use official VDD other/wall names for an explicitly corrected vocabulary.",
    )
    parser.add_argument("--geometry-depth", type=int, choices=(1, 2), default=1)
    parser.add_argument("--prefix-policy", choices=("preserve", "block"), default="preserve")
    parser.add_argument(
        "--methods",
        default="G_all20_uniform,BoundedUnion",
        help="Comma-separated grounded-context methods to materialize.",
    )
    return parser.parse_args()


@torch.inference_mode()
def predict_image(model, image, banks, args, work_dir, diagnostics, methods=METHODS):
    height, width = image.shape[-2:]
    if getattr(model, "needs_whole", True):
        whole = model.prepare_overview(image[None].to(model.device))
        whole_maps = {p: model.overview_alias_map(whole, bank) for p, bank in banks.items()}
        del whole
    else:
        whole_maps = {p: None for p in banks}
    context_cache = {}
    if (getattr(model, "needs_whole", True)
            and height == width and height <= args.context_extent):
        context_cache[(0, 0, height)] = whole_maps
    blend = hann_blend_window(args.tile_size)
    forward_counts = {"local": 0, "context": int(model.needs_whole) if hasattr(model, "needs_whole") else 1}
    with ExitStack() as stack:
        accumulators = {
            (p, m): stack.enter_context(ProbabilityAccumulator(
                bank.class_count, height, width, 2048, work_dir))
            for p, bank in banks.items() for m in methods
        }
        for top in tile_starts(height, args.tile_size, args.overlap):
            for left in tile_starts(width, args.tile_size, args.overlap):
                box = context_box(height, width, top, left, args.tile_size, args.context_extent)
                if box not in context_cache:
                    context = model.prepare_overview(context_crop(image, box).to(model.device))
                    context_cache[box] = {
                        p: model.overview_alias_map(context, bank) for p, bank in banks.items()
                    }
                    forward_counts["context"] += 1
                    del context
                h, w = min(args.tile_size, height - top), min(args.tile_size, width - left)
                prepared = model.prepare_image(crop_tile(image, left, top, args.tile_size).to(model.device))
                forward_counts["local"] += 1
                valid = patch_valid_mask(h, w, args.tile_size, model.patch_size, model.device)
                for p, bank in banks.items():
                    fields, diag = model.read_grounded(
                        prepared, whole_maps[p], context_cache[box][p], bank,
                        height=height, width=width, top=top, left=left, box=box, valid_mask=valid,
                    )
                    diagnostics[p]["tiles"] += 1
                    for key, value in diag.items():
                        diagnostics[p][key] = diagnostics[p].get(key, 0.0) + value
                    for method, logits in fields.items():
                        if method not in methods:
                            continue
                        if not torch.isfinite(logits).all():
                            raise FloatingPointError(
                                f"Non-finite {method} logits for {sample.key} at tile ({top}, {left})."
                            )
                        dense = F.interpolate(logits, size=(args.tile_size, args.tile_size),
                                              mode="bilinear", align_corners=False)
                        prob = (dense.float() / args.output_temperature).softmax(dim=1)
                        accumulators[(p, method)].add(
                            prob[0, :, :h, :w].cpu().numpy(), blend[:h, :w], left, top)
        predictions = {
            p: {m: accumulators[(p, m)].finalize(None)[0] for m in methods} for p in banks
        }
    return predictions, forward_counts


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def proposal_quality(base, proposal, output, target, class_count):
    """How many useful/harmful BoundedUnion changes survive the final decision?"""
    valid = (target >= 0) & (target < class_count)
    changed = valid & (proposal != base)
    beneficial = changed & (base != target) & (proposal == target)
    harmful = changed & (base == target) & (proposal != target)
    return {
        "proposed_changed": int(changed.sum()),
        "proposed_beneficial": int(beneficial.sum()),
        "proposed_harmful": int(harmful.sum()),
        "beneficial_retained": int((beneficial & (output == target)).sum()),
        "harmful_rejected": int((harmful & (output == base)).sum()),
    }


def proposal_quality_summary(counts):
    return {
        **counts,
        "beneficial_retention": counts["beneficial_retained"] / max(counts["proposed_beneficial"], 1),
        "harmful_rejection": counts["harmful_rejected"] / max(counts["proposed_harmful"], 1),
    }


def main(args, *, model_type=GroundedContextSegmenter,
         config_type=GroundedContextConfig, methods=METHODS,
         implementation="gar-gcr-context-gated-v3-20260928"):
    if not 0 < args.memory_fraction <= 1 or args.progress_every < 1:
        raise ValueError("Invalid memory fraction or progress interval.")
    if args.vdd_ontology != "legacy" and args.dataset != "vdd":
        raise ValueError("--vdd-ontology=official is only valid for VDD.")
    if torch.device(args.device).type == "cuda":
        device_index = torch.device(args.device).index
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0 if device_index is None else device_index)
    if args.tile_size != 512 or args.overview_size != 512 or args.overlap != 128:
        raise ValueError("This experiment fixes local/overview size=512 and overlap=128.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index.")
    requested_methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    if not requested_methods or any(item not in methods for item in requested_methods):
        raise ValueError(f"--methods must be a nonempty subset of {methods}")
    if "G_all20_uniform" not in requested_methods or "BoundedUnion" not in requested_methods:
        raise ValueError("The external protocol requires G_all20_uniform and BoundedUnion.")
    methods = requested_methods
    if args.dataset == "loveda":
        samples = discover_loveda_samples(args.data_root)
        random.Random(args.sample_seed).shuffle(samples)
        external = None
    elif args.dataset in ("udd5", "oem"):
        from eval_cver_external import (
            add_non_residual_metric, class_names, discover_samples,
            load_rgb as external_rgb, load_target, residual_names,
        )
        external = (add_non_residual_metric, class_names, external_rgb, load_target, residual_names)
        samples = discover_samples(args.dataset, Path(args.data_root))
        if args.max_images > 0:
            random.Random(args.sample_seed).shuffle(samples)
    else:
        from dinotool.rs_external import (
            add_non_residual_metric, class_names, discover_samples,
            load_rgb as external_rgb, load_target, residual_names,
        )
        external = (add_non_residual_metric, class_names, external_rgb, load_target, residual_names)
        samples = discover_samples(args.dataset, Path(args.data_root))
        if args.max_images > 0:
            random.Random(args.sample_seed).shuffle(samples)
    if args.max_images > 0:
        samples = samples[:args.max_images]
    global_keys = [s.key for s in samples]
    samples = samples[args.shard_index::args.num_shards]
    if not samples:
        raise ValueError("Empty shard.")
    vocabulary_path = Path(args.vocabulary_config).resolve()
    vocabulary = load_class_specs(vocabulary_path)
    if args.dataset == "loveda":
        specs = {p: protocol_classes(p, vocabulary) for p in ("P", "D")}
    else:
        _, class_names, _, _, _ = external
        specs = {args.dataset: vocabulary}
        expected_names = (
            ("other", "wall", "road", "vegetation", "vehicle", "roof", "water")
            if args.dataset == "vdd" and args.vdd_ontology == "official"
            else class_names(args.dataset)
        )
        if tuple(s.name for s in vocabulary) != expected_names:
            raise ValueError("Vocabulary order differs from the locked label mapping.")
    if any(len(s.synonyms) != 20 for values in specs.values() for s in values):
        raise ValueError("Exactly 20 aliases per class are required.")
    config = config_type(context_extent=args.context_extent)
    config.validate()
    contextual = ContextualPhraseConfig()
    tcpr = TCPRConfig(
        maximum_aliases_per_class=20,
        geometry_depth=args.geometry_depth,
        prefix_policy=args.prefix_policy,
    )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise ValueError("Use a new output directory; do not overwrite an evaluation.")
    work = output / ".accumulators"
    work.mkdir(exist_ok=True)
    (output / "per_image").mkdir(exist_ok=True)
    checkpoints = make_checkpoints(args)
    model = model_type(
        DINOTextSegmenter(checkpoints, device=args.device, amp=not args.no_amp),
        contextual, tcpr, config,
    )
    banks = {p: model.encode_text(classes) for p, classes in specs.items()}
    counts = {p: [int((bank.parent_indices == c).sum()) for c in range(bank.class_count)]
              for p, bank in banks.items()}
    if any(n != 20 for values in counts.values() for n in values):
        raise ValueError("Encoded alias count changed.")
    digest = lambda keys: hashlib.sha256("\n".join(keys).encode()).hexdigest()
    names = {p: list(bank.class_names) for p, bank in banks.items()}
    signature = {
        "implementation": implementation, "dataset": args.dataset,
        "methods": list(methods), "classes": names,
        "sample_keys": [s.key for s in samples], "sample_count": len(samples),
        "sample_keys_sha256": digest([s.key for s in samples]),
        "global_sample_count": len(global_keys), "global_sample_keys_sha256": digest(global_keys),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "config": {k: v for k, v in vars(args).items() if k != "device"},
        "grounded": asdict(config), "contextual_phrase": asdict(contextual), "tcpr": asdict(tcpr),
        "vocabulary": {"sha256": hashlib.sha256(vocabulary_path.read_bytes()).hexdigest(),
                       "classes": {p: serialize_class_specs(s) for p, s in specs.items()},
                       "alias_counts": counts},
        "checkpoints": checkpoint_manifest(checkpoints),
        "protocol_note": "Exploratory validation. Frozen weights and fixed vocabularies. "
                         "Earlier target evaluation informed the hypothesis, not inference inputs.",
    }
    write_json(output / "signature.json", signature)
    matrices = {p: {m: Confusion(names[p]) for m in methods} for p in banks}
    changes = {p: {m: _empty_change() for m in methods[1:]} for p in banks}
    proposal_totals = {
        p: {m: {k: 0 for k in ("proposed_changed", "proposed_beneficial", "proposed_harmful",
                                 "beneficial_retained", "harmful_rejected")}
            for m in methods if m not in (methods[0], "BoundedUnion")}
        for p in banks
    } if "BoundedUnion" in methods else {}
    diagnostics = {p: {"tiles": 0} for p in banks}
    forward_counts = {"local": 0, "context": 0}
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()
    for index, sample in enumerate(samples, 1):
        image = load_rgb(sample.image_path) if args.dataset == "loveda" else external[2](sample, args.dataset)
        predictions, forwards = predict_image(model, image, banks, args, work, diagnostics, methods)
        for key, value in forwards.items():
            forward_counts[key] += value
        image_metrics = {}
        for p in banks:
            target = target_ids(sample.mask_path, p) if args.dataset == "loveda" else external[3](
                sample, args.dataset, tuple(image.shape[-2:]))
            image_metrics[p] = {}
            for m in methods:
                metric = Confusion(names[p])
                metric.update(predictions[p][m], target)
                matrices[p][m].matrix += metric.matrix
                matrices[p][m].ignored += metric.ignored
                image_metrics[p][m] = metric.summary()
                if m != methods[0]:
                    _update_change(changes[p][m], predictions[p][methods[0]], predictions[p][m], target)
                if p in proposal_totals and m in proposal_totals[p]:
                    counts = proposal_quality(predictions[p][methods[0]],
                                              predictions[p]["BoundedUnion"],
                                              predictions[p][m], target, len(names[p]))
                    for key, count in counts.items():
                        proposal_totals[p][m][key] += count
        write_json(output / "per_image" / f"{index:05}.json",
                   {"key": sample.key, "shape": list(image.shape[-2:]), "metrics": image_metrics})
        if index % args.progress_every == 0 or index == len(samples):
            metrics = {p: {m: matrices[p][m].summary() for m in methods} for p in banks}
            if args.dataset != "loveda":
                add_non_residual_metric, _, _, _, residual_names = external
                for summary in metrics[args.dataset].values():
                    add_non_residual_metric(summary, tuple(names[args.dataset]), residual_names(args.dataset))
            result = {
                "status": "complete" if index == len(samples) else "running",
                "processed_images": index, "total_images": len(samples), "metrics": metrics,
                "changes_vs_geometry": {p: {m: _change_summary(v) for m, v in c.items()}
                                        for p, c in changes.items()},
                "proposal_quality": {
                    p: {m: proposal_quality_summary(v) for m, v in arms.items()}
                    for p, arms in proposal_totals.items()
                },
                "diagnostics": {p: {k: v if k == "tiles" else v / max(d["tiles"], 1)
                                    for k, v in d.items()} for p, d in diagnostics.items()},
                "forward_counts": forward_counts,
                "wall_seconds": time.perf_counter() - started,
                "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(model.device) / 1048576
                                       if model.device.type == "cuda" else 0.0,
                "signature": signature,
            }
            write_json(output / "results.json", result)
            print(json.dumps({"dataset": args.dataset, "processed": index, "total": len(samples),
                              "metrics": {p: {m: x["mean_iou_percent"] for m, x in v.items()}
                                          for p, v in metrics.items()}}), flush=True)
    return result


if __name__ == "__main__":
    main(parse_args())
