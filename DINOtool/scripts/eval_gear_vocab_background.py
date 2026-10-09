#!/usr/bin/env python3
"""Evaluate VDD/Potsdam vocabulary and background rules with fixed Geometry/Multiscale."""
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

from dinotool.competitive_ownership import CompetitiveOwnershipReadout
from dinotool.shared_family_ownership import SharedFamilyOwnershipReadout
from dinotool.gear_ov import GearOVSegmenter
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.loveda import discover_loveda_samples
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import (ClassSpec, load_class_specs, load_vip_official_aliases,
                             serialize_class_specs)
from eval_competitive_evidence_loveda_e1 import protocol_classes
from eval_grounded_context import write_json
from eval_stride_ov_loveda_e1 import Confusion, load_rgb, make_checkpoints, target_ids


METHODS = ("Geometry", "Multiscale")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("vdd", "potsdam"),
                        required=True)
    for name in ("dinov3-repo", "checkpoint-dir", "data-root", "vocabulary-config", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.60)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--output-temperature", type=float, default=0.07)
    parser.add_argument("--vdd-ontology", choices=("legacy", "official"), default="official")
    parser.add_argument("--vip-official-classes", action="store_true",
                        help="Use the exact VIP text-file query groups without adding canonical names.")
    parser.add_argument("--exact-alias-groups", action="store_true",
                        help="Use JSON alias groups exactly, including when the class name is absent.")
    parser.add_argument("--competitive-ownership", action="store_true",
                        help="Evaluate the frozen full-bank competitive alias readout as COR.")
    parser.add_argument("--shared-family-ownership", action="store_true",
                        help="Evaluate the family-shared ownership readout on the matched views.")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def digest(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def protocol(args: argparse.Namespace, vocabulary):
    if args.dataset == "loveda":
        samples = discover_loveda_samples(args.data_root)
        random.Random(args.sample_seed).shuffle(samples)
        specs = {key: protocol_classes(key, vocabulary) for key in ("P", "D")}
        return samples, specs, load_rgb, lambda sample, key, shape: target_ids(sample.mask_path, key)
    if args.dataset in ("udd5", "oem"):
        from eval_cver_external import class_names, discover_samples
        from eval_cver_external import load_rgb as external_rgb, load_target
    elif args.dataset in ("flair1", "landcoverai"):
        from dinotool.gear_datasets import class_names, discover_samples
        from dinotool.gear_datasets import load_rgb as external_rgb, load_target
    else:
        from dinotool.rs_external import class_names, discover_samples
        from dinotool.rs_external import load_rgb as external_rgb, load_target
    samples = discover_samples(args.dataset, Path(args.data_root))
    expected = (("other", "wall", "road", "vegetation", "vehicle", "roof", "water")
                if args.dataset == "vdd" and args.vdd_ontology == "official"
                else class_names(args.dataset))
    if tuple(spec.name for spec in vocabulary) != expected:
        raise ValueError("Vocabulary order differs from the locked label mapping.")
    return (samples, {args.dataset: vocabulary},
            lambda sample: external_rgb(sample, args.dataset),
            lambda sample, key, shape: load_target(sample, args.dataset, shape))


def predict_image(model: GearOVSegmenter, image: torch.Tensor,
                  banks: dict[str, object], args: argparse.Namespace,
                  work: Path, competitive: dict[str, CompetitiveOwnershipReadout] | None = None,
                  shared: dict[str, SharedFamilyOwnershipReadout] | None = None
                  ) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, float]]]:
    height, width = image.shape[-2:]
    blend = hann_blend_window(args.tile_size)
    totals = {key: {"tiles": 0, "iterations": 0.0, "mean_abs_alias_delta": 0.0,
                    "negative_alias_delta_fraction": 0.0, "support_edges": 0.0,
                    "objective_initial": 0.0, "objective_final": 0.0}
              for key in banks}
    if competitive is not None:
        for values in totals.values():
            values.update(mean_rejection=0.0, mean_abs_correction=0.0,
                          identifiable_alias_fraction=0.0)
    if shared is not None:
        for values in totals.values():
            values.update(mean_rejection=0.0, mean_abs_correction=0.0,
                          mean_unknown=0.0, identifiable_family_fraction=0.0)
    methods = (("Geometry", "Multiscale", "COR_Shared") if shared is not None
               else METHODS + (("COR",) if competitive is not None else ()))
    text = {key: model.text_basis(bank) for key, bank in banks.items()}
    with ExitStack() as stack:
        accumulators = {
            (key, method): stack.enter_context(ProbabilityAccumulator(
                bank.class_count, height, width, 2048, work
            ))
            for key, bank in banks.items() for method in methods
        }
        for top in tile_starts(height, args.tile_size, args.overlap):
            for left in tile_starts(width, args.tile_size, args.overlap):
                views = model.prepare_views(image, top, left)
                weights = blend[:views.actual_height, :views.actual_width]
                for key in banks:
                    result = model.read_views(views, text[key], reconstruct=False)
                    values = {"Geometry": result.local_logits,
                              "Multiscale": result.multiscale_logits}
                    cor_diagnostics = {}
                    if competitive is not None:
                        cor_logits, cor_diagnostics = competitive[key].solve(
                            views, text[key], result.multiscale_logits)
                        values["COR"] = cor_logits
                    if shared is not None:
                        cor_logits, cor_diagnostics = shared[key].solve(
                            views, text[key], result.multiscale_logits)
                        values["COR_Shared"] = cor_logits
                    for method, logits in values.items():
                        probability = (logits.float() / args.output_temperature).softmax(1)
                        accumulators[(key, method)].add(
                            probability[0, :, :views.actual_height, :views.actual_width]
                            .detach().cpu().numpy(), weights, left, top,
                        )
                    totals[key]["tiles"] += 1
                    for field in totals[key]:
                        if field != "tiles":
                            diagnostics = cor_diagnostics if field in cor_diagnostics \
                                else result.diagnostics
                            totals[key][field] += float(diagnostics[field])
                del views
        threshold, background = ((0.35, 0) if args.dataset == "vdd" else (0.25, 5))
        predictions = {}
        for key in banks:
            predictions[key] = {}
            for method in methods:
                accumulator = accumulators[(key, method)]
                predictions[key][method] = accumulator.finalize(None)[0]
                predictions[key][method + "_BG"] = accumulator.finalize(
                    threshold, background
                )[0]
    return predictions, totals


def main(args: argparse.Namespace) -> dict[str, object]:
    if args.tile_size != 512 or args.overlap != 128 or args.output_temperature != 0.07:
        raise ValueError("This evaluation fixes 512 tiles, 128 overlap and output temperature 0.07.")
    if args.competitive_ownership and args.shared_family_ownership:
        raise ValueError("Select only one ownership readout.")
    if not 0 < args.memory_fraction <= 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid memory fraction or shard index.")
    if args.progress_every < 1:
        raise ValueError("Progress interval must be positive.")
    if args.dataset != "vdd" and args.vdd_ontology != "official":
        raise ValueError("VDD ontology flag applies only to VDD.")
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.memory_fraction, torch.device(args.device).index or 0
        )
    vocabulary_path = Path(args.vocabulary_config).resolve()
    if args.vip_official_classes and args.exact_alias_groups:
        raise ValueError("Choose only one exact vocabulary input format.")
    if args.vip_official_classes:
        if args.dataset not in ("vdd", "potsdam"):
            raise ValueError("Exact VIP query groups are supported only for VDD and Potsdam.")
        from dinotool.rs_external import class_names
        names = (("other", "wall", "road", "vegetation", "vehicle", "roof", "water")
                 if args.dataset == "vdd" else class_names(args.dataset))
        groups = load_vip_official_aliases(vocabulary_path)
        if len(groups) != len(names):
            raise ValueError("VIP query group count does not match the dataset taxonomy.")
        vocabulary = [ClassSpec(name, group) for name, group in zip(names, groups)]
    elif args.exact_alias_groups:
        payload = json.loads(vocabulary_path.read_text(encoding="utf-8"))
        vocabulary = [ClassSpec(str(item["name"]), tuple(item["synonyms"]))
                      for item in payload["classes"]]
        if any(not spec.synonyms or len(set(spec.synonyms)) != len(spec.synonyms)
               for spec in vocabulary):
            raise ValueError("Exact alias groups must be nonempty and duplicate-free.")
    else:
        vocabulary = load_class_specs(vocabulary_path)
    all_samples, specs, load_image, load_mask = protocol(args, vocabulary)
    if args.max_images:
        if args.dataset != "loveda":
            random.Random(args.sample_seed).shuffle(all_samples)
        all_samples = all_samples[:args.max_images]
    global_keys = [sample.key for sample in all_samples]
    samples = all_samples[args.shard_index::args.num_shards]
    if not samples:
        raise ValueError("Empty evaluation shard.")
    if not (args.vip_official_classes or args.exact_alias_groups) and any(
        not 1 <= len(spec.synonyms) <= 20 for group in specs.values() for spec in group
    ):
        raise ValueError("Every class must retain between 1 and 20 fixed aliases.")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise ValueError("Use a new output directory; do not overwrite results.")
    work = output / ".accumulators"
    work.mkdir(exist_ok=True)
    checkpoints = make_checkpoints(args)
    backbone = DINOTextSegmenter(checkpoints, device=args.device, amp=not args.no_amp)
    model = GearOVSegmenter(backbone)
    banks = {
        key: (model.encode_alias_groups([spec.name for spec in group],
                                        [spec.synonyms for spec in group])
              if args.vip_official_classes or args.exact_alias_groups
              else model.encode_text(group))
        for key, group in specs.items()
    }
    inference_methods = (("Geometry", "Multiscale", "COR_Shared") if args.shared_family_ownership
                         else METHODS + (("COR",) if args.competitive_ownership else ()))
    methods = inference_methods + tuple(method + "_BG" for method in inference_methods)
    competitive = ({key: CompetitiveOwnershipReadout(bank)
                    for key, bank in banks.items()} if args.competitive_ownership else None)
    shared = ({key: SharedFamilyOwnershipReadout(bank)
               for key, bank in banks.items()} if args.shared_family_ownership else None)
    counts = {key: [int((bank.parent_indices == class_index).sum())
                     for class_index in range(bank.class_count)] for key, bank in banks.items()}
    expected_counts = {key: [len(spec.synonyms) for spec in group]
                       for key, group in specs.items()}
    if counts != expected_counts:
        raise ValueError("Text encoder changed the fixed per-class alias count.")
    names = {key: list(bank.class_names) for key, bank in banks.items()}
    signature = {
        "implementation": "gear-vocabulary-background-local-multiscale-20260930",
        "dataset": args.dataset, "methods": list(methods), "classes": names,
        "sample_keys": [sample.key for sample in samples],
        "sample_count": len(samples), "sample_keys_sha256": digest([s.key for s in samples]),
        "global_sample_count": len(global_keys),
        "global_sample_keys_sha256": digest(global_keys),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "config": {key: value for key, value in vars(args).items() if key != "device"},
        "gear": model.config_dict(),
        "competitive": (asdict(next(iter(competitive.values())).config)
                        if competitive is not None else
                        next(iter(shared.values())).config_dict() if shared is not None else None),
        "vocabulary": {"sha256": hashlib.sha256(vocabulary_path.read_bytes()).hexdigest(),
                       "classes": {key: serialize_class_specs(group) for key, group in specs.items()},
                       "alias_counts": counts,
                       "exact_vip_query_groups": args.vip_official_classes,
                       "exact_alias_groups": args.exact_alias_groups},
        "background_threshold": 0.35 if args.dataset == "vdd" else 0.25,
        "background_index": 0 if args.dataset == "vdd" else 5,
        "checkpoints": checkpoint_manifest(checkpoints),
        "note": "Frozen DINOv3/DINO.text; image-local optimization only. Three views are "
                "shared by Geometry, matched Multiscale and GEAR-OV. Target masks enter metrics only.",
    }
    write_json(output / "signature.json", signature)
    matrices = {key: {method: Confusion(tuple(group)) for method in methods}
                for key, group in names.items()}
    diagnostics = {key: {"tiles": 0, "iterations": 0.0, "mean_abs_alias_delta": 0.0,
                         "negative_alias_delta_fraction": 0.0, "support_edges": 0.0,
                         "objective_initial": 0.0, "objective_final": 0.0}
                   for key in banks}
    if competitive is not None:
        for values in diagnostics.values():
            values.update(mean_rejection=0.0, mean_abs_correction=0.0,
                          identifiable_alias_fraction=0.0)
    if shared is not None:
        for values in diagnostics.values():
            values.update(mean_rejection=0.0, mean_abs_correction=0.0,
                          mean_unknown=0.0, identifiable_family_fraction=0.0)
    if backbone.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backbone.device)
    started = time.perf_counter()
    for number, sample in enumerate(samples, 1):
        image = load_image(sample.image_path) if args.dataset == "loveda" else load_image(sample)
        predictions, current = predict_image(model, image, banks, args, work,
                                             competitive=competitive, shared=shared)
        for key in diagnostics:
            for field in diagnostics[key]:
                diagnostics[key][field] += current[key][field]
            target = load_mask(sample, key, tuple(image.shape[-2:]))
            for method in methods:
                matrices[key][method].update(predictions[key][method], target)
        if number % args.progress_every == 0 or number == len(samples):
            summary = {}
            for key, group in names.items():
                summary[key] = {}
                for method in methods:
                    metric = matrices[key][method].summary()
                    if key in ("udd5", "landcoverai"):
                        values = [item["iou_percent"] for item in metric["per_class"]
                                  if item["name"] not in ("other", "background")
                                  and item["iou_percent"] is not None]
                        metric["non_residual_mean_iou_percent"] = (
                            round(sum(values) / len(values), 4) if values else None
                        )
                    summary[key][method] = metric
            result = {
                "status": "complete" if number == len(samples) else "running",
                "processed_images": number, "total_images": len(samples),
                "metrics": summary, "diagnostics": {
                    key: {field: value if field == "tiles" else value / max(diagnostics[key]["tiles"], 1)
                          for field, value in totals.items()}
                    for key, totals in diagnostics.items()
                },
                "wall_seconds": time.perf_counter() - started,
                "peak_cuda_memory_mb": (torch.cuda.max_memory_allocated(backbone.device) / 1048576
                                        if backbone.device.type == "cuda" else 0.0),
                "signature": signature,
            }
            write_json(output / "results.json", result)
            print(json.dumps({"dataset": args.dataset, "processed": number,
                              "total": len(samples), "miou": {
                                  key: {method: summary[key][method]["mean_iou_percent"]
                                        for method in methods} for key in banks
                              }}), flush=True)
    return result


if __name__ == "__main__":
    main(parse_args())
