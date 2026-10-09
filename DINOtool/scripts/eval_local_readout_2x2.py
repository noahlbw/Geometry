#!/usr/bin/env python3
"""Matched canonical/20-alias by dense/sparse local Geometry diagnosis."""
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

from dinotool.hypothesis_readout import uniform_subset_scores
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.local_readout_audit import ARMS, error_counts, summarize_error_counts
from dinotool.loveda import discover_loveda_samples
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs, serialize_class_specs
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from eval_competitive_evidence_loveda_e1 import protocol_classes
from eval_grounded_context import write_json
from eval_stride_ov_loveda_e1 import (
    Confusion, crop_tile, load_rgb, make_checkpoints, target_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("vdd", "potsdam", "udd5", "oem", "loveda"), required=True)
    for name in ("dinov3-repo", "checkpoint-dir", "data-root", "vocabulary-config", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.20)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--output-temperature", type=float, default=0.07)
    parser.add_argument("--alias-temperature", type=float, default=0.07)
    parser.add_argument("--geometry-depth", type=int, choices=(1, 2), default=2)
    parser.add_argument("--prefix-policy", choices=("preserve", "block"), default="preserve")
    parser.add_argument("--vdd-ontology", choices=("legacy", "official"), default="legacy")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def protocol(args, vocabulary):
    if args.dataset == "loveda":
        samples = discover_loveda_samples(args.data_root)
        random.Random(args.sample_seed).shuffle(samples)
        specs = {key: protocol_classes(key, vocabulary) for key in ("P", "D")}
        return samples, specs, load_rgb, lambda sample, key, shape: target_ids(sample.mask_path, key)
    if args.dataset in ("udd5", "oem"):
        from eval_cver_external import class_names, discover_samples, load_rgb as external_rgb, load_target
    else:
        from dinotool.rs_external import class_names, discover_samples, load_rgb as external_rgb, load_target
    samples = discover_samples(args.dataset, Path(args.data_root))
    expected = (("other", "wall", "road", "vegetation", "vehicle", "roof", "water")
                if args.dataset == "vdd" and args.vdd_ontology == "official"
                else class_names(args.dataset))
    if tuple(spec.name for spec in vocabulary) != expected:
        raise ValueError("Vocabulary order differs from the locked label mapping.")
    return (samples, {args.dataset: vocabulary},
            lambda sample: external_rgb(sample, args.dataset),
            lambda sample, key, shape: load_target(sample, args.dataset, shape))


@torch.inference_mode()
def predict_image(models, image, banks, args, work):
    height, width = image.shape[-2:]
    blend = hann_blend_window(args.tile_size)
    counts = {"tiles": 0, "sparse_active_fraction_sum": 0.0}
    with ExitStack() as stack:
        accumulators = {
            (key, arm): stack.enter_context(ProbabilityAccumulator(
                bank.class_count, height, width, 2048, work
            ))
            for key, bank in banks.items() for arm in ARMS
        }
        for top in tile_starts(height, args.tile_size, args.overlap):
            for left in tile_starts(width, args.tile_size, args.overlap):
                h = min(args.tile_size, height - top)
                w = min(args.tile_size, width - left)
                rgb = crop_tile(image, left, top, args.tile_size).to(models["dense"].device)
                counts["tiles"] += 1
                for relation, model in models.items():
                    prepared = model.prepare_image(rgb)
                    if relation == "sparse":
                        counts["sparse_active_fraction_sum"] += float(
                            (prepared.geometry_patch_conditional > 0).float().mean()
                        )
                    for key, bank in banks.items():
                        alias_scores = prepared.geometry_projected @ F.normalize(bank.features.float(), dim=-1).T
                        for subset_name, subset in (("C", bank.canonical_mask),
                                                    ("A", torch.ones_like(bank.canonical_mask))):
                            arm = f"{subset_name}_{relation}"
                            scores = uniform_subset_scores(
                                alias_scores, bank.parent_indices, bank.class_count,
                                subset, args.alias_temperature,
                            )
                            logits = scores.transpose(1, 2).reshape(
                                1, bank.class_count, prepared.grid_height, prepared.grid_width
                            )
                            dense = F.interpolate(logits, size=(args.tile_size, args.tile_size),
                                                  mode="bilinear", align_corners=False)
                            probability = (dense.float() / args.output_temperature).softmax(1)
                            accumulators[(key, arm)].add(
                                probability[0, :, :h, :w].cpu().numpy(), blend[:h, :w], left, top
                            )
                    del prepared
        predictions = {key: {arm: accumulators[(key, arm)].finalize(None)[0]
                             for arm in ARMS} for key in banks}
    counts["sparse_active_fraction"] = counts.pop("sparse_active_fraction_sum") / counts["tiles"]
    return predictions, counts


def main(args: argparse.Namespace) -> dict[str, object]:
    if args.tile_size != 512 or args.overlap != 128 or args.geometry_depth != 2:
        raise ValueError("This diagnosis fixes 512 tiles, 128 overlap and two head blocks.")
    if not 0 < args.memory_fraction <= 1 or args.progress_every < 1:
        raise ValueError("Invalid memory fraction or progress interval.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index.")
    if args.vdd_ontology != "legacy" and args.dataset != "vdd":
        raise ValueError("Official VDD ontology is only valid for VDD.")
    if torch.device(args.device).type == "cuda":
        device_index = torch.device(args.device).index or 0
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device_index)
    vocabulary_path = Path(args.vocabulary_config).resolve()
    vocabulary = load_class_specs(vocabulary_path)
    samples, specs, load_image, load_mask = protocol(args, vocabulary)
    if args.max_images:
        if args.dataset != "loveda":
            random.Random(args.sample_seed).shuffle(samples)
        samples = samples[:args.max_images]
    global_keys = [sample.key for sample in samples]
    samples = samples[args.shard_index::args.num_shards]
    if not samples:
        raise ValueError("Empty shard.")
    if any(len(spec.synonyms) != 20 for group in specs.values() for spec in group):
        raise ValueError("All classes must have exactly 20 fixed aliases.")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise ValueError("Use a new output directory; do not overwrite a result.")
    work = output / ".accumulators"
    work.mkdir(exist_ok=True)
    checkpoints = make_checkpoints(args)
    backbone = DINOTextSegmenter(checkpoints, device=args.device, amp=not args.no_amp)
    configs = {relation: TCPRConfig(
        maximum_aliases_per_class=20, geometry_depth=2,
        prefix_policy=args.prefix_policy, relation_policy=relation,
    ) for relation in ("dense", "sparse")}
    models = {relation: TCPRSegmenter(backbone, config) for relation, config in configs.items()}
    banks = {key: models["dense"].encode_text(classes) for key, classes in specs.items()}
    alias_counts = {key: [int((bank.parent_indices == index).sum())
                          for index in range(bank.class_count)] for key, bank in banks.items()}
    if any(count != 20 for group in alias_counts.values() for count in group):
        raise ValueError("Encoded alias count changed after deduplication.")
    names = {key: list(bank.class_names) for key, bank in banks.items()}
    digest = lambda keys: hashlib.sha256("\n".join(keys).encode()).hexdigest()
    signature = {
        "implementation": "local-readout-2x2-20260929", "dataset": args.dataset,
        "arms": list(ARMS), "classes": names,
        "sample_keys": [s.key for s in samples], "sample_count": len(samples),
        "sample_keys_sha256": digest([s.key for s in samples]),
        "global_sample_count": len(global_keys), "global_sample_keys_sha256": digest(global_keys),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "config": {key: value for key, value in vars(args).items() if key != "device"},
        "readouts": {name: asdict(config) for name, config in configs.items()},
        "vocabulary": {"sha256": hashlib.sha256(vocabulary_path.read_bytes()).hexdigest(),
                       "classes": {key: serialize_class_specs(group) for key, group in specs.items()},
                       "alias_counts": alias_counts},
        "checkpoints": checkpoint_manifest(checkpoints),
        "note": "Sparse uses per-query cosine > 1.5x mean as a VIP-style connectivity control; "
                "it retains this pipeline's Geometry scores, spatial prior and evaluator. "
                "This is not an official VIP reproduction. GT enters only metrics and error intersections.",
    }
    write_json(output / "signature.json", signature)
    matrices = {key: {arm: Confusion(tuple(group)) for arm in ARMS}
                for key, group in names.items()}
    intersections = {key: {"valid_pixels": 0, "error_patterns": [0] * 16,
                           "error_patterns_by_true_class": [[0] * 16 for _ in group],
                           "pairwise": {}} for key, group in names.items()}
    tile_count = 0
    sparse_fraction_sum = 0.0
    if backbone.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backbone.device)
    started = time.perf_counter()
    for index, sample in enumerate(samples, 1):
        image = load_image(sample.image_path) if args.dataset == "loveda" else load_image(sample)
        predictions, diagnostics = predict_image(models, image, banks, args, work)
        tile_count += diagnostics["tiles"]
        sparse_fraction_sum += diagnostics["tiles"] * diagnostics["sparse_active_fraction"]
        per_image = {}
        for key, group in names.items():
            target = load_mask(sample, key, tuple(image.shape[-2:]))
            for arm in ARMS:
                matrices[key][arm].update(predictions[key][arm], target)
            current = error_counts(predictions[key], target, len(group))
            total = intersections[key]
            total["valid_pixels"] += current["valid_pixels"]
            total["error_patterns"] = [a + b for a, b in zip(total["error_patterns"],
                                                               current["error_patterns"])]
            total["error_patterns_by_true_class"] = [
                [a + b for a, b in zip(row_a, row_b)]
                for row_a, row_b in zip(total["error_patterns_by_true_class"],
                                        current["error_patterns_by_true_class"])
            ]
            for pair, values in current["pairwise"].items():
                saved = total["pairwise"].setdefault(pair, {field: 0 for field in values})
                for field, value in values.items():
                    saved[field] += value
            per_image[key] = {"all_four_wrong_pixels": current["error_patterns"][15],
                              "valid_pixels": current["valid_pixels"]}
        write_json(output / f"image_{index:05}.json", {"key": sample.key, "errors": per_image})
        if index % args.progress_every == 0 or index == len(samples):
            result = {
                "status": "complete" if index == len(samples) else "running",
                "processed_images": index, "total_images": len(samples),
                "metrics": {key: {arm: matrices[key][arm].summary() for arm in ARMS} for key in names},
                "error_intersections": {key: summarize_error_counts(intersections[key], group)
                                        for key, group in names.items()},
                "diagnostics": {"tiles": tile_count,
                                "sparse_active_fraction": sparse_fraction_sum / max(tile_count, 1)},
                "wall_seconds": time.perf_counter() - started,
                "peak_cuda_memory_mb": (torch.cuda.max_memory_allocated(backbone.device) / 1048576
                                        if backbone.device.type == "cuda" else 0.0),
                "signature": signature,
            }
            write_json(output / "results.json", result)
            print(json.dumps({"dataset": args.dataset, "processed": index, "total": len(samples),
                              "miou": {key: {arm: result["metrics"][key][arm]["mean_iou_percent"]
                                             for arm in ARMS} for key in names}}), flush=True)
    return result


if __name__ == "__main__":
    main(parse_args())
