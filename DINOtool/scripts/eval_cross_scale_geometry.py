#!/usr/bin/env python3
"""Evaluate the complete frozen Geometry + cross-scale relation model only."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from dinotool.cross_scale_geometry import (
    CrossScaleGeometrySegmenter, DIAGNOSTICS, IMPLEMENTATION, METHOD,
)
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs, serialize_class_specs
from eval_gear_ov import digest, protocol
from eval_grounded_context import write_json
from eval_stride_ov_loveda_e1 import Confusion, make_checkpoints


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("loveda", "udd5", "oem", "vdd", "potsdam",
                                             "vaihingen", "landcoverai", "flair1"), required=True)
    for field in ("dinov3-repo", "checkpoint-dir", "data-root", "vocabulary-config", "output-dir",
                  "reference-manifest"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=.60)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--output-temperature", type=float, default=.07)
    parser.add_argument("--vdd-ontology", choices=("official",), default="official")
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def predict_image(model, image, banks, args, work):
    height, width = image.shape[-2:]
    blend = hann_blend_window(512)
    totals = {"tiles": 0, **dict.fromkeys(DIAGNOSTICS, 0.0)}
    with ExitStack() as stack:
        buffers = {key: stack.enter_context(ProbabilityAccumulator(
            bank.class_count, height, width, 2048, work)) for key, bank in banks.items()}
        for top in tile_starts(height, 512, 128):
            for left in tile_starts(width, 512, 128):
                views = model.prepare_views(image, top, left)
                for key, bank in banks.items():
                    scores = model.read_views(views, bank)
                    if not bool(torch.isfinite(scores).all()):
                        raise ValueError("Non-finite final class scores.")
                    probability = (scores / args.output_temperature).softmax(1)
                    buffers[key].add(probability[0, :, :views.actual_height, :views.actual_width]
                                     .cpu().numpy(),
                                     blend[:views.actual_height, :views.actual_width], left, top)
                totals["tiles"] += 1
                for field in DIAGNOSTICS:
                    totals[field] += views.diagnostics[field]
                del views
        predictions = {key: buffer.finalize(None)[0] for key, buffer in buffers.items()}
    return predictions, totals


def main(args):
    if (args.tile_size, args.overlap, args.output_temperature, args.sample_seed) != (512, 128, .07, 20260923):
        raise ValueError("The complete evaluation uses the fixed historical tiling and seed.")
    if not 0 < args.memory_fraction <= 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid memory or shard settings.")
    if args.max_images < 0 or args.progress_every < 1:
        raise ValueError("Invalid image/progress limits.")
    vocab_path = Path(args.vocabulary_config).resolve()
    vocabulary = load_class_specs(vocab_path)
    samples, specs, load_image, load_mask = protocol(args, vocabulary)
    manifest = json.loads(Path(args.reference_manifest).read_text())[args.dataset]
    vocab_sha = hashlib.sha256(vocab_path.read_bytes()).hexdigest()
    if vocab_sha != manifest["vocabulary_sha256"]:
        raise ValueError("Vocabulary differs from the historical fixed-20 comparison.")
    keys = [sample.key for sample in samples]
    if len(keys) != manifest["global_sample_count"] or digest(keys) != manifest["global_sample_keys_sha256"]:
        raise ValueError("Dataset does not match the complete historical global sample set.")
    if args.max_images:
        if args.dataset != "loveda":
            random.Random(args.sample_seed).shuffle(samples)
        samples = samples[:args.max_images]
    keys = [sample.key for sample in samples]
    shard = samples[args.shard_index::args.num_shards]
    if not shard:
        raise ValueError("Empty evaluation shard.")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise ValueError("Refusing to overwrite an existing result.")
    work = output / ".accumulators"
    work.mkdir(exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device.index or 0)
    checkpoints = make_checkpoints(args)
    backbone = DINOTextSegmenter(checkpoints, device=args.device, amp=not args.no_amp)
    model = CrossScaleGeometrySegmenter(backbone)
    banks = {key: model.encode_text(group) for key, group in specs.items()}
    names = {key: list(bank.class_names) for key, bank in banks.items()}
    counts = {key: [int((bank.parent_indices == c).sum()) for c in range(bank.class_count)]
              for key, bank in banks.items()}
    signature = {
        "implementation": IMPLEMENTATION, "dataset": args.dataset, "methods": [METHOD],
        "classes": names, "sample_keys": [sample.key for sample in shard],
        "sample_count": len(shard), "sample_keys_sha256": digest([sample.key for sample in shard]),
        "global_sample_count": len(keys), "global_sample_keys_sha256": digest(keys),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "config": {key: value for key, value in vars(args).items() if key != "device"},
        "gear": model.config_dict(), "competitive": None,
        "vocabulary": {"sha256": vocab_sha, "alias_counts": counts,
                       "classes": {key: serialize_class_specs(group) for key, group in specs.items()}},
        "checkpoints": checkpoint_manifest(checkpoints),
        "note": "One complete frozen model; fixed20 aliases; six backbone views; "
                "coarse relative relations read fine Values in two frozen head blocks; "
                "target masks enter metrics after predictions only; no parameter selection.",
    }
    write_json(output / "signature.json", signature)
    matrices = {key: Confusion(tuple(group)) for key, group in names.items()}
    totals = {"tiles": 0, **dict.fromkeys(DIAGNOSTICS, 0.0)}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for number, sample in enumerate(shard, 1):
        image = load_image(sample.image_path) if args.dataset == "loveda" else load_image(sample)
        predictions, current = predict_image(model, image, banks, args, work)
        for field in totals:
            totals[field] += current[field]
        for key in banks:
            target = load_mask(sample, key, tuple(image.shape[-2:]))
            matrices[key].update(predictions[key], target)
        if number % args.progress_every == 0 or number == len(shard):
            summary = {}
            for key, matrix in matrices.items():
                metric = matrix.summary()
                if key in ("udd5", "landcoverai"):
                    values = [item["iou_percent"] for item in metric["per_class"]
                              if item["name"] not in ("other", "background")
                              and item["iou_percent"] is not None]
                    metric["non_residual_mean_iou_percent"] = round(sum(values) / len(values), 4)
                summary[key] = {METHOD: metric}
            diagnostics = {field: value if field == "tiles" else value / max(totals["tiles"], 1)
                           for field, value in totals.items()}
            result = {"status": "complete" if number == len(shard) else "running",
                      "processed_images": number, "total_images": len(shard), "metrics": summary,
                      "diagnostics": {key: diagnostics for key in banks},
                      "wall_seconds": time.perf_counter() - started,
                      "peak_cuda_memory_mb": (torch.cuda.max_memory_allocated(device) / 1048576
                                              if device.type == "cuda" else 0),
                      "signature": signature}
            write_json(output / "results.json", result)
            print(json.dumps({"dataset": args.dataset, "processed": number, "total": len(shard),
                              "miou": {key: group[METHOD]["mean_iou_percent"]
                                       for key, group in summary.items()},
                              "grounding_seconds_per_tile": diagnostics["grounding_seconds"]}), flush=True)
    return result


if __name__ == "__main__":
    main(parse_args())
