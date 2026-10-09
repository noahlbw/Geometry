from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image
import torch

from dinotool.config import CheckpointConfig
from dinotool.fast_dense import FastDenseConfig, FastDenseSession
from dinotool.loveda import LoveDAConfusionMatrix, default_loveda_classes, discover_loveda_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark one-pass DINO dense OVSS on LoveDA validation.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-root")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--dinov3-repo")
    parser.add_argument("--cost-aggregation-checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dino-input-resolution", type=int, default=512)
    parser.add_argument("--output-temperature", type=float, default=0.07)
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_images < 1 or args.start_index < 0:
        raise ValueError("max-images must be positive and start-index must be non-negative.")

    checkpoints = CheckpointConfig.from_roots(args.code_root, args.checkpoint_dir)
    if args.dinov3_repo:
        checkpoints = checkpoints.__class__(
            dinov3_repo=Path(args.dinov3_repo),
            checkpoint_dir=checkpoints.checkpoint_dir,
            dinotxt_weights=checkpoints.dinotxt_weights,
            lvd_weights=checkpoints.lvd_weights,
            sat_weights=checkpoints.sat_weights,
            bpe_path=checkpoints.bpe_path,
        )
    config = FastDenseConfig(
        input_resolution=args.dino_input_resolution,
        output_temperature=args.output_temperature,
        image_cache_size=1,
    )
    session = FastDenseSession(
        checkpoints,
        config,
        device=args.device,
        amp=not args.no_amp,
        cost_aggregation_checkpoint=args.cost_aggregation_checkpoint,
    )
    try:
        samples = discover_loveda_samples(args.data_root)
        selected = samples[args.start_index : args.start_index + args.max_images]
        if not selected:
            raise ValueError("The selected LoveDA range is empty.")
        classes = default_loveda_classes()

        if session.device.type == "cuda":
            torch.cuda.synchronize(session.device)
            model_memory_allocated = torch.cuda.memory_allocated(session.device)
            model_memory_reserved = torch.cuda.memory_reserved(session.device)
            torch.cuda.reset_peak_memory_stats(session.device)

        # Initialize kernels without polluting the selected image cache.  The
        # first measured LoveDA call still includes its actual text encoding.
        warmup_started = time.perf_counter()
        session.warmup()
        if session.device.type == "cuda":
            torch.cuda.synchronize(session.device)
        warmup_seconds = time.perf_counter() - warmup_started

        matrix = LoveDAConfusionMatrix()
        timings: list[float] = []
        image_prepare_seconds: list[float] = []
        cache_hits = 0
        started = time.perf_counter()
        for index, sample in enumerate(selected, start=1):
            with Image.open(sample.image_path) as source:
                image = source.convert("RGB")
            if session.device.type == "cuda":
                torch.cuda.synchronize(session.device)
            image_started = time.perf_counter()
            prediction, _, diagnostics = session.segment(image, classes, image_key=sample.key)
            if session.device.type == "cuda":
                torch.cuda.synchronize(session.device)
            elapsed = time.perf_counter() - image_started
            timings.append(elapsed)
            timing = diagnostics.get("timing_seconds", {})
            if isinstance(timing, dict):
                image_prepare_seconds.append(float(timing.get("image_prepare", 0.0)))
            cache = diagnostics.get("cache", {})
            if isinstance(cache, dict) and cache.get("image_hit"):
                cache_hits += 1
            with Image.open(sample.mask_path) as mask:
                target = np.asarray(mask).copy()
            matrix.update(prediction, target)
            if index % 100 == 0 or index == len(selected):
                print(
                    json.dumps(
                        {
                            "processed": index,
                            "total": len(selected),
                            "mean_seconds": round(statistics.mean(timings), 4),
                            "mIoU": matrix.summary()["mean_iou"],
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )

        wall_seconds = time.perf_counter() - started
        if session.device.type == "cuda":
            torch.cuda.synchronize(session.device)
            peak_allocated = torch.cuda.max_memory_allocated(session.device)
            peak_reserved = torch.cuda.max_memory_reserved(session.device)
        else:
            model_memory_allocated = model_memory_reserved = 0
            peak_allocated = peak_reserved = 0

        result = {
            "method": "DINOv3 dino.txt one-pass dense OVSS",
            "quality_decoder": session.quality_decoder,
            "dataset": "LoveDA validation",
            "data_root": str(Path(args.data_root).expanduser().resolve()),
            "start_index": args.start_index,
            "images": len(selected),
            "input_resolution": args.dino_input_resolution,
            "classes": [spec.name for spec in classes],
            "metrics": matrix.summary(),
            "timing": {
                "warmup_seconds": round(warmup_seconds, 4),
                "wall_seconds": round(wall_seconds, 4),
                "total_inference_seconds": round(sum(timings), 4),
                "mean_seconds": round(statistics.mean(timings), 4),
                "median_seconds": round(statistics.median(timings), 4),
                "p95_seconds": round(_percentile(timings, 95), 4),
                "first_image_seconds": round(timings[0], 4),
                "steady_mean_seconds": round(statistics.mean(timings[1:]), 4) if len(timings) > 1 else None,
                "images_per_second": round(len(timings) / wall_seconds, 4),
                "cache_hits": cache_hits,
                "mean_image_prepare_seconds": round(statistics.mean(image_prepare_seconds), 4),
            },
            "memory_mb": {
                "model_allocated": round(model_memory_allocated / 2**20, 2),
                "model_reserved": round(model_memory_reserved / 2**20, 2),
                "peak_allocated": round(peak_allocated / 2**20, 2),
                "peak_reserved": round(peak_reserved / 2**20, 2),
            },
            "checkpoints": {
                "dinov3_repo": str(checkpoints.dinov3_repo),
                "checkpoint_dir": str(checkpoints.checkpoint_dir),
            },
        }
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=True))
    finally:
        session.close()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


if __name__ == "__main__":
    main()
