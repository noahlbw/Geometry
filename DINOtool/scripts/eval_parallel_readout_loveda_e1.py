#!/usr/bin/env python3
"""Screen frozen DINO.text parallel-readout arms on the fixed LoveDA E1 sample."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.loveda import discover_loveda_samples
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.parallel_readout import ParallelReadoutConfig

from eval_stride_ov_loveda_e1 import (
    Confusion,
    D_CLASSES,
    P_CLASSES,
    crop_tile,
    load_rgb,
    make_checkpoints,
    protocol_classes,
    save_prediction,
    target_ids,
)


METHOD_CONFIGS = {
    "N_native": ParallelReadoutConfig(mode="native"),
    "G_geometry": ParallelReadoutConfig(mode="geometry"),
    "Mix_arithmetic": ParallelReadoutConfig(mode="mixture", beta=0.5),
    "Temp_native": ParallelReadoutConfig(mode="temperature", beta=0.5),
    "Joint_log": ParallelReadoutConfig(mode="joint", beta=0.5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov3-repo", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--output-temperature", type=float, default=0.07)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _load_prediction(path: Path, shape: tuple[int, int], classes: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        prediction = np.asarray(Image.open(path)).copy()
    except OSError:
        return None
    if prediction.ndim != 2 or prediction.shape != shape or np.any(prediction >= classes):
        return None
    return prediction


@torch.inference_mode()
def predict_image(
    model: DINOTextSegmenter,
    image_path: Path,
    texts: dict[str, torch.Tensor],
    args: argparse.Namespace,
    work_dir: Path,
    check_native: bool,
) -> tuple[dict[str, dict[str, np.ndarray]], float | None]:
    image = load_rgb(image_path)
    height, width = image.shape[-2:]
    blend = hann_blend_window(args.tile_size)
    starts_x = tile_starts(width, args.tile_size, args.overlap)
    starts_y = tile_starts(height, args.tile_size, args.overlap)
    parity_max_abs: float | None = None
    with ExitStack() as stack:
        accumulators = {
            (protocol, method): stack.enter_context(
                ProbabilityAccumulator(len(P_CLASSES if protocol == "P" else D_CLASSES), height, width, 2048, work_dir)
            )
            for protocol in texts for method in METHOD_CONFIGS
        }
        for top in starts_y:
            for left in starts_x:
                actual_h = min(args.tile_size, height - top)
                actual_w = min(args.tile_size, width - left)
                rgb = crop_tile(image, left, top, args.tile_size).to(model.device, non_blocking=True)
                maps = model.encode_image_parallel_readouts(rgb, METHOD_CONFIGS)
                if check_native and parity_max_abs is None:
                    original, _ = model.encode_image(rgb)
                    parity_max_abs = float((maps["N_native"] - original).abs().max().item())
                    if parity_max_abs > 1e-5:
                        raise RuntimeError(f"Native parallel path differs from frozen vision head: {parity_max_abs:.8g}")
                weights = blend[:actual_h, :actual_w]
                for protocol, text_features in texts.items():
                    for method, patch_features in maps.items():
                        logits = model.similarity_logits(patch_features, text_features)
                        dense = F.interpolate(logits, size=(args.tile_size, args.tile_size), mode="bilinear", align_corners=False)
                        probabilities = torch.softmax(dense.float() / args.output_temperature, dim=1)
                        accumulators[(protocol, method)].add(
                            probabilities[0, :, :actual_h, :actual_w].cpu().numpy(), weights, left, top
                        )
        predictions = {
            protocol: {method: accumulators[(protocol, method)].finalize(None)[0] for method in METHOD_CONFIGS}
            for protocol in texts
        }
    return predictions, parity_max_abs


def main(args: argparse.Namespace) -> dict[str, object]:
    if args.tile_size % 16 or not 0 <= args.overlap < args.tile_size:
        raise ValueError("tile-size must be patch aligned and overlap must be smaller than it.")
    samples = discover_loveda_samples(args.data_root)
    random.Random(args.sample_seed).shuffle(samples)
    samples = samples[: args.max_images]
    sample_hash = hashlib.sha256("\n".join(sample.key for sample in samples).encode()).hexdigest()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work_dir = output / ".accumulators"
    work_dir.mkdir(exist_ok=True)
    model_checkpoints = make_checkpoints(args)
    model = DINOTextSegmenter(model_checkpoints, device=args.device, amp=not args.no_amp)
    texts = {protocol: model.encode_text(protocol_classes(protocol)) for protocol in ("P", "D")}
    matrices = {
        (protocol, method): Confusion(P_CLASSES if protocol == "P" else D_CLASSES)
        for protocol in ("P", "D") for method in METHOD_CONFIGS
    }
    signature = {
        "method": "training-free parallel DINO.text readout E1",
        "sample_count": len(samples),
        "sample_keys_sha256": sample_hash,
        "configs": {name: asdict(config) for name, config in METHOD_CONFIGS.items()},
        "config": {key: value for key, value in vars(args).items() if key != "device"},
        "checkpoints": checkpoint_manifest(model_checkpoints),
        "protocol_note": "P excludes background labels. D includes it; D foreground mIoU only excludes background from class averaging.",
    }
    signature_path = output / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Output directory is bound to a different experiment signature.")
    signature_path.write_text(json.dumps(signature, indent=2, ensure_ascii=True))
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()
    parity_max_abs: float | None = None
    for index, sample in enumerate(samples, start=1):
        with Image.open(sample.image_path) as image:
            shape = (image.height, image.width)
        predictions: dict[str, dict[str, np.ndarray]] = {"P": {}, "D": {}}
        complete = True
        for protocol in predictions:
            classes = len(P_CLASSES if protocol == "P" else D_CLASSES)
            for method in METHOD_CONFIGS:
                loaded = _load_prediction(output / "predictions" / protocol / method / sample.key, shape, classes)
                complete = complete and loaded is not None
                if loaded is not None:
                    predictions[protocol][method] = loaded
        if not complete:
            predictions, parity = predict_image(model, sample.image_path, texts, args, work_dir, parity_max_abs is None)
            if parity is not None:
                parity_max_abs = parity
            for protocol, per_method in predictions.items():
                for method, prediction in per_method.items():
                    save_prediction(output / "predictions" / protocol / method / sample.key, prediction)
        for protocol, per_method in predictions.items():
            target = target_ids(sample.mask_path, protocol)
            for method, prediction in per_method.items():
                matrices[(protocol, method)].update(prediction, target)
        if index % max(args.progress_every, 1) == 0 or index == len(samples):
            result = {
                "status": "complete" if index == len(samples) else "running",
                "processed_images": index,
                "total_images": len(samples),
                "metrics": {
                    protocol: {method: matrices[(protocol, method)].summary() for method in METHOD_CONFIGS}
                    for protocol in ("P", "D")
                },
                "native_parity_max_abs": parity_max_abs,
                "wall_seconds": round(time.perf_counter() - started, 3),
                "peak_cuda_memory_mb": round(torch.cuda.max_memory_allocated(model.device) / 1048576, 2)
                if model.device.type == "cuda" else None,
                "signature": signature,
            }
            (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=True))
            print(json.dumps({
                "processed": index,
                "mIoU": {protocol: {method: result["metrics"][protocol][method]["mean_iou_percent"] for method in METHOD_CONFIGS} for protocol in ("P", "D")},
                "native_parity_max_abs": parity_max_abs,
            }), flush=True)
    return result


if __name__ == "__main__":
    main(parse_args())
