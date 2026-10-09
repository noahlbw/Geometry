#!/usr/bin/env python3
"""Evaluate Connected-Context OV on fixed or full LoveDA P/D protocols."""
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

from dinotool.connected_context_ov import (
    ConnectedContextOVConfig,
    ConnectedContextOVSegmenter,
    support_conditioned_decision,
)
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.loveda import discover_loveda_samples
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.structure_interfaces import StructureInterfaceConfig, build_structure_interfaces

from eval_stride_ov_loveda_e1 import (
    Confusion,
    D_CLASSES,
    P_CLASSES,
    crop_tile,
    make_checkpoints,
    patch_valid_mask,
    protocol_classes,
    save_prediction,
    target_ids,
    load_rgb,
)


METHODS = ("Local", "Context", "Final", "Oracle")
REAL_METHODS = METHODS[:-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov3-repo", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--max-context-regions", type=int, default=32)
    parser.add_argument("--context-batch-size", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


class ComplementTotals:
    def __init__(self) -> None:
        self.valid = 0
        self.both = 0
        self.local_only = 0
        self.context_only = 0
        self.neither = 0

    def update(self, local: np.ndarray, context: np.ndarray, target: np.ndarray) -> np.ndarray:
        valid = target != 255
        local_ok = valid & (local == target)
        context_ok = valid & (context == target)
        self.valid += int(valid.sum())
        self.both += int((local_ok & context_ok).sum())
        self.local_only += int((local_ok & ~context_ok).sum())
        self.context_only += int((~local_ok & context_ok).sum())
        self.neither += int((valid & ~local_ok & ~context_ok).sum())
        return np.where(local_ok | context_ok, target, local).astype(np.uint8)

    def summary(self) -> dict[str, float | int]:
        denominator = max(self.valid, 1)
        return {
            "valid_pixels": self.valid,
            "both_correct_percent": round(self.both / denominator * 100.0, 4),
            "local_only_correct_percent": round(self.local_only / denominator * 100.0, 4),
            "context_only_correct_percent": round(self.context_only / denominator * 100.0, 4),
            "both_wrong_percent": round(self.neither / denominator * 100.0, 4),
            "pixel_oracle_accuracy_percent": round((self.both + self.local_only + self.context_only) / denominator * 100.0, 4),
        }


def _load_prediction(path: Path, shape: tuple[int, int], classes: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        value = np.asarray(Image.open(path)).copy()
    except OSError:
        return None
    if value.ndim != 2 or value.shape != shape or np.any(value >= classes):
        return None
    return value


def _accumulate_diagnostics(total: dict[str, float], diagnostics) -> None:
    for item in diagnostics:
        total["tiles"] += 1
        total["supports"] += item.supports
        total["context_regions"] += item.context_regions
        total["context_coverage"] += item.context_coverage
        total["branch_disagreement"] += item.branch_disagreement
        total["mean_js_divergence"] += item.mean_js_divergence
        total["final_changed_from_local"] += item.final_changed_from_local
        total["final_kept_local_against_context"] += item.final_kept_local_against_context
        total["local_visual_coherence"] += item.local_visual_coherence
        total["context_visual_coherence"] += item.context_visual_coherence
        total["final_visual_coherence"] += item.final_visual_coherence


def _diagnostic_summary(total: dict[str, float]) -> dict[str, float]:
    count = max(total["tiles"], 1.0)
    return {key if key == "tiles" else f"mean_{key}": round(value if key == "tiles" else value / count, 6)
            for key, value in total.items()}


@torch.inference_mode()
def predict_image(
    model: ConnectedContextOVSegmenter,
    image_path: Path,
    d_text: torch.Tensor,
    args: argparse.Namespace,
    work_dir: Path,
    diagnostic_totals: dict[str, dict[str, float]],
) -> dict[str, dict[str, np.ndarray]]:
    image = load_rgb(image_path)
    height, width = image.shape[-2:]
    starts_x = tile_starts(width, args.tile_size, args.overlap)
    starts_y = tile_starts(height, args.tile_size, args.overlap)
    blend = hann_blend_window(args.tile_size)
    with ExitStack() as stack:
        accumulators = {
            (protocol, method): stack.enter_context(
                ProbabilityAccumulator(
                    len(P_CLASSES if protocol == "P" else D_CLASSES),
                    height,
                    width,
                    2048,
                    work_dir,
                )
            )
            for protocol in ("P", "D") for method in REAL_METHODS
        }
        for top in starts_y:
            for left in starts_x:
                actual_h = min(args.tile_size, height - top)
                actual_w = min(args.tile_size, width - left)
                rgb = crop_tile(image, left, top, args.tile_size).to(model.device, non_blocking=True)
                local_features, raw_features, _ = model.backbone.encode_image_with_structure(rgb)
                valid = patch_valid_mask(actual_h, actual_w, args.tile_size, model.patch_size, model.device)
                partition = build_structure_interfaces(
                    raw_features[0],
                    StructureInterfaceConfig(model.config.atom_similarity_threshold),
                    valid,
                )
                d_local = model.backbone.similarity_logits(local_features, d_text).float()
                d_context, context_valid = model._context_observations(rgb, d_text, (partition,))
                protocol_scores = {
                    "D": (d_local, d_context),
                    "P": (d_local[:, 1:], d_context[:, 1:]),
                }
                weights = blend[:actual_h, :actual_w]
                for protocol, (local_logits, context_logits) in protocol_scores.items():
                    final_logits, diagnostics = support_conditioned_decision(
                        local_logits,
                        context_logits,
                        context_valid,
                        raw_features,
                        (partition,),
                        model.config,
                    )
                    _accumulate_diagnostics(diagnostic_totals[protocol], diagnostics)
                    filled_context = torch.where(context_valid[:, None], context_logits, local_logits)
                    score_maps = {
                        "Local": local_logits / model.config.branch_temperature,
                        "Context": filled_context / model.config.branch_temperature,
                        "Final": final_logits,
                    }
                    for method, patch_logits in score_maps.items():
                        dense = F.interpolate(
                            patch_logits,
                            size=(args.tile_size, args.tile_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                        probabilities = torch.softmax(dense.float(), dim=1)
                        accumulators[(protocol, method)].add(
                            probabilities[0, :, :actual_h, :actual_w].cpu().numpy(),
                            weights,
                            left,
                            top,
                        )
        return {
            protocol: {
                method: accumulators[(protocol, method)].finalize(None)[0]
                for method in REAL_METHODS
            }
            for protocol in ("P", "D")
        }


def main(args: argparse.Namespace) -> dict[str, object]:
    if args.tile_size % 16 or not 0 <= args.overlap < args.tile_size:
        raise ValueError("Invalid patch-aligned tile or overlap setting.")
    samples = discover_loveda_samples(args.data_root)
    random.Random(args.sample_seed).shuffle(samples)
    samples = samples[: args.max_images] if args.max_images > 0 else samples
    sample_hash = hashlib.sha256("\n".join(sample.key for sample in samples).encode()).hexdigest()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = output / ".accumulators"
    work.mkdir(exist_ok=True)

    checkpoints = make_checkpoints(args)
    backbone = DINOTextSegmenter(checkpoints, device=args.device, amp=not args.no_amp)
    config = ConnectedContextOVConfig(
        context_size=args.context_size,
        maximum_context_regions=args.max_context_regions,
        context_batch_size=args.context_batch_size,
    )
    model = ConnectedContextOVSegmenter(backbone=backbone, config=config)
    d_text = model.encode_text(protocol_classes("D"))
    matrices = {
        (protocol, method): Confusion(P_CLASSES if protocol == "P" else D_CLASSES)
        for protocol in ("P", "D") for method in METHODS
    }
    complement = {protocol: ComplementTotals() for protocol in ("P", "D")}
    diagnostic_keys = (
        "tiles", "supports", "context_regions", "context_coverage", "branch_disagreement",
        "mean_js_divergence", "final_changed_from_local", "final_kept_local_against_context",
        "local_visual_coherence", "context_visual_coherence", "final_visual_coherence",
    )
    diagnostics = {protocol: {key: 0.0 for key in diagnostic_keys} for protocol in ("P", "D")}
    signature = {
        "method": "Connected-Context OV training-free LoveDA evaluation",
        "sample_count": len(samples),
        "sample_keys_sha256": sample_hash,
        "config": {key: value for key, value in vars(args).items() if key != "device"},
        "model": asdict(config),
        "checkpoints": checkpoint_manifest(checkpoints),
        "protocol_note": "P excludes background pixels and labels. D includes background; D foreground mIoU excludes it only from averaging.",
    }
    signature_path = output / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Output directory is bound to a different experiment signature.")
    signature_path.write_text(json.dumps(signature, indent=2, ensure_ascii=True))

    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()
    result: dict[str, object] = {}
    for index, sample in enumerate(samples, start=1):
        with Image.open(sample.image_path) as image:
            shape = (image.height, image.width)
        predictions: dict[str, dict[str, np.ndarray]] = {"P": {}, "D": {}}
        complete = True
        for protocol in ("P", "D"):
            classes = len(P_CLASSES if protocol == "P" else D_CLASSES)
            for method in REAL_METHODS:
                loaded = _load_prediction(output / "predictions" / protocol / method / sample.key, shape, classes)
                complete = complete and loaded is not None
                if loaded is not None:
                    predictions[protocol][method] = loaded
        if not complete:
            predictions = predict_image(model, sample.image_path, d_text, args, work, diagnostics)
            for protocol, per_method in predictions.items():
                for method, prediction in per_method.items():
                    save_prediction(output / "predictions" / protocol / method / sample.key, prediction)

        for protocol in ("P", "D"):
            target = target_ids(sample.mask_path, protocol)
            oracle = complement[protocol].update(
                predictions[protocol]["Local"], predictions[protocol]["Context"], target
            )
            predictions[protocol]["Oracle"] = oracle
            for method in METHODS:
                matrices[(protocol, method)].update(predictions[protocol][method], target)

        if index % max(args.progress_every, 1) == 0 or index == len(samples):
            result = {
                "status": "complete" if index == len(samples) else "running",
                "processed_images": index,
                "total_images": len(samples),
                "metrics": {
                    protocol: {method: matrices[(protocol, method)].summary() for method in METHODS}
                    for protocol in ("P", "D")
                },
                "complementarity": {protocol: complement[protocol].summary() for protocol in ("P", "D")},
                "diagnostics": {protocol: _diagnostic_summary(diagnostics[protocol]) for protocol in ("P", "D")},
                "wall_seconds": round(time.perf_counter() - started, 3),
                "peak_cuda_memory_mb": round(torch.cuda.max_memory_allocated(model.device) / 1048576, 2)
                if model.device.type == "cuda" else None,
                "signature": signature,
            }
            (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=True))
            print(json.dumps({
                "processed": index,
                "P": {method: result["metrics"]["P"][method]["mean_iou_percent"] for method in METHODS},
                "D": {method: result["metrics"]["D"][method]["mean_iou_percent"] for method in METHODS},
                "seconds": result["wall_seconds"],
            }), flush=True)
    return result


if __name__ == "__main__":
    main(parse_args())
