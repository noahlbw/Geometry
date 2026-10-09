#!/usr/bin/env python3
"""Evaluate TCPR and its Geometry base on the locked LoveDA E1 P/D sample."""
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
from dinotool.prompts import ClassSpec, load_class_specs, serialize_class_specs
from dinotool.tcpr import TCPRConfig, TCPRSegmenter

from eval_stride_ov_loveda_e1 import (
    Confusion,
    D_CLASSES,
    P_CLASSES,
    crop_tile,
    load_rgb,
    make_checkpoints,
    patch_valid_mask,
    save_prediction,
    target_ids,
)


METHODS = ("G_geometry", "TCPR")
DEFAULT_VOCABULARY = Path(__file__).resolve().parents[1] / "configs" / "tcpr_loveda_vip_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov3-repo", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocabulary-config", default=str(DEFAULT_VOCABULARY))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--output-temperature", type=float, default=0.07)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def protocol_classes(protocol: str, vocabulary: list[ClassSpec]) -> list[ClassSpec]:
    names = P_CLASSES if protocol == "P" else D_CLASSES
    by_name = {spec.name: spec for spec in vocabulary}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"VIP-style vocabulary is missing LoveDA classes: {missing}")
    return [by_name[name] for name in names]


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


class DiagnosticTotals:
    def __init__(self) -> None:
        self.tiles = 0
        self.reference_coverage = 0.0
        self.evidence_coverage = 0.0
        self.evidence_mass = 0.0
        self.changed = 0.0
        self.pilot_disagreement = 0.0

    def add(self, summary: dict[str, object]) -> None:
        self.tiles += 1
        self.reference_coverage += float(summary["reference_class_coverage"])
        self.evidence_coverage += float(summary["evidence_patch_coverage"])
        self.evidence_mass += float(summary["mean_evidence_mass"])
        self.changed += float(summary["changed_from_geometry"])
        self.pilot_disagreement += float(summary["mean_geometry_native_disagreement"])

    def summary(self) -> dict[str, float | int | None]:
        if not self.tiles:
            return {"new_tiles": 0}
        return {
            "new_tiles": self.tiles,
            "mean_reference_class_coverage": self.reference_coverage / self.tiles,
            "mean_evidence_patch_coverage": self.evidence_coverage / self.tiles,
            "mean_evidence_mass": self.evidence_mass / self.tiles,
            "mean_changed_from_geometry": self.changed / self.tiles,
            "mean_geometry_native_disagreement": self.pilot_disagreement / self.tiles,
        }


@torch.inference_mode()
def predict_image(
    model: TCPRSegmenter,
    image_path: Path,
    text_banks: dict[str, object],
    args: argparse.Namespace,
    work_dir: Path,
    diagnostics: dict[str, DiagnosticTotals],
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
                    height, width, 2048, work_dir,
                )
            )
            for protocol in text_banks for method in METHODS
        }
        for top in starts_y:
            for left in starts_x:
                actual_h = min(args.tile_size, height - top)
                actual_w = min(args.tile_size, width - left)
                rgb = crop_tile(image, left, top, args.tile_size).to(model.device, non_blocking=True)
                valid = patch_valid_mask(
                    actual_h, actual_w, args.tile_size, model.patch_size, model.device
                )
                prepared = model.prepare_image(rgb)
                weights = blend[:actual_h, :actual_w]
                for protocol, text_bank in text_banks.items():
                    result = model.read_prepared(prepared, text_bank, valid_mask=valid)
                    diagnostics[protocol].add(result.diagnostics.summary())
                    for method, logits in (
                        ("G_geometry", result.geometry_logits),
                        ("TCPR", result.logits),
                    ):
                        dense = F.interpolate(
                            logits, size=(args.tile_size, args.tile_size),
                            mode="bilinear", align_corners=False,
                        )
                        probability = torch.softmax(dense.float() / args.output_temperature, dim=1)
                        accumulators[(protocol, method)].add(
                            probability[0, :, :actual_h, :actual_w].cpu().numpy(),
                            weights, left, top,
                        )
        return {
            protocol: {
                method: accumulators[(protocol, method)].finalize(None)[0]
                for method in METHODS
            }
            for protocol in text_banks
        }


def _empty_changes() -> dict[str, int]:
    return {
        "valid": 0,
        "both_correct": 0,
        "geometry_only_correct": 0,
        "tcpr_only_correct": 0,
        "both_wrong": 0,
        "changed": 0,
        "beneficial": 0,
        "harmful": 0,
        "wrong_to_wrong": 0,
    }


def _update_changes(values: dict[str, int], geometry: np.ndarray, tcpr: np.ndarray, target: np.ndarray) -> None:
    valid = target != 255
    geometry_ok = valid & (geometry == target)
    tcpr_ok = valid & (tcpr == target)
    changed = valid & (geometry != tcpr)
    values["valid"] += int(valid.sum())
    values["both_correct"] += int((geometry_ok & tcpr_ok).sum())
    values["geometry_only_correct"] += int((geometry_ok & ~tcpr_ok).sum())
    values["tcpr_only_correct"] += int((~geometry_ok & tcpr_ok).sum())
    values["both_wrong"] += int((valid & ~geometry_ok & ~tcpr_ok).sum())
    values["changed"] += int(changed.sum())
    values["beneficial"] += int((changed & ~geometry_ok & tcpr_ok).sum())
    values["harmful"] += int((changed & geometry_ok & ~tcpr_ok).sum())
    values["wrong_to_wrong"] += int((changed & ~geometry_ok & ~tcpr_ok).sum())


def _change_summary(values: dict[str, int]) -> dict[str, int | float]:
    valid = max(values["valid"], 1)
    changed = max(values["changed"], 1)
    result: dict[str, int | float] = dict(values)
    for key in ("both_correct", "geometry_only_correct", "tcpr_only_correct", "both_wrong", "changed"):
        result[f"{key}_fraction"] = values[key] / valid
    for key in ("beneficial", "harmful", "wrong_to_wrong"):
        result[f"{key}_among_changes"] = values[key] / changed
    return result


def main(args: argparse.Namespace) -> dict[str, object]:
    if args.tile_size % 16 or not 0 <= args.overlap < args.tile_size:
        raise ValueError("tile-size must be patch aligned and overlap must be smaller than it.")
    samples = discover_loveda_samples(args.data_root)
    random.Random(args.sample_seed).shuffle(samples)
    samples = samples[:args.max_images]
    sample_hash = hashlib.sha256("\n".join(sample.key for sample in samples).encode()).hexdigest()
    vocabulary_path = Path(args.vocabulary_config).resolve()
    vocabulary = load_class_specs(vocabulary_path)
    vocabulary_hash = hashlib.sha256(vocabulary_path.read_bytes()).hexdigest()

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work_dir = output / ".accumulators"
    work_dir.mkdir(exist_ok=True)
    checkpoints = make_checkpoints(args)
    backbone = DINOTextSegmenter(checkpoints, device=args.device, amp=not args.no_amp)
    config = TCPRConfig()
    model = TCPRSegmenter(backbone, config)
    classes = {protocol: protocol_classes(protocol, vocabulary) for protocol in ("P", "D")}
    text_banks = {protocol: model.encode_text(specs) for protocol, specs in classes.items()}
    matrices = {
        (protocol, method): Confusion(P_CLASSES if protocol == "P" else D_CLASSES)
        for protocol in ("P", "D") for method in METHODS
    }
    changes = {protocol: _empty_changes() for protocol in ("P", "D")}
    diagnostics = {protocol: DiagnosticTotals() for protocol in ("P", "D")}
    signature = {
        "method": "TCPR v1 training-free LoveDA E1",
        "date": "2026-09-24",
        "sample_count": len(samples),
        "sample_keys_sha256": sample_hash,
        "methods": list(METHODS),
        "config": {key: value for key, value in vars(args).items() if key != "device"},
        "tcpr": asdict(config),
        "vocabulary": {
            "path": str(vocabulary_path),
            "sha256": vocabulary_hash,
            "classes": {protocol: serialize_class_specs(specs) for protocol, specs in classes.items()},
            "note": "Fixed VIP-style candidates; canonical retained; image-conditioned filtering; no LLM call.",
        },
        "checkpoints": checkpoint_manifest(checkpoints),
        "protocol_note": "P excludes background. D includes background; both share one backbone/pilot pass per tile.",
    }
    signature_path = output / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Output directory is bound to a different experiment signature.")
    signature_path.write_text(json.dumps(signature, indent=2, ensure_ascii=True))

    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        with Image.open(sample.image_path) as image:
            shape = (image.height, image.width)
        predictions: dict[str, dict[str, np.ndarray]] = {"P": {}, "D": {}}
        complete = True
        for protocol in predictions:
            class_count = len(P_CLASSES if protocol == "P" else D_CLASSES)
            for method in METHODS:
                cached = _load_prediction(
                    output / "predictions" / protocol / method / sample.key,
                    shape, class_count,
                )
                complete = complete and cached is not None
                if cached is not None:
                    predictions[protocol][method] = cached
        if not complete:
            predictions = predict_image(model, sample.image_path, text_banks, args, work_dir, diagnostics)
            for protocol in predictions:
                for method, prediction in predictions[protocol].items():
                    save_prediction(output / "predictions" / protocol / method / sample.key, prediction)

        for protocol in ("P", "D"):
            target = target_ids(sample.mask_path, protocol)
            for method in METHODS:
                matrices[(protocol, method)].update(predictions[protocol][method], target)
            _update_changes(
                changes[protocol], predictions[protocol]["G_geometry"],
                predictions[protocol]["TCPR"], target,
            )

        if index % max(args.progress_every, 1) == 0 or index == len(samples):
            metrics = {
                protocol: {method: matrices[(protocol, method)].summary() for method in METHODS}
                for protocol in ("P", "D")
            }
            result = {
                "status": "complete" if index == len(samples) else "running",
                "processed_images": index,
                "total_images": len(samples),
                "metrics": metrics,
                "tcpr_minus_geometry_miou": {
                    protocol: round(
                        metrics[protocol]["TCPR"]["mean_iou_percent"]
                        - metrics[protocol]["G_geometry"]["mean_iou_percent"], 4,
                    )
                    for protocol in ("P", "D")
                },
                "complementarity": {
                    protocol: _change_summary(changes[protocol]) for protocol in ("P", "D")
                },
                "diagnostics": {
                    protocol: diagnostics[protocol].summary() for protocol in ("P", "D")
                },
                "wall_seconds": round(time.perf_counter() - started, 3),
                "peak_cuda_memory_mb": (
                    round(torch.cuda.max_memory_allocated(model.device) / 1048576, 2)
                    if model.device.type == "cuda" else None
                ),
                "signature": signature,
            }
            (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=True))
            print(json.dumps({
                "processed": index,
                "mIoU": {
                    protocol: {method: metrics[protocol][method]["mean_iou_percent"] for method in METHODS}
                    for protocol in ("P", "D")
                },
                "delta": result["tcpr_minus_geometry_miou"],
            }), flush=True)
    return result


if __name__ == "__main__":
    main(parse_args())
