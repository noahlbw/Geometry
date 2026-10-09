#!/usr/bin/env python3
"""Evaluate STRIDE-OV on the fixed 64-image LoveDA E1 P/D protocol."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from dinotool.config import CheckpointConfig
from dinotool.contrast_flow import solve_contrast_flow
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.interface_budget import build_interface_budgets
from dinotool.loveda import discover_loveda_samples
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import ClassSpec
from dinotool.stride_ov import StrideOVConfig, _restrict_state_to_valid_mask
from dinotool.structure_interfaces import build_structure_interfaces
from dinotool.tlp import apply_tlp_state, build_tlp_state


METHODS = ("B0_raw", "B1_tlp", "STRIDE_OV")
P_CLASSES = ("building", "road", "water", "barren", "tree", "farm")
D_CLASSES = ("background", *P_CLASSES)


class Confusion:
    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = names
        self.matrix = np.zeros((len(names), len(names)), dtype=np.int64)
        self.ignored = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        valid = (target >= 0) & (target < len(self.names))
        self.ignored += int((~valid).sum())
        encoded = target[valid].astype(np.int64) * len(self.names) + prediction[valid].astype(np.int64)
        self.matrix += np.bincount(encoded, minlength=self.matrix.size).reshape(self.matrix.shape)

    def summary(self) -> dict[str, object]:
        intersection = np.diag(self.matrix)
        target = self.matrix.sum(1)
        predicted = self.matrix.sum(0)
        union = target + predicted - intersection
        iou = np.divide(intersection, union, out=np.full(len(self.names), np.nan), where=union > 0)
        result: dict[str, object] = {
            "mean_iou_percent": round(float(np.nanmean(iou)) * 100.0, 4),
            "pixel_accuracy_percent": round(float(intersection.sum() / max(target.sum(), 1)) * 100.0, 4),
            "per_class": [
                {"name": name, "iou_percent": None if np.isnan(iou[i]) else round(float(iou[i]) * 100.0, 4)}
                for i, name in enumerate(self.names)
            ],
            "confusion_matrix": self.matrix.tolist(),
            "ignored_pixels": self.ignored,
        }
        if self.names[0] == "background":
            result["foreground_mean_iou_percent"] = round(float(np.nanmean(iou[1:])) * 100.0, 4)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov3-repo", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--output-temperature", type=float, default=0.07)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--early-stop-after", type=int, default=16)
    parser.add_argument("--early-stop-margin", type=float, default=1.0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def make_checkpoints(args: argparse.Namespace) -> CheckpointConfig:
    root = Path(args.checkpoint_dir).resolve()
    return CheckpointConfig(
        dinov3_repo=Path(args.dinov3_repo).resolve(),
        checkpoint_dir=root,
        dinotxt_weights=root / "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
        lvd_weights=root / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        sat_weights=root / "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        bpe_path=root / "bpe_simple_vocab_16e6.txt.gz",
    )


def protocol_classes(protocol: str) -> list[ClassSpec]:
    names = P_CLASSES if protocol == "P" else D_CLASSES
    return [ClassSpec.from_name(name) for name in names]


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def crop_tile(image: torch.Tensor, left: int, top: int, tile: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    source = image[:, top : min(top + tile, height), left : min(left + tile, width)].unsqueeze(0)
    pad_h = tile - source.shape[-2]
    pad_w = tile - source.shape[-1]
    if pad_h or pad_w:
        source = F.pad(source, (0, pad_w, 0, pad_h), mode="replicate")
    return source


def patch_valid_mask(actual_h: int, actual_w: int, tile: int, patch: int, device: torch.device) -> torch.Tensor:
    height = tile // patch
    width = tile // patch
    valid_h = min(height, math.ceil(actual_h / patch))
    valid_w = min(width, math.ceil(actual_w / patch))
    valid = torch.zeros((height, width), device=device, dtype=torch.bool)
    valid[:valid_h, :valid_w] = True
    return valid


class DiagnosticTotals:
    def __init__(self) -> None:
        self.tiles = 0
        self.atoms = 0
        self.interfaces = 0
        self.eligible_interfaces = 0
        self.active_budgets = 0
        self.constrained_edges = 0
        self.flow_iterations = 0
        self.nonconverged = 0
        self.max_capacity_violation = 0.0
        self.changed_patches = 0
        self.total_patches = 0

    def summary(self) -> dict[str, float | int]:
        count = max(self.tiles, 1)
        return {
            "tiles": self.tiles,
            "mean_atoms": round(self.atoms / count, 3),
            "mean_interfaces": round(self.interfaces / count, 3),
            "mean_eligible_interfaces": round(self.eligible_interfaces / count, 3),
            "mean_active_budgets": round(self.active_budgets / count, 3),
            "mean_constrained_edges": round(self.constrained_edges / count, 3),
            "mean_flow_iterations": round(self.flow_iterations / count, 3),
            "nonconverged_tiles": self.nonconverged,
            "maximum_capacity_violation": self.max_capacity_violation,
            "changed_patch_fraction_vs_tlp": round(self.changed_patches / max(self.total_patches, 1), 7),
        }


@torch.inference_mode()
def predict_image(
    model: DINOTextSegmenter,
    image_path: Path,
    text: dict[str, torch.Tensor],
    args: argparse.Namespace,
    config: StrideOVConfig,
    work_dir: Path,
    diagnostics: dict[str, DiagnosticTotals],
) -> dict[str, np.ndarray]:
    image = load_rgb(image_path)
    height, width = image.shape[-2:]
    starts_x = tile_starts(width, args.tile_size, args.overlap)
    starts_y = tile_starts(height, args.tile_size, args.overlap)
    blend = hann_blend_window(args.tile_size)
    with ExitStack() as stack:
        accumulators = {
            protocol: stack.enter_context(
                ProbabilityAccumulator(len(P_CLASSES if protocol == "P" else D_CLASSES), height, width, 2048, work_dir)
            )
            for protocol in text
        }
        for top in starts_y:
            for left in starts_x:
                actual_h = min(args.tile_size, height - top)
                actual_w = min(args.tile_size, width - left)
                rgb = crop_tile(image, left, top, args.tile_size).to(model.device, non_blocking=True)
                patch_features, raw_features, _ = model.encode_image_with_structure(rgb)
                valid = patch_valid_mask(actual_h, actual_w, args.tile_size, model.patch_size, model.device)
                partition = build_structure_interfaces(raw_features[0], config.structure, valid)
                grid_rgb = F.interpolate(rgb, size=patch_features.shape[-2:], mode="area")
                weights = blend[:actual_h, :actual_w]
                for protocol, text_features in text.items():
                    initial = model.similarity_logits(patch_features, text_features)
                    state = build_tlp_state(initial, grid_rgb, text_features, config.tlp)
                    state = _restrict_state_to_valid_mask(state, valid)
                    proposal, _ = apply_tlp_state(initial, state)
                    budgets, budget_diag = build_interface_budgets(
                        initial, proposal, state, partition, config.budget
                    )
                    output, flow_diag = solve_contrast_flow(
                        initial, proposal, state, partition, budgets, config.flow
                    )
                    dense = F.interpolate(
                        output, size=(args.tile_size, args.tile_size), mode="bilinear", align_corners=False
                    )
                    probabilities = torch.softmax(dense.float() / args.output_temperature, dim=1)
                    accumulators[protocol].add(
                        probabilities[0, :, :actual_h, :actual_w].cpu().numpy(), weights, left, top
                    )
                    totals = diagnostics[protocol]
                    totals.tiles += 1
                    totals.atoms += partition.atom_count
                    totals.interfaces += len(partition.interfaces)
                    totals.eligible_interfaces += budget_diag.eligible_interfaces
                    totals.active_budgets += budget_diag.active_budgets
                    totals.constrained_edges += flow_diag.constrained_edges
                    totals.flow_iterations += flow_diag.iterations
                    totals.nonconverged += int(not flow_diag.converged)
                    totals.max_capacity_violation = max(
                        totals.max_capacity_violation, flow_diag.maximum_capacity_violation
                    )
                    valid_flat = valid.reshape(-1)
                    proposal_label = proposal.argmax(1).reshape(-1)
                    output_label = output.argmax(1).reshape(-1)
                    totals.changed_patches += int(((proposal_label != output_label) & valid_flat).sum().item())
                    totals.total_patches += int(valid_flat.sum().item())
        result = {}
        for protocol, accumulator in accumulators.items():
            labels, _ = accumulator.finalize(None)
            result[protocol] = labels
        return result


def target_ids(path: Path, protocol: str) -> np.ndarray:
    with Image.open(path) as image:
        raw = np.asarray(image).copy()
    if raw.ndim == 3:
        raw = raw[..., 0]
    result = np.full(raw.shape, 255, dtype=np.uint8)
    if protocol == "P":
        valid = (raw >= 2) & (raw <= 7)
        result[valid] = raw[valid] - 2
    else:
        valid = (raw >= 1) & (raw <= 7)
        result[valid] = raw[valid] - 1
    return result


def load_prediction(path: Path, shape: tuple[int, int], classes: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required E1 baseline prediction is missing: {path}")
    prediction = np.asarray(Image.open(path)).copy()
    if prediction.shape != shape or prediction.ndim != 2 or np.any(prediction >= classes):
        raise ValueError(f"Invalid cached prediction: {path}")
    return prediction


def save_prediction(path: Path, prediction: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.png")
    Image.fromarray(prediction, mode="L").save(temporary)
    temporary.replace(path)


def summaries(matrices: dict[tuple[str, str], Confusion]) -> dict[str, dict[str, dict[str, object]]]:
    return {
        protocol: {method: matrices[(protocol, method)].summary() for method in METHODS}
        for protocol in ("P", "D")
    }


def main(args: argparse.Namespace) -> dict[str, object]:
    if args.tile_size % 16 or not 0 <= args.overlap < args.tile_size:
        raise ValueError("Invalid patch-aligned tile or overlap setting.")
    samples = discover_loveda_samples(args.data_root)
    random.Random(args.sample_seed).shuffle(samples)
    samples = samples[: args.max_images]
    sample_hash = hashlib.sha256("\n".join(sample.key for sample in samples).encode()).hexdigest()
    baseline_dir = Path(args.baseline_dir).resolve()
    baseline_signature = json.loads((baseline_dir / "signature.json").read_text())
    if sample_hash != baseline_signature["sample_keys_sha256"]:
        raise ValueError("The requested sample is not the locked LoveDA E1 sample.")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = output / ".accumulators"
    work.mkdir(exist_ok=True)
    config = StrideOVConfig()
    checkpoints = make_checkpoints(args)
    model = DINOTextSegmenter(checkpoints, device=args.device, amp=not args.no_amp)
    text = {protocol: model.encode_text(protocol_classes(protocol)) for protocol in ("P", "D")}
    matrices = {
        (protocol, method): Confusion(P_CLASSES if protocol == "P" else D_CLASSES)
        for protocol in ("P", "D") for method in METHODS
    }
    changes = {
        protocol: {"updated": 0, "beneficial": 0, "harmful": 0, "wrong_to_wrong": 0}
        for protocol in ("P", "D")
    }
    diagnostics = {protocol: DiagnosticTotals() for protocol in ("P", "D")}
    signature = {
        "method": "STRIDE-OV v1 training-free E1 screen",
        "sample_count": len(samples),
        "sample_keys_sha256": sample_hash,
        "baseline_signature": baseline_signature,
        "config": {key: value for key, value in vars(args).items() if key != "device"},
        "stride": asdict(config),
        "checkpoints": checkpoint_manifest(checkpoints),
    }
    signature_path = output / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Output directory has a different experiment signature.")
    signature_path.write_text(json.dumps(signature, indent=2, ensure_ascii=True))

    started = time.perf_counter()
    stopped_reason = None
    processed = 0
    for index, sample in enumerate(samples, start=1):
        with Image.open(sample.image_path) as image:
            shape = (image.height, image.width)
        predictions: dict[str, dict[str, np.ndarray]] = {"P": {}, "D": {}}
        missing_stride = False
        for protocol in ("P", "D"):
            classes = len(P_CLASSES if protocol == "P" else D_CLASSES)
            for method in ("B0_raw", "B1_tlp"):
                predictions[protocol][method] = load_prediction(
                    baseline_dir / "predictions" / protocol / method / sample.key, shape, classes
                )
            stride_path = output / "predictions" / protocol / "STRIDE_OV" / sample.key
            if stride_path.is_file():
                predictions[protocol]["STRIDE_OV"] = load_prediction(stride_path, shape, classes)
            else:
                missing_stride = True
        if missing_stride:
            stride = predict_image(model, sample.image_path, text, args, config, work, diagnostics)
            for protocol in ("P", "D"):
                predictions[protocol]["STRIDE_OV"] = stride[protocol]
                save_prediction(
                    output / "predictions" / protocol / "STRIDE_OV" / sample.key,
                    stride[protocol],
                )

        for protocol in ("P", "D"):
            target = target_ids(sample.mask_path, protocol)
            valid = target != 255
            raw = predictions[protocol]["B0_raw"]
            stride = predictions[protocol]["STRIDE_OV"]
            for method in METHODS:
                matrices[(protocol, method)].update(predictions[protocol][method], target)
            updated = valid & (stride != raw)
            changes[protocol]["updated"] += int(updated.sum())
            changes[protocol]["beneficial"] += int((updated & (raw != target) & (stride == target)).sum())
            changes[protocol]["harmful"] += int((updated & (raw == target) & (stride != target)).sum())
            changes[protocol]["wrong_to_wrong"] += int((updated & (raw != target) & (stride != target)).sum())
        processed = index

        if index % max(args.progress_every, 1) == 0 or index == len(samples):
            metrics = summaries(matrices)
            deltas = {
                protocol: round(
                    metrics[protocol]["STRIDE_OV"]["mean_iou_percent"]
                    - metrics[protocol]["B1_tlp"]["mean_iou_percent"],
                    4,
                )
                for protocol in ("P", "D")
            }
            if index >= args.early_stop_after:
                failed = [protocol for protocol, delta in deltas.items() if delta < -args.early_stop_margin]
                if failed:
                    stopped_reason = (
                        f"STRIDE_OV trails B1_tlp by more than {args.early_stop_margin} mIoU "
                        f"for {','.join(failed)} after {index} images."
                    )
            result = {
                "status": "stopped_early" if stopped_reason else ("complete" if index == len(samples) else "running"),
                "processed_images": index,
                "total_images": len(samples),
                "metrics": metrics,
                "delta_vs_tlp_miou": deltas,
                "change_audit_vs_raw": changes,
                "diagnostics": {protocol: totals.summary() for protocol, totals in diagnostics.items()},
                "wall_seconds": round(time.perf_counter() - started, 3),
                "peak_cuda_memory_mb": round(torch.cuda.max_memory_allocated(model.device) / 1048576, 2),
                "stopped_reason": stopped_reason,
                "signature": signature,
            }
            (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=True))
            print(json.dumps({
                "processed": index,
                "mIoU": {
                    protocol: {method: metrics[protocol][method]["mean_iou_percent"] for method in METHODS}
                    for protocol in ("P", "D")
                },
                "delta_vs_tlp": deltas,
                "status": result["status"],
            }), flush=True)
            if stopped_reason:
                break
    return result


if __name__ == "__main__":
    main(parse_args())
