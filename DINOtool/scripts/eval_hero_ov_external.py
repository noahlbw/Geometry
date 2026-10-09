#!/usr/bin/env python3
"""Evaluate frozen Geometry-20 and HERO-OV on UDD5 or OpenEarthMap validation."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from dinotool.hypothesis_readout import HEROConfig, HEROSegmenter, deterministic_alias_split
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.oem import (
    OEM_CLASSES,
    OEM_RAW_TO_CLASS_INDEX,
    _read_mask as read_oem_mask,
    _read_rgb as read_oem_rgb,
    _validate_raw_labels,
    discover_oem_samples,
)
from dinotool.prompts import load_class_specs, serialize_class_specs
from dinotool.tcpr import TCPRConfig

from eval_competitive_evidence_loveda_e1 import (
    _change_summary,
    _empty_change,
    _update_change,
)
from eval_stride_ov_loveda_e1 import (
    Confusion,
    crop_tile,
    make_checkpoints,
    patch_valid_mask,
)


METHODS = ("G_all20_uniform", "HERO_Cross", "HERO_Consensus", "HERO_SameGroup")
UDD5_CLASSES = ("vegetation", "building", "road", "vehicle", "other")
OEM_CLASS_NAMES = tuple(spec.name for spec in OEM_CLASSES)


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path
    key: str


class Diagnostics:
    FIELDS = (
        "mean_route_kl",
        "maximum_route_kl",
        "fold_sign_agreement",
        "mean_cross_delta",
        "mean_consensus_delta",
        "changed_cross_from_geometry",
        "changed_consensus_from_geometry",
    )

    def __init__(self) -> None:
        self.tiles = 0
        self.values = {field: 0.0 for field in self.FIELDS}

    def add(self, value) -> None:
        self.tiles += 1
        summary = value.summary()
        for field in self.FIELDS:
            if field == "maximum_route_kl":
                self.values[field] = max(self.values[field], summary[field])
            else:
                self.values[field] += summary[field]

    def summary(self) -> dict[str, float | int]:
        count = max(self.tiles, 1)
        return {
            "tiles": self.tiles,
            **{
                field: value if field == "maximum_route_kl" else value / count
                for field, value in self.values.items()
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("udd5", "oem"), required=True)
    parser.add_argument("--dinov3-repo", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocabulary-config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--output-temperature", type=float, default=0.07)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def discover_samples(dataset: str, root: Path) -> list[Sample]:
    if dataset == "oem":
        return [
            Sample(item.image_path, item.mask_path, item.key)
            for item in discover_oem_samples(root, "val", allow_missing_images=True)
        ]
    metadata = root / "metadata" / "val.txt"
    if root.name != "UDD5" or not metadata.is_file():
        raise ValueError("UDD5 data-root must contain metadata/val.txt.")
    samples: list[Sample] = []
    for line_number, line in enumerate(metadata.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Invalid UDD5 val row {line_number}: {line!r}")
        image_path, mask_path = (root / fields[0]).resolve(), (root / fields[1]).resolve()
        if not image_path.is_file() or not mask_path.is_file() or image_path.stem != mask_path.stem:
            raise ValueError(f"Invalid UDD5 pair at row {line_number}.")
        samples.append(Sample(image_path, mask_path, image_path.name))
    if len(samples) != 40 or len({item.key for item in samples}) != 40:
        raise ValueError(f"Expected 40 unique UDD5 validation images, found {len(samples)}.")
    return samples


def class_names(dataset: str) -> tuple[str, ...]:
    return UDD5_CLASSES if dataset == "udd5" else OEM_CLASS_NAMES


def residual_names(dataset: str) -> tuple[str, ...]:
    return ("other",) if dataset == "udd5" else ()


def load_rgb(sample: Sample, dataset: str) -> torch.Tensor:
    if dataset == "oem":
        array = np.asarray(read_oem_rgb(sample.image_path), dtype=np.uint8)
    else:
        with Image.open(sample.image_path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)


def load_target(sample: Sample, dataset: str, shape: tuple[int, int]) -> np.ndarray:
    if dataset == "udd5":
        with Image.open(sample.mask_path) as mask:
            target = np.asarray(mask, dtype=np.uint8).copy()
        unexpected = sorted(set(np.unique(target).tolist()) - set(range(len(UDD5_CLASSES))))
        if unexpected:
            raise ValueError(f"Unexpected UDD5 label IDs for {sample.key}: {unexpected}")
    else:
        raw = read_oem_mask(sample.mask_path)
        _validate_raw_labels(raw, sample.key)
        lut = np.full(256, 255, dtype=np.uint8)
        for raw_id, index in OEM_RAW_TO_CLASS_INDEX.items():
            lut[raw_id] = index
        target = lut[raw]
    if target.shape != shape:
        raise ValueError(f"Image/mask size mismatch for {sample.key}: {shape} != {target.shape}")
    return target.astype(np.int64, copy=False)


@torch.inference_mode()
def predict_image(
    model: HEROSegmenter,
    image: torch.Tensor,
    text_bank,
    args: argparse.Namespace,
    work_dir: Path,
    diagnostics: Diagnostics,
) -> dict[str, np.ndarray]:
    height, width = image.shape[-2:]
    starts_x = tile_starts(width, args.tile_size, args.overlap)
    starts_y = tile_starts(height, args.tile_size, args.overlap)
    blend = hann_blend_window(args.tile_size)
    with ExitStack() as stack:
        accumulators = {
            method: stack.enter_context(
                ProbabilityAccumulator(
                    text_bank.class_count, height, width, 2048, work_dir
                )
            )
            for method in METHODS
        }
        for top in starts_y:
            for left in starts_x:
                actual_h = min(args.tile_size, height - top)
                actual_w = min(args.tile_size, width - left)
                rgb = crop_tile(image, left, top, args.tile_size).to(
                    model.device, non_blocking=True
                )
                valid = patch_valid_mask(
                    actual_h, actual_w, args.tile_size, model.patch_size, model.device
                )
                result = model.read_prepared(model.prepare_image(rgb), text_bank, valid_mask=valid)
                diagnostics.add(result.diagnostics)
                logits_by_method = {
                    "G_all20_uniform": result.geometry_logits,
                    "HERO_Cross": result.cross_logits,
                    "HERO_Consensus": result.consensus_logits,
                    "HERO_SameGroup": result.same_group_logits,
                }
                weights = blend[:actual_h, :actual_w]
                for method, logits in logits_by_method.items():
                    dense = F.interpolate(
                        logits,
                        size=(args.tile_size, args.tile_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    probability = torch.softmax(
                        dense.float() / args.output_temperature, dim=1
                    )
                    accumulators[method].add(
                        probability[0, :, :actual_h, :actual_w].cpu().numpy(),
                        weights,
                        left,
                        top,
                    )
        return {method: accumulators[method].finalize(None)[0] for method in METHODS}


def add_non_residual_metric(summary: dict[str, object], names: tuple[str, ...], residual: tuple[str, ...]) -> None:
    if not residual:
        return
    excluded = {name.casefold() for name in residual}
    values = [
        item["iou_percent"] for item in summary["per_class"]
        if item["name"].casefold() not in excluded and item["iou_percent"] is not None
    ]
    summary["non_residual_mean_iou_percent"] = round(float(np.mean(values)), 4)


def main(args: argparse.Namespace) -> dict[str, object]:
    if args.tile_size % 16 or not 0 <= args.overlap < args.tile_size:
        raise ValueError("tile-size must be patch aligned and overlap smaller than tile-size.")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= shard-index < num-shards.")
    dataset_root = Path(args.data_root).resolve()
    all_samples = discover_samples(args.dataset, dataset_root)
    if args.max_images > 0:
        all_samples = all_samples[: args.max_images]
    global_keys = [sample.key for sample in all_samples]
    global_hash = hashlib.sha256("\n".join(global_keys).encode()).hexdigest()
    samples = all_samples[args.shard_index :: args.num_shards]
    names = class_names(args.dataset)
    residual = residual_names(args.dataset)

    vocabulary_path = Path(args.vocabulary_config).resolve()
    specs = load_class_specs(vocabulary_path)
    if tuple(spec.name for spec in specs) != names:
        raise ValueError(
            f"Vocabulary order must exactly match {args.dataset} classes: "
            f"{tuple(spec.name for spec in specs)} != {names}"
        )
    if any(len(spec.synonyms) != 20 for spec in specs):
        raise ValueError("External HERO evaluation requires exactly 20 aliases per class.")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work_dir = output / ".accumulators"
    work_dir.mkdir(exist_ok=True)
    checkpoints = make_checkpoints(args)
    backbone = DINOTextSegmenter(checkpoints, device=args.device, amp=not args.no_amp)
    hero_config = HEROConfig(residual_class_names=residual)
    tcpr_config = TCPRConfig(maximum_aliases_per_class=20)
    model = HEROSegmenter(backbone, hero_config, tcpr_config)
    text_bank = model.encode_text(specs)
    alias_counts = [
        int((text_bank.parent_indices == index).sum())
        for index in range(text_bank.class_count)
    ]
    split = deterministic_alias_split(text_bank, hero_config.split_salt)
    split_counts = [
        [
            int((split[fold] & (text_bank.parent_indices == index)).sum())
            for index in range(text_bank.class_count)
        ]
        for fold in range(2)
    ]
    matrices = {method: Confusion(names) for method in METHODS}
    changes = {method: _empty_change() for method in METHODS if method != METHODS[0]}
    diagnostics = Diagnostics()
    sample_hash = hashlib.sha256("\n".join(item.key for item in samples).encode()).hexdigest()
    signature = {
        "method": "Frozen Geometry-20 and HERO-OV external dataset evaluation",
        "date": "2026-09-26",
        "dataset": args.dataset,
        "split": "val",
        "sample_count": len(samples),
        "sample_keys_sha256": sample_hash,
        "sample_keys": [sample.key for sample in samples],
        "global_sample_count": len(all_samples),
        "global_sample_keys_sha256": global_hash,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "methods": list(METHODS),
        "classes": list(names),
        "residual_class_names": list(residual),
        "config": {key: value for key, value in vars(args).items() if key != "device"},
        "hero": asdict(hero_config),
        "tcpr": asdict(tcpr_config),
        "vocabulary": {
            "path": str(vocabulary_path),
            "sha256": hashlib.sha256(vocabulary_path.read_bytes()).hexdigest(),
            "classes": serialize_class_specs(specs),
            "alias_counts_per_class": alias_counts,
            "split_counts_per_class": split_counts,
        },
        "checkpoints": checkpoint_manifest(checkpoints),
        "protocol_note": (
            "Frozen DINO.text, Geometry and HERO; 20 fixed aliases per class; native "
            "resolution sliding-window inference; no dataset labels used for parameter or "
            "hyperparameter selection. Labels are read only by the evaluator."
        ),
    }
    signature_path = output / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Output directory is bound to another experiment signature.")
    signature_path.write_text(json.dumps(signature, indent=2, ensure_ascii=True))

    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()
    result: dict[str, object] = {}
    for index, sample in enumerate(samples, 1):
        image = load_rgb(sample, args.dataset)
        target = load_target(sample, args.dataset, tuple(image.shape[-2:]))
        predictions = predict_image(model, image, text_bank, args, work_dir, diagnostics)
        base = predictions[METHODS[0]]
        for method, prediction in predictions.items():
            matrices[method].update(prediction, target)
            if method != METHODS[0]:
                _update_change(changes[method], base, prediction, target)
        if index % max(args.progress_every, 1) == 0 or index == len(samples):
            metrics = {method: matrices[method].summary() for method in METHODS}
            for summary in metrics.values():
                add_non_residual_metric(summary, names, residual)
            result = {
                "status": "complete" if index == len(samples) else "running",
                "processed_images": index,
                "total_images": len(samples),
                "metrics": metrics,
                "delta_miou_vs_geometry": {
                    method: round(
                        metrics[method]["mean_iou_percent"]
                        - metrics[METHODS[0]]["mean_iou_percent"],
                        4,
                    )
                    for method in METHODS
                },
                "changes_vs_geometry": {
                    method: _change_summary(changes[method])
                    for method in METHODS if method != METHODS[0]
                },
                "diagnostics": diagnostics.summary(),
                "wall_seconds": round(time.perf_counter() - started, 3),
                "peak_cuda_memory_mb": (
                    round(torch.cuda.max_memory_allocated(model.device) / 1048576, 2)
                    if model.device.type == "cuda" else None
                ),
                "signature": signature,
            }
            (output / "results.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=True)
            )
            print(json.dumps({
                "dataset": args.dataset,
                "processed": index,
                "metrics": {
                    method: metrics[method]["mean_iou_percent"] for method in METHODS
                },
            }), flush=True)
    return result


if __name__ == "__main__":
    main(parse_args())
