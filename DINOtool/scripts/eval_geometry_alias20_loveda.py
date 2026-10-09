#!/usr/bin/env python3
"""Compare Geometry readout with top-8 and all-20 LoveDA aliases."""
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
from dinotool.tcpr import TCPRConfig, TCPRPreparedImage, TCPRTextBank, TCPRSegmenter
from dinotool.tcpr import aggregate_alias_scores, filter_aliases, _normalize_valid_mask, _tokens_to_map

from eval_stride_ov_loveda_e1 import (
    Confusion, D_CLASSES, P_CLASSES, crop_tile, load_rgb, make_checkpoints,
    patch_valid_mask, save_prediction, target_ids,
)


METHODS = ("N_all20_uniform", "G_top8", "G_all20_weighted", "G_all20_uniform")
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
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def protocol_classes(protocol: str, vocabulary: list[ClassSpec]) -> list[ClassSpec]:
    names = P_CLASSES if protocol == "P" else D_CLASSES
    by_name = {spec.name: spec for spec in vocabulary}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Expanded vocabulary is missing LoveDA classes: {missing}")
    return [by_name[name] for name in names]


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


def _uniform_alias_weights(bank: TCPRTextBank, batch: int, device: torch.device) -> torch.Tensor:
    weights = torch.zeros((batch, bank.features.shape[0]), dtype=torch.float32, device=device)
    for class_index in range(bank.class_count):
        members = bank.parent_indices == class_index
        weights[:, members] = 1.0 / int(members.sum())
    return weights


@torch.inference_mode()
def geometry_readouts(
    prepared: TCPRPreparedImage,
    bank: TCPRTextBank,
    top8_config: TCPRConfig,
    all20_config: TCPRConfig,
    valid_mask: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Compare native and Geometry readouts under matched alias aggregation."""
    visual = F.normalize(prepared.geometry_projected.float(), dim=-1)
    batch, patches, _ = visual.shape
    valid = _normalize_valid_mask(
        valid_mask, batch, prepared.grid_height, prepared.grid_width, visual.device
    ).reshape(batch, patches)
    text = F.normalize(bank.features.float(), dim=-1)
    geometry_alias = visual @ text.T
    native_alias = prepared.native_projected.float() @ text.T
    weights_top8 = filter_aliases(
        geometry_alias, native_alias, bank.parent_indices, bank.canonical_mask,
        bank.class_count, valid, top8_config,
    )
    weights_all20 = filter_aliases(
        geometry_alias, native_alias, bank.parent_indices, bank.canonical_mask,
        bank.class_count, valid, all20_config,
    )
    weights_uniform = _uniform_alias_weights(bank, batch, visual.device)
    outputs: dict[str, torch.Tensor] = {}
    weights_by_method = {
        "N_all20_uniform": weights_uniform,
        "G_top8": weights_top8,
        "G_all20_weighted": weights_all20,
        "G_all20_uniform": weights_uniform,
    }
    for method, weights in weights_by_method.items():
        alias_scores = native_alias if method == "N_all20_uniform" else geometry_alias
        scores = aggregate_alias_scores(
            alias_scores, weights, bank.parent_indices, bank.class_count,
            top8_config.alias_temperature,
        )
        outputs[method] = _tokens_to_map(scores, prepared.grid_height, prepared.grid_width)
    diagnostics = {
        "alias_counts": {
            method: [
                int(((bank.parent_indices == cls) & (weights_by_method[method][0] > 0)).sum())
                for cls in range(bank.class_count)
            ]
            for method in METHODS
        },
        "weight_sums_per_class": {
            method: [
                round(float(weights_by_method[method][0, bank.parent_indices == cls].sum().item()), 7)
                for cls in range(bank.class_count)
            ]
            for method in METHODS
        },
    }
    return outputs, diagnostics


@torch.inference_mode()
def predict_image(
    model: TCPRSegmenter,
    image_path: Path,
    banks: dict[str, TCPRTextBank],
    args: argparse.Namespace,
    top8_config: TCPRConfig,
    all20_config: TCPRConfig,
    work_dir: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    image = load_rgb(image_path)
    height, width = image.shape[-2:]
    starts_x = tile_starts(width, args.tile_size, args.overlap)
    starts_y = tile_starts(height, args.tile_size, args.overlap)
    blend = hann_blend_window(args.tile_size)
    diagnostics: dict[str, object] = {}
    with ExitStack() as stack:
        accumulators = {
            (protocol, method): stack.enter_context(
                ProbabilityAccumulator(
                    len(P_CLASSES if protocol == "P" else D_CLASSES), height, width, 2048, work_dir
                )
            )
            for protocol in banks for method in METHODS
        }
        for top in starts_y:
            for left in starts_x:
                actual_h = min(args.tile_size, height - top)
                actual_w = min(args.tile_size, width - left)
                rgb = crop_tile(image, left, top, args.tile_size).to(model.device, non_blocking=True)
                valid = patch_valid_mask(actual_h, actual_w, args.tile_size, model.patch_size, model.device)
                prepared = model.prepare_image(rgb)
                weights = blend[:actual_h, :actual_w]
                for protocol, bank in banks.items():
                    logits_by_method, readout_diagnostics = geometry_readouts(
                        prepared, bank, top8_config, all20_config, valid
                    )
                    diagnostics.setdefault(protocol, readout_diagnostics)
                    for method, logits in logits_by_method.items():
                        dense = F.interpolate(
                            logits, size=(args.tile_size, args.tile_size),
                            mode="bilinear", align_corners=False,
                        )
                        probability = torch.softmax(dense.float() / args.output_temperature, dim=1)
                        accumulators[(protocol, method)].add(
                            probability[0, :, :actual_h, :actual_w].cpu().numpy(), weights, left, top
                        )
        return {
            protocol: {
                method: accumulators[(protocol, method)].finalize(None)[0]
                for method in METHODS
            }
            for protocol in banks
        }, diagnostics


def _empty_change() -> dict[str, int]:
    return {"valid": 0, "changed": 0, "beneficial": 0, "harmful": 0, "wrong_to_wrong": 0}


def _update_change(values: dict[str, int], base: np.ndarray, output: np.ndarray, target: np.ndarray) -> None:
    valid = target != 255
    changed = valid & (base != output)
    base_ok = base == target
    output_ok = output == target
    values["valid"] += int(valid.sum())
    values["changed"] += int(changed.sum())
    values["beneficial"] += int((changed & ~base_ok & output_ok).sum())
    values["harmful"] += int((changed & base_ok & ~output_ok).sum())
    values["wrong_to_wrong"] += int((changed & ~base_ok & ~output_ok).sum())


def _change_summary(values: dict[str, int]) -> dict[str, float | int]:
    valid = max(values["valid"], 1)
    changed = max(values["changed"], 1)
    return {
        **values,
        "changed_fraction": values["changed"] / valid,
        "beneficial_fraction": values["beneficial"] / valid,
        "harmful_fraction": values["harmful"] / valid,
        "beneficial_among_changes": values["beneficial"] / changed,
        "harmful_among_changes": values["harmful"] / changed,
        "wrong_to_wrong_among_changes": values["wrong_to_wrong"] / changed,
    }


def main(args: argparse.Namespace) -> dict[str, object]:
    if args.tile_size % 16 or not 0 <= args.overlap < args.tile_size:
        raise ValueError("tile-size must be patch aligned and overlap must be smaller than it.")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= shard-index < num-shards.")
    samples = discover_loveda_samples(args.data_root)
    random.Random(args.sample_seed).shuffle(samples)
    if args.max_images > 0:
        samples = samples[:args.max_images]
    global_keys = [sample.key for sample in samples]
    global_hash = hashlib.sha256("\n".join(global_keys).encode()).hexdigest()
    samples = samples[args.shard_index::args.num_shards]
    sample_hash = hashlib.sha256("\n".join(sample.key for sample in samples).encode()).hexdigest()

    vocabulary_path = Path(args.vocabulary_config).resolve()
    vocabulary = load_class_specs(vocabulary_path)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work_dir = output / ".accumulators"
    work_dir.mkdir(exist_ok=True)
    checkpoints = make_checkpoints(args)
    backbone = DINOTextSegmenter(checkpoints, device=args.device, amp=not args.no_amp)
    top8_config = TCPRConfig(maximum_aliases_per_class=8)
    all20_config = TCPRConfig(maximum_aliases_per_class=20)
    model = TCPRSegmenter(backbone, top8_config)
    specs = {protocol: protocol_classes(protocol, vocabulary) for protocol in ("P", "D")}
    banks = {protocol: model.encode_text(values) for protocol, values in specs.items()}
    alias_counts = {
        protocol: [int((bank.parent_indices == cls).sum()) for cls in range(bank.class_count)]
        for protocol, bank in banks.items()
    }
    if any(count != 20 for counts in alias_counts.values() for count in counts):
        raise ValueError(f"Expected 20 unique aliases per class, found {alias_counts}.")
    matrices = {
        (protocol, method): Confusion(P_CLASSES if protocol == "P" else D_CLASSES)
        for protocol in ("P", "D") for method in METHODS
    }
    changes = {
        (protocol, method): _empty_change()
        for protocol in ("P", "D") for method in METHODS if method != "G_top8"
    }
    signature = {
        "method": "Geometry-only alias-count intervention",
        "date": "2026-09-26",
        "sample_count": len(samples),
        "sample_keys_sha256": sample_hash,
        "methods": list(METHODS),
        "global_sample_count": len(global_keys),
        "global_sample_keys_sha256": global_hash,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "sample_keys": [sample.key for sample in samples],
        "config": {key: value for key, value in vars(args).items() if key != "device"},
        "top8_config": asdict(top8_config),
        "all20_config": asdict(all20_config),
        "vocabulary": {
            "path": str(vocabulary_path),
            "sha256": hashlib.sha256(vocabulary_path.read_bytes()).hexdigest(),
            "expanded": {protocol: serialize_class_specs(values) for protocol, values in specs.items()},
            "alias_counts_per_class": alias_counts,
        },
        "checkpoints": checkpoint_manifest(checkpoints),
        "protocol_note": (
            "All arms share frozen DINO.text, text embeddings, resolution, tiling, and aggregation. "
            "N_all20_uniform reads native DINO self-attention features; Geometry arms read the "
            "structurally modified features. G_top8 uses image-wise reliability top-8; "
            "G_all20_weighted retains all aliases with the same reliability weights; "
            "G_all20_uniform retains all aliases uniformly."
        ),
    }
    signature_path = output / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Output directory is bound to another experiment signature.")
    signature_path.write_text(json.dumps(signature, indent=2, ensure_ascii=True))

    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()
    tile_diagnostics: dict[str, object] = {}
    for index, sample in enumerate(samples, start=1):
        with Image.open(sample.image_path) as image:
            shape = (image.height, image.width)
        predictions: dict[str, dict[str, np.ndarray]] = {"P": {}, "D": {}}
        complete = True
        for protocol in ("P", "D"):
            class_count = len(P_CLASSES if protocol == "P" else D_CLASSES)
            for method in METHODS:
                cached = _load_prediction(output / "predictions" / protocol / method / sample.key, shape, class_count)
                complete = complete and cached is not None
                if cached is not None:
                    predictions[protocol][method] = cached
        if not complete:
            predictions, tile_diagnostics = predict_image(
                model, sample.image_path, banks, args, top8_config, all20_config, work_dir
            )
            for protocol in ("P", "D"):
                for method, prediction in predictions[protocol].items():
                    save_prediction(output / "predictions" / protocol / method / sample.key, prediction)

        for protocol in ("P", "D"):
            target = target_ids(sample.mask_path, protocol)
            base = predictions[protocol]["G_top8"]
            for method in METHODS:
                matrices[(protocol, method)].update(predictions[protocol][method], target)
                if method != "G_top8":
                    _update_change(changes[(protocol, method)], base, predictions[protocol][method], target)

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
                "delta_miou_vs_top8": {
                    protocol: {
                        method: round(metrics[protocol][method]["mean_iou_percent"] - metrics[protocol]["G_top8"]["mean_iou_percent"], 4)
                        for method in METHODS
                    }
                    for protocol in ("P", "D")
                },
                "changes_vs_top8": {
                    protocol: {
                        method: _change_summary(changes[(protocol, method)])
                        for method in METHODS if method != "G_top8"
                    }
                    for protocol in ("P", "D")
                },
                "readout_diagnostics": tile_diagnostics,
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
                "P": {method: metrics["P"][method]["mean_iou_percent"] for method in METHODS},
                "D": {method: metrics["D"][method]["mean_iou_percent"] for method in METHODS},
            }), flush=True)
    return result


if __name__ == "__main__":
    main(parse_args())
