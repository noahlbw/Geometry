#!/usr/bin/env python3
"""Matched Geometry alias experiments for visual and VIP-prompt vocabularies.

``select`` performs VIP Eq. 3--6 style offline alias distillation on the
unlabeled evaluation images. ``evaluate`` compares all 20 aliases, the current
per-tile GAR weighting, and the frozen offline VIP selection.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs, serialize_class_specs
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from dinotool.vip_alias_distillation import (
    VIPAliasAccumulator,
    VIPDistillationConfig,
    gar_geometry_alias_reliability,
    gar_geometry_alias_weights,
    gar_weighted_alias_scores,
    multilayer_attention_affinity,
    selected_mask_from_report,
)

from eval_competitive_evidence_loveda_e1 import _change_summary, _empty_change, _update_change
from eval_cver_external import (
    add_non_residual_metric,
    class_names,
    discover_samples,
    load_rgb,
    load_target,
    residual_names,
)
from eval_stride_ov_loveda_e1 import Confusion, crop_tile, make_checkpoints, patch_valid_mask
from dinotool.contextual_phrase_readout import masked_geometry
from dinotool.hypothesis_readout import uniform_subset_scores


VOCABULARIES = (("visual", "Visual"), ("vip_prompt", "VIPPrompt"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("select", "evaluate"), required=True)
    parser.add_argument("--dataset", choices=("udd5", "oem"), required=True)
    for name in ("dinov3-repo", "checkpoint-dir", "data-root", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--visual-vocabulary-config", required=True)
    parser.add_argument("--vip-prompt-vocabulary-config", required=True)
    parser.add_argument("--selection-file")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.20)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--output-temperature", type=float, default=0.07)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _vocabulary_inputs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "visual": Path(args.visual_vocabulary_config).resolve(),
        "vip_prompt": Path(args.vip_prompt_vocabulary_config).resolve(),
    }


def _load_banks(model: TCPRSegmenter, args: argparse.Namespace):
    expected = class_names(args.dataset)
    banks, metadata = {}, {}
    for key, path in _vocabulary_inputs(args).items():
        specs = load_class_specs(path)
        if tuple(spec.name for spec in specs) != expected:
            raise ValueError(f"{key} vocabulary order does not match {args.dataset} labels.")
        if any(len(spec.synonyms) != 20 for spec in specs):
            raise ValueError(f"{key} vocabulary must contain exactly 20 aliases per class.")
        bank = model.encode_text(specs)
        counts = [int((bank.parent_indices == index).sum()) for index in range(bank.class_count)]
        if any(count != 20 for count in counts):
            raise ValueError(f"{key} encoded alias counts changed: {counts}")
        banks[key] = bank
        metadata[key] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "classes": serialize_class_specs(specs),
            "alias_counts_per_class": counts,
            "alias_names": list(bank.alias_names),
        }
    return banks, metadata


def _samples(args: argparse.Namespace):
    all_samples = discover_samples(args.dataset, Path(args.data_root).resolve())
    if args.max_images > 0:
        all_samples = all_samples[:args.max_images]
    if not all_samples:
        raise ValueError("No input samples.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index.")
    return all_samples, all_samples[args.shard_index::args.num_shards]


def _signature(args: argparse.Namespace, all_samples, samples, vocabularies, checkpoints) -> dict[str, object]:
    return {
        "implementation": "vip-alias-ablation-v1-20260928",
        "stage": args.stage,
        "dataset": args.dataset,
        "split": "val",
        "sample_count": len(samples),
        "sample_keys": [sample.key for sample in samples],
        "sample_keys_sha256": _digest([sample.key for sample in samples]),
        "global_sample_count": len(all_samples),
        "global_sample_keys_sha256": _digest([sample.key for sample in all_samples]),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "classes": list(class_names(args.dataset)),
        "residual_class_names": list(residual_names(args.dataset)),
        "config": {key: value for key, value in vars(args).items() if key != "device"},
        "vocabularies": vocabularies,
        "checkpoints": checkpoint_manifest(checkpoints),
        "protocol_note": (
            "Frozen DINOv3/DINO.text and fixed native-resolution Geometry inference. "
            "VIP-style distillation reads only unlabeled images from this evaluation split; "
            "masks are read only after prediction for metrics."
        ),
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


def select(args: argparse.Namespace) -> dict[str, object]:
    all_samples, samples = _samples(args)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = make_checkpoints(args)
    if torch.device(args.device).type == "cuda":
        index = torch.device(args.device).index or 0
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, index)
    model = TCPRSegmenter(
        DINOTextSegmenter(checkpoints, device=args.device, amp=not args.no_amp),
        TCPRConfig(maximum_aliases_per_class=20),
    )
    banks, vocabulary_metadata = _load_banks(model, args)
    config = VIPDistillationConfig()
    global_stats = {key: VIPAliasAccumulator(bank.features.shape[0], device=model.device)
                    for key, bank in banks.items()}
    signature = _signature(args, all_samples, samples, vocabulary_metadata, checkpoints)
    started = time.perf_counter()
    for sample_index, sample in enumerate(samples, 1):
        image = load_rgb(sample, args.dataset)
        height, width = image.shape[-2:]
        image_stats = {key: VIPAliasAccumulator(bank.features.shape[0], device=model.device)
                       for key, bank in banks.items()}
        for top in tile_starts(height, args.tile_size, args.overlap):
            for left in tile_starts(width, args.tile_size, args.overlap):
                actual_h, actual_w = min(args.tile_size, height - top), min(args.tile_size, width - left)
                tile = crop_tile(image, left, top, args.tile_size).to(model.device, non_blocking=True)
                prepared = model.prepare_image(tile)
                attention = multilayer_attention_affinity(model.backbone, tile)
                valid = patch_valid_mask(
                    actual_h, actual_w, args.tile_size, model.patch_size, model.device
                ).reshape(1, -1)
                for key, bank in banks.items():
                    text = torch.nn.functional.normalize(bank.features.float(), dim=-1)
                    aliases = torch.nn.functional.normalize(prepared.geometry_projected.float(), dim=-1) @ text.T
                    image_stats[key].update_tile(
                        aliases, attention, bank.parent_indices, bank.class_count, valid, config
                    )
        for key in banks:
            image_stats[key].finalize_image()
            global_stats[key].merge(image_stats[key])
        if sample_index % args.progress_every == 0 or sample_index == len(samples):
            _write_json(output / "results.json", {
                "status": "complete" if sample_index == len(samples) else "running",
                "processed_images": sample_index,
                "total_images": len(samples),
                "states": {key: stats.state_dict() for key, stats in global_stats.items()},
                "distillation": {"config": vars(config)},
                "wall_seconds": round(time.perf_counter() - started, 3),
                "peak_cuda_memory_mb": round(torch.cuda.max_memory_allocated(model.device) / 1048576, 2),
                "signature": signature,
            })
            print(json.dumps({"stage": "select", "dataset": args.dataset, "processed": sample_index}), flush=True)
    return json.loads((output / "results.json").read_text())


def _selection_for_banks(path: Path, banks, metadata) -> dict[str, Tensor]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError("VIP selection file is incomplete.")
    selections = payload.get("selections")
    if not isinstance(selections, dict):
        raise ValueError("VIP selection file has no selections.")
    masks = {}
    for key, bank in banks.items():
        selected = selections.get(key)
        if not isinstance(selected, dict) or selected.get("vocabulary_sha256") != metadata[key]["sha256"]:
            raise ValueError(f"VIP selection does not match the {key} vocabulary.")
        mask = selected_mask_from_report(selected["report"], device=bank.features.device)
        if mask.shape[0] != bank.features.shape[0]:
            raise ValueError(f"VIP selection alias width mismatch for {key}.")
        for class_index in range(bank.class_count):
            if not bool((mask & (bank.parent_indices == class_index)).any()):
                raise ValueError(f"VIP selection removed every alias for class {class_index}.")
        masks[key] = mask
    return masks


def _method_name(vocabulary_key: str, suffix: str) -> str:
    prefix = dict(VOCABULARIES)[vocabulary_key]
    return f"{prefix}_{suffix}"


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    if not args.selection_file:
        raise ValueError("--selection-file is required for evaluate.")
    all_samples, samples = _samples(args)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    work = output / ".accumulators"
    work.mkdir()
    checkpoints = make_checkpoints(args)
    if torch.device(args.device).type == "cuda":
        index = torch.device(args.device).index or 0
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, index)
    model = TCPRSegmenter(
        DINOTextSegmenter(checkpoints, device=args.device, amp=not args.no_amp),
        TCPRConfig(maximum_aliases_per_class=20),
    )
    banks, vocabulary_metadata = _load_banks(model, args)
    selected_masks = _selection_for_banks(Path(args.selection_file).resolve(), banks, vocabulary_metadata)
    names = class_names(args.dataset)
    methods = tuple(_method_name(key, suffix) for key, _ in VOCABULARIES for suffix in ("All20", "GAR", "VIPDistilled"))
    matched_base = {
        _method_name(key, suffix): _method_name(key, "All20")
        for key, _ in VOCABULARIES for suffix in ("GAR", "VIPDistilled")
    }
    matrices = {method: Confusion(names) for method in methods}
    changes = {method: _empty_change() for method in matched_base}
    signature = _signature(args, all_samples, samples, vocabulary_metadata, checkpoints)
    signature["methods"] = list(methods)
    signature["selection_file_sha256"] = hashlib.sha256(Path(args.selection_file).read_bytes()).hexdigest()
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()
    for sample_index, sample in enumerate(samples, 1):
        image = load_rgb(sample, args.dataset)
        target = load_target(sample, args.dataset, tuple(image.shape[-2:]))
        height, width = image.shape[-2:]
        blend = hann_blend_window(args.tile_size)
        with ExitStack() as stack:
            accumulators = {
                method: stack.enter_context(ProbabilityAccumulator(len(names), height, width, 2048, work))
                for method in methods
            }
            for top in tile_starts(height, args.tile_size, args.overlap):
                for left in tile_starts(width, args.tile_size, args.overlap):
                    actual_h, actual_w = min(args.tile_size, height - top), min(args.tile_size, width - left)
                    tile = crop_tile(image, left, top, args.tile_size).to(model.device, non_blocking=True)
                    prepared = model.prepare_image(tile)
                    valid = patch_valid_mask(actual_h, actual_w, args.tile_size, model.patch_size, model.device).reshape(1, -1)
                    geometry = masked_geometry(prepared.geometry_patch_conditional.float(), valid, 1e-6)
                    for key, bank in banks.items():
                        text = torch.nn.functional.normalize(bank.features.float(), dim=-1)
                        aliases = torch.nn.functional.normalize(prepared.geometry_projected.float(), dim=-1) @ text.T
                        all_aliases = torch.ones(aliases.shape[-1], device=model.device, dtype=torch.bool)
                        all_scores = uniform_subset_scores(
                            aliases, bank.parent_indices, bank.class_count, all_aliases, 0.07
                        )
                        reliability = gar_geometry_alias_reliability(aliases, geometry, bank, valid, 0.07)
                        weights = gar_geometry_alias_weights(
                            reliability, bank, valid, temperature=0.20, uniform_prior=0.35, canonical_prior=0.20
                        )
                        gar_scores = gar_weighted_alias_scores(aliases, bank, weights, 0.07)
                        vip_scores = uniform_subset_scores(
                            aliases, bank.parent_indices, bank.class_count, selected_masks[key], 0.07
                        )
                        for suffix, logits in (("All20", all_scores), ("GAR", gar_scores), ("VIPDistilled", vip_scores)):
                            dense = F.interpolate(logits.transpose(1, 2).reshape(1, len(names), prepared.grid_height, prepared.grid_width), size=(args.tile_size, args.tile_size), mode="bilinear", align_corners=False)
                            probability = torch.softmax(dense.float() / args.output_temperature, dim=1)
                            accumulators[_method_name(key, suffix)].add(
                                probability[0, :, :actual_h, :actual_w].cpu().numpy(), blend[:actual_h, :actual_w], left, top
                            )
            predictions = {method: accumulators[method].finalize(None)[0] for method in methods}
        for method, prediction in predictions.items():
            matrices[method].update(prediction, target)
        for method, base in matched_base.items():
            _update_change(changes[method], predictions[base], predictions[method], target)
        if sample_index % args.progress_every == 0 or sample_index == len(samples):
            metrics = {method: matrices[method].summary() for method in methods}
            for summary in metrics.values():
                add_non_residual_metric(summary, names, residual_names(args.dataset))
            _write_json(output / "results.json", {
                "status": "complete" if sample_index == len(samples) else "running",
                "processed_images": sample_index,
                "total_images": len(samples),
                "metrics": metrics,
                "delta_miou_vs_matched_all": {
                    method: round(metrics[method]["mean_iou_percent"] - metrics[base]["mean_iou_percent"], 4)
                    for method, base in matched_base.items()
                },
                "changes_vs_matched_all": {method: _change_summary(value) for method, value in changes.items()},
                "selection_counts_per_class": {
                    key: [int((selected_masks[key] & (banks[key].parent_indices == index)).sum()) for index in range(banks[key].class_count)]
                    for key in banks
                },
                "wall_seconds": round(time.perf_counter() - started, 3),
                "peak_cuda_memory_mb": round(torch.cuda.max_memory_allocated(model.device) / 1048576, 2),
                "signature": signature,
            })
            print(json.dumps({"stage": "evaluate", "dataset": args.dataset, "processed": sample_index, "mIoU": {key: value["mean_iou_percent"] for key, value in metrics.items()}}), flush=True)
    return json.loads((output / "results.json").read_text())


def main(args: argparse.Namespace) -> dict[str, object]:
    if args.tile_size != 512 or args.overlap != 128:
        raise ValueError("This protocol fixes tile=512 and overlap=128.")
    if not 0 < args.memory_fraction <= 1 or args.progress_every < 1:
        raise ValueError("Invalid memory fraction or progress interval.")
    return select(args) if args.stage == "select" else evaluate(args)


if __name__ == "__main__":
    main(parse_args())
