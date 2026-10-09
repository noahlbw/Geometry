#!/usr/bin/env python3
"""Screen matched text/vision interaction mechanisms on LoveDA E1."""
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
from dinotool.text_visual_intervention import (
    TextVisualInterventionConfig,
    text_visual_intervention_readouts,
)

from eval_stride_ov_loveda_e1 import (
    Confusion, D_CLASSES, P_CLASSES, crop_tile, load_rgb, make_checkpoints,
    patch_valid_mask, save_prediction, target_ids,
)


INTERVENTION_METHODS = (
    "G_canonical", "G_expanded", "TextGraph_logits", "ImageSelf_50",
    "TextSelf_25", "TextSelf_50", "Cross_25", "Cross_50",
    "Parallel_25", "Parallel_50", "TextSelf_shuffled_50", "Serial_50",
)
METHODS = (*INTERVENTION_METHODS, "TCPR")
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
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="Comma-separated subset of methods to evaluate; must include G_canonical and G_expanded.",
    )
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--semantic-relation-strength", type=float, default=4.0)
    parser.add_argument("--weak-update", type=float, default=0.25)
    parser.add_argument("--strong-update", type=float, default=0.50)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def protocol_classes(protocol: str, vocabulary: list[ClassSpec]) -> tuple[list[ClassSpec], list[ClassSpec]]:
    names = P_CLASSES if protocol == "P" else D_CLASSES
    by_name = {spec.name: spec for spec in vocabulary}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Expanded vocabulary is missing LoveDA classes: {missing}")
    return [ClassSpec.from_name(name) for name in names], [by_name[name] for name in names]


class MechanismDiagnostics:
    def __init__(self) -> None:
        self.tiles = 0
        self.relation_kl = 0.0
        self.alias_kept = 0.0
        self.feature_shift: dict[str, float] = {}
        self.tcpr_changed = 0.0
        self.tcpr_evidence = 0.0

    def add(self, intervention, tcpr) -> None:
        self.tiles += 1
        self.relation_kl += intervention.relation_kl_from_geometry
        self.alias_kept += intervention.alias_kept_mean
        for name, value in intervention.feature_shift.items():
            self.feature_shift[name] = self.feature_shift.get(name, 0.0) + value
        self.tcpr_changed += tcpr.changed_from_geometry
        self.tcpr_evidence += tcpr.mean_evidence_mass

    def summary(self) -> dict[str, object]:
        count = max(self.tiles, 1)
        return {
            "new_tiles": self.tiles,
            "mean_relation_kl_from_geometry": self.relation_kl / count,
            "mean_aliases_kept": self.alias_kept / count,
            "mean_feature_shift": {name: value / count for name, value in self.feature_shift.items()},
            "mean_tcpr_changed": self.tcpr_changed / count,
            "mean_tcpr_evidence_mass": self.tcpr_evidence / count,
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


@torch.inference_mode()
def predict_image(
    model: TCPRSegmenter,
    image_path: Path,
    canonical_banks: dict[str, object],
    expanded_banks: dict[str, object],
    args: argparse.Namespace,
    intervention_config: TextVisualInterventionConfig,
    work_dir: Path,
    diagnostics: dict[str, MechanismDiagnostics],
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
            for protocol in canonical_banks for method in METHODS
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
                for protocol in canonical_banks:
                    intervention = text_visual_intervention_readouts(
                        prepared,
                        canonical_banks[protocol],
                        expanded_banks[protocol],
                        model.config,
                        intervention_config,
                        valid_mask=valid,
                    )
                    tcpr = model.read_prepared(prepared, expanded_banks[protocol], valid_mask=valid)
                    diagnostics[protocol].add(intervention, tcpr.diagnostics)
                    logits_by_method = dict(intervention.logits)
                    logits_by_method["TCPR"] = tcpr.logits
                    for method, logits in logits_by_method.items():
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
            for protocol in canonical_banks
        }


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
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    if not {"G_canonical", "G_expanded"}.issubset(methods):
        raise ValueError("Selected methods must include G_canonical and G_expanded.")
    samples = discover_loveda_samples(args.data_root)
    random.Random(args.sample_seed).shuffle(samples)
    if args.max_images > 0:
        samples = samples[:args.max_images]
    global_sample_keys = [sample.key for sample in samples]
    global_sample_hash = hashlib.sha256("\n".join(global_sample_keys).encode()).hexdigest()
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
    tcpr_config = TCPRConfig()
    intervention_config = TextVisualInterventionConfig(
        semantic_relation_strength=args.semantic_relation_strength,
        weak_update=args.weak_update,
        strong_update=args.strong_update,
    )
    model = TCPRSegmenter(backbone, tcpr_config)
    class_specs = {protocol: protocol_classes(protocol, vocabulary) for protocol in ("P", "D")}
    canonical_banks = {
        protocol: model.encode_text(specs[0]) for protocol, specs in class_specs.items()
    }
    expanded_banks = {
        protocol: model.encode_text(specs[1]) for protocol, specs in class_specs.items()
    }
    matrices = {
        (protocol, method): Confusion(P_CLASSES if protocol == "P" else D_CLASSES)
        for protocol in ("P", "D") for method in methods
    }
    changes = {
        (protocol, method): _empty_change()
        for protocol in ("P", "D") for method in methods if method != "G_expanded"
    }
    diagnostics = {protocol: MechanismDiagnostics() for protocol in ("P", "D")}
    signature = {
        "method": "matched text-visual intervention screen",
        "date": "2026-09-25",
        "sample_count": len(samples),
        "sample_keys_sha256": sample_hash,
        "methods": list(methods),
        "global_sample_count": len(global_sample_keys),
        "global_sample_keys_sha256": global_sample_hash,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "sample_keys": [sample.key for sample in samples],
        "config": {key: value for key, value in vars(args).items() if key != "device"},
        "tcpr": asdict(tcpr_config),
        "intervention": asdict(intervention_config),
        "vocabulary": {
            "path": str(vocabulary_path),
            "sha256": hashlib.sha256(vocabulary_path.read_bytes()).hexdigest(),
            "canonical": {
                protocol: serialize_class_specs(specs[0]) for protocol, specs in class_specs.items()
            },
            "expanded": {
                protocol: serialize_class_specs(specs[1]) for protocol, specs in class_specs.items()
            },
        },
        "checkpoints": checkpoint_manifest(checkpoints),
        "protocol_note": "All arms share the same image, Geometry state, resolution, and sliding-window protocol.",
    }
    signature_path = output / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Output directory is bound to another experiment signature.")
    signature_path.write_text(json.dumps(signature, indent=2, ensure_ascii=True))

    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        with Image.open(sample.image_path) as image:
            shape = (image.height, image.width)
        predictions: dict[str, dict[str, np.ndarray]] = {"P": {}, "D": {}}
        complete = True
        for protocol in ("P", "D"):
            class_count = len(P_CLASSES if protocol == "P" else D_CLASSES)
            for method in methods:
                cached = _load_prediction(
                    output / "predictions" / protocol / method / sample.key,
                    shape, class_count,
                )
                complete = complete and cached is not None
                if cached is not None:
                    predictions[protocol][method] = cached
        if not complete:
            predictions = predict_image(
                model, sample.image_path, canonical_banks, expanded_banks,
                args, intervention_config, work_dir, diagnostics,
            )
            for protocol in ("P", "D"):
                predictions[protocol] = {
                    method: predictions[protocol][method] for method in methods
                }
                for method, prediction in predictions[protocol].items():
                    save_prediction(output / "predictions" / protocol / method / sample.key, prediction)

        for protocol in ("P", "D"):
            target = target_ids(sample.mask_path, protocol)
            base = predictions[protocol]["G_expanded"]
            for method in methods:
                matrices[(protocol, method)].update(predictions[protocol][method], target)
                if method != "G_expanded":
                    _update_change(changes[(protocol, method)], base, predictions[protocol][method], target)

        if index % max(args.progress_every, 1) == 0 or index == len(samples):
            metrics = {
                protocol: {method: matrices[(protocol, method)].summary() for method in methods}
                for protocol in ("P", "D")
            }
            result = {
                "status": "complete" if index == len(samples) else "running",
                "processed_images": index,
                "total_images": len(samples),
                "metrics": metrics,
                "delta_miou_vs_canonical": {
                    protocol: {
                        method: round(
                            metrics[protocol][method]["mean_iou_percent"]
                            - metrics[protocol]["G_canonical"]["mean_iou_percent"], 4,
                        )
                        for method in methods
                    }
                    for protocol in ("P", "D")
                },
                "delta_miou_vs_expanded": {
                    protocol: {
                        method: round(
                            metrics[protocol][method]["mean_iou_percent"]
                            - metrics[protocol]["G_expanded"]["mean_iou_percent"], 4,
                        )
                        for method in methods
                    }
                    for protocol in ("P", "D")
                },
                "changes_vs_expanded": {
                    protocol: {
                        method: _change_summary(changes[(protocol, method)])
                        for method in methods if method != "G_expanded"
                    }
                    for protocol in ("P", "D")
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
                "P": {method: metrics["P"][method]["mean_iou_percent"] for method in methods},
                "D": {method: metrics["D"][method]["mean_iou_percent"] for method in methods},
            }), flush=True)
    return result


if __name__ == "__main__":
    main(parse_args())
