#!/usr/bin/env python3
"""Merge non-overlapping CAFe-DINO depth-diagnosis evaluation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cafedino_depth_diagnosis import Confusion, _write_json, _write_tables  # noqa: E402
from dinotool.depth_diagnosis import oracle_pair_matrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-images", type=int)
    parser.add_argument("--locked-selection")
    return parser.parse_args()


def _same_payload(shards: Sequence[dict[str, Any]], key: str) -> Any:
    reference = shards[0][key]
    encoded = json.dumps(reference, sort_keys=True)
    if any(json.dumps(shard[key], sort_keys=True) != encoded for shard in shards[1:]):
        raise ValueError(f"Shard field differs: {key}")
    return reference


def _summary(class_names: Sequence[str], matrices: Sequence[Any]) -> dict[str, Any]:
    confusion = Confusion(class_names)
    confusion.matrix = sum((np.asarray(matrix, dtype=np.int64) for matrix in matrices), np.zeros_like(confusion.matrix))
    return confusion.summary()


def _merge_states(shards: Sequence[dict[str, Any]], state_name: str) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for shard in shards:
        for key, value in shard["merge_state"][state_name].items():
            row = output.setdefault(key, {"sum": 0.0, "count": 0})
            row["sum"] = float(row["sum"]) + float(value["sum"])
            row["count"] = int(row["count"]) + int(value["count"])
    return output


def merge(args: argparse.Namespace) -> dict[str, Any]:
    paths = [Path(path).expanduser().resolve() for path in args.inputs]
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not shards or any(shard.get("status") != "complete" for shard in shards):
        raise ValueError("Every input must be a completed result shard.")
    dataset_name = shards[0]["dataset"]["name"]
    class_names = tuple(shards[0]["dataset"]["classes"])
    config = _same_payload(shards, "config")
    alignment_manifest = _same_payload(shards, "alignment_manifest")
    model_checkpoint = _same_payload(shards, "model_checkpoint")
    for shard in shards:
        if shard["dataset"]["name"] != dataset_name or tuple(shard["dataset"]["classes"]) != class_names:
            raise ValueError("Dataset identity or class order differs across shards.")
    ranges = sorted(
        (int(shard["dataset"]["start_index"]), int(shard["dataset"]["images"]))
        for shard in shards
    )
    for (start, count), (next_start, _) in zip(ranges, ranges[1:]):
        if start + count > next_start:
            raise ValueError("Evaluation shards overlap.")
    image_count = sum(count for _, count in ranges)
    if args.expected_images is not None and image_count != args.expected_images:
        raise ValueError(f"Merged {image_count} images, expected {args.expected_images}.")

    layers = tuple(int(layer) for layer in config["layers"])
    semantic = {
        method: {
            str(layer): _summary(
                class_names,
                [shard["semantic"][method][str(layer)]["confusion_matrix"] for shard in shards],
            )
            for layer in layers
        }
        for method in config["semantic_methods"]
    }
    region_semantic = {}
    for method in config["semantic_methods"]:
        region_semantic[method] = {}
        for layer in layers:
            key = f"{method}:{layer}"
            matrix = sum(
                (
                    np.asarray(
                        shard["merge_state"]["region_semantic_confusions"][key],
                        dtype=np.int64,
                    )
                    for shard in shards
                ),
                np.zeros((len(class_names), len(class_names)), dtype=np.int64),
            )
            region_semantic[method][str(layer)] = _summary(class_names, [matrix])
    spatial_state = _merge_states(shards, "spatial_metrics")
    spatial = {}
    for kind in config["affinity_kinds"]:
        spatial[kind] = {}
        for layer in layers:
            summary = _summary(
                class_names,
                [shard["spatial"][kind][str(layer)]["confusion_matrix"] for shard in shards],
            )
            per_class = {}
            for name in class_names:
                per_class[name] = {}
                for metric in ("region_iou", "boundary_f1", "connectivity_recall", "leakage_ratio"):
                    value = spatial_state[f"{kind}:{layer}:{name}:{metric}"]
                    per_class[name][metric] = round(float(value["sum"]) / max(int(value["count"]), 1), 6)
            spatial[kind][str(layer)] = {**summary, "mean_per_class_metrics": per_class}

    pair_matrices = {}
    pair_scores = {}
    for semantic_layer in layers:
        for spatial_layer in layers:
            key = f"{semantic_layer}:{spatial_layer}"
            matrix = sum(
                (np.asarray(shard["merge_state"]["pair_confusions"][key], dtype=np.int64) for shard in shards),
                np.zeros((len(class_names), len(class_names)), dtype=np.int64),
            )
            pair_matrices[key] = matrix.tolist()
            pair_scores[(semantic_layer, spatial_layer)] = _summary(class_names, [matrix])["mean_iou_percent"]
    oracle_pairs = oracle_pair_matrix(pair_scores, layers)
    oracle_state = {
        key: sum(float(shard["merge_state"]["per_image_oracle"][key]) for shard in shards)
        for key in ("unrestricted_sum", "dual_sum", "same_sum")
    }
    oracle_count = sum(int(shard["merge_state"]["per_image_oracle"]["count"]) for shard in shards)
    oracle_pairs["per_image_oracle"] = {
        "adaptive_unrestricted_miou_percent": round(oracle_state["unrestricted_sum"] / oracle_count, 4),
        "adaptive_dual_depth_miou_percent": round(oracle_state["dual_sum"] / oracle_count, 4),
        "adaptive_same_depth_miou_percent": round(oracle_state["same_sum"] / oracle_count, 4),
        "adaptive_dual_gap_points": round(
            (oracle_state["dual_sum"] - oracle_state["same_sum"]) / oracle_count,
            4,
        ),
    }
    adaptive_confusions = {}
    for name in ("unrestricted", "dual_depth", "same_depth"):
        matrix = sum(
            (
                np.asarray(
                    shard["merge_state"]["adaptive_confusions"][name],
                    dtype=np.int64,
                )
                for shard in shards
            ),
            np.zeros((len(class_names), len(class_names)), dtype=np.int64),
        )
        adaptive_confusions[name] = matrix.tolist()
    oracle_pairs["dataset_oracle"] = {
        name: _summary(class_names, [matrix])
        for name, matrix in adaptive_confusions.items()
    }

    morphology_state = _merge_states(shards, "morphology")
    morphology: dict[str, Any] = {}
    for key, value in morphology_state.items():
        method, layer, group = key.split(":", 2)
        morphology.setdefault(method, {}).setdefault(layer, {})[group] = {
            "components": int(value["count"]),
            "mean_correct_pixel_fraction": round(float(value["sum"]) / max(int(value["count"]), 1), 6),
        }

    locked = None
    if args.locked_selection:
        source_path = Path(args.locked_selection).expanduser().resolve()
        source = json.loads(source_path.read_text(encoding="utf-8"))
        same = int(source["oracle_pairs"]["best_same_depth"]["layer"])
        dual_row = source["oracle_pairs"]["best_off_diagonal_pair"]
        dual = (int(dual_row["semantic_layer"]), int(dual_row["spatial_layer"]))
        locked = {
            "source_result": str(source_path),
            "fixed_same_depth": {
                "layer": same,
                "target_metrics": _summary(class_names, [pair_matrices[f"{same}:{same}"]]),
            },
            "fixed_dual_depth": {
                "semantic_layer": dual[0],
                "spatial_layer": dual[1],
                "target_metrics": _summary(class_names, [pair_matrices[f"{dual[0]}:{dual[1]}"]]),
            },
        }

    fusion = _summary(
        class_names,
        [shard["static_multilayer_fusion"]["confusion_matrix"] for shard in shards],
    )
    seed_coverage = {}
    for layer in config["layers"]:
        rows = [shard["semantic_seed_coverage"][str(layer)] for shard in shards]
        windows = sum(int(row["windows"]) for row in rows)
        windows_with_any = sum(int(row["windows_with_any_seed"]) for row in rows)
        seed_pixels = sum(int(row["seed_pixels"]) for row in rows)
        correct_seed_pixels = sum(int(row["correct_seed_pixels"]) for row in rows)
        class_seed_pixels = [
            sum(int(row["class_seed_pixels"][index]) for row in rows)
            for index in range(len(class_names))
        ]
        class_correct_seed_pixels = [
            sum(int(row["class_correct_seed_pixels"][index]) for row in rows)
            for index in range(len(class_names))
        ]
        patch_count = windows * (int(config["window_size"]) // 16) ** 2
        seed_coverage[str(layer)] = {
            "windows": windows,
            "windows_with_any_seed": windows_with_any,
            "seed_pixels": seed_pixels,
            "correct_seed_pixels": correct_seed_pixels,
            "class_seed_pixels": class_seed_pixels,
            "class_correct_seed_pixels": class_correct_seed_pixels,
            "seed_fraction": round(seed_pixels / max(patch_count, 1), 8),
            "seed_precision": round(correct_seed_pixels / max(seed_pixels, 1), 8),
            "window_seed_rate": round(windows_with_any / max(windows, 1), 8),
        }
    fused_rows = [shard["fused_seed_coverage"] for shard in shards]
    fused_windows = sum(int(row["windows"]) for row in fused_rows)
    fused_windows_with_any = sum(int(row["windows_with_any_seed"]) for row in fused_rows)
    fused_seed_pixels = sum(int(row["seed_pixels"]) for row in fused_rows)
    fused_correct_seed_pixels = sum(
        int(row["correct_seed_pixels"]) for row in fused_rows
    )
    fused_class_seed_pixels = [
        sum(int(row["class_seed_pixels"][index]) for row in fused_rows)
        for index in range(len(class_names))
    ]
    fused_class_correct_seed_pixels = [
        sum(int(row["class_correct_seed_pixels"][index]) for row in fused_rows)
        for index in range(len(class_names))
    ]
    fused_patch_count = fused_windows * (int(config["window_size"]) // 16) ** 2
    fused_seed_coverage = {
        "windows": fused_windows,
        "windows_with_any_seed": fused_windows_with_any,
        "seed_pixels": fused_seed_pixels,
        "correct_seed_pixels": fused_correct_seed_pixels,
        "class_seed_pixels": fused_class_seed_pixels,
        "class_correct_seed_pixels": fused_class_correct_seed_pixels,
        "seed_fraction": round(fused_seed_pixels / max(fused_patch_count, 1), 8),
        "seed_precision": round(
            fused_correct_seed_pixels / max(fused_seed_pixels, 1), 8
        ),
        "window_seed_rate": round(fused_windows_with_any / max(fused_windows, 1), 8),
    }
    result = {
        "status": "complete",
        "dataset": {
            **shards[0]["dataset"],
            "images": image_count,
            "start_index": min(start for start, _ in ranges),
            "merged_ranges": [{"start": start, "count": count} for start, count in ranges],
        },
        "config": config,
        "alignment_manifest": alignment_manifest,
        "model_checkpoint": model_checkpoint,
        "semantic": semantic,
        "region_semantic": region_semantic,
        "spatial": spatial,
        "oracle_pairs": oracle_pairs,
        "locked_source_selection": locked,
        "static_multilayer_fusion": fusion,
        "semantic_seed_coverage": seed_coverage,
        "fused_seed_coverage": fused_seed_coverage,
        "morphology_component_recall": morphology,
        "merge_state": {
            "pair_confusions": pair_matrices,
            "region_semantic_confusions": {
                f"{method}:{layer}": sum(
                    (
                        np.asarray(
                            shard["merge_state"]["region_semantic_confusions"][
                                f"{method}:{layer}"
                            ],
                            dtype=np.int64,
                        )
                        for shard in shards
                    ),
                    np.zeros((len(class_names), len(class_names)), dtype=np.int64),
                ).tolist()
                for method in config["semantic_methods"]
                for layer in layers
            },
            "adaptive_confusions": adaptive_confusions,
            "spatial_metrics": spatial_state,
            "morphology": morphology_state,
            "per_image_oracle": {**oracle_state, "count": oracle_count},
        },
        "timing_seconds": round(sum(float(shard["timing_seconds"]) for shard in shards), 4),
        "parallel_wall_upper_bound_seconds": round(max(float(shard["timing_seconds"]) for shard in shards), 4),
        "shards": [str(path) for path in paths],
        "interpretation_guardrails": shards[0]["interpretation_guardrails"],
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "results.json", result)
    _write_tables(output_dir, result, layers, class_names)
    return result


def main() -> None:
    args = parse_args()
    result = merge(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset": result["dataset"]["name"],
                "images": result["dataset"]["images"],
                "output": str(Path(args.output_dir).expanduser().resolve() / "results.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
