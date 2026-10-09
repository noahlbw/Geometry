#!/usr/bin/env python3
"""Merge sharded Geometry alias-count LoveDA evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def metric_summary(matrix: np.ndarray, names: list[str], ignored: int) -> dict[str, object]:
    intersection = np.diag(matrix)
    target = matrix.sum(1)
    predicted = matrix.sum(0)
    union = target + predicted - intersection
    iou = np.divide(intersection, union, out=np.full(len(names), np.nan), where=union > 0)
    result: dict[str, object] = {
        "mean_iou_percent": round(float(np.nanmean(iou)) * 100.0, 4),
        "pixel_accuracy_percent": round(float(intersection.sum() / max(int(target.sum()), 1)) * 100.0, 4),
        "per_class": [
            {"name": name, "iou_percent": None if np.isnan(iou[index]) else round(float(iou[index]) * 100.0, 4)}
            for index, name in enumerate(names)
        ],
        "confusion_matrix": matrix.tolist(),
        "ignored_pixels": int(ignored),
    }
    if names and names[0] == "background":
        result["foreground_mean_iou_percent"] = round(float(np.nanmean(iou[1:])) * 100.0, 4)
    return result


def change_summary(values: dict[str, int]) -> dict[str, float | int]:
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
    paths = [Path(item) / "results.json" if Path(item).is_dir() else Path(item) for item in args.inputs]
    shards = [json.loads(path.read_text()) for path in paths]
    if not shards or any(item.get("status") != "complete" for item in shards):
        raise ValueError("Every shard must contain a complete results.json.")
    signatures = [item["signature"] for item in shards]
    first = signatures[0]
    indices = sorted(signature["shard_index"] for signature in signatures)
    if len(shards) != first["num_shards"] or indices != list(range(first["num_shards"])):
        raise ValueError(f"Expected shard indices 0..{first['num_shards'] - 1}, received {indices}.")
    keys = [key for signature in signatures for key in signature["sample_keys"]]
    if len(keys) != len(set(keys)) or len(keys) != first["global_sample_count"]:
        raise ValueError("Shard sample keys overlap or do not cover the complete global sample set.")
    for signature in signatures[1:]:
        for field in ("global_sample_count", "global_sample_keys_sha256", "methods", "checkpoints", "vocabulary"):
            if signature[field] != first[field]:
                raise ValueError(f"Shard signatures disagree on {field}.")

    metrics: dict[str, dict[str, object]] = {"P": {}, "D": {}}
    for protocol in ("P", "D"):
        for method in first["methods"]:
            source = shards[0]["metrics"][protocol][method]
            names = [item["name"] for item in source["per_class"]]
            matrix = sum(
                (np.asarray(shard["metrics"][protocol][method]["confusion_matrix"], dtype=np.int64) for shard in shards),
                np.zeros((len(names), len(names)), dtype=np.int64),
            )
            ignored = sum(int(shard["metrics"][protocol][method]["ignored_pixels"]) for shard in shards)
            metrics[protocol][method] = metric_summary(matrix, names, ignored)
    count_fields = ("valid", "changed", "beneficial", "harmful", "wrong_to_wrong")
    changes = {"P": {}, "D": {}}
    for protocol in ("P", "D"):
        for method in first["methods"]:
            if method == "G_top8":
                continue
            totals = {
                field: sum(int(shard["changes_vs_top8"][protocol][method][field]) for shard in shards)
                for field in count_fields
            }
            changes[protocol][method] = change_summary(totals)
    result = {
        "status": "complete",
        "processed_images": len(keys),
        "total_images": first["global_sample_count"],
        "metrics": metrics,
        "delta_miou_vs_top8": {
            protocol: {
                method: round(metrics[protocol][method]["mean_iou_percent"] - metrics[protocol]["G_top8"]["mean_iou_percent"], 4)
                for method in first["methods"]
            }
            for protocol in ("P", "D")
        },
        "changes_vs_top8": changes,
        "parallel_wall_seconds": max(float(shard["wall_seconds"]) for shard in shards),
        "aggregate_gpu_seconds": sum(float(shard["wall_seconds"]) for shard in shards),
        "peak_cuda_memory_mb": max(float(shard["peak_cuda_memory_mb"]) for shard in shards),
        "shards": [str(path) for path in paths],
        "signature": {**first, "sample_count": first["global_sample_count"], "sample_keys_sha256": first["global_sample_keys_sha256"], "shard_index": None, "sample_keys": None},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True))
    print(json.dumps({
        "images": len(keys),
        "P": {method: metrics["P"][method]["mean_iou_percent"] for method in first["methods"]},
        "D": {method: metrics["D"][method]["mean_iou_percent"] for method in first["methods"]},
    }))
    return result


if __name__ == "__main__":
    main(parse_args())
