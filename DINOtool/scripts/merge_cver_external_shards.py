#!/usr/bin/env python3
"""Merge exact confusion matrices from sharded external CVER evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from merge_competitive_evidence_shards import change_summary, metric_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main(args: argparse.Namespace) -> dict[str, object]:
    paths = [Path(value) / "results.json" if Path(value).is_dir() else Path(value) for value in args.inputs]
    shards = [json.loads(path.read_text()) for path in paths]
    if not shards or any(item.get("status") != "complete" for item in shards):
        raise ValueError("Every shard must contain a complete results.json.")
    signatures = [item["signature"] for item in shards]
    first = signatures[0]
    expected = first["num_shards"]
    indices = sorted(signature["shard_index"] for signature in signatures)
    if len(shards) != expected or indices != list(range(expected)):
        raise ValueError(f"Expected shard indices 0..{expected - 1}, received {indices}.")
    keys = [key for signature in signatures for key in signature["sample_keys"]]
    if len(keys) != len(set(keys)) or len(keys) != first["global_sample_count"]:
        raise ValueError("Shard sample keys overlap or do not cover the global sample set.")
    for signature in signatures[1:]:
        for field in (
            "dataset", "split", "global_sample_count", "global_sample_keys_sha256",
            "methods", "classes", "residual_class_names", "checkpoints", "vocabulary",
            "cver", "tcpr",
        ):
            if signature[field] != first[field]:
                raise ValueError(f"Shard signatures disagree on {field}.")

    methods = first["methods"]
    names = first["classes"]
    metrics: dict[str, object] = {}
    for method in methods:
        matrix = sum(
            (np.asarray(shard["metrics"][method]["confusion_matrix"], dtype=np.int64) for shard in shards),
            np.zeros((len(names), len(names)), dtype=np.int64),
        )
        ignored = sum(int(shard["metrics"][method]["ignored_pixels"]) for shard in shards)
        summary = metric_summary(matrix, names, ignored)
        residual = {name.casefold() for name in first["residual_class_names"]}
        if residual:
            values = [
                item["iou_percent"] for item in summary["per_class"]
                if item["name"].casefold() not in residual and item["iou_percent"] is not None
            ]
            summary["non_residual_mean_iou_percent"] = round(float(np.mean(values)), 4)
        metrics[method] = summary

    count_fields = ("valid", "changed", "beneficial", "harmful", "wrong_to_wrong")
    changes: dict[str, object] = {}
    for method in methods:
        if method == "G_all20_uniform":
            continue
        totals = {
            field: sum(int(shard["changes_vs_geometry"][method][field]) for shard in shards)
            for field in count_fields
        }
        changes[method] = change_summary(totals)

    diagnostic_fields = tuple(field for field in shards[0]["diagnostics"] if field != "tiles")
    tile_count = sum(int(shard["diagnostics"]["tiles"]) for shard in shards)
    diagnostics: dict[str, object] = {"tiles": tile_count}
    for field in diagnostic_fields:
        diagnostics[field] = sum(
            float(shard["diagnostics"][field]) * int(shard["diagnostics"]["tiles"])
            for shard in shards
        ) / max(tile_count, 1)

    result = {
        "status": "complete",
        "processed_images": len(keys),
        "total_images": first["global_sample_count"],
        "metrics": metrics,
        "delta_miou_vs_geometry": {
            method: round(
                metrics[method]["mean_iou_percent"]
                - metrics["G_all20_uniform"]["mean_iou_percent"],
                4,
            )
            for method in methods
        },
        "changes_vs_geometry": changes,
        "diagnostics": diagnostics,
        "parallel_wall_seconds": max(float(shard["wall_seconds"]) for shard in shards),
        "aggregate_gpu_seconds": sum(float(shard["wall_seconds"]) for shard in shards),
        "peak_cuda_memory_mb": max(float(shard["peak_cuda_memory_mb"]) for shard in shards),
        "shards": [str(path) for path in paths],
        "signature": {
            **first,
            "sample_count": first["global_sample_count"],
            "sample_keys_sha256": first["global_sample_keys_sha256"],
            "shard_index": None,
            "sample_keys": None,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True))
    print(json.dumps({
        "dataset": first["dataset"],
        "images": len(keys),
        "metrics": {method: metrics[method]["mean_iou_percent"] for method in methods},
    }))
    return result


if __name__ == "__main__":
    main(parse_args())
