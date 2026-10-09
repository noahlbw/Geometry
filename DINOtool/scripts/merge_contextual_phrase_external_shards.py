#!/usr/bin/env python3
"""Merge exact external-dataset confusion matrices for multiscale alias-union runs."""
from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
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
    ordered = sorted(signatures, key=lambda item: item["shard_index"])
    global_keys = [key for group in zip_longest(*(x["sample_keys"] for x in ordered)) for key in group if key is not None]
    if hashlib.sha256("\n".join(global_keys).encode()).hexdigest() != first["global_sample_keys_sha256"]:
        raise ValueError("Actual shard coverage differs from the fixed global sequence.")
    for shard, signature in zip(shards, signatures):
        count = len(signature["sample_keys"])
        if shard["processed_images"] != count or shard["total_images"] != count:
            raise ValueError("A complete shard has inconsistent processed/total counts.")
        if hashlib.sha256("\n".join(signature["sample_keys"]).encode()).hexdigest() != signature["sample_keys_sha256"]:
            raise ValueError("A shard sample-key digest is invalid.")
    for signature in signatures[1:]:
        for field in (
            "dataset", "split", "global_sample_count", "global_sample_keys_sha256",
            "methods", "classes", "residual_class_names", "checkpoints", "vocabulary",
            "contextual_phrase", "tcpr", "implementation",
        ):
            if signature[field] != first[field]:
                raise ValueError(f"Shard signatures disagree on {field}.")
        ignored = {"output_dir", "shard_index"}
        current = {key: value for key, value in signature["config"].items() if key not in ignored}
        reference = {key: value for key, value in first["config"].items() if key not in ignored}
        if current != reference:
            raise ValueError("Shard inference configurations differ.")

    methods, names = first["methods"], first["classes"]
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
            values = [x["iou_percent"] for x in summary["per_class"] if x["name"].casefold() not in residual and x["iou_percent"] is not None]
            summary["non_residual_mean_iou_percent"] = round(float(np.mean(values)), 4)
        metrics[method] = summary

    fields = ("valid", "changed", "beneficial", "harmful", "wrong_to_wrong")
    changes = {}
    for method in methods[1:]:
        changes[method] = change_summary({
            field: sum(int(shard["changes_vs_geometry"][method][field]) for shard in shards)
            for field in fields
        })
    diagnostic_fields = tuple(field for field in shards[0]["diagnostics"] if field != "tiles")
    tile_count = sum(int(shard["diagnostics"]["tiles"]) for shard in shards)
    diagnostics = {"tiles": tile_count}
    for field in diagnostic_fields:
        diagnostics[field] = sum(
            float(shard["diagnostics"][field]) * int(shard["diagnostics"]["tiles"])
            for shard in shards
        ) / max(tile_count, 1)
    result = {
        "status": "complete", "processed_images": len(keys),
        "total_images": first["global_sample_count"], "metrics": metrics,
        "delta_miou_vs_geometry": {
            method: round(metrics[method]["mean_iou_percent"] - metrics[methods[0]]["mean_iou_percent"], 4)
            for method in methods
        },
        "changes_vs_geometry": changes, "diagnostics": diagnostics,
        "parallel_wall_seconds": max(float(shard["wall_seconds"]) for shard in shards),
        "aggregate_gpu_seconds": sum(float(shard["wall_seconds"]) for shard in shards),
        "peak_cuda_memory_mb": max(float(shard["peak_cuda_memory_mb"]) for shard in shards),
        "coverage_verified": True, "shards": [str(path) for path in paths],
        "signature": {**first, "sample_count": first["global_sample_count"],
                      "sample_keys_sha256": first["global_sample_keys_sha256"],
                      "shard_index": None, "sample_keys": None},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True))
    print(json.dumps({"dataset": first["dataset"], "images": len(keys),
                      "metrics": {method: metrics[method]["mean_iou_percent"] for method in methods}}))
    return result


if __name__ == "__main__":
    main(parse_args())
