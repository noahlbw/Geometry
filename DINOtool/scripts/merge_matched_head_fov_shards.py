#!/usr/bin/env python3
"""Merge matched-head field-of-view shards with full sample coverage checks."""
from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np

from merge_competitive_evidence_shards import metric_summary


def digest(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def merge(inputs: list[Path], output: Path) -> dict:
    shards = [json.loads((path / "results.json").read_text(encoding="utf-8")) for path in inputs]
    if not shards or any(shard["status"] != "complete" or
                         shard["processed_images"] != shard["total_images"] for shard in shards):
        raise ValueError("Every shard must have complete results.")
    signatures = [shard["signature"] for shard in shards]
    first = signatures[0]
    if (len(shards) != first["num_shards"] or
            sorted(item["shard_index"] for item in signatures) != list(range(len(shards)))):
        raise ValueError("Missing or duplicate shard index.")
    excluded = ("sample_keys", "sample_keys_sha256", "shard_index")
    fixed = {key: value for key, value in first.items() if key not in excluded}
    for shard, signature in zip(shards, signatures):
        if {key: value for key, value in signature.items() if key not in excluded} != fixed:
            raise ValueError("Shard inference signatures differ.")
        keys = signature["sample_keys"]
        if len(keys) != shard["total_images"] or digest(keys) != signature["sample_keys_sha256"]:
            raise ValueError("Invalid shard sample keys.")
    arms = tuple(first["arms"])
    if len(arms) != 4 or len(set(arms)) != 4:
        raise ValueError("Expected four distinct experiment arms.")
    ordered = sorted(signatures, key=lambda item: item["shard_index"])
    keys = [key for row in zip_longest(*(item["sample_keys"] for item in ordered))
            for key in row if key is not None]
    if (len(keys) != len(set(keys)) or len(keys) != first["global_sample_count"] or
            digest(keys) != first["global_sample_keys_sha256"]):
        raise ValueError("Shards do not uniquely cover the fixed sample sequence.")
    classes = len(first["class_names"])
    metrics = {}
    for arm in arms:
        parts = [shard["metrics"][arm] for shard in shards]
        matrix = sum((np.asarray(part["confusion_matrix"], dtype=np.int64)
                      for part in parts), np.zeros((classes, classes), dtype=np.int64))
        metrics[arm] = metric_summary(matrix, list(first["class_names"]),
                                      sum(part["ignored_pixels"] for part in parts))
    pairs = {}
    for key in shards[0]["pairwise"]:
        pairs[key] = {}
        for field in shards[0]["pairwise"][key]:
            values = [shard["pairwise"][key][field] for shard in shards]
            pairs[key][field] = ([sum(group) for group in zip(*values)]
                                 if isinstance(values[0], list) else sum(values))
    result = {
        "status": "complete", "coverage_verified": True,
        "processed_images": len(keys), "total_images": len(keys),
        "metrics": metrics, "pairwise": pairs,
        "crops": {edge: sum(shard["crops"][edge] for shard in shards)
                  for edge in shards[0]["crops"]},
        "parallel_wall_seconds": max(shard["wall_seconds"] for shard in shards),
        "aggregate_gpu_seconds": sum(shard["wall_seconds"] for shard in shards),
        "peak_cuda_memory_mb": max(shard["peak_cuda_memory_mb"] for shard in shards),
        "signature": {**first, "sample_keys": keys, "sample_keys_sha256": digest(keys),
                      "shard_index": None},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"Refusing existing merge output: {output}")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": first["dataset"], "images": len(keys),
                      "miou": {arm: metrics[arm]["mean_iou_percent"] for arm in arms}}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merge(args.inputs, args.output)
