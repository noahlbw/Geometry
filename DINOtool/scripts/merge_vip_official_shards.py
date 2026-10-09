#!/usr/bin/env python3
"""Merge pinned VIP shards after checking exact sample coverage and settings."""
from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np

from eval_vip_official_eight import Confusion


def digest(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def merge(inputs: list[str], output: str) -> dict[str, object]:
    paths = [Path(directory) / "results.json" for directory in inputs]
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not shards or any(shard.get("status") != "complete" for shard in shards):
        raise ValueError("Every VIP shard must have a complete results.json.")
    signatures = [shard["signature"] for shard in shards]
    first = signatures[0]
    if len(shards) != first["num_shards"] or sorted(item["shard_index"] for item in signatures) != list(range(len(shards))):
        raise ValueError("VIP shard indices are incomplete or duplicated.")
    ordered = sorted(signatures, key=lambda item: item["shard_index"])
    keys = [key for group in zip_longest(*(item["sample_keys"] for item in ordered))
            for key in group if key is not None]
    if (len(keys) != first["global_sample_count"] or len(keys) != len(set(keys))
            or digest(keys) != first["global_sample_keys_sha256"]):
        raise ValueError("VIP shards do not uniquely cover the fixed global sequence.")
    fixed = ("implementation", "upstream_commit", "dataset", "source_type", "scored_classes",
             "query_classes", "alias_counts", "settings", "template_count", "vocabulary_sha256",
             "checkpoint_manifest", "data_root", "short_edge_policy", "global_sample_count",
             "global_sample_keys_sha256", "num_shards")
    for shard, signature in zip(shards, signatures):
        if (shard["processed_images"] != shard["total_images"]
                or shard["total_images"] != len(signature["sample_keys"])
                or digest(signature["sample_keys"]) != signature["sample_keys_sha256"]):
            raise ValueError("VIP per-shard count or key digest is invalid.")
        if any(signature[field] != first[field] for field in fixed):
            raise ValueError("VIP shards have different inference settings or inputs.")
    metrics = {}
    for key, names in first["scored_classes"].items():
        collector = Confusion(tuple(names), len(first["query_classes"][key]))
        for shard in shards:
            source = shard["metrics"][key]
            collector.matrix += np.asarray(source["confusion_matrix"], dtype=np.int64)
            collector.ignored += int(source["ignored_pixels"])
        metrics[key] = collector.summary()
    merged = {
        "status": "complete", "coverage_verified": True,
        "processed_images": len(keys), "total_images": len(keys),
        "metrics": metrics,
        "short_edge_padded_crops": {
            key: sum(shard["short_edge_padded_crops"][key] for shard in shards)
            for key in first["scored_classes"]
        },
        "parallel_wall_seconds": max(shard["wall_seconds"] for shard in shards),
        "aggregate_gpu_seconds": sum(shard["wall_seconds"] for shard in shards),
        "peak_cuda_memory_mb": max(shard["peak_cuda_memory_mb"] for shard in shards),
        "signature": {**first, "shard_index": None, "sample_count": len(keys),
                      "sample_keys": keys, "sample_keys_sha256": digest(keys)},
        "shards": [str(path) for path in paths],
    }
    Path(output).write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": first["dataset"], "images": len(keys),
                      "miou": {key: value["mean_iou_percent"] for key, value in metrics.items()}}))
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merge(args.inputs, args.output)
