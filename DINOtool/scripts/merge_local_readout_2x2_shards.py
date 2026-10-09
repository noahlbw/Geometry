#!/usr/bin/env python3
"""Merge exact confusion and error-intersection counts from local readout shards."""
from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np

from dinotool.local_readout_audit import ARMS, summarize_error_counts
from merge_competitive_evidence_shards import metric_summary


def main(inputs, output):
    paths = [Path(item) / "results.json" if Path(item).is_dir() else Path(item) for item in inputs]
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not shards or any(shard["status"] != "complete" for shard in shards):
        raise ValueError("Every shard must be complete.")
    signatures = [shard["signature"] for shard in shards]
    first = signatures[0]
    if len(shards) != first["num_shards"] or sorted(s["shard_index"] for s in signatures) != list(range(len(shards))):
        raise ValueError("Missing or duplicate shard index.")
    digest = lambda keys: hashlib.sha256("\n".join(keys).encode()).hexdigest()
    ordered = sorted(signatures, key=lambda signature: signature["shard_index"])
    keys = [key for group in zip_longest(*(s["sample_keys"] for s in ordered)) for key in group if key is not None]
    if len(keys) != len(set(keys)) or len(keys) != first["global_sample_count"]:
        raise ValueError("Incomplete or overlapping sample keys.")
    if digest(keys) != first["global_sample_keys_sha256"]:
        raise ValueError("Shards do not cover the fixed global sample sequence.")
    for shard, signature in zip(shards, signatures):
        if shard["processed_images"] != shard["total_images"] or shard["total_images"] != len(signature["sample_keys"]):
            raise ValueError("Shard image count mismatch.")
        if digest(signature["sample_keys"]) != signature["sample_keys_sha256"]:
            raise ValueError("Shard sample digest mismatch.")
        for field in ("implementation", "dataset", "arms", "classes", "readouts", "vocabulary",
                      "checkpoints", "global_sample_count", "global_sample_keys_sha256"):
            if signature[field] != first[field]:
                raise ValueError(f"Shards disagree on {field}.")
        normalize = lambda data: {key: value for key, value in data.items()
                                  if key not in ("output_dir", "shard_index")}
        if normalize(signature["config"]) != normalize(first["config"]):
            raise ValueError("Shards use different inference configurations.")
    if tuple(first["arms"]) != ARMS:
        raise ValueError("Unexpected four-arm ordering.")
    metrics, intersections = {}, {}
    for protocol, names in first["classes"].items():
        metrics[protocol] = {}
        for arm in ARMS:
            sources = [shard["metrics"][protocol][arm] for shard in shards]
            matrix = sum((np.asarray(item["confusion_matrix"], dtype=np.int64) for item in sources),
                         np.zeros((len(names), len(names)), dtype=np.int64))
            metrics[protocol][arm] = metric_summary(matrix, names,
                                                    sum(item["ignored_pixels"] for item in sources))
        sources = [shard["error_intersections"][protocol] for shard in shards]
        raw = {
            "valid_pixels": sum(item["valid_pixels"] for item in sources),
            "error_patterns": [sum(item["error_patterns"][index] for item in sources) for index in range(16)],
            "error_patterns_by_true_class": [
                [sum(item["error_patterns_by_true_class"][cls][index] for item in sources)
                 for index in range(16)] for cls in range(len(names))
            ],
            "pairwise": {pair: {field: sum(item["pairwise"][pair][field] for item in sources)
                                for field in sources[0]["pairwise"][pair]}
                         for pair in sources[0]["pairwise"]},
        }
        intersections[protocol] = summarize_error_counts(raw, names)
        for arm in ARMS:
            if sum(sum(row) for row in metrics[protocol][arm]["confusion_matrix"]) != raw["valid_pixels"]:
                raise ValueError("Error intersections and confusion matrices disagree.")
    tiles = sum(shard["diagnostics"]["tiles"] for shard in shards)
    result = {
        "status": "complete", "coverage_verified": True,
        "processed_images": len(keys), "total_images": len(keys),
        "metrics": metrics, "error_intersections": intersections,
        "diagnostics": {"tiles": tiles,
                        "sparse_active_fraction": sum(shard["diagnostics"]["tiles"] *
                                                      shard["diagnostics"]["sparse_active_fraction"]
                                                      for shard in shards) / max(tiles, 1)},
        "parallel_wall_seconds": max(shard["wall_seconds"] for shard in shards),
        "aggregate_gpu_seconds": sum(shard["wall_seconds"] for shard in shards),
        "peak_cuda_memory_mb": max(shard["peak_cuda_memory_mb"] for shard in shards),
        "signature": {**first, "sample_keys": keys, "sample_count": len(keys),
                      "sample_keys_sha256": digest(keys), "shard_index": None},
        "shards": [str(path) for path in paths],
    }
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": first["dataset"], "images": len(keys),
                      "miou": {key: {arm: value[arm]["mean_iou_percent"] for arm in ARMS}
                               for key, value in metrics.items()}}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    main(arguments.inputs, arguments.output)
