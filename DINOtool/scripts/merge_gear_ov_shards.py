#!/usr/bin/env python3
"""Merge GEAR-OV shards after verifying the complete global sample sequence."""
from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np

from merge_competitive_evidence_shards import metric_summary


def _digest(keys) -> str:
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def merge(inputs: list[str], output: str, *, diagnostic_weight: str = "tiles") -> dict[str, object]:
    if diagnostic_weight not in ("tiles", "images"):
        raise ValueError("Diagnostic weight must be tiles or images.")
    paths = [Path(path) / "results.json" for path in inputs]
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not shards or any(item.get("status") != "complete" for item in shards):
        raise ValueError("All shards must have complete results.json files.")
    signatures = [item["signature"] for item in shards]
    first = signatures[0]
    if sorted(item["shard_index"] for item in signatures) != list(range(first["num_shards"])):
        raise ValueError("Shard indices are incomplete or duplicated.")
    if len(shards) != first["num_shards"]:
        raise ValueError("Wrong number of shards.")
    ordered = sorted(signatures, key=lambda item: item["shard_index"])
    keys = [key for group in zip_longest(*(item["sample_keys"] for item in ordered))
            for key in group if key is not None]
    if (len(keys) != len(set(keys)) or len(keys) != first["global_sample_count"]
            or _digest(keys) != first["global_sample_keys_sha256"]):
        raise ValueError("Shard keys do not uniquely cover the global sample sequence.")
    for shard, item in zip(shards, signatures):
        if (shard["processed_images"] != shard["total_images"]
                or shard["total_images"] != len(item["sample_keys"])
                or _digest(item["sample_keys"]) != item["sample_keys_sha256"]):
            raise ValueError("Incorrect per-shard count or sample key digest.")
        for field in ("implementation", "dataset", "methods", "classes", "gear",
                      "competitive",
                      "vocabulary", "checkpoints", "global_sample_count",
                      "global_sample_keys_sha256"):
            if item.get(field) != first.get(field):
                raise ValueError(f"Shards have inconsistent {field}.")
        trim = lambda cfg: {key: val for key, val in cfg.items()
                            if key not in ("output_dir", "shard_index")}
        if trim(item["config"]) != trim(first["config"]):
            raise ValueError("Shards use different inference settings.")
    metrics = {}
    diagnostics = {}
    for key, names in first["classes"].items():
        metrics[key] = {}
        for method in first["methods"]:
            sources = [shard["metrics"][key][method] for shard in shards]
            matrix = sum((np.asarray(source["confusion_matrix"], dtype=np.int64)
                          for source in sources), np.zeros((len(names), len(names)), np.int64))
            summary = metric_summary(matrix, names,
                                     sum(source["ignored_pixels"] for source in sources))
            if key in ("udd5", "landcoverai"):
                values = [item["iou_percent"] for item in summary["per_class"]
                          if item["name"] not in ("other", "background")
                          and item["iou_percent"] is not None]
                summary["non_residual_mean_iou_percent"] = (
                    round(float(np.mean(values)), 4) if values else None
                )
            metrics[key][method] = summary
        weights = [shard["diagnostics"][key]["tiles"] if diagnostic_weight == "tiles"
                   else shard["processed_images"] for shard in shards]
        weight_sum = sum(weights)
        diagnostics[key] = {"tiles": weight_sum} if diagnostic_weight == "tiles" else {}
        for field in shards[0]["diagnostics"][key]:
            if field != "tiles" or diagnostic_weight == "images":
                diagnostics[key][field] = (
                    sum(shard["diagnostics"][key][field] * weight
                        for shard, weight in zip(shards, weights)) / max(weight_sum, 1)
                )
    result = {
        "status": "complete", "coverage_verified": True,
        "processed_images": len(keys), "total_images": len(keys),
        "metrics": metrics, "diagnostics": diagnostics,
        "parallel_wall_seconds": max(shard["wall_seconds"] for shard in shards),
        "aggregate_gpu_seconds": sum(shard["wall_seconds"] for shard in shards),
        "peak_cuda_memory_mb": max(shard["peak_cuda_memory_mb"] for shard in shards),
        "signature": {**first, "shard_index": None, "sample_count": len(keys),
                      "sample_keys": keys, "sample_keys_sha256": _digest(keys)},
        "shards": [str(path) for path in paths],
    }
    if diagnostic_weight == "images":
        result["diagnostic_weighting"] = "processed_images"
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": first["dataset"], "images": len(keys),
                      "miou": {key: {method: item["mean_iou_percent"]
                                      for method, item in values.items()}
                               for key, values in metrics.items()}}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merge(args.inputs, args.output)
