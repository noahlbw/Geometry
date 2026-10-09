#!/usr/bin/env python3
"""Merge the four fixed observer shards and verify the original Geometry control."""
from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np

from merge_competitive_evidence_shards import metric_summary


def merge(root, source_path):
    source = json.loads(source_path.read_text())
    shards = [json.loads((root / f"s{i}" / "results.json").read_text()) for i in range(4)]
    first = shards[0]["signature"]
    keys = [key for group in zip_longest(*(row["signature"]["sample_keys"] for row in shards))
            for key in group if key is not None]
    if (keys != source["signature"]["samples"] or len(keys) != 8 or len(set(keys)) != 8
            or hashlib.sha256("\n".join(keys).encode()).hexdigest() != first["global_sample_keys_sha256"]):
        raise ValueError("Observer shards do not exactly cover the original diagnostic sequence.")
    for i, shard in enumerate(shards):
        signature = shard["signature"]
        if (shard["status"] != "complete" or shard["processed_images"] != shard["total_images"]
                or shard["total_images"] != len(signature["sample_keys"])
                or signature["shard_index"] != i or signature["num_shards"] != 4
                or [row["sample_key"] for row in shard["images"]] != signature["sample_keys"]):
            raise ValueError("Incomplete or mismatched shard.")
        for field in first:
            if field not in ("shard_index", "sample_keys") and signature[field] != first[field]:
                raise ValueError(f"Mismatched observer configuration: {field}")
    names, methods = first["class_names"], first["methods"]
    metrics = {}
    for method in methods:
        sources = [row["metrics"][method] for row in shards]
        matrix = sum(np.asarray(row["confusion_matrix"], dtype=np.int64) for row in sources)
        metrics[method] = metric_summary(matrix, names, sum(row["ignored_pixels"] for row in sources))
        for i, entry in enumerate(metrics[method]["per_class"]):
            tp, predicted, truth = int(matrix[i, i]), int(matrix[:, i].sum()), int(matrix[i].sum())
            entry.update(precision_percent=100*tp/max(predicted, 1), recall_percent=100*tp/max(truth, 1),
                         target_pixels=truth, predicted_pixels=predicted,
                         predicted_area_percent=100*predicted/max(int(matrix.sum()), 1))
    if metrics["Geometry"]["confusion_matrix"] != source["stage_probe_metrics"]["Geometry_original"]["confusion_matrix"]:
        raise ValueError("Original Geometry whole-image predictions changed.")
    transitions = {}
    for method in methods:
        if method == "Geometry":
            continue
        counts = sum(np.asarray(row["transitions"][method]["counts"], dtype=np.int64) for row in shards)
        if (counts.sum(1).T.tolist() != metrics["Geometry"]["confusion_matrix"]
                or counts.sum(0).T.tolist() != metrics[method]["confusion_matrix"]):
            raise ValueError("Transition tensor does not reproduce both confusion matrices.")
        old, new, truth = np.indices(counts.shape)
        changed = old != new
        transitions[method] = {"counts": counts.tolist(), "changed": int(counts[changed].sum()),
                               "beneficial": int(counts[changed & (new == truth)].sum()),
                               "harmful": int(counts[changed & (old == truth)].sum()),
                               "wrong_to_wrong": int(counts[changed & (old != truth) & (new != truth)].sum())}
    if sum(row["windows"] for row in shards) != source["windows"]:
        raise ValueError("Original windows changed.")
    return {"status": "complete", "coverage_verified": True, "geometry_original_verified": True,
            "processed_images": 8, "total_images": 8, "metrics": metrics, "transitions": transitions,
            "windows": source["windows"], "signature": {**first, "shard_index": None, "sample_keys": keys},
            "images": sorted([record for row in shards for record in row["images"]], key=lambda r: r["image_index"]),
            "parallel_wall_seconds": max(row["wall_seconds"] for row in shards),
            "aggregate_gpu_seconds": sum(row["wall_seconds"] for row in shards),
            "peak_cuda_memory_mb": max(row["peak_cuda_memory_mb"] for row in shards)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    output = args.root / "merged.json"
    if output.exists():
        raise ValueError("Refusing an existing merged output.")
    result = merge(args.root, args.source)
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"dataset": result["signature"]["dataset"], "geometry_original_verified": True,
                      "miou": {key: value["mean_iou_percent"] for key, value in result["metrics"].items()}}))
