#!/usr/bin/env python3
"""Merge complete alias counterfactual shards with sample and rule verification."""
from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np


def digest(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def scores(matrix: np.ndarray) -> dict:
    intersection = np.diag(matrix).astype(np.float64)
    target = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    union = target + predicted - intersection
    iou = np.divide(intersection, union, out=np.full(len(target), np.nan), where=union > 0)
    precision = np.divide(intersection, predicted, out=np.zeros(len(target)), where=predicted > 0)
    recall = np.divide(intersection, target, out=np.zeros(len(target)), where=target > 0)
    return {
        "mean_iou_percent": round(float(np.nanmean(iou) * 100), 4),
        "per_class_iou_percent": [None if np.isnan(value) else round(float(value * 100), 4)
                                  for value in iou],
        "per_class_precision_percent": [round(float(value * 100), 4) for value in precision],
        "per_class_recall_percent": [round(float(value * 100), 4) for value in recall],
        "predicted_area_percent": [round(float(value / max(matrix.sum(), 1) * 100), 4)
                                   for value in predicted],
    }


def merge(inputs: list[Path], output: Path) -> dict:
    shards = [json.loads((path / "results.json").read_text(encoding="utf-8")) for path in inputs]
    if not shards or any(item["status"] != "complete" or
                         item["processed_images"] != item["total_images"] for item in shards):
        raise ValueError("Every alias-audit shard must be complete.")
    signatures = [item["signature"] for item in shards]
    first = signatures[0]
    if len(shards) != first["num_shards"] or sorted(s["shard_index"] for s in signatures) != list(range(len(shards))):
        raise ValueError("Missing or duplicate shard indices.")
    fixed_fields = ("implementation", "dataset", "readout", "tile_size", "overlap",
                    "sample_stride", "sample_offset", "alias_temperature", "output_temperature",
                    "class_names", "alias_names", "alias_counts", "vocabulary_sha256",
                    "checkpoint_manifest", "global_sample_count", "global_sample_keys_sha256",
                    "num_shards")
    for shard, signature in zip(shards, signatures):
        if any(signature[field] != first[field] for field in fixed_fields):
            raise ValueError("Shard signatures differ.")
        keys = signature["sample_keys"]
        if len(keys) != shard["total_images"] or digest(keys) != signature["sample_keys_sha256"]:
            raise ValueError("Invalid shard sample keys.")
        if len(shard["aliases"]) != len(first["alias_names"]):
            raise ValueError("Incomplete alias list.")
    ordered = sorted(signatures, key=lambda item: item["shard_index"])
    keys = [key for row in zip_longest(*(s["sample_keys"] for s in ordered))
            for key in row if key is not None]
    if (len(keys) != len(set(keys)) or len(keys) != first["global_sample_count"]
            or digest(keys) != first["global_sample_keys_sha256"]):
        raise ValueError("Shard keys do not uniquely cover the fixed global set.")
    classes = len(first["class_names"])
    baseline = sum((np.asarray(shard["baseline_confusion"], dtype=np.int64)
                    for shard in shards), np.zeros((classes, classes), dtype=np.int64))
    baseline_metrics = scores(baseline)
    aliases = []
    for index, name in enumerate(first["alias_names"]):
        parts = [shard["aliases"][index] for shard in shards]
        if any(part["index"] != index or part["alias"] != name or
               part["class_index"] != parts[0]["class_index"] for part in parts):
            raise ValueError("Alias order differs between shards.")
        matrix = sum((np.asarray(part["confusion_matrix"], dtype=np.int64)
                      for part in parts), np.zeros_like(baseline))
        removed = sum((np.asarray(part["rival_fp_removed"], dtype=np.int64)
                       for part in parts), np.zeros(classes, dtype=np.int64))
        added = sum((np.asarray(part["rival_fp_added"], dtype=np.int64)
                     for part in parts), np.zeros(classes, dtype=np.int64))
        metrics = scores(matrix)
        parent = parts[0]["class_index"]
        aliases.append({
            "index": index, "class_index": parent, "class": parts[0]["class"], "alias": name,
            "without_alias_confusion": matrix.tolist(),
            "without_alias_metrics": metrics,
            "without_minus_full_miou_pp": round(metrics["mean_iou_percent"] -
                                                  baseline_metrics["mean_iou_percent"], 4),
            "without_minus_full_parent_iou_pp": round(
                metrics["per_class_iou_percent"][parent] -
                baseline_metrics["per_class_iou_percent"][parent], 4),
            "own_tp_lost": sum(part["own_tp_lost"] for part in parts),
            "own_tp_gained": sum(part["own_tp_gained"] for part in parts),
            "rival_fp_removed": removed.tolist(), "rival_fp_added": added.tolist(),
            "beneficial_flips": sum(part["beneficial_flips"] for part in parts),
            "harmful_flips": sum(part["harmful_flips"] for part in parts),
            "changed_predictions": sum(part["changed_predictions"] for part in parts),
        })
    result = {
        "status": "complete", "coverage_verified": True,
        "processed_images": len(keys), "total_images": len(keys),
        "sampled_pixels": sum(item["sampled_pixels"] for item in shards),
        "baseline_confusion": baseline.tolist(), "baseline_metrics": baseline_metrics,
        "aliases": aliases,
        "parallel_wall_seconds": max(item["wall_seconds"] for item in shards),
        "aggregate_gpu_seconds": sum(item["wall_seconds"] for item in shards),
        "peak_cuda_memory_mb": max(item["peak_cuda_memory_mb"] for item in shards),
        "signature": {**first, "sample_keys": keys, "sample_keys_sha256": digest(keys),
                      "shard_index": None},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": first["dataset"], "images": len(keys),
                      "sampled_pixels": result["sampled_pixels"],
                      "baseline_sampled_miou": baseline_metrics["mean_iou_percent"]}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merge(args.inputs, args.output)
