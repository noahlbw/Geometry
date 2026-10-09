#!/usr/bin/env python3
"""Merge VIP selection shards or six-arm evaluation shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dinotool.prompts import load_class_specs
from dinotool.vip_alias_distillation import VIPAliasAccumulator
from merge_competitive_evidence_shards import change_summary, metric_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("select", "evaluate"), required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load(paths):
    results = [json.loads((Path(value) / "results.json" if Path(value).is_dir() else Path(value)).read_text()) for value in paths]
    if not results or any(value.get("status") != "complete" for value in results):
        raise ValueError("Every shard must contain a complete results.json.")
    signatures = [value["signature"] for value in results]
    first = signatures[0]
    expected = first["num_shards"]
    indices = sorted(signature["shard_index"] for signature in signatures)
    if len(results) != expected or indices != list(range(expected)):
        raise ValueError(f"Expected exactly shards 0..{expected - 1}, got {indices}.")
    keys = [key for signature in signatures for key in signature["sample_keys"]]
    if len(keys) != len(set(keys)) or len(keys) != first["global_sample_count"]:
        raise ValueError("Shard sample keys overlap or do not cover the locked global sequence.")
    for signature in signatures[1:]:
        for field in ("implementation", "stage", "dataset", "global_sample_count", "global_sample_keys_sha256", "classes", "vocabularies", "checkpoints"):
            if signature[field] != first[field]:
                raise ValueError(f"Shard signatures disagree on {field}.")
    return results, first, keys


def _bank(metadata):
    specs = load_class_specs(metadata["path"])
    parents = torch.tensor([index for index, spec in enumerate(specs) for _ in spec.synonyms])
    canonical = torch.tensor([alias_index == 0 for spec in specs for alias_index in range(len(spec.synonyms))])
    return SimpleNamespace(
        parent_indices=parents,
        canonical_mask=canonical,
        class_count=len(specs),
        alias_names=tuple(alias for spec in specs for alias in spec.synonyms),
    )


def merge_select(results, first, keys):
    selections = {}
    for key, metadata in first["vocabularies"].items():
        aggregate = None
        for item in results:
            current = VIPAliasAccumulator.from_state_dict(item["states"][key], device=torch.device("cpu"))
            if aggregate is None:
                aggregate = current
            else:
                aggregate.merge(current)
        selections[key] = {
            "vocabulary_sha256": metadata["sha256"],
            "report": aggregate.report(_bank(metadata)),
        }
    return {
        "status": "complete",
        "stage": "select",
        "dataset": first["dataset"],
        "processed_images": len(keys),
        "total_images": first["global_sample_count"],
        "coverage_verified": True,
        "selections": selections,
        "signature": {**first, "sample_count": len(keys), "sample_keys": None, "sample_keys_sha256": first["global_sample_keys_sha256"], "shard_index": None},
    }


def merge_evaluate(results, first, keys):
    methods = first["methods"]
    names = first["classes"]
    metrics = {}
    for method in methods:
        matrix = sum((np.asarray(item["metrics"][method]["confusion_matrix"], dtype=np.int64) for item in results), np.zeros((len(names), len(names)), dtype=np.int64))
        ignored = sum(int(item["metrics"][method]["ignored_pixels"]) for item in results)
        metrics[method] = metric_summary(matrix, names, ignored)
        residual = {name.casefold() for name in first["residual_class_names"]}
        if residual:
            values = [entry["iou_percent"] for entry in metrics[method]["per_class"] if entry["name"].casefold() not in residual]
            metrics[method]["non_residual_mean_iou_percent"] = round(float(np.mean(values)), 4)
    suffix_base = {method: method.rsplit("_", 1)[0] + "_All20" for method in methods if not method.endswith("_All20")}
    count_fields = ("valid", "changed", "beneficial", "harmful", "wrong_to_wrong")
    changes = {
        method: change_summary({field: sum(int(item["changes_vs_matched_all"][method][field]) for item in results) for field in count_fields})
        for method in suffix_base
    }
    return {
        "status": "complete",
        "stage": "evaluate",
        "dataset": first["dataset"],
        "processed_images": len(keys),
        "total_images": first["global_sample_count"],
        "coverage_verified": True,
        "metrics": metrics,
        "delta_miou_vs_matched_all": {method: round(metrics[method]["mean_iou_percent"] - metrics[base]["mean_iou_percent"], 4) for method, base in suffix_base.items()},
        "changes_vs_matched_all": changes,
        "selection_counts_per_class": results[0]["selection_counts_per_class"],
        "parallel_wall_seconds": max(float(item["wall_seconds"]) for item in results),
        "aggregate_gpu_seconds": sum(float(item["wall_seconds"]) for item in results),
        "peak_cuda_memory_mb": max(float(item["peak_cuda_memory_mb"]) for item in results),
        "signature": {**first, "sample_count": len(keys), "sample_keys": None, "sample_keys_sha256": first["global_sample_keys_sha256"], "shard_index": None},
    }


def main(args: argparse.Namespace):
    results, first, keys = _load(args.inputs)
    merged = merge_select(results, first, keys) if args.stage == "select" else merge_evaluate(results, first, keys)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"stage": args.stage, "dataset": first["dataset"], "images": len(keys)}))
    return merged


if __name__ == "__main__":
    main(parse_args())
