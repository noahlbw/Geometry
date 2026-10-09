#!/usr/bin/env python3
"""Coverage-checked merge of the image-only Geometry/VIP reliability audit."""
from __future__ import annotations

import argparse
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np

from dinotool.semantic_correction_audit import transition_summary
from eval_geometry_vip_reliability import ARMS, IMPLEMENTATION, add_counts, summary
from eval_matched_head_fov import digest


def merge(inputs, output):
    if output.exists():
        raise ValueError(f"Refusing existing output: {output}")
    shards = [json.loads((path / "results.json").read_text()) for path in inputs]
    if not shards or any(j["status"] != "complete" or j["processed_images"] != j["total_images"] for j in shards):
        raise ValueError("Every shard must be complete.")
    shards.sort(key=lambda j: j["signature"]["shard_index"])
    signatures = [j["signature"] for j in shards]
    first = signatures[0]
    if (first["implementation"] != IMPLEMENTATION or tuple(first["arms"]) != ARMS
            or len(shards) != first["num_shards"]
            or [s["shard_index"] for s in signatures] != list(range(len(shards)))):
        raise ValueError("Missing shards or unexpected experiment.")
    skip = ("sample_keys", "sample_keys_sha256", "shard_index")
    fixed = {k: v for k, v in first.items() if k not in skip}
    for j, sig in zip(shards, signatures):
        if {k: v for k, v in sig.items() if k not in skip} != fixed:
            raise ValueError("Shard rules, vocabulary or checkpoints differ.")
        if len(sig["sample_keys"]) != j["total_images"] or digest(sig["sample_keys"]) != sig["sample_keys_sha256"]:
            raise ValueError("Invalid shard keys.")
    keys = [key for row in zip_longest(*(s["sample_keys"] for s in signatures)) for key in row if key is not None]
    if (len(keys) != len(set(keys)) or len(keys) != first["global_sample_count"]
            or digest(keys) != first["global_sample_keys_sha256"]):
        raise ValueError("Incomplete, duplicate or reordered global coverage.")
    names = tuple(first["class_names"])
    metrics = {}
    for arm in ARMS:
        matrix = sum((np.asarray(j["metrics"][arm]["confusion_matrix"], dtype=np.int64) for j in shards),
                     np.zeros((len(names), len(names)), dtype=np.int64))
        metrics[arm] = summary(matrix, names, sum(j["metrics"][arm]["ignored_pixels"] for j in shards))
    audit, diagnostics = {}, {}
    with_transitions = ["pair_transition_counts" in j["audit"] for j in shards]
    if any(with_transitions) and not all(with_transitions):
        raise ValueError("Do not mix old and transition-instrumented audit shards.")
    for j in shards:
        audit = add_counts(audit, j["audit"])
        diagnostics = add_counts(diagnostics, j["diagnostics"])
    counts = np.asarray(audit["correctness_by_true_class"], dtype=np.int64)
    valid = int(counts.sum())
    audit["pattern_order"] = ["both_wrong", "Multiscale_only_correct", "VIP_only_correct", "both_correct"]
    if all(with_transitions):
        transitions = transition_summary(np.asarray(audit["pair_transition_counts"]))
        if (transitions["valid"] != valid
                or transitions["base_confusion"] != metrics["Multiscale"]["confusion_matrix"]
                or transitions["proposal_confusion"] != metrics["VIP20"]["confusion_matrix"]):
            raise ValueError("Transition tensor does not reproduce both branch confusions.")
        audit["pair_transition_axes"] = ["old_Multiscale", "proposal_VIP20", "ground_truth"]
    audit["oracle_any_correct_accuracy_percent"] = 100 * int(counts[:, 1:].sum()) / max(valid, 1)
    audit["oracle_note"] = "GT-only pixel-accuracy audit, not a deployable method or an mIoU upper bound."
    for record in audit["routing"].values():
        for output_key, numerator, denominator in (
            ("beneficial_retention", "retained_beneficial", "available_beneficial"),
            ("harmful_rejection", "rejected_harmful", "available_harmful"),
            ("xor_route_accuracy", "correct_xor_routes", "xor_total"),
        ):
            record[output_key] = record[numerator] / record[denominator] if record[denominator] else None
    result = {"status": "complete", "coverage_verified": True,
              "processed_images": len(keys), "total_images": len(keys), "metrics": metrics,
              "audit": audit, "diagnostics": diagnostics,
              "parallel_wall_seconds": max(j["wall_seconds"] for j in shards),
              "aggregate_gpu_seconds": sum(j["wall_seconds"] for j in shards),
              "peak_cuda_memory_mb": max(j["peak_cuda_memory_mb"] for j in shards),
              "signature": {**first, "sample_keys": keys, "sample_keys_sha256": digest(keys), "shard_index": None}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": first["dataset"], "images": len(keys),
                      "miou": {key: metrics[key]["mean_iou_percent"] for key in ARMS}}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merge(args.inputs, args.output)
