#!/usr/bin/env python3
"""Merge frozen context-innovation evaluations with exact sample coverage."""
from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np

from merge_competitive_evidence_shards import metric_summary, change_summary


def main(inputs, output):
    paths = [Path(p) / "results.json" for p in inputs]
    shards = [json.loads(p.read_text()) for p in paths]
    if not shards or any(s["status"] != "complete" for s in shards):
        raise ValueError("Every shard must be complete.")
    signatures = [s["signature"] for s in shards]
    ref = signatures[0]
    if sorted(s["shard_index"] for s in signatures) != list(range(ref["num_shards"])):
        raise ValueError("Missing or duplicate shard index.")
    ordered = sorted(signatures, key=lambda s: s["shard_index"])
    keys = [k for group in zip_longest(*(s["sample_keys"] for s in ordered)) for k in group if k is not None]
    digest = lambda seq: hashlib.sha256("\n".join(seq).encode()).hexdigest()
    if len(keys) != len(set(keys)) or len(keys) != ref["global_sample_count"]:
        raise ValueError("Incomplete or overlapping coverage.")
    if digest(keys) != ref["global_sample_keys_sha256"]:
        raise ValueError("Sample sequence differs from the locked protocol.")
    for shard, sig in zip(shards, signatures):
        if shard["processed_images"] != shard["total_images"] or shard["total_images"] != len(sig["sample_keys"]):
            raise ValueError("Incorrect shard counts.")
        if digest(sig["sample_keys"]) != sig["sample_keys_sha256"]:
            raise ValueError("Incorrect shard digest.")
        for field in ("implementation", "dataset", "methods", "classes", "grounded",
                      "contextual_phrase", "tcpr", "vocabulary", "checkpoints",
                      "global_sample_count", "global_sample_keys_sha256"):
            if sig[field] != ref[field]:
                raise ValueError(f"Inconsistent {field}.")
        normalize = lambda c: {k: v for k, v in c.items() if k not in ("output_dir", "shard_index")}
        if normalize(sig["config"]) != normalize(ref["config"]):
            raise ValueError("Inference configurations differ.")
    methods = ref["methods"]
    metrics, changes, diagnostics = {}, {}, {}
    for protocol, names in ref["classes"].items():
        metrics[protocol], changes[protocol] = {}, {}
        for method in methods:
            sources = [s["metrics"][protocol][method] for s in shards]
            matrix = sum((np.asarray(s["confusion_matrix"], dtype=np.int64) for s in sources),
                         np.zeros((len(names), len(names)), dtype=np.int64))
            summary = metric_summary(matrix, names, sum(s["ignored_pixels"] for s in sources))
            if protocol == "udd5":
                summary["non_residual_mean_iou_percent"] = round(float(np.mean(
                    [c["iou_percent"] for c in summary["per_class"] if c["name"] != "other"])), 4)
            metrics[protocol][method] = summary
            if method != methods[0]:
                totals = {field: sum(s["changes_vs_geometry"][protocol][method][field] for s in shards)
                          for field in ("valid", "changed", "beneficial", "harmful", "wrong_to_wrong")}
                changes[protocol][method] = change_summary(totals)
        tiles = sum(s["diagnostics"][protocol]["tiles"] for s in shards)
        diagnostics[protocol] = {"tiles": tiles}
        for field in shards[0]["diagnostics"][protocol]:
            if field != "tiles":
                diagnostics[protocol][field] = sum(s["diagnostics"][protocol][field] *
                    s["diagnostics"][protocol]["tiles"] for s in shards) / max(tiles, 1)
    result = {
        "status": "complete", "coverage_verified": True,
        "processed_images": len(keys), "total_images": len(keys),
        "metrics": metrics, "changes_vs_geometry": changes, "diagnostics": diagnostics,
        "parallel_wall_seconds": max(s["wall_seconds"] for s in shards),
        "aggregate_gpu_seconds": sum(s["wall_seconds"] for s in shards),
        "peak_cuda_memory_mb": max(s["peak_cuda_memory_mb"] for s in shards),
        "forward_counts": {k: sum(s["forward_counts"][k] for s in shards) for k in ("local", "context")},
        "signature": {**ref, "shard_index": None, "sample_count": len(keys), "sample_keys": keys,
                      "sample_keys_sha256": digest(keys)},
        "shards": [str(p) for p in paths],
    }
    if any(s.get("proposal_quality") for s in shards):
        if not all(s.get("proposal_quality") for s in shards):
            raise ValueError("Proposal-quality counts are missing from a shard.")
        result["proposal_quality"] = {}
        for protocol in ref["classes"]:
            result["proposal_quality"][protocol] = {}
            for method in shards[0]["proposal_quality"][protocol]:
                keys = ("proposed_changed", "proposed_beneficial", "proposed_harmful",
                        "beneficial_retained", "harmful_rejected")
                totals = {key: sum(s["proposal_quality"][protocol][method][key] for s in shards)
                          for key in keys}
                result["proposal_quality"][protocol][method] = {
                    **totals,
                    "beneficial_retention": totals["beneficial_retained"] / max(totals["proposed_beneficial"], 1),
                    "harmful_rejection": totals["harmful_rejected"] / max(totals["proposed_harmful"], 1),
                }
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": ref["dataset"], "images": len(keys),
                      "metrics": {p: {m: v["mean_iou_percent"] for m, v in s.items()}
                                  for p, s in metrics.items()}}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.inputs, args.output)
