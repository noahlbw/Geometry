#!/usr/bin/env python3
"""Merge exact confusion matrices from sharded CIDER E1 evaluations."""
from __future__ import annotations

import argparse
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
    paths = [
        Path(value) / "results.json" if Path(value).is_dir() else Path(value)
        for value in args.inputs
    ]
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
    for signature in signatures[1:]:
        for field in (
            "global_sample_count",
            "global_sample_keys_sha256",
            "methods",
            "checkpoints",
            "vocabulary",
            "cider",
        ):
            if signature[field] != first[field]:
                raise ValueError(f"Shard signatures disagree on {field}.")

    methods = first["methods"]
    metrics: dict[str, dict[str, object]] = {"P": {}, "D": {}}
    for protocol in ("P", "D"):
        for method in methods:
            source = shards[0]["metrics"][protocol][method]
            names = [item["name"] for item in source["per_class"]]
            matrix = sum(
                (
                    np.asarray(
                        shard["metrics"][protocol][method]["confusion_matrix"],
                        dtype=np.int64,
                    )
                    for shard in shards
                ),
                np.zeros((len(names), len(names)), dtype=np.int64),
            )
            ignored = sum(
                int(shard["metrics"][protocol][method]["ignored_pixels"])
                for shard in shards
            )
            metrics[protocol][method] = metric_summary(matrix, names, ignored)

    count_fields = ("valid", "changed", "beneficial", "harmful", "wrong_to_wrong")
    changes: dict[str, dict[str, object]] = {"P": {}, "D": {}}
    for protocol in ("P", "D"):
        for method in methods:
            if method == "S0_TextGraph":
                continue
            totals = {
                field: sum(
                    int(shard["changes_vs_s0"][protocol][method][field])
                    for shard in shards
                )
                for field in count_fields
            }
            changes[protocol][method] = change_summary(totals)

    arms = ("CIDER", "CIDER_PairShuffled")
    diagnostics: dict[str, object] = {}
    for protocol in ("P", "D"):
        tile_count = sum(int(shard["diagnostics"][protocol]["tiles"]) for shard in shards)
        protocol_summary: dict[str, object] = {"tiles": tile_count}
        for arm in arms:
            fields = shards[0]["diagnostics"][protocol][arm]
            arm_summary = {}
            for field in fields:
                if field == "maximum_attention_kl":
                    arm_summary[field] = max(
                        float(shard["diagnostics"][protocol][arm][field]) for shard in shards
                    )
                else:
                    arm_summary[field] = sum(
                        float(shard["diagnostics"][protocol][arm][field])
                        * int(shard["diagnostics"][protocol]["tiles"])
                        for shard in shards
                    ) / max(tile_count, 1)
            protocol_summary[arm] = arm_summary
        diagnostics[protocol] = protocol_summary

    result = {
        "status": "complete",
        "processed_images": len(keys),
        "total_images": first["global_sample_count"],
        "metrics": metrics,
        "delta_miou_vs_s0": {
            protocol: {
                method: round(
                    metrics[protocol][method]["mean_iou_percent"]
                    - metrics[protocol]["S0_TextGraph"]["mean_iou_percent"],
                    4,
                )
                for method in methods
            }
            for protocol in ("P", "D")
        },
        "changes_vs_s0": changes,
        "diagnostics": diagnostics,
        "parallel_wall_seconds": max(float(shard["wall_seconds"]) for shard in shards),
        "aggregate_gpu_seconds": sum(float(shard["wall_seconds"]) for shard in shards),
        "peak_cuda_memory_mb": max(float(shard["peak_cuda_memory_mb"]) for shard in shards),
        "shards": [str(path) for path in paths],
        "signature": {
            **first,
            "sample_count": first["global_sample_count"],
            "sample_keys_sha256": first["global_sample_keys_sha256"],
            "shard_index": None,
            "sample_keys": None,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True))
    print(json.dumps({
        "images": len(keys),
        "P": {method: metrics["P"][method]["mean_iou_percent"] for method in methods},
        "D": {method: metrics["D"][method]["mean_iou_percent"] for method in methods},
    }))
    return result


if __name__ == "__main__":
    main(parse_args())
