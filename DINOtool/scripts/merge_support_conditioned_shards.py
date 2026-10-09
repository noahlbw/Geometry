#!/usr/bin/env python3
"""Use the coverage-verified merger and retain paired changes vs Multiscale."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_grounded_context import write_json
from merge_gear_ov_shards import merge


def merge_supported(inputs: list[str], output: str):
    if Path(output).exists():
        raise ValueError("Refusing to overwrite a merged result.")
    shards = [json.loads((Path(path) / "results.json").read_text()) for path in inputs]
    for shard in shards:
        for key, names in shard["signature"]["classes"].items():
            counts = shard["changes"][key]
            if (counts["changed"] > counts["valid"] or counts["changed"] != sum(
                    counts[name] for name in ("beneficial", "harmful", "wrong_to_wrong"))):
                raise ValueError("Paired changes do not partition valid changed pixels.")
            for kind in ("beneficial", "harmful"):
                per_class = counts[kind + "_by_true_class"]
                if len(per_class) != len(names) or sum(per_class) != counts[kind]:
                    raise ValueError("Per-class changes do not agree with totals.")
    result = merge(inputs, output)
    changes = {}
    for key in result["signature"]["classes"]:
        changes[key] = {}
        for name, value in shards[0]["changes"][key].items():
            changes[key][name] = ([sum(shard["changes"][key][name][i] for shard in shards)
                                   for i in range(len(value))] if isinstance(value, list)
                                  else sum(shard["changes"][key][name] for shard in shards))
    result["changes"] = changes
    write_json(Path(output), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merge_supported(args.inputs, args.output)
