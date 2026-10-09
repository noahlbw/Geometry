#!/usr/bin/env python3
"""Merge matched innovation and direct candidate-versus-fusion audits."""
import argparse
import json
from pathlib import Path

import numpy as np

from dinotool.semantic_correction_audit import transition_summary
from merge_geometry_semantic_innovation import main as merge


def main(args):
    rows = [json.loads((Path(path)/"results.json").read_text()) for path in args.inputs]
    comparisons = {}
    for key, methods in rows[0]["direct_comparisons"].items():
        comparisons[key] = {}
        for baseline in methods:
            counts = sum(np.asarray(row["direct_comparisons"][key][baseline]["counts"], np.int64) for row in rows)
            cm_base = sum(np.asarray(row["metrics"][key][baseline]["confusion_matrix"], np.int64) for row in rows)
            cm_new = sum(np.asarray(row["metrics"][key]["CounterfactualGSI"]["confusion_matrix"], np.int64) for row in rows)
            if not np.array_equal(counts.sum(1).T, cm_base) or not np.array_equal(counts.sum(0).T, cm_new):
                raise ValueError("Direct audit does not reproduce both confusion matrices.")
            comparisons[key][baseline] = {"counts": counts.tolist(), **transition_summary(counts)}
    merge(args)
    output = Path(args.output)
    result = json.loads(output.read_text())
    result["direct_comparisons"] = comparisons
    output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
