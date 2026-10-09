#!/usr/bin/env python3
"""Coverage-verified innovation merge including exact A/B/truth transitions."""
import argparse
import json
from pathlib import Path

import numpy as np

from dinotool.semantic_correction_audit import transition_summary
from merge_gear_ov_shards import merge


def main(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError("Refusing existing merged output.")
    shards = [json.loads((Path(path)/"results.json").read_text()) for path in args.inputs]
    transitions = {}
    for key, methods in shards[0]["metrics"].items():
        transitions[key] = {}
        baseline = sum(np.asarray(row["metrics"][key]["Geometry"]["confusion_matrix"], np.int64)
                       for row in shards)
        for method in methods:
            if method == "Geometry":
                continue
            counts = sum(np.asarray(row["transitions"][key][method]["counts"], np.int64) for row in shards)
            proposal = sum(np.asarray(row["metrics"][key][method]["confusion_matrix"], np.int64)
                           for row in shards)
            if (not np.array_equal(counts.sum(1).T, baseline)
                    or not np.array_equal(counts.sum(0).T, proposal)):
                raise ValueError("Transitions do not reproduce the two confusion matrices.")
            transitions[key][method] = {"counts": counts.tolist(), **transition_summary(counts)}
    result = merge(args.inputs, args.output)
    for methods in result["metrics"].values():
        for metric in methods.values():
            matrix = np.asarray(metric["confusion_matrix"], np.int64)
            for index, entry in enumerate(metric["per_class"]):
                tp, predicted, target = int(matrix[index, index]), int(matrix[:, index].sum()), int(matrix[index].sum())
                entry.update(precision_percent=100*tp/max(predicted, 1), recall_percent=100*tp/max(target, 1),
                             target_pixels=target, predicted_pixels=predicted,
                             predicted_area_percent=100*predicted/max(int(matrix.sum()), 1))
    result["transitions"] = transitions
    output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
