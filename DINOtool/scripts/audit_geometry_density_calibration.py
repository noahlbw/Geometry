#!/usr/bin/env python3
"""Fixed unlabeled density calibration, then labeled audit of saved snapshots."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dinotool.geometry_density_calibration import IMPLEMENTATION, calibrate_margins
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from merge_competitive_evidence_shards import metric_summary


METHODS = ("Geometry", "DensityShift", "GeoDensityShift")


def main(args):
    root, output = Path(args.input), Path(args.output)
    if output.exists():
        raise ValueError("Preserve existing audits.")
    reference = json.loads((root/"results.json").read_text())
    images = {row["sample_key"]: row for row in reference["images"]}
    names = reference["signature"]["class_names"]
    classes = len(names)
    matrices = {method: np.zeros((classes, classes), np.int64) for method in METHODS}
    transitions = {method: np.zeros((classes,)*3, np.int64) for method in METHODS if method != "Geometry"}
    records = []
    torch.set_num_threads(2)
    for path in sorted((root/"features").glob("*.pt")):
        cache = torch.load(path, map_location="cpu", weights_only=True)
        image = images[cache["sample_key"]]
        yy = cache["top"]+np.arange(32)*16+8
        xx = cache["left"]+np.arange(32)*16+8
        geometric_valid = ((yy[:, None] < image["height"]) & (xx[None] < image["width"])).reshape(-1)
        scores = cache["stage_scores"]["Geometry_original"][0].numpy()
        relation = cache["geometry_relation"][0].float().numpy()
        proposals, fits = {"Geometry": scores.argmax(-1)}, {}
        for method, graph in (("DensityShift", None), ("GeoDensityShift", relation)):
            corrected, result = calibrate_margins(scores, geometric_valid, graph)
            proposals[method], fits[method] = corrected.argmax(-1), result
        # The calibration functions receive only scores, image bounds and relations.
        truth = cache["target_patch_centers"][0].numpy()
        valid = geometric_valid & (truth >= 0) & (truth < classes)
        for method, prediction in proposals.items():
            encoded = truth[valid].astype(np.int64)*classes+prediction[valid]
            matrices[method] += np.bincount(encoded, minlength=classes**2).reshape(classes, classes)
            if method != "Geometry":
                transitions[method] += transition_counts(proposals["Geometry"], prediction, np.where(valid, truth, 255), classes)
        records.append({"sample_key": cache["sample_key"], "top": cache["top"], "left": cache["left"], "fits": fits})
    if len(records) != 16:
        raise ValueError("Expected the fixed 16 snapshots.")
    result = {"implementation": IMPLEMENTATION, "dataset": reference["signature"]["dataset"],
              "snapshots": len(records), "metrics": {method: metric_summary(cm, names, 0) for method, cm in matrices.items()},
              "transitions": {method: {"counts": counts.tolist(), **transition_summary(counts)} for method, counts in transitions.items()},
              "records": records,
              "note": "Image-score-only fixed GMM/BIC rule; labels are audit-only. Overlapping patch centers of two cached windows per image, not full-image mIoU. Development data; no class-specific rule or coefficient chosen."}
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"dataset": result["dataset"], "snapshots": len(records),
                      "miou": {method: row["mean_iou_percent"] for method, row in result["metrics"].items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
