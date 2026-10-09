#!/usr/bin/env python3
"""Use saved visual interventions to test score influence without new forwards."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.geometry_support_erasure import (
    IMPLEMENTATION, balanced_support_scores, reconstruct_support_scores,
    semantic_score_influence, support_operator,
)
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from dinotool.support_conditioned_geometry import SupportConfig
from eval_geometry_vip_reliability import summary


@torch.inference_mode()
def main(args):
    source, observations, output = Path(args.source), Path(args.observations), Path(args.output)
    if output.exists():
        raise ValueError("Refusing existing audit.")
    original = json.loads((source/"results.json").read_text())
    observed = json.loads(observations.read_text())
    if observed["status"] != "complete" or observed["snapshots"] != 16 or observed["maximum_control_error"] != 0.:
        raise ValueError("Requires complete verified V2 measurements.")
    records = {(row["sample_key"], row["top"], row["left"]): row for row in observed["records"]}
    images = {row["sample_key"]: row for row in original["images"]}
    names = original["signature"]["class_names"]
    classes = len(names)
    methods = ("Geometry", "BalancedErasure", "ScoreGain", "ScoreInfluence")
    matrices = {method: np.zeros((classes, classes), np.int64) for method in methods}
    transitions = {method: np.zeros((classes,)*3, np.int64) for method in methods if method != "Geometry"}
    maximum_reference_error, seen, new_records = 0., set(), []
    config = SupportConfig()
    for path in sorted((source/"features").glob("*.pt")):
        cache = torch.load(path, map_location="cpu", weights_only=True)
        key = (cache["sample_key"], cache["top"], cache["left"])
        if key not in records or key in seen:
            raise ValueError("Unexpected/duplicate cache.")
        seen.add(key)
        row, image = records[key], images[key[0]]
        yy = key[1]+torch.arange(32)*16+8
        xx = key[2]+torch.arange(32)*16+8
        valid = ((yy[:, None] < image["height"]) & (xx[None] < image["width"])).to(args.device)
        raw = F.normalize(cache["features"]["backbone_tokens"][:, cache["prefix_tokens"]:].to(args.device).float(), dim=-1)[0]
        local = cache["stage_scores"]["Geometry_original"][0].to(args.device)
        relation = cache["geometry_relation"][0].to(args.device)
        operator = support_operator(raw, relation, local, valid, config)
        measured = row["diagnostics"]["semantic_observations"]
        reference = torch.tensor(row["diagnostics"]["support_reference"], device=args.device)
        error = float((operator @ local-reference).abs().max()) if len(operator) else 0.
        maximum_reference_error = max(error, maximum_reference_error)
        if error > 1e-6 or len(operator) != row["diagnostics"]["regions"]:
            raise ValueError(f"Original support operator not reproduced: {error}")
        b = {method: torch.tensor(values, device=args.device) for method, values in measured.items()}
        if not len(operator):
            b = {method: local.new_zeros((0, classes)) for method in b}
        gain = b["SupportGlobal"]-b["SupportErasedGlobal"]
        scores = {"Geometry": local,
                  "BalancedErasure": balanced_support_scores(local, operator, b["SupportErasure"]),
                  "ScoreGain": semantic_score_influence(local, operator, gain, valid.reshape(-1), normalize_area=False),
                  "ScoreInfluence": semantic_score_influence(local, operator, gain, valid.reshape(-1))}
        predictions = {method: score.argmax(-1).cpu().numpy() for method, score in scores.items()}
        # Saved target centers are used only after observer reconstruction.
        truth = cache["target_patch_centers"][0].numpy()
        audit_valid = valid.reshape(-1).cpu().numpy() & (truth >= 0) & (truth < classes)
        for method, prediction in predictions.items():
            matrices[method] += np.bincount(truth[audit_valid].astype(np.int64)*classes+prediction[audit_valid], minlength=classes**2).reshape(classes, classes)
            if method != "Geometry":
                transitions[method] += transition_counts(predictions["Geometry"], prediction, np.where(audit_valid, truth, 255), classes)
        area = operator.sum(-1)/operator.amax(-1).clamp_min(1e-12)/valid.sum() if len(operator) else operator.new_zeros((0,))
        new_records.append({"sample_key": key[0], "top": key[1], "left": key[2],
                            "score_gain": gain.cpu().tolist(), "erasure_area_fraction": area.cpu().tolist()})
        del cache
    if seen != set(records) or len(seen) != 16:
        raise ValueError("Incomplete cache/measurement pairing.")
    for method in ("Geometry", "BalancedErasure"):
        if not np.array_equal(matrices[method], observed["metrics"][method]["confusion_matrix"]):
            raise ValueError(f"Saved observer/control not exactly reproduced: {method}")
    result = {"status": "complete", "implementation": IMPLEMENTATION, "dataset": observed["dataset"],
              "snapshots": 16, "source_observation_controls_exact": True,
              "maximum_support_reference_error": maximum_reference_error,
              "signature": {"config": asdict(config), "primary": "ScoreInfluence",
                            "aliases": observed["signature"]["aliases"], "source_samples": original["signature"]["samples"],
                            "rule": "native full class-score change / soft-erasure area; balanced support-space incremental writeback"},
              "metrics": {method: summary(matrix, names, 0) for method, matrix in matrices.items()},
              "transitions": {method: {"counts": tensor.tolist(), **transition_summary(tensor)} for method, tensor in transitions.items()},
              "records": new_records,
              "note": "No new model/image forwards. Fixed rule applied to the same saved image interventions. Geometry and V2 controls reproduce exactly. Overlapping patch centers, not full-image mIoU; target labels audit-only. No per-class sign, threshold or fitted coefficient."}
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"dataset": result["dataset"], "controls_exact": True,
                      "metrics": {method: row["mean_iou_percent"] for method, row in result["metrics"].items()},
                      "changes": {method: {key: row[key] for key in ("beneficial", "harmful", "wrong_to_wrong")}
                                  for method, row in result["transitions"].items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
