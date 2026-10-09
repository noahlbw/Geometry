#!/usr/bin/env python3
"""Audit saved head/text factorial counts; no inference or parameter selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_semantic_correction_operating_point import class_statistics, iou_decomposition


ARMS = ("Geometry_RS", "VIP_RS", "Geometry_ImageNet", "VIP_ImageNet")


def rates(stats: dict, total: int) -> dict:
    return {
        "precision_percent": 100 * stats["tp"] / stats["predicted"] if stats["predicted"] else None,
        "recall_percent": 100 * stats["tp"] / stats["gt"] if stats["gt"] else None,
        "predicted_area_percent": 100 * stats["predicted"] / total,
        "gt_area_percent": 100 * stats["gt"] / total,
    }


def audit(payload: dict) -> dict:
    if (payload.get("status") != "complete" or not payload.get("coverage_verified")
            or not payload.get("processed_images")
            or payload["processed_images"] != payload.get("total_images")):
        raise ValueError("Requires a complete coverage-verified saved result.")
    metrics = payload["metrics"]
    if any(arm not in metrics for arm in ARMS):
        raise ValueError("Missing a head/text factorial arm.")
    names = [row["name"] for row in metrics[ARMS[0]]["per_class"]]
    matrices = {arm: metrics[arm]["confusion_matrix"] for arm in ARMS}
    stats = {arm: class_statistics(matrices[arm]) for arm in ARMS}
    gt = [row["gt"] for row in stats[ARMS[0]]]
    total = sum(gt)
    if not total:
        raise ValueError("No evaluated pixels.")
    for arm in ARMS:
        if [row["name"] for row in metrics[arm]["per_class"]] != names:
            raise ValueError("Class order changed between factorial arms.")
        if len(stats[arm]) != len(names) or [row["gt"] for row in stats[arm]] != gt:
            raise ValueError("Ground-truth totals changed between factorial arms.")

    def contrast(before: str, after: str) -> dict:
        decomposition = iou_decomposition(matrices[before], matrices[after], names)
        for row in decomposition:
            row["base_rates"] = rates(row["base"], total)
            row["candidate_rates"] = rates(row["candidate"], total)
        valid = [row for row in decomposition if row["delta_iou_pp"] is not None]
        return {
            "before": before,
            "after": after,
            "delta_miou_pp": sum(row["delta_iou_pp"] for row in valid) / len(valid),
            "per_class": decomposition,
        }

    text_geometry = contrast("Geometry_RS", "Geometry_ImageNet")
    text_vip = contrast("VIP_RS", "VIP_ImageNet")
    head_rs = contrast("Geometry_RS", "VIP_RS")
    head_imagenet = contrast("Geometry_ImageNet", "VIP_ImageNet")
    interactions = []
    for geo, vip, rs, imagenet in zip(text_geometry["per_class"], text_vip["per_class"],
                                     head_rs["per_class"], head_imagenet["per_class"]):
        interaction = (vip["delta_iou_pp"] - geo["delta_iou_pp"]
                       if vip["delta_iou_pp"] is not None and geo["delta_iou_pp"] is not None
                       else None)
        if interaction is not None:
            if abs(interaction - (imagenet["delta_iou_pp"] - rs["delta_iou_pp"])) > 1e-10:
                raise AssertionError("The two factorial paths disagree.")
        interactions.append({
            "class": geo["class"],
            "iou_interaction_pp": interaction,
            "tp_interaction": vip["delta_tp"] - geo["delta_tp"],
            "fp_interaction": vip["delta_fp"] - geo["delta_fp"],
        })
    signature = payload.get("signature", {})
    return {
        "dataset": signature.get("dataset"),
        "processed_images": payload["processed_images"],
        "evaluated_pixels": total,
        "sample_keys_sha256": signature.get("global_sample_keys_sha256"),
        "vocabulary_sha256": signature.get("vocabulary_sha256"),
        "protocol": {key: signature.get(key) for key in
                     ("implementation", "crop", "stride", "long_edges", "threshold",
                      "rs_templates", "vip_templates")},
        "contrasts": {"text_geometry": text_geometry, "text_vip": text_vip,
                      "head_rs": head_rs, "head_imagenet": head_imagenet},
        "miou_interaction_pp": text_vip["delta_miou_pp"] - text_geometry["delta_miou_pp"],
        "per_class_interactions": interactions,
        "note": ("Descriptive interactions on already evaluated labels, not calibrated correction "
                 "signals, independent validation, or evidence for choosing a head by true class."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {
        "status": "complete",
        "note": "Saved-output audit only; no inference, tuning, or vocabulary output.",
        "datasets": [{"source": str(path), **audit(json.loads(path.read_text(encoding="utf-8-sig")))}
                     for path in args.inputs],
    }
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)


if __name__ == "__main__":
    main()
