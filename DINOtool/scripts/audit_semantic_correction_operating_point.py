#!/usr/bin/env python3
"""Read-only correction audit from saved paired counts and confusion matrices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def class_statistics(matrix: list[list[int]]) -> list[dict]:
    count = len(matrix)
    if not count or any(len(row) != count for row in matrix):
        raise ValueError("Expected a square confusion matrix.")
    if any(value < 0 for row in matrix for value in row):
        raise ValueError("Negative confusion count.")
    rows = [sum(row) for row in matrix]
    columns = [sum(matrix[i][j] for i in range(count)) for j in range(count)]
    return [{"tp": matrix[c][c], "gt": rows[c],
             "fp": columns[c] - matrix[c][c],
             "fn": rows[c] - matrix[c][c], "predicted": columns[c]}
            for c in range(count)]


def iou_decomposition(base: list[list[int]], candidate: list[list[int]],
                      names: list[str]) -> list[dict]:
    before, after = class_statistics(base), class_statistics(candidate)
    if len(before) != len(after) or len(names) != len(before):
        raise ValueError("Incompatible class dimensions.")
    result = []
    for name, old, new in zip(names, before, after):
        if old["gt"] != new["gt"]:
            raise ValueError("Ground-truth totals changed between methods.")
        d0, d1 = old["gt"] + old["fp"], new["gt"] + new["fp"]
        dt, df = new["tp"] - old["tp"], new["fp"] - old["fp"]
        old_iou = old["tp"] / d0 if d0 else None
        new_iou = new["tp"] / d1 if d1 else None
        tp_term = 100 * dt / d1 if d0 and d1 else None
        fp_term = -100 * old["tp"] * df / (d0 * d1) if d0 and d1 else None
        delta = 100 * (new_iou - old_iou) if d0 and d1 else None
        if delta is not None and abs(delta - tp_term - fp_term) > 1e-10:
            raise AssertionError("Exact IoU decomposition failed.")
        result.append({"class": name, "base": old, "candidate": new,
                       "delta_tp": dt, "delta_fp": df,
                       "base_iou_percent": 100 * old_iou if d0 else None,
                       "candidate_iou_percent": 100 * new_iou if d1 else None,
                       "delta_iou_pp": delta, "tp_contribution_pp": tp_term,
                       "fp_contribution_pp": fp_term})
    return result


def routing_summary(available_beneficial: int, available_harmful: int,
                    retained_beneficial: int, accepted_harmful: int) -> dict:
    b, h, rb, ah = (available_beneficial, available_harmful,
                     retained_beneficial, accepted_harmful)
    if min(b, h, rb, ah) < 0 or rb > b or ah > h:
        raise ValueError("Invalid paired routing counts.")
    retention = rb / b if b else None
    acceptance = ah / h if h else None
    # B * retention > H * harmful_acceptance is pixel-accuracy break-even,
    # not an mIoU or branch-confidence threshold.
    required_rejection = max(0.0, 1 - rb / h) if h else None
    return {"available_beneficial": b, "available_harmful": h,
            "retained_beneficial": rb, "accepted_harmful": ah,
            "beneficial_retention": retention,
            "harmful_rejection": 1 - acceptance if h else None,
            "net_correct_pixels": rb - ah,
            "beneficial_fraction_among_accepted_xor": rb / (rb + ah) if rb + ah else None,
            "harmful_to_beneficial_base_ratio": h / b if b else None,
            "minimum_harmful_rejection_for_nonnegative_accuracy": required_rejection,
            "note": "Break-even of paired correct pixels only; wrong-to-wrong can change IoU."}


def audit(payload: dict) -> dict:
    if payload.get("status") != "complete" or not payload.get("coverage_verified"):
        raise ValueError("Requires a complete coverage-verified saved result.")
    metrics = payload["metrics"]
    names = [row["name"] for row in metrics["Multiscale"]["per_class"]]
    base = metrics["Multiscale"]["confusion_matrix"]
    counts = payload["audit"]["correctness_by_true_class"]
    if any(len(row) != 4 for row in counts) or len(counts) != len(names):
        raise ValueError("Invalid paired correctness counts.")
    b, h = sum(row[2] for row in counts), sum(row[1] for row in counts)
    if sum(sum(row) for row in base) != sum(sum(row) for row in counts):
        raise ValueError("Paired counts and confusion totals disagree.")
    available = payload["audit"]["changes"]["Multiscale"]["VIP20"]
    if (available["beneficial"], available["harmful"]) != (b, h):
        raise ValueError("Paired correctness direction disagrees with changes.")
    all_arms = {}
    changes = payload["audit"]["changes"]["Multiscale"]
    for name, metric in metrics.items():
        if name == "Multiscale":
            continue
        exact = iou_decomposition(base, metric["confusion_matrix"], names)
        all_arms[name] = {"decomposition": exact,
                         "paired_net_correct": changes[name]["beneficial"] - changes[name]["harmful"],
                         "wrong_to_wrong": changes[name]["wrong_to_wrong"]}
    routes = {}
    for name, row in payload["audit"]["routing"].items():
        if (row["available_beneficial"], row["available_harmful"]) != (b, h):
            raise ValueError("Routing availability differs from paired branches.")
        routes[name] = routing_summary(b, h, row["retained_beneficial"],
                                       h - row["rejected_harmful"])
    return {"dataset": payload["signature"]["dataset"],
            "images": payload["processed_images"],
            "pattern_order_used": ["both_wrong", "Multiscale_only_correct",
                                   "VIP_only_correct", "both_correct"],
            "pattern_note": ("Some saved merge metadata calls column1 Geometry_only_correct. "
                             "The producer compares Multiscale and VIP20, not Geometry."),
            "routes": routes, "arms": all_arms,
            "proposal_classes": [
                {"class": name, "beneficial": row[2], "harmful": row[1],
                 "harmful_to_beneficial_ratio": row[1] / row[2] if row[2] else None}
                for name, row in zip(names, counts)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise ValueError(f"Refusing existing output: {args.output}")
    result = {"status": "complete",
              "note": "Labeled saved-output audit only; no inference, tuning, or vocabulary output.",
              "datasets": [{"source": str(path), **audit(json.loads(path.read_text(encoding="utf-8-sig")))}
                           for path in args.inputs]}
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
