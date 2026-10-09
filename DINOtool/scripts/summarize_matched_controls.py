#!/usr/bin/env python3
"""Summarize source-only controls and apply the pre-registered screen gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_arm(root: Path, arm: str) -> dict[str, Any]:
    path = (root / arm / "status.json").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing status file for {arm}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") not in {"paused_at_validation", "complete"}:
        raise ValueError(f"{arm} is not at a terminal source-validation state: {payload.get('status')!r}")
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        raise ValueError(f"{arm} has no source validation in {path}")
    if validation.get("validation_step") != payload.get("step"):
        raise ValueError(f"{arm} status step and validation step disagree")
    try:
        coco = float(validation["coco"]["mean_iou"])
        oem = float(validation["oem"]["mean_iou"])
        score = float(validation["selection_score"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{arm} has malformed source metrics") from error
    if abs(score - 0.5 * (coco + oem)) > 1e-6:
        raise ValueError(f"{arm} selection score is not the declared COCO/OEM mean")
    return {"arm": arm, "status": payload["status"], "step": int(payload["step"]),
            "elapsed_seconds": float(payload.get("elapsed_seconds", float("nan"))),
            "coco_raw": coco, "oem_raw": oem, "source_mean_raw": score}


def _metric_scale(records: dict[str, dict[str, Any]]) -> float:
    """Return the common conversion from stored mIoU to percentage points.

    A status file stores either all metrics as fractions (the current trainer)
    or all metrics directly in percentage points. Infer that convention from
    absolute scores once, before calculating deltas: inferring it from a
    delta would turn a valid 0.5-point percentage improvement into 50 points.
    """
    values = [record[key] for record in records.values()
              for key in ("coco_raw", "oem_raw", "source_mean_raw")]
    fractional = [abs(value) <= 1.5 for value in values]
    if all(fractional):
        return 100.0
    if not any(fractional):
        return 1.0
    raise ValueError("mixed fractional and percentage mIoU conventions across source status files")


def _points(value: float, scale: float) -> float:
    """Present a score or delta in percentage points using a known scale."""
    return scale * value


def build_report(root: Path, parallel: str, epl: str, candidate: str | None, required_gain: float,
                 max_source_drop: float, max_time_ratio: float) -> dict[str, Any]:
    records = {arm: _read_arm(root, arm) for arm in (parallel, epl)}
    if candidate:
        records[candidate] = _read_arm(root, candidate)
    scale = _metric_scale(records)
    report: dict[str, Any] = {
        "results_root": str(root.resolve()),
        "protocol": "source-only COCOStuff2017 + OEM; LoveDA/external targets excluded",
        "stored_miou_convention": "fraction" if scale == 100.0 else "percentage_points",
        "arms": {arm: {**record, "coco_miou": _points(record["coco_raw"], scale),
                         "oem_miou": _points(record["oem_raw"], scale),
                         "source_mean_miou": _points(record["source_mean_raw"], scale)}
                 for arm, record in records.items()},
    }
    if not candidate:
        report["screen_gate"] = {"state": "awaiting_candidate", "passed": False}
        return report
    base = max(records[parallel]["source_mean_raw"], records[epl]["source_mean_raw"])
    cand = records[candidate]
    gain = _points(cand["source_mean_raw"] - base, scale)
    coco_delta = _points(cand["coco_raw"] - max(records[parallel]["coco_raw"], records[epl]["coco_raw"]), scale)
    oem_delta = _points(cand["oem_raw"] - max(records[parallel]["oem_raw"], records[epl]["oem_raw"]), scale)
    ratio = cand["elapsed_seconds"] / records[epl]["elapsed_seconds"]
    passed = gain >= required_gain and coco_delta >= -max_source_drop and oem_delta >= -max_source_drop and ratio <= max_time_ratio
    report["screen_gate"] = {
        "state": "screen_pass" if passed else "screen_fail", "passed": passed, "candidate": candidate,
        "source_mean_gain_vs_best_control": gain, "coco_delta_vs_best_control": coco_delta,
        "oem_delta_vs_best_control": oem_delta, "training_time_ratio_vs_epl": ratio,
        "thresholds": {"required_source_mean_gain": required_gain, "max_per_source_drop": max_source_drop, "max_time_ratio": max_time_ratio},
        "remaining_requirements": ["paired three-seed comparison against pair-independent transport", "untouched external-data evaluation"],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--parallel", default="parallel_corrected_w0")
    parser.add_argument("--epl", default="epl_corrected_w0")
    parser.add_argument("--candidate", help="Candidate arm directory; omit while controls are running")
    parser.add_argument("--required-gain", type=float, default=0.5)
    parser.add_argument("--max-source-drop", type=float, default=0.5)
    parser.add_argument("--max-time-ratio", type=float, default=1.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.results_root, args.parallel, args.epl, args.candidate, args.required_gain,
                          args.max_source_drop, args.max_time_ratio)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
