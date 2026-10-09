from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_matched_controls import build_report


def _write_status(root: Path, arm: str, coco: float, oem: float, elapsed: float) -> None:
    path = root / arm
    path.mkdir(parents=True)
    validation = {"coco": {"mean_iou": coco}, "oem": {"mean_iou": oem}, "selection_score": 0.5 * (coco + oem), "validation_step": 2612}
    (path / "status.json").write_text(json.dumps({"status": "paused_at_validation", "step": 2612, "elapsed_seconds": elapsed, "validation": validation}))


def test_report_keeps_target_data_out_and_applies_screen_gate(tmp_path: Path) -> None:
    _write_status(tmp_path, "parallel", 0.55, 0.56, 10.0)
    _write_status(tmp_path, "epl", 0.56, 0.55, 12.0)
    _write_status(tmp_path, "candidate", 0.57, 0.565, 17.0)
    report = build_report(tmp_path, "parallel", "epl", "candidate", 0.5, 0.5, 1.5)
    assert report["protocol"].startswith("source-only")
    assert abs(report["arms"]["candidate"]["source_mean_miou"] - 56.75) < 1e-9
    assert report["screen_gate"]["passed"] is True
    assert report["screen_gate"]["remaining_requirements"]


def test_report_rejects_incomplete_arm(tmp_path: Path) -> None:
    _write_status(tmp_path, "parallel", 0.55, 0.56, 10.0)
    _write_status(tmp_path, "epl", 0.56, 0.55, 12.0)
    data = json.loads((tmp_path / "epl" / "status.json").read_text())
    data["status"] = "training"
    (tmp_path / "epl" / "status.json").write_text(json.dumps(data))
    try:
        build_report(tmp_path, "parallel", "epl", None, 0.5, 0.5, 1.5)
    except ValueError as error:
        assert "terminal source-validation" in str(error)
    else:
        raise AssertionError("incomplete arm was accepted")


def test_report_keeps_percentage_point_deltas_in_percentage_points(tmp_path: Path) -> None:
    _write_status(tmp_path, "parallel", 55.0, 55.0, 10.0)
    _write_status(tmp_path, "epl", 55.5, 55.5, 12.0)
    _write_status(tmp_path, "candidate", 56.0, 56.0, 17.0)
    report = build_report(tmp_path, "parallel", "epl", "candidate", 0.5, 0.5, 1.5)
    assert report["stored_miou_convention"] == "percentage_points"
    assert report["arms"]["candidate"]["source_mean_miou"] == 56.0
    assert report["screen_gate"]["source_mean_gain_vs_best_control"] == 0.5
    assert report["screen_gate"]["passed"] is True
