"""CPU-only contracts for the fixed Astra Q-Lift target-screen logic."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cafe_qlift_gate", ROOT / "scripts" / "analyze_cafe_qlift_gate.py")
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


P_NAMES = ("building", "road", "water", "barren", "tree", "farm")
D_NAMES = ("background", "building", "road", "water", "barren", "forest", "agricultural")


def result(protocol: str, score: float, *, query: bool) -> dict:
    """Build a small row-normalized confusion fixture with directional gains."""

    names = P_NAMES if protocol == "P" else D_NAMES
    error = 1 if query else 2
    matrix = []
    per_class = []
    for index, name in enumerate(names):
        row = [0] * len(names)
        row[index] = 100 - error
        row[(index + 1) % len(names)] = error
        matrix.append(row)
        per_class.append({
            "name": name,
            "intersection_pixels": row[index],
            "target_pixels": 100,
            "predicted_pixels": 100,
            "iou_percent": 100.0 * score,
        })
    payload = {"mean_iou": score, "per_class": per_class, "confusion_matrix": matrix}
    if protocol == "D":
        payload["foreground_mean_with_background_false_positives"] = score
    return payload


def raw(query_p: float, query_d: float, *, control: float = 0.50) -> dict:
    return {
        "P": {
            "skip": result("P", control, query=False),
            "haar": result("P", control - 0.01, query=False),
            "image_lift": result("P", control - 0.02, query=False),
            "query_lift": result("P", query_p, query=True),
        },
        "D": {
            "skip": result("D", control, query=False),
            "haar": result("D", control - 0.01, query=False),
            "image_lift": result("D", control - 0.02, query=False),
            "query_lift": result("D", query_d, query=True),
        },
    }


def test_fixed_screen_holds_threshold_passing_run_pending_mechanism_and_spatial_evidence() -> None:
    decision = gate.preliminary_target_gate(raw(0.52, 0.52))
    assert decision["decision"] == "HOLD_PENDING_VERIFIED_MECHANISM_AND_SPATIAL_AUDITS"
    assert decision["reference_controls"] == {"P": "skip", "D_foreground": "skip", "D_seven_class": "skip"}
    assert all(decision["checks"].values())


def test_fixed_screen_stops_when_both_primary_metrics_lose() -> None:
    decision = gate.preliminary_target_gate(raw(0.49, 0.49))
    assert decision["decision"] == "STOP_CURRENT_Q_LIFT_TRANSFER_CLAIM"
    assert decision["primary_metric_deltas_pp"]["P_six_class_miou_pp"] == -1.0
    assert decision["primary_metric_deltas_pp"]["D_foreground_miou_pp"] == -1.0


def test_fixed_screen_holds_small_gains_without_independent_ranking_evidence() -> None:
    decision = gate.preliminary_target_gate(raw(0.503, 0.503))
    assert decision["decision"] == "HOLD_NO_INDEPENDENT_DISCRIMINATION_EVIDENCE"
    assert decision["primary_metric_deltas_pp"]["P_six_class_miou_pp"] == 0.3
    assert gate.research_disposition(decision)["code"] == "STOP_CURRENT_Q_LIFT_TRANSFER_CLAIM"


def test_verified_null_pu_interventions_block_a_metric_only_go() -> None:
    decision = gate.preliminary_target_gate(raw(0.52, 0.52))
    mechanism = {
        protocol: {
            mode: {
                "native_minus_counterfactual_primary_metric_raw_pp": 0.1,
                "counterfactual_minus_native_directional_confusion_raw_pp": {"x->y": 0.1},
            }
            for mode in ("pu-shuffled", "pu-image-only", "details-zero")
        }
        for protocol in ("P", "D")
    }
    gate.apply_mechanism_gate(decision, mechanism)
    gate.finalize_after_spatial_replay(decision)
    assert decision["decision"] == "HOLD_QUERY_PU_PATH_EFFECTIVELY_NULL"
    assert gate.research_disposition(decision)["code"] == "STOP_QUERY_RECONSTRUCTION_MECHANISM_CLAIM"


def test_nonnull_pu_interventions_and_valid_spatial_replay_enable_only_provisional_go() -> None:
    decision = gate.preliminary_target_gate(raw(0.52, 0.52))
    mechanism = {
        protocol: {
            mode: {
                "native_minus_counterfactual_primary_metric_raw_pp": 0.3 if mode != "details-zero" else 2.0,
                "counterfactual_minus_native_directional_confusion_raw_pp": {"x->y": 0.3},
            }
            for mode in ("pu-shuffled", "pu-image-only", "details-zero")
        }
        for protocol in ("P", "D")
    }
    gate.apply_mechanism_gate(decision, mechanism)
    gate.finalize_after_spatial_replay(decision)
    assert decision["decision"] == "PROVISIONAL_GO_FOR_LIMITED_ATTRIBUTION_ONLY"
    assert gate.research_disposition(decision)["code"] == "PROCEED_TO_REGISTERED_FACTORIAL_ONLY"


def test_native_target_identity_rejects_a_historical_checkpoint_binding() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint = root / "corrected.pt"
        checkpoint.write_bytes(b"corrected-checkpoint")
        source = {
            "architecture": {"name": "Q-Lift-DINO-v1", "arm": "skip"},
            "selected_checkpoint": gate.file_binding(checkpoint),
        }
        native = root / "P" / "skip"
        native.mkdir(parents=True)
        sample_keys = [f"sample-{index}" for index in range(1669)]
        config = {
            "protocol": "P", "qlift_audit": "native", "model_variant": "skip",
            "target_used_for_training_or_selection": False, "max_images": None,
            "images": 1669, "world_size": 4, "architecture": source["architecture"],
            "sample_keys": sample_keys, "weights": str(checkpoint.resolve()),
            "integrity_bindings": {
                "selected_checkpoint": dict(source["selected_checkpoint"]),
                "native_argmax_digests": {"entries": 1669, "sha256": "a" * 64},
            },
            **gate.LOCKED_LOVEDA_SETTINGS["P"],
        }
        result = {
            "status": "complete", "protocol": "P", "qlift_audit": "native", "model_variant": "skip",
            "images": 1669, "world_size": 4, "weights": str(checkpoint.resolve()),
            **gate.LOCKED_LOVEDA_SETTINGS["P"],
        }
        (native / "evaluation_config.json").write_text(json.dumps(config), encoding="utf-8")
        (native / "results.json").write_text(json.dumps(result), encoding="utf-8")
        gate.validate_native_target_evaluation(native, "P", "skip", source)
        config["integrity_bindings"]["selected_checkpoint"]["sha256"] = "b" * 64
        (native / "evaluation_config.json").write_text(json.dumps(config), encoding="utf-8")
        try:
            gate.validate_native_target_evaluation(native, "P", "skip", source)
        except ValueError as error:
            assert "corrected source-selected checkpoint" in str(error)
        else:
            raise AssertionError("historical/native checkpoint mismatch was accepted")
