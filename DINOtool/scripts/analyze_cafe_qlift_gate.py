#!/usr/bin/env python3
"""Strict, read-only summary for the capacity-matched Q-Lift gate.

The report is deliberately conservative. LoveDA P/D are reported as external
cross-dataset RS transfer, never as a semantic-unseen-class result. Target
scores are read only after all source-selected checkpoints are fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ARMS = ("skip", "haar", "image_lift", "query_lift")
PREDECLARED_PAIRS_BY_PROTOCOL = {
    # P deliberately uses the fixed foreground-only CAFe vocabulary, whose
    # cultivated-land query is named ``farm`` rather than ``agricultural``.
    # Keep the protocol-specific surface forms separate: silently querying a
    # label absent from a result would drop the very ambiguity being audited.
    "P": (("tree", "farm"), ("barren", "tree"), ("barren", "road"), ("building", "road")),
    # D uses the native LoveDA vocabulary and includes a background class.
    "D": (("forest", "agricultural"), ("barren", "forest"), ("barren", "road"), ("building", "road")),
}
LOCKED_LOVEDA_SETTINGS = {
    "P": {
        "evaluation_size": 512,
        "window_size": 224,
        "stride": 112,
        "classes": ["building", "road", "water", "barren", "tree", "farm"],
        "include_background": False,
    },
    "D": {
        "evaluation_size": 0,
        "window_size": 448,
        "stride": 224,
        "classes": ["background", "building", "road", "water", "barren", "forest", "agricultural"],
        "include_background": True,
    },
}
LOCKED_ARGS = (
    "updates", "validate_every", "checkpoint_every", "warmup_steps", "stop_after_validation",
    "batch_size", "accum_steps", "workers",
    "new_lr", "head_lr", "visual_lr", "weight_decay", "kd_weight", "label_smoothing",
    "seed", "amp", "memory_fraction",
    "qlift_levels", "qlift_guide_dim", "qlift_hidden_dim", "qlift_kernel",
    "qlift_query_chunk", "qlift_tune_visual_blocks",
)

EXPECTED_CORRECTED_ARGS = {
    "updates": 20896, "validate_every": 2612, "checkpoint_every": 500,
    "warmup_steps": 500, "stop_after_validation": 2612,
    "batch_size": 2, "accum_steps": 4, "workers": 4,
    "new_lr": 1e-4, "head_lr": 2e-5, "visual_lr": 2e-6,
    "weight_decay": 0.01, "kd_weight": 0.02, "label_smoothing": 0.1,
    "seed": 20260921, "amp": "bf16", "memory_fraction": 0.75,
    "qlift_levels": 2, "qlift_guide_dim": 128, "qlift_hidden_dim": 256,
    "qlift_kernel": 3, "qlift_query_chunk": 8, "qlift_tune_visual_blocks": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--loveda-root", type=Path,
                        help="Optional root containing P/<arm>/results.json and D/<arm>/results.json.")
    parser.add_argument("--mechanism-root", type=Path,
                        help="Optional root containing P/D/<counterfactual>/results.json.")
    parser.add_argument("--spatial-root", type=Path,
                        help="Optional root containing P/D/<arm>/strict spatial-audit outputs.")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": fingerprint(path)}


def same_file_binding(actual: Any, expected: dict[str, Any], label: str) -> None:
    if not isinstance(actual, dict):
        raise ValueError(f"{label} is missing")
    if (
        not isinstance(actual.get("path"), str)
        or Path(actual["path"]).resolve() != Path(expected["path"]).resolve()
        or actual.get("bytes") != expected["bytes"]
        or actual.get("sha256") != expected["sha256"]
    ):
        raise ValueError(f"{label} does not bind to the corrected source-selected checkpoint")


def last_json_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    records = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(records[-1]) if records else {}


def last_training_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    training = [record for record in records if record.get("status") == "training"]
    if not training:
        raise ValueError(f"No training LR record: {path}")
    return training[-1]


def pct(value: float | None) -> float | None:
    return None if value is None else round(100.0 * float(value), 4)


def class_stats(result: dict[str, Any]) -> dict[str, dict[str, float | int | None]]:
    output = {}
    for entry in result["per_class"]:
        tp, predicted, target = (int(entry[key]) for key in ("intersection_pixels", "predicted_pixels", "target_pixels"))
        output[entry["name"]] = {
            "iou_percent": entry["iou_percent"],
            "precision_percent": pct(tp / predicted) if predicted else None,
            "recall_percent": pct(tp / target) if target else None,
            "prediction_to_gt": round(predicted / target, 6) if target else None,
            "true_positive_pixels": tp,
            "false_positive_pixels": predicted - tp,
            "false_negative_pixels": target - tp,
        }
    return output


def target_summary(result: dict[str, Any]) -> dict[str, Any]:
    required = ("status", "protocol", "mean_iou", "per_class", "confusion_matrix")
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(f"Target result misses {missing}")
    if result["status"] != "complete":
        raise ValueError(f"Target evaluation is not complete: {result['status']}")
    summary = {
        "mean_iou_percent": result.get("mean_iou_percent", pct(result["mean_iou"])),
        "pixel_accuracy_percent": result.get("pixel_accuracy_percent", pct(result.get("pixel_accuracy"))),
        "foreground_mean_with_background_false_positives_percent": pct(result.get("foreground_mean_with_background_false_positives")),
        "per_class": class_stats(result),
        "confusion_matrix": result["confusion_matrix"],
        "images": result.get("images"),
        "seconds": result.get("seconds"),
        "qlift_audit": result.get("qlift_audit", "native"),
    }
    return summary


def pairwise_error(result: dict[str, Any], first: str, second: str) -> float | None:
    names = [entry["name"] for entry in result["per_class"]]
    if first not in names or second not in names:
        return None
    matrix = result["confusion_matrix"]
    i, j = names.index(first), names.index(second)
    denominator = sum(matrix[i]) + sum(matrix[j])
    return None if not denominator else (matrix[i][j] + matrix[j][i]) / denominator


def finite_value(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is missing or non-numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite")
    return result


def protocol_metric(result: dict[str, Any], protocol: str) -> float:
    """Use the pre-registered primary metric, never a silent fallback."""

    if protocol == "P":
        return finite_value(result.get("mean_iou"), "P mean_iou")
    if protocol == "D":
        # D foreground mIoU keeps background false positives in each
        # foreground denominator and is therefore the primary D screen.
        return finite_value(
            result.get("foreground_mean_with_background_false_positives"),
            "D foreground_mean_with_background_false_positives",
        )
    raise ValueError(f"Unknown target protocol {protocol!r}")


def delta(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else round(float(left) - float(right), 6)


def finite_percent(value: Any) -> float | None:
    """Return a fraction as percentage without silently coercing absent diagnostics."""

    return None if value is None else round(100.0 * float(value), 4)


def spatial_metric(summary: dict[str, Any], *path: str) -> Any:
    value: Any = summary
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Spatial summary misses {'/'.join(path)}")
        value = value[key]
    return value


def same_float(left: Any, right: Any) -> bool:
    """Use the replay's published numerical precision, never a rounded display metric."""

    return left is not None and right is not None and math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=1e-12,
    )


def validate_native_target_evaluation(
    native_dir: Path,
    protocol: str,
    arm: str,
    source_run: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind one LoveDA result to the corrected source-selected checkpoint.

    This prevents an internally valid but historical compressed-schedule target
    result from being combined with a later corrected source gate.
    """

    config = load(native_dir / "evaluation_config.json")
    result = load(native_dir / "results.json")
    settings = LOCKED_LOVEDA_SETTINGS[protocol]
    checks = {
        "config.protocol": (config.get("protocol"), protocol),
        "result.protocol": (result.get("protocol"), protocol),
        "config.qlift_audit": (config.get("qlift_audit"), "native"),
        "result.qlift_audit": (result.get("qlift_audit"), "native"),
        "result.status": (result.get("status"), "complete"),
        "config.model_variant": (config.get("model_variant"), arm),
        "result.model_variant": (result.get("model_variant"), arm),
        "config.target_used_for_training_or_selection": (
            config.get("target_used_for_training_or_selection"), False,
        ),
        "config.max_images": (config.get("max_images"), None),
        "config.images": (config.get("images"), 1669),
        "result.images": (result.get("images"), 1669),
        "config.world_size": (config.get("world_size"), 4),
        "result.world_size": (result.get("world_size"), config.get("world_size")),
        "config.architecture": (config.get("architecture"), source_run["architecture"]),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"Native target contract mismatch for {protocol}/{arm}: {label}")
    if not isinstance(config.get("sample_keys"), list) or len(config["sample_keys"]) != 1669:
        raise ValueError(f"Native target has no complete sample ordering for {protocol}/{arm}")
    if len(set(config["sample_keys"])) != 1669:
        raise ValueError(f"Native target sample ordering is not unique for {protocol}/{arm}")
    for name, expected in settings.items():
        if config.get(name) != expected or result.get(name) != expected:
            raise ValueError(f"Native target does not use locked {protocol} setting {name!r} for {arm}")
    weights = config.get("weights")
    if not isinstance(weights, str) or Path(weights).resolve() != Path(source_run["selected_checkpoint"]["path"]).resolve():
        raise ValueError(f"Native target weights path does not select corrected {arm} checkpoint")
    if result.get("weights") != weights:
        raise ValueError(f"Native target result weights differ from config for {protocol}/{arm}")
    bindings = config.get("integrity_bindings")
    if not isinstance(bindings, dict):
        raise ValueError(f"Native target lacks integrity bindings for {protocol}/{arm}")
    same_file_binding(bindings.get("selected_checkpoint"), source_run["selected_checkpoint"],
                      f"Native target selected checkpoint for {protocol}/{arm}")
    digests = bindings.get("native_argmax_digests")
    if not isinstance(digests, dict) or digests.get("entries") != 1669:
        raise ValueError(f"Native target lacks complete argmax-digest binding for {protocol}/{arm}")
    if not isinstance(digests.get("sha256"), str) or len(digests["sha256"]) != 64:
        raise ValueError(f"Native target argmax-digest hash is invalid for {protocol}/{arm}")
    return config, result


def validate_mechanism_evaluation(
    mechanism_dir: Path,
    protocol: str,
    mode: str,
    native_dir: Path,
    native_config: dict[str, Any],
    source_run: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require each counterfactual to cite the exact native query-lift run."""

    config = load(mechanism_dir / "evaluation_config.json")
    result = load(mechanism_dir / "results.json")
    settings = LOCKED_LOVEDA_SETTINGS[protocol]
    checks = {
        "config.protocol": (config.get("protocol"), protocol),
        "result.protocol": (result.get("protocol"), protocol),
        "config.qlift_audit": (config.get("qlift_audit"), mode),
        "result.qlift_audit": (result.get("qlift_audit"), mode),
        "result.status": (result.get("status"), "complete"),
        "config.model_variant": (config.get("model_variant"), "query_lift"),
        "result.model_variant": (result.get("model_variant"), "query_lift"),
        "config.architecture": (config.get("architecture"), source_run["architecture"]),
        "config.target_used_for_training_or_selection": (
            config.get("target_used_for_training_or_selection"), False,
        ),
        "config.max_images": (config.get("max_images"), None),
        "config.images": (config.get("images"), 1669),
        "result.images": (result.get("images"), 1669),
        "config.world_size": (config.get("world_size"), 4),
        "result.world_size": (result.get("world_size"), config.get("world_size")),
        "config.sample_keys": (config.get("sample_keys"), native_config.get("sample_keys")),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"Mechanism contract mismatch for {protocol}/{mode}: {label}")
    for name, expected in settings.items():
        if config.get(name) != expected or result.get(name) != expected:
            raise ValueError(f"Mechanism does not use locked {protocol} setting {name!r} for {mode}")
    if (
        not isinstance(config.get("weights"), str)
        or Path(config["weights"]).resolve() != Path(source_run["selected_checkpoint"]["path"]).resolve()
        or result.get("weights") != config["weights"]
    ):
        raise ValueError(f"Mechanism does not use corrected query-lift checkpoint for {protocol}/{mode}")
    reference = config.get("mechanism_native_reference")
    if not isinstance(reference, dict):
        raise ValueError(f"Mechanism lacks native-reference binding for {protocol}/{mode}")
    if (
        reference.get("evaluation_config_sha256") != fingerprint(native_dir / "evaluation_config.json")
        or reference.get("results_sha256") != fingerprint(native_dir / "results.json")
        or reference.get("sample_keys_sha256") != hashlib.sha256(
            "\n".join(native_config["sample_keys"]).encode("utf-8")
        ).hexdigest()
    ):
        raise ValueError(f"Mechanism native reference does not match corrected target run for {protocol}/{mode}")
    native_sources = native_config.get("integrity_bindings", {}).get("evaluator_source_sha256")
    if reference.get("counterfactual_evaluator_source_sha256") != native_sources:
        raise ValueError(f"Mechanism evaluator source differs from native reference for {protocol}/{mode}")
    same_file_binding(reference.get("selected_checkpoint"), source_run["selected_checkpoint"],
                      f"Mechanism selected checkpoint for {protocol}/{mode}")
    return config, result


def validate_spatial_audit(
    spatial_root: Path,
    protocol: str,
    arm: str,
    native_dir: Path,
    native_config: dict[str, Any],
    native: dict[str, Any],
    source_run: dict[str, Any],
) -> dict[str, Any]:
    """Load a strict replay only after proving it is bound to the native run.

    The native evaluator is the semantic metric authority. Spatial diagnostics
    are published only if the independent replay reproduces every argmax map,
    support pixel and global confusion matrix claimed by that native result.
    """

    root = spatial_root / protocol / arm
    integrity = load(root / "integrity.json")
    status = load(root / "status.json")
    manifest = load(root / "audit_manifest.json")
    summary = load(root / "spatial_summary.json")
    if integrity.get("status") != "complete" or status.get("status") != "complete":
        raise ValueError(f"Incomplete spatial audit for {protocol}/{arm}")
    for key in (
        "reference_integrity_bindings_exact_match",
        "native_per_image_argmax_digest_exact_match",
        "reference_confusion_exact_match",
        "per_image_confusion_equals_ddp_reduce",
        "ignored_pixels_exact_match",
    ):
        if integrity.get(key) is not True:
            raise ValueError(f"Spatial audit integrity failed for {protocol}/{arm}: {key}")
    if status.get("protocol") != protocol or status.get("model_variant") != arm:
        raise ValueError(f"Spatial audit protocol/model mismatch for {protocol}/{arm}")
    if manifest.get("reference_protocol") != protocol or manifest.get("reference_model_variant") != arm:
        raise ValueError(f"Spatial audit reference protocol/model mismatch for {protocol}/{arm}")
    if manifest.get("reference_evaluation_config_sha256") != fingerprint(native_dir / "evaluation_config.json"):
        raise ValueError(f"Spatial audit is not bound to this native config for {protocol}/{arm}")
    if manifest.get("reference_results_sha256") != fingerprint(native_dir / "results.json"):
        raise ValueError(f"Spatial audit is not bound to this native result for {protocol}/{arm}")
    if (
        not isinstance(manifest.get("reference_weights"), str)
        or Path(manifest["reference_weights"]).resolve() != Path(source_run["selected_checkpoint"]["path"]).resolve()
        or manifest.get("reference_weights_sha256") != source_run["selected_checkpoint"]["sha256"]
    ):
        raise ValueError(f"Spatial audit is not bound to corrected {arm} checkpoint for {protocol}")
    if native_config.get("integrity_bindings", {}).get("selected_checkpoint") != source_run["selected_checkpoint"]:
        raise ValueError(f"Spatial audit native config is not source-checkpoint bound for {protocol}/{arm}")
    global_summary = spatial_metric(summary, "global")
    if global_summary.get("confusion_matrix") != native.get("confusion_matrix"):
        raise ValueError(f"Spatial confusion does not equal native result for {protocol}/{arm}")
    if summary.get("ignored_pixels") != native.get("ignored_pixels"):
        raise ValueError(f"Spatial ignored-pixel support differs for {protocol}/{arm}")
    if not same_float(global_summary.get("mean_iou"), native.get("mean_iou")):
        raise ValueError(f"Spatial mIoU does not equal native result for {protocol}/{arm}")
    if protocol == "D" and (
        not same_float(
            global_summary.get("foreground_mean_with_background_false_positives"),
            native.get("foreground_mean_with_background_false_positives"),
        )
    ):
        raise ValueError(f"Spatial foreground mIoU does not equal native result for {protocol}/{arm}")

    components = spatial_metric(summary, "components")
    return {
        "integrity": {
            "images": integrity.get("images"),
            "components": integrity.get("components"),
            "native_argmax_digest_exact_match": True,
            "native_confusion_exact_match": True,
            "ignored_pixels_exact_match": True,
        },
        "gt_boundary": {
            "pixels": spatial_metric(summary, "gt_boundary", "pixels"),
            "accuracy_percent": finite_percent(spatial_metric(summary, "gt_boundary", "accuracy")),
            "per_class_recall_percent": {
                name: finite_percent(value)
                for name, value in spatial_metric(summary, "gt_boundary", "per_class_recall").items()
            },
        },
        "gt_interior": {
            "pixels": spatial_metric(summary, "gt_interior", "pixels"),
            "accuracy_percent": finite_percent(spatial_metric(summary, "gt_interior", "accuracy")),
            "per_class_recall_percent": {
                name: finite_percent(value)
                for name, value in spatial_metric(summary, "gt_interior", "per_class_recall").items()
            },
        },
        "error_decomposition": summary.get("error_decomposition"),
        "components": {
            "definition": components.get("definition"),
            "all": components.get("all"),
            "tiny": spatial_metric(summary, "components", "by_area_bin", "tiny"),
            "small": spatial_metric(summary, "components", "by_area_bin", "small"),
            "by_class": components.get("by_class"),
        },
        "focal_pairwise_confusion": spatial_metric(summary, "pairwise_confusion", "global", "focal_pairs"),
    }


def spatial_delta(query: dict[str, Any], control: dict[str, Any]) -> dict[str, float | None]:
    """Comparable spatial deltas in percentage points for the fixed controls."""

    return {
        "gt_boundary_accuracy_pp": delta(
            query["gt_boundary"]["accuracy_percent"], control["gt_boundary"]["accuracy_percent"]
        ),
        "gt_interior_accuracy_pp": delta(
            query["gt_interior"]["accuracy_percent"], control["gt_interior"]["accuracy_percent"]
        ),
        "tiny_macro_same_class_coverage_pp": delta(
            finite_percent(query["components"]["tiny"].get("macro_same_class_coverage")),
            finite_percent(control["components"]["tiny"].get("macro_same_class_coverage")),
        ),
        "small_macro_same_class_coverage_pp": delta(
            finite_percent(query["components"]["small"].get("macro_same_class_coverage")),
            finite_percent(control["components"]["small"].get("macro_same_class_coverage")),
        ),
        "tiny_hit_at_50_pp": delta(
            finite_percent(query["components"]["tiny"].get("hit_at_50_rate")),
            finite_percent(control["components"]["tiny"].get("hit_at_50_rate")),
        ),
        "small_hit_at_50_pp": delta(
            finite_percent(query["components"]["small"].get("hit_at_50_rate")),
            finite_percent(control["components"]["small"].get("hit_at_50_rate")),
        ),
    }


def per_class_entry(result: dict[str, Any], name: str) -> dict[str, Any]:
    for entry in result["per_class"]:
        if entry.get("name") == name:
            return entry
    raise ValueError(f"Result does not contain registered class {name!r}")


def class_recall(result: dict[str, Any], name: str) -> float | None:
    entry = per_class_entry(result, name)
    target = int(entry["target_pixels"])
    return None if target == 0 else float(entry["intersection_pixels"]) / target


def directional_confusion(result: dict[str, Any], source: str, destination: str) -> float | None:
    """GT-row-normalized error; unlike mutual error, direction is not hidden."""

    names = [entry["name"] for entry in result["per_class"]]
    source_index, destination_index = names.index(source), names.index(destination)
    matrix = result["confusion_matrix"]
    denominator = sum(matrix[source_index])
    return None if denominator == 0 else float(matrix[source_index][destination_index]) / denominator


def best_control_arm(result_by_arm: dict[str, dict[str, Any]], metric: str) -> str:
    if metric == "P":
        score = lambda result: protocol_metric(result, "P")
    elif metric == "Dfg":
        score = lambda result: protocol_metric(result, "D")
    elif metric == "D7":
        score = lambda result: finite_value(result.get("mean_iou"), "D seven-class mean_iou")
    else:
        raise ValueError(f"Unknown target metric {metric}")
    return max((arm for arm in ARMS if arm != "query_lift"), key=lambda arm: score(result_by_arm[arm]))


def preliminary_target_gate(raw: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Apply Astra's fixed one-seed screen without upgrading it to a paper claim."""

    p_control = best_control_arm(raw["P"], "P")
    dfg_control = best_control_arm(raw["D"], "Dfg")
    d7_control = best_control_arm(raw["D"], "D7")
    raw_deltas = {
        "P_six_class_miou_pp": 100.0 * (protocol_metric(raw["P"]["query_lift"], "P") - protocol_metric(raw["P"][p_control], "P")),
        "D_foreground_miou_pp": 100.0 * (protocol_metric(raw["D"]["query_lift"], "D") - protocol_metric(raw["D"][dfg_control], "D")),
        "D_seven_class_miou_pp": 100.0 * (
            finite_value(raw["D"]["query_lift"].get("mean_iou"), "D query seven-class mean_iou")
            - finite_value(raw["D"][d7_control].get("mean_iou"), "D control seven-class mean_iou")
        ),
    }
    deltas = {name: round(value, 4) for name, value in raw_deltas.items()}
    references = {"P": p_control, "D_foreground": dfg_control, "D_seven_class": d7_control}

    risk_deltas: dict[str, dict[str, float | None]] = {}
    directional: dict[str, dict[str, float | None]] = {}
    for protocol, control in (("P", p_control), ("D", dfg_control)):
        query, baseline = raw[protocol]["query_lift"], raw[protocol][control]
        risk_names = tuple(dict.fromkeys(name for pair in PREDECLARED_PAIRS_BY_PROTOCOL[protocol] for name in pair))
        risk_deltas[protocol] = {}
        for name in risk_names:
            query_recall, baseline_recall = class_recall(query, name), class_recall(baseline, name)
            if query_recall is None or baseline_recall is None:
                risk_deltas[protocol][name] = None
            else:
                risk_deltas[protocol][name] = 100.0 * (query_recall - baseline_recall)
        for first, second in PREDECLARED_PAIRS_BY_PROTOCOL[protocol]:
            for source, destination in ((first, second), (second, first)):
                query_error = directional_confusion(query, source, destination)
                baseline_error = directional_confusion(baseline, source, destination)
                directional[f"{protocol}:{source}->{destination}"] = {
                    "query_percent": finite_percent(query_error),
                    "control_percent": finite_percent(baseline_error),
                    "query_minus_control_pp": None if query_error is None or baseline_error is None
                    else 100.0 * (query_error - baseline_error),
                }

    observed_risk = [value for group in risk_deltas.values() for value in group.values()]
    directional_by_protocol = {
        protocol: [entry["query_minus_control_pp"] for key, entry in directional.items()
                   if key.startswith(protocol + ":")]
        for protocol in ("P", "D")
    }
    checks = {
        "P_at_least_plus_1pp": raw_deltas["P_six_class_miou_pp"] >= 1.0,
        "D_foreground_at_least_plus_2pp": raw_deltas["D_foreground_miou_pp"] >= 2.0,
        "D_seven_class_not_below_minus_0_5pp": raw_deltas["D_seven_class_miou_pp"] >= -0.5,
        "all_registered_risk_recalls_not_below_minus_2pp": bool(observed_risk) and all(
            value is not None and value >= -2.0 for value in observed_risk
        ),
        "P_at_least_two_registered_directional_confusions_not_worse": all(
            value is not None for value in directional_by_protocol["P"]
        ) and sum(value <= 0.0 for value in directional_by_protocol["P"] if value is not None) >= 2,
        "P_at_least_one_registered_directional_confusion_improved": any(
            value is not None and value < 0.0 for value in directional_by_protocol["P"]
        ),
        "D_at_least_two_registered_directional_confusions_not_worse": all(
            value is not None for value in directional_by_protocol["D"]
        ) and sum(value <= 0.0 for value in directional_by_protocol["D"] if value is not None) >= 2,
        "D_at_least_one_registered_directional_confusion_improved": any(
            value is not None and value < 0.0 for value in directional_by_protocol["D"]
        ),
    }
    screen_go = all(checks.values())
    both_primary_negative = (
        raw_deltas["P_six_class_miou_pp"] < 0.0 and raw_deltas["D_foreground_miou_pp"] < 0.0
    )
    weak_primary = (
        raw_deltas["P_six_class_miou_pp"] <= 0.5 and raw_deltas["D_foreground_miou_pp"] <= 0.5
    )
    if screen_go:
        decision = "HOLD_PENDING_VERIFIED_MECHANISM_AND_SPATIAL_AUDITS"
        rationale = "The fixed metric/error screen passes, but verified P/U mechanism and spatial diagnostics remain required."
    elif both_primary_negative:
        decision = "STOP_CURRENT_Q_LIFT_TRANSFER_CLAIM"
        rationale = "Query-Lift loses to the strongest control on both P and D foreground primary metrics."
    elif weak_primary:
        decision = "HOLD_NO_INDEPENDENT_DISCRIMINATION_EVIDENCE"
        rationale = "Both primary gains are at most +0.5 pp; this report does not contain the required AP/matched-recall evidence to override that stop condition."
    else:
        decision = "HOLD"
        rationale = "The fixed screen is mixed or below its required thresholds; do not expand the architecture claim."
    return {
        "scope": "fixed, one-seed Astra screening rule; not a statistical, semantic-unseen, causal, or paper-level conclusion",
        "reference_controls": references,
        "primary_metric_deltas_pp": deltas,
        "risk_class_recall_deltas_pp": risk_deltas,
        "directional_confusion_diagnostics": directional,
        "checks": checks,
        "metric_screen_pass": screen_go,
        "verified_mechanism": False,
        "strict_spatial_replay_validated": False,
        "decision": decision,
        "rationale": rationale,
    }


def mechanism_effect(
    native: dict[str, Any], counterfactual: dict[str, Any], protocol: str,
) -> dict[str, Any]:
    """Summarize a fixed counterfactual without mistaking it for causality."""

    directional: dict[str, float | None] = {}
    for first, second in PREDECLARED_PAIRS_BY_PROTOCOL[protocol]:
        for source, destination in ((first, second), (second, first)):
            native_error = directional_confusion(native, source, destination)
            altered_error = directional_confusion(counterfactual, source, destination)
            directional[f"{source}->{destination}"] = None if native_error is None or altered_error is None else (
                100.0 * (altered_error - native_error)
            )
    raw_primary_delta = 100.0 * (protocol_metric(native, protocol) - protocol_metric(counterfactual, protocol))
    return {
        "native_minus_counterfactual_primary_metric_pp": round(raw_primary_delta, 4),
        "native_minus_counterfactual_primary_metric_raw_pp": raw_primary_delta,
        "counterfactual_minus_native_directional_confusion_raw_pp": directional,
    }


def apply_mechanism_gate(decision: dict[str, Any], mechanism: dict[str, Any]) -> None:
    """Hold a metric screen when P/U interventions are effectively null."""

    decision["verified_mechanism"] = True
    pu_modes = ("pu-shuffled", "pu-image-only")
    scalar_values = [
        mechanism[protocol][mode]["native_minus_counterfactual_primary_metric_raw_pp"]
        for protocol in ("P", "D") for mode in pu_modes
    ]
    directional_values = [
        value
        for protocol in ("P", "D") for mode in pu_modes
        for value in mechanism[protocol][mode]["counterfactual_minus_native_directional_confusion_raw_pp"].values()
    ]
    null_primary = all(abs(value) <= 0.2 for value in scalar_values)
    null_directional = all(value is not None and abs(value) <= 0.2 for value in directional_values)
    decision["mechanism_summary"] = {
        "null_threshold_pp": 0.2,
        "pu_interventions_effectively_null_on_primary_metrics": null_primary,
        "registered_directional_confusions_effectively_null": null_directional,
    }
    if decision["metric_screen_pass"] and null_primary and null_directional:
        decision["decision"] = "HOLD_QUERY_PU_PATH_EFFECTIVELY_NULL"
        decision["rationale"] = (
            "Both verified P/U interventions are within ±0.2 pp on P and D primary metrics and have no "
            "registered directional-confusion response; the query-conditioned path is not independently evidenced."
        )


def finalize_after_spatial_replay(decision: dict[str, Any]) -> None:
    """Turn only a fully audited metric screen into a limited provisional GO."""

    decision["strict_spatial_replay_validated"] = True
    if not decision["metric_screen_pass"]:
        return
    mechanism = decision.get("mechanism_summary")
    if not decision.get("verified_mechanism"):
        decision["decision"] = "HOLD_PENDING_VERIFIED_MECHANISM_EVIDENCE"
        decision["rationale"] = "The metric screen passes, but verified P/U counterfactual evidence is absent."
    elif mechanism["pu_interventions_effectively_null_on_primary_metrics"] and mechanism[
        "registered_directional_confusions_effectively_null"
    ]:
        # apply_mechanism_gate already wrote the stronger explanatory HOLD.
        return
    else:
        decision["decision"] = "PROVISIONAL_GO_FOR_LIMITED_ATTRIBUTION_ONLY"
        decision["rationale"] = (
            "The fixed metric/error screen passes, P/U counterfactuals are not effectively null, and strict "
            "spatial replay identity is validated. This is still neither a causal mechanism nor paper-level claim."
        )


def research_disposition(decision: dict[str, Any]) -> dict[str, str]:
    """Translate a software gate label into an unambiguous research action.

    ``HOLD`` is intentionally used by the report state machine for both
    incomplete evidence and failed-but-not-double-negative metric screens. It
    must never be read as permission to tune v1, add layers, or tell a positive
    reconstruction story. This supplemental field does not change any
    threshold, checkpoint selection, or the conditional factorial trigger.
    """
    code = decision["decision"]
    if code == "PROVISIONAL_GO_FOR_LIMITED_ATTRIBUTION_ONLY":
        return {
            "code": "PROCEED_TO_REGISTERED_FACTORIAL_ONLY",
            "action": "Run only the pre-registered P/U x update-gain factorial; do not claim causal, semantic-unseen, multi-seed, or paper-level success.",
        }
    if code in {
        "HOLD_PENDING_VERIFIED_MECHANISM_AND_SPATIAL_AUDITS",
        "HOLD_PENDING_VERIFIED_MECHANISM_EVIDENCE",
        "HOLD_PENDING_STRICT_SPATIAL_REPLAY",
    }:
        return {
            "code": "INCOMPLETE_AUDIT_NOT_A_GO",
            "action": "Complete only the already-registered missing audit; do not tune, extend, or reinterpret v1 while evidence is incomplete.",
        }
    if code == "HOLD_QUERY_PU_PATH_EFFECTIVELY_NULL":
        return {
            "code": "STOP_QUERY_RECONSTRUCTION_MECHANISM_CLAIM",
            "action": "Do not run the factorial or expand v1. A future model must be newly pre-registered with a non-degenerate query-to-region constraint.",
        }
    if code in {"STOP_CURRENT_Q_LIFT_TRANSFER_CLAIM", "HOLD_NO_INDEPENDENT_DISCRIMINATION_EVIDENCE", "HOLD"}:
        return {
            "code": "STOP_CURRENT_Q_LIFT_TRANSFER_CLAIM",
            "action": "Do not optimize v1 or present a positive transfer claim. Any follow-up is a new architecture and a new registered protocol, not a rescue run.",
        }
    return {
        "code": "INCOMPLETE_OR_UNRECOGNIZED_DECISION",
        "action": "Do not proceed until the decision state is resolved under the registered protocol.",
    }


def main() -> None:
    args = parse_args()
    gate_root, output = args.gate_root.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output must be fresh: {output}")
    if not gate_root.is_dir():
        raise FileNotFoundError(gate_root)

    runs: dict[str, dict[str, Any]] = {}
    reference_args: dict[str, Any] | None = None
    reference_world: int | None = None
    reference_base_checkpoint: dict[str, Any] | None = None
    manifests: set[str] = set()
    parameter_counts: set[int] = set()
    for arm in ARMS:
        root = gate_root / arm
        if (root / "failure.json").exists():
            raise RuntimeError(f"Source gate failed for {arm}: {load(root / 'failure.json')}")
        status, training = load(root / "status.json"), load(root / "training_config.json")
        history = last_json_record(root / "history.jsonl")
        if status.get("status") != "paused_at_validation" or status.get("step") != 2612:
            raise ValueError(f"{arm} is not the locked source stopping point: {status}")
        architecture, run_args = training["architecture"], training["args"]
        if architecture.get("name") != "Q-Lift-DINO-v1" or architecture.get("arm") != arm:
            raise ValueError(f"Unexpected architecture for {arm}: {architecture}")
        if "resume" not in run_args or run_args["resume"] is not None:
            raise ValueError(f"{arm} is not a fresh-start corrected source run: resume={run_args.get('resume')!r}")
        if architecture.get("tune_visual_blocks") != 0 or architecture.get("tuned_visual_indices") != []:
            raise ValueError(f"The primary gate requires a frozen DINO backbone: {arm}")
        parameter_counts.add(int(architecture["trainable_parameters"]))
        selected_args = {key: run_args[key] for key in LOCKED_ARGS}
        expected_args = {key: EXPECTED_CORRECTED_ARGS[key] for key in LOCKED_ARGS}
        if selected_args != expected_args:
            raise ValueError(f"{arm} is not the corrected frozen source schedule: {selected_args}")
        if reference_args is None:
            reference_args = selected_args
        elif reference_args != selected_args:
            raise ValueError(f"Training protocol differs for {arm}: {selected_args} != {reference_args}")
        if reference_world is None:
            reference_world = int(training["world_size"])
        elif reference_world != int(training["world_size"]):
            raise ValueError(f"World size differs for {arm}: {training['world_size']} != {reference_world}")
        base_checkpoint = training["base_checkpoint"]
        if reference_base_checkpoint is None:
            reference_base_checkpoint = base_checkpoint
        elif reference_base_checkpoint != base_checkpoint:
            raise ValueError(f"Frozen CAFe base checkpoint provenance differs for {arm}")
        manifests.add(fingerprint(root / "source_manifest.json"))
        validation = status.get("validation")
        if not isinstance(validation, dict):
            raise ValueError(f"Missing source validation for {arm}")
        coco, oem = validation["coco"], validation["oem"]
        if status.get("total_updates") != EXPECTED_CORRECTED_ARGS["updates"]:
            raise ValueError(f"{arm} did not preserve the full cosine schedule")
        if training.get("world_size") != 4:
            raise ValueError(f"{arm} has unexpected world size: {training.get('world_size')}")
        if coco.get("images") != 2500 or oem.get("images") != 384:
            raise ValueError(f"{arm} source validation sample count differs from the frozen contract")
        if abs(float(validation["selection_score"]) - 0.5 * (float(coco["mean_iou"]) + float(oem["mean_iou"]))) > 1e-10:
            raise ValueError(f"{arm} source selection score is not mean(COCO, OEM)")
        last_train = last_training_record(root / "steps.jsonl")
        learning_rates = last_train.get("learning_rates", {})
        if last_train.get("total_updates") != EXPECTED_CORRECTED_ARGS["updates"] or int(last_train.get("step", 0)) < 2600:
            raise ValueError(f"{arm} does not have a valid final gate LR record")
        if not (9e-5 <= float(learning_rates.get("new", 0.0)) <= EXPECTED_CORRECTED_ARGS["new_lr"] and
                1.8e-5 <= float(learning_rates.get("head", 0.0)) <= EXPECTED_CORRECTED_ARGS["head_lr"]):
            raise ValueError(f"{arm} has compressed-schedule learning rates: {learning_rates}")
        selected_checkpoint = file_binding(root / "best_inference.pt")
        runs[arm] = {
            "source_selection_score": status["best_source_score"],
            "architecture": architecture,
            "selected_checkpoint": selected_checkpoint,
            "source": {
                "coco_source_dev_miou_percent": pct(coco["mean_iou"]),
                "oem_native_val_miou_percent": pct(oem["mean_iou"]),
                "selection_score_percent": pct(validation["selection_score"]),
                "coco_per_class_iou": coco.get("per_class_iou"),
                "oem_per_class_iou": oem.get("per_class_iou"),
            },
            "training": {
                "elapsed_seconds": status.get("elapsed_seconds"),
                "rank0_peak_gib": status.get("rank0_peak_gib", history.get("rank0_peak_gib")),
                "trainable_parameters": architecture["trainable_parameters"],
                "frozen_parameters": architecture["frozen_parameters"],
            },
        }
    if len(parameter_counts) != 1:
        raise ValueError(f"Capacity is not matched: {parameter_counts}")
    if len(manifests) != 1:
        raise ValueError("Source manifests differ across arms")

    source_query = runs["query_lift"]["source_selection_score"]
    source_delta = {arm: round(source_query - runs[arm]["source_selection_score"], 6) for arm in ARMS if arm != "query_lift"}
    report: dict[str, Any] = {
        "scope": {
            "source_gate": "nominal-parameter-matched source-only decoder comparison",
            "target_scope": "LoveDA P/D, if supplied, are external cross-dataset RS transfer tests; not a semantic-unseen-class protocol",
            "not_proven_by_this_report": [
                "training-seed robustness", "statistical confidence intervals", "true semantic-unseen classes",
                "Boundary F1, component precision, AP, matched-recall precision, or spatial uncertainty intervals",
            ],
        },
        "contract": {
            "arms": list(ARMS),
            "source_manifest_sha256": next(iter(manifests)),
            "locked_args": reference_args,
            "world_size": reference_world,
            "base_checkpoint": reference_base_checkpoint,
            "nominal_trainable_parameters": next(iter(parameter_counts)),
            "dino_frozen": True,
            "checkpoint_selection": "single source-only validation at update 2612; no target metric used for selection",
        },
        "source_runs": runs,
        "source_query_lift_delta_vs_controls": source_delta,
    }

    if args.loveda_root is not None:
        loveda_root = args.loveda_root.resolve()
        target: dict[str, Any] = {}
        raw: dict[str, dict[str, dict[str, Any]]] = {}
        native_configs: dict[str, dict[str, dict[str, Any]]] = {}
        for protocol in ("P", "D"):
            raw[protocol] = {}
            native_configs[protocol] = {}
            target[protocol] = {}
            for arm in ARMS:
                native_dir = loveda_root / protocol / arm
                config, result = validate_native_target_evaluation(native_dir, protocol, arm, runs[arm])
                native_configs[protocol][arm] = config
                raw[protocol][arm] = result
                target[protocol][arm] = {
                    **target_summary(result),
                    "evaluation_config_sha256": fingerprint(native_dir / "evaluation_config.json"),
                    "selected_checkpoint_sha256": runs[arm]["selected_checkpoint"]["sha256"],
                }
            query = protocol_metric(raw[protocol]["query_lift"], protocol)
            target[protocol]["query_lift_delta_vs_controls_pp"] = {
                arm: round(100.0 * (query - protocol_metric(raw[protocol][arm], protocol)), 4)
                for arm in ARMS if arm != "query_lift"
            }
            pairs = PREDECLARED_PAIRS_BY_PROTOCOL[protocol]
            target[protocol]["predeclared_pairwise_error_rate_percent"] = {
                f"{first}<->{second}": {
                    arm: pct(pairwise_error(raw[protocol][arm], first, second)) for arm in ARMS
                }
                for first, second in pairs
                if pairwise_error(raw[protocol]["query_lift"], first, second) is not None
            }
        report["loveda_external_transfer"] = target
        report["preliminary_target_decision"] = preliminary_target_gate(raw)

        if args.mechanism_root is not None:
            mechanism_root = args.mechanism_root.resolve()
            audit: dict[str, Any] = {}
            mechanism_effects: dict[str, dict[str, dict[str, Any]]] = {}
            for protocol in ("P", "D"):
                native = raw[protocol]["query_lift"]
                native_metric = protocol_metric(native, protocol)
                audit[protocol] = {"native_primary_metric_percent": round(100 * native_metric, 4)}
                mechanism_effects[protocol] = {}
                for mode in ("pu-shuffled", "pu-image-only", "details-zero"):
                    _, result = validate_mechanism_evaluation(
                        mechanism_root / protocol / mode, protocol, mode,
                        loveda_root / protocol / "query_lift", native_configs[protocol]["query_lift"],
                        runs["query_lift"],
                    )
                    effect = mechanism_effect(native, result, protocol)
                    mechanism_effects[protocol][mode] = effect
                    audit[protocol][mode] = {
                        **target_summary(result),
                        **effect,
                    }
            report["query_lift_counterfactuals"] = audit
            apply_mechanism_gate(report["preliminary_target_decision"], mechanism_effects)

        if args.spatial_root is not None:
            spatial_root = args.spatial_root.resolve()
            spatial: dict[str, Any] = {}
            for protocol in ("P", "D"):
                spatial[protocol] = {
                    arm: validate_spatial_audit(
                        spatial_root, protocol, arm, loveda_root / protocol / arm,
                        native_configs[protocol][arm], raw[protocol][arm], runs[arm],
                    )
                    for arm in ARMS
                }
                query = spatial[protocol]["query_lift"]
                spatial[protocol]["query_lift_delta_vs_controls_pp"] = {
                    arm: spatial_delta(query, spatial[protocol][arm])
                    for arm in ARMS if arm != "query_lift"
                }
            report["strict_spatial_audits"] = spatial
            report["scope"]["spatial_diagnostics"] = (
                "strict replay of each native result with exact per-image argmax, global confusion, and "
                "ignored-pixel bindings; replay validity is not by itself positive boundary/component evidence"
            )
            finalize_after_spatial_replay(report["preliminary_target_decision"])
        elif report["preliminary_target_decision"]["metric_screen_pass"] and report[
            "preliminary_target_decision"
        ].get("verified_mechanism"):
            report["preliminary_target_decision"]["decision"] = "HOLD_PENDING_STRICT_SPATIAL_REPLAY"
            report["preliminary_target_decision"]["rationale"] = (
                "The metric screen and P/U counterfactual validation are available, but strict fixed-checkpoint "
                "spatial replay is absent."
            )
    elif args.spatial_root is not None:
        raise ValueError("--spatial-root requires --loveda-root so strict replays can bind to native metrics")

    if "preliminary_target_decision" in report:
        report["preliminary_target_decision"]["research_disposition"] = research_disposition(
            report["preliminary_target_decision"]
        )

    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Q-Lift controlled gate audit", "", "## Contract", "",
             f"- Arms: {', '.join(ARMS)}", f"- Nominal matched trainable parameters: {report['contract']['nominal_trainable_parameters']:,}",
             "- DINO backbone and DINO.text: frozen", "- Selection: source-only at update 2,612; no LoveDA target selection", "",
             "## Source selection", "", "| Arm | Source score | COCO dev | OEM native val | Peak GiB |", "| --- | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        run = runs[arm]
        peak = run["training"]["rank0_peak_gib"]
        lines.append("| {arm} | {score:.4f} | {coco:.4f} | {oem:.4f} | {peak} |".format(
            arm=arm, score=100 * run["source_selection_score"], coco=run["source"]["coco_source_dev_miou_percent"],
            oem=run["source"]["oem_native_val_miou_percent"], peak="n/a" if peak is None else f"{peak:.2f}",
        ))
    if "strict_spatial_audits" in report:
        lines += ["", "## Strict spatial replays", "",
                  "Each entry re-ran native argmax inference and was published only after exact per-image "
                  "prediction digests, confusion matrix, and ignored-pixel support matched the native evaluator.", "",
                  "| Protocol | Query-Lift Δ boundary accuracy vs skip | Δ interior accuracy vs skip | Δ tiny coverage vs skip | Δ small coverage vs skip |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for protocol in ("P", "D"):
            values = report["strict_spatial_audits"][protocol]["query_lift_delta_vs_controls_pp"]["skip"]
            lines.append(
                "| {protocol} | {boundary} | {interior} | {tiny} | {small} |".format(
                    protocol=protocol,
                    boundary=values["gt_boundary_accuracy_pp"],
                    interior=values["gt_interior_accuracy_pp"],
                    tiny=values["tiny_macro_same_class_coverage_pp"],
                    small=values["small_macro_same_class_coverage_pp"],
                )
            )
    if "preliminary_target_decision" in report:
        decision = report["preliminary_target_decision"]
        deltas = decision["primary_metric_deltas_pp"]
        lines += ["", "## Fixed preliminary decision", "",
                  f"- Decision: `{decision['decision']}`", f"- Rationale: {decision['rationale']}",
                  f"- Research disposition: `{decision['research_disposition']['code']}`",
                  f"- Permitted next action: {decision['research_disposition']['action']}",
                  "- Primary deltas versus the strongest metric-matched control: "
                  f"P {deltas['P_six_class_miou_pp']:+.4f} pp; "
                  f"D foreground {deltas['D_foreground_miou_pp']:+.4f} pp; "
                  f"D seven-class {deltas['D_seven_class_miou_pp']:+.4f} pp.",
                  "- This fixed single-seed screen is not evidence of semantic-unseen OVSS, a causal P/U effect, "
                  "statistical robustness, or a paper-level result."]
    lines += ["", "The JSON report contains per-class precision/recall/area ratios, full confusion matrices, and all target/audit deltas when supplied."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
