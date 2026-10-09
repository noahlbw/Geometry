#!/usr/bin/env python3
"""Validate and summarize the pre-registered Q-Lift P/U × gain factorial audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from analyze_cafe_qlift_gate import (
    LOCKED_LOVEDA_SETTINGS,
    PREDECLARED_PAIRS_BY_PROTOCOL,
    directional_confusion,
    fingerprint,
    load,
    protocol_metric,
    same_file_binding,
    validate_mechanism_evaluation,
    validate_native_target_evaluation,
)


CELLS = {
    "Q,Q": "native",
    "Q,I": "pu-query-gain-image",
    "I,Q": "pu-image-gain-query",
    "I,I": "pu-image-gain-image",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-report", required=True, type=Path)
    parser.add_argument("--v1-native-root", required=True, type=Path)
    parser.add_argument("--factorial-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _metric_pp(result: dict[str, Any], protocol: str) -> float:
    return 100.0 * protocol_metric(result, protocol)


def _check_revised_native_equivalence(
    revised_dir: Path,
    original_dir: Path,
    original_config: dict[str, Any],
    original_result: dict[str, Any],
    source_run: dict[str, Any],
    protocol: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Accept code revision only when default Q,Q output is bitwise-identical."""
    config, result = load(revised_dir / "evaluation_config.json"), load(revised_dir / "results.json")
    settings = LOCKED_LOVEDA_SETTINGS[protocol]
    checks = {
        "config.protocol": (config.get("protocol"), protocol),
        "result.protocol": (result.get("protocol"), protocol),
        "config.qlift_audit": (config.get("qlift_audit"), "native"),
        "result.qlift_audit": (result.get("qlift_audit"), "native"),
        "result.status": (result.get("status"), "complete"),
        "config.model_variant": (config.get("model_variant"), "query_lift"),
        "result.model_variant": (result.get("model_variant"), "query_lift"),
        "config.target_used_for_training_or_selection": (config.get("target_used_for_training_or_selection"), False),
        "config.max_images": (config.get("max_images"), None),
        "config.images": (config.get("images"), 1669),
        "result.images": (result.get("images"), 1669),
        "config.world_size": (config.get("world_size"), 4),
        "result.world_size": (result.get("world_size"), 4),
        "config.architecture": (config.get("architecture"), source_run["architecture"]),
        "config.sample_keys": (config.get("sample_keys"), original_config.get("sample_keys")),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"Factorial Q,Q contract mismatch for {protocol}: {label}")
    for name, expected in settings.items():
        if config.get(name) != expected or result.get(name) != expected:
            raise ValueError(f"Factorial Q,Q does not use locked {protocol} setting {name!r}")
    if result.get("weights") != config.get("weights") or Path(config.get("weights", "")).resolve() != Path(
        source_run["selected_checkpoint"]["path"]
    ).resolve():
        raise ValueError(f"Factorial Q,Q does not use the original source-selected checkpoint for {protocol}")
    revised_bindings = config.get("integrity_bindings")
    original_bindings = original_config.get("integrity_bindings")
    if not isinstance(revised_bindings, dict) or not isinstance(original_bindings, dict):
        raise ValueError(f"Q,Q native integrity bindings are missing for {protocol}")
    same_file_binding(revised_bindings.get("selected_checkpoint"), source_run["selected_checkpoint"],
                      f"Factorial Q,Q selected checkpoint for {protocol}")
    revised_digest, original_digest = revised_bindings.get("native_argmax_digests"), original_bindings.get("native_argmax_digests")
    if not isinstance(revised_digest, dict) or not isinstance(original_digest, dict):
        raise ValueError(f"Q,Q native argmax digest binding is missing for {protocol}")
    if revised_digest.get("entries") != 1669 or revised_digest.get("sha256") != original_digest.get("sha256"):
        raise ValueError(f"Factorial Q,Q argmax maps do not exactly reproduce v1 native {protocol}")
    if result.get("confusion_matrix") != original_result.get("confusion_matrix"):
        raise ValueError(f"Factorial Q,Q confusion does not exactly reproduce v1 native {protocol}")
    if result.get("ignored_pixels") != original_result.get("ignored_pixels"):
        raise ValueError(f"Factorial Q,Q ignored pixel count differs from v1 native {protocol}")
    if not math.isclose(protocol_metric(result, protocol), protocol_metric(original_result, protocol), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Factorial Q,Q primary metric differs from v1 native {protocol}")
    return config, result


def _directional_contrast(cells: dict[str, dict[str, Any]], protocol: str) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    for first, second in PREDECLARED_PAIRS_BY_PROTOCOL[protocol]:
        for origin, destination in ((first, second), (second, first)):
            rates = {cell: directional_confusion(result, origin, destination) for cell, result in cells.items()}
            def contrast(left: str, right: str) -> float | None:
                if rates[left] is None or rates[right] is None:
                    return None
                return round(100.0 * (rates[left] - rates[right]), 6)
            output[f"{origin}->{destination}"] = {
                "pu_effect_under_image_gain_pp": contrast("Q,I", "I,I"),
                "gain_effect_under_image_pu_pp": contrast("I,Q", "I,I"),
                "interaction_pp": (
                    None if any(rates[cell] is None for cell in CELLS) else round(
                        100.0 * (rates["Q,Q"] - rates["Q,I"] - rates["I,Q"] + rates["I,I"]), 6
                    )
                ),
            }
    return output


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output must be fresh: {output}")
    report = load(args.v1_report.resolve())
    decision = report.get("preliminary_target_decision", {})
    if decision.get("decision") != "PROVISIONAL_GO_FOR_LIMITED_ATTRIBUTION_ONLY":
        raise ValueError("P/U × gain factorial is pre-registered only after a v1 provisional GO")
    source_run = report.get("source_runs", {}).get("query_lift")
    if not isinstance(source_run, dict):
        raise ValueError("v1 report lacks the verified source-selected query_lift run")
    result: dict[str, Any] = {
        "status": "complete",
        "scope": (
            "one-seed, target-only inference intervention after a v1 provisional gate; "
            "not causal proof, semantic-unseen OVSS, or a paper result"
        ),
        "v1_report": {"path": str(args.v1_report.resolve()), "sha256": fingerprint(args.v1_report.resolve())},
        "cells": {"Q,Q": "query P/U + query gain", "Q,I": "query P/U + image gain",
                  "I,Q": "image P/U + query gain", "I,I": "image P/U + image gain"},
        "protocols": {},
    }
    for protocol in ("P", "D"):
        original_dir = args.v1_native_root.resolve() / protocol / "query_lift"
        original_config, original_result = validate_native_target_evaluation(
            original_dir, protocol, "query_lift", source_run
        )
        revised_dir = args.factorial_root.resolve() / protocol / "Q,Q"
        revised_config, revised_result = _check_revised_native_equivalence(
            revised_dir, original_dir, original_config, original_result, source_run, protocol
        )
        cells: dict[str, dict[str, Any]] = {"Q,Q": revised_result}
        for cell in ("Q,I", "I,Q", "I,I"):
            mode = CELLS[cell]
            _, cells[cell] = validate_mechanism_evaluation(
                args.factorial_root.resolve() / protocol / cell, protocol, mode,
                revised_dir, revised_config, source_run,
            )
        metrics = {cell: _metric_pp(value, protocol) for cell, value in cells.items()}
        pu_effect = metrics["Q,I"] - metrics["I,I"]
        gain_effect = metrics["I,Q"] - metrics["I,I"]
        interaction = metrics["Q,Q"] - metrics["Q,I"] - metrics["I,Q"] + metrics["I,I"]
        directional = _directional_contrast(cells, protocol)
        result["protocols"][protocol] = {
            "original_native_reference": str(original_dir),
            "original_native_config_sha256": fingerprint(original_dir / "evaluation_config.json"),
            "factorial_native_config_sha256": fingerprint(revised_dir / "evaluation_config.json"),
            "primary_metric_percent": {cell: round(value, 4) for cell, value in metrics.items()},
            "pu_effect_under_image_gain_pp": round(pu_effect, 4),
            "gain_effect_under_image_pu_pp": round(gain_effect, 4),
            "interaction_pp": round(interaction, 4),
            "directional_confusion_contrasts_pp": directional,
            "pu_effect_effectively_null": abs(pu_effect) <= 0.2,
        }
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
