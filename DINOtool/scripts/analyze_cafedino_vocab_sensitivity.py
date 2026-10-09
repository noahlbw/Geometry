#!/usr/bin/env python3
"""Compare locked CAFe-DINO LoveDA vocabulary experiments.

Only results that used identical data, metric class order, image resize,
striding, model mode, and background threshold are accepted. This keeps the
reported delta attributable to the prompt vocabulary rather than an inference
or benchmark change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COMPARABILITY_KEYS = (
    "method",
    "data_root",
    "sample_count",
    "sample_keys_sha256",
    "classes",
    "include_background",
    "evaluation_size",
    "window_size",
    "stride",
    "background_threshold",
    "model_mode",
    "seed",
    "ground_truth_mapping",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="results.json for the official one-prompt baseline.")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="NAME=RESULTS_JSON",
        help="Named variant result. Can be supplied more than once.",
    )
    parser.add_argument("--output", required=True, help="Output JSON summary path.")
    return parser.parse_args()


def load_result(path_text: str) -> dict[str, Any]:
    path = Path(path_text).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid result JSON at {path}: {error}") from error
    if payload.get("status") != "complete":
        raise ValueError(f"Result is not complete: {path}")
    if not isinstance(payload.get("metrics"), dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"Not a locked CAFe-DINO result: {path}")
    payload["_path"] = str(path)
    return payload


def parse_variant(value: str) -> tuple[str, str]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise ValueError(f"Variant must have the form NAME=RESULTS_JSON, got {value!r}.")
    return name.strip(), path.strip()


def ensure_comparable(baseline: dict[str, Any], variant: dict[str, Any]) -> None:
    expected = baseline["config"]
    observed = variant["config"]
    mismatches = [
        key for key in COMPARABILITY_KEYS if expected.get(key) != observed.get(key)
    ]
    base_checkpoint = baseline.get("checkpoints", {}).get("checkpoint")
    variant_checkpoint = variant.get("checkpoints", {}).get("checkpoint")
    if base_checkpoint != variant_checkpoint:
        mismatches.append("checkpoint")
    if mismatches:
        raise ValueError(
            f"Non-comparable result {variant['_path']}; differs from baseline in: {', '.join(mismatches)}."
        )


def result_row(name: str, payload: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    metrics = payload["metrics"]
    vocabulary = payload["config"].get("vocabulary", {})
    baseline_ious = {
        row["name"]: row["iou_percent"]
        for row in baseline["metrics"]["per_class"]
    } if baseline is not None else {}
    per_class = []
    for row in metrics["per_class"]:
        value = row["iou_percent"]
        reference = baseline_ious.get(row["name"])
        per_class.append(
            {
                "name": row["name"],
                "iou_percent": value,
                "delta_from_baseline_points": (
                    None if baseline is None or value is None or reference is None else round(value - reference, 4)
                ),
            }
        )
    base_miou = None if baseline is None else baseline["metrics"]["mean_iou_percent"]
    return {
        "name": name,
        "result_path": payload["_path"],
        "aggregation": vocabulary.get("aggregation"),
        "prompt_count": vocabulary.get("prompt_count"),
        "mean_iou_percent": metrics["mean_iou_percent"],
        "delta_from_baseline_points": (
            None if base_miou is None else round(metrics["mean_iou_percent"] - base_miou, 4)
        ),
        "pixel_accuracy_percent": metrics["pixel_accuracy_percent"],
        "per_class": per_class,
        "wall_seconds": payload.get("timing", {}).get("wall_seconds"),
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# CAFe-DINO vocabulary-sensitivity summary",
        "",
        "| condition | aggregation | prompt entries | mIoU (%) | delta (points) | wall time (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        delta = "—" if row["delta_from_baseline_points"] is None else f"{row['delta_from_baseline_points']:+.4f}"
        wall = "—" if row["wall_seconds"] is None else f"{row['wall_seconds']:.1f}"
        lines.append(
            f"| {row['name']} | {row['aggregation']} | {row['prompt_count']} | "
            f"{row['mean_iou_percent']:.4f} | {delta} | {wall} |"
        )
    lines.extend(["", "Per-class deltas are in the JSON summary.", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    baseline = load_result(args.baseline)
    rows = [result_row("official_single_prompt", baseline, None)]
    seen_names = {rows[0]["name"]}
    for raw_variant in args.variant:
        name, path = parse_variant(raw_variant)
        if name in seen_names:
            raise ValueError(f"Duplicate variant name: {name}")
        variant = load_result(path)
        ensure_comparable(baseline, variant)
        rows.append(result_row(name, variant, baseline))
        seen_names.add(name)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"baseline": rows[0], "variants": rows[1:]}, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown_table(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
