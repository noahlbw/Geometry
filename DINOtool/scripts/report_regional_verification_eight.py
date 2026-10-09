#!/usr/bin/env python3
"""Verify and summarize the frozen eight-dataset regional verification run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = (
    ("udd5", 40), ("oem", 384), ("vdd", 80), ("potsdam", 504),
    ("vaihingen", 113), ("landcoverai", 1602), ("loveda", 1669),
    ("flair1", 15700),
)
METHODS = ("Geometry", "Multiscale", "ScreenOnly", "ContextOnly",
           "VerifiedGeometry")
IMPLEMENTATION = "geometry-regional-alias-verification-v1-20260930"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(dataset: str, count: int, result: dict, previous: dict) -> None:
    signature = result["signature"]
    old = previous["signature"]
    if (result.get("status") != "complete" or not result.get("coverage_verified")
            or result["processed_images"] != result["total_images"]
            or result["processed_images"] != count):
        raise ValueError(f"{dataset}: incomplete merged result")
    if signature["implementation"] != IMPLEMENTATION or tuple(signature["methods"]) != METHODS:
        raise ValueError(f"{dataset}: wrong implementation or arms")
    if signature["global_sample_keys_sha256"] != old["global_sample_keys_sha256"]:
        raise ValueError(f"{dataset}: sample sequence differs from the historical control")
    if signature["vocabulary"]["sha256"] != old["vocabulary"]["sha256"]:
        raise ValueError(f"{dataset}: vocabulary changed")
    if signature["checkpoints"] != old["checkpoints"]:
        raise ValueError(f"{dataset}: checkpoints changed")
    if any(value != 20 for counts in signature["vocabulary"]["alias_counts"].values()
           for value in counts):
        raise ValueError(f"{dataset}: expected exactly 20 aliases per class")


def miou(result: dict, protocol: str, method: str) -> float:
    return result["metrics"][protocol][method]["mean_iou_percent"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lines = [
        "# Regional alias verification: eight-dataset frozen evaluation",
        "",
        "Full-set results, frozen DINOv3/DINO.text, exactly 20 aliases per class. "
        "All five arms use the same Geometry views, masks and 512/128 sliding-window protocol. "
        "Target labels enter metrics only. This is exploratory validation on datasets "
        "used during method development, not independent test performance.",
        "",
        "| Dataset | Images | Geometry | Multiscale | Screen only | Correction only | "
        "Verified Geometry | Final - Multiscale |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    completed = []
    deltas = []
    results = {}
    for dataset, count in DATASETS:
        result = load(args.results / f"{dataset}_merged.json")
        previous = load(args.baseline / f"gear_ov_v2_{dataset}_full_20260929_results.json")
        validate(dataset, count, result, previous)
        results[dataset] = result
        completed.append(dataset)
        protocol = "D" if dataset == "loveda" else dataset
        values = [miou(result, protocol, method) for method in METHODS]
        old_geometry = miou(previous, protocol, "Geometry")
        old_multiscale = miou(previous, protocol, "Multiscale")
        if abs(values[0] - old_geometry) > 0.0002 or abs(values[1] - old_multiscale) > 0.0002:
            raise ValueError(f"{dataset}: matched Geometry/Multiscale controls changed")
        delta = values[-1] - values[1]
        deltas.append(delta)
        lines.append("| " + " | ".join([
            dataset, str(count), *(f"{value:.4f}" for value in values), f"{delta:+.4f}"
        ]) + " |")
    loveda = results["loveda"]
    p_values = [miou(loveda, "P", method) for method in METHODS]
    lines.append("| LoveDA P (same images) | 1669 | " + " | ".join(
        f"{value:.4f}" for value in p_values) + f" | {p_values[-1] - p_values[1]:+.4f} |")
    lines.extend([
        "",
        f"Datasets improved over matched Multiscale: {sum(value > 0 for value in deltas)}/8; "
        f"worsened: {sum(value < 0 for value in deltas)}/8. "
        f"Unweighted eight-dataset mean delta: {sum(deltas) / len(deltas):+.4f} pp "
        "(LoveDA D counted once; this mixed-taxonomy average is auxiliary).",
        "",
        "Every merged output verified complete unique global-sample coverage, exact sample sequence, "
        "checkpoint manifest, vocabulary SHA, 20 aliases per class and matching Geometry/Multiscale controls.",
        "",
        "## Per-class IoU",
        "",
    ])
    for dataset, _ in DATASETS:
        result = results[dataset]
        for protocol in (("P", "D") if dataset == "loveda" else (dataset,)):
            lines.extend([f"### {dataset} / {protocol}", "", "| Class | Multiscale | Final | Delta |",
                          "|---|---:|---:|---:|"])
            base = result["metrics"][protocol]["Multiscale"]["per_class"]
            final = result["metrics"][protocol]["VerifiedGeometry"]["per_class"]
            for left, right in zip(base, final):
                if left["name"] != right["name"]:
                    raise ValueError(f"{dataset}: class ordering changed")
                a, b = left["iou_percent"], right["iou_percent"]
                if a is None or b is None:
                    lines.append(f"| {left['name']} | {a} | {b} | n/a |")
                else:
                    lines.append(f"| {left['name']} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
            lines.append("")
    lines.extend(["## Diagnostics and cost", "",
                  "| Dataset | Alias rejection | Active regions | Changed patches | "
                  "Parallel wall (s) | Peak GPU MiB |",
                  "|---|---:|---:|---:|---:|---:|"])
    for dataset, _ in DATASETS:
        result = results[dataset]
        protocol = "D" if dataset == "loveda" else dataset
        values = result["diagnostics"][protocol]
        lines.append(
            f"| {dataset} | {values['mean_alias_rejection']:.4f} | "
            f"{values['active_region_fraction']:.4f} | "
            f"{values['changed_patch_fraction']:.4f} | "
            f"{result['parallel_wall_seconds']:.1f} | "
            f"{result['peak_cuda_memory_mb']:.1f} |"
        )
    lines.extend([
        "",
        "LoveDA P scores six foreground classes; D scores seven including background. "
        "UDD5 and LandCover.ai also provide non-residual foreground metrics in the merged JSON. "
        "OEM contains 384 available validation images; Potsdam comprises 504 prepared tiles; "
        "Vaihingen retains the historical first-three-band input; LandCover.ai is the eighth "
        "labeled set in place of unavailable iSAID masks.",
        "",
    ])
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Verified {', '.join(completed)}; report: {args.output}")


if __name__ == "__main__":
    main()
