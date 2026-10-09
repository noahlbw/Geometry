#!/usr/bin/env python3
"""Report verified full results without fitting or selecting a routing rule."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ARMS = ("Geometry", "Multiscale", "VIP20", "MeanProb50", "MaxConfidence",
        "LeaveFamilyOut", "GroundedLeaveFamilyOut")
TOTALS = {"vdd": 80, "potsdam": 504}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def report(research):
    results = {}
    controls = {}
    shared_rule = None
    for dataset, expected in TOTALS.items():
        j = read(research / "geometry_vip_reliability_20260930" / f"{dataset}_merged.json")
        sig = j["signature"]
        keys = sig["sample_keys"]
        if (j["status"] != "complete" or not j["coverage_verified"]
                or j["processed_images"] != j["total_images"] or j["total_images"] != expected
                or len(keys) != len(set(keys)) or len(keys) != expected
                or hashlib.sha256("\n".join(keys).encode()).hexdigest() != sig["global_sample_keys_sha256"]
                or any(count != 20 for count in sig["alias_counts"])):
            raise ValueError(f"Invalid full coverage or vocabulary: {dataset}")
        rule = {key: value for key, value in sig["selector"].items() if key != "family_count"}
        if shared_rule is not None and rule != shared_rule:
            raise ValueError("The two datasets used different selection rules.")
        shared_rule = rule
        historical = read(research / f"gear_ov_v2_{dataset}_full_20260929_results.json")
        if sig["global_sample_keys_sha256"] != historical["signature"]["global_sample_keys_sha256"]:
            raise ValueError(f"Historical sample sequence differs: {dataset}")
        if (sig["vocabulary_sha256"] != historical["signature"]["vocabulary"]["sha256"]
                or sig["checkpoint_manifest"] != historical["signature"]["checkpoints"]):
            raise ValueError(f"Historical vocabulary or checkpoint manifest differs: {dataset}")
        native_vip = read(research / f"matched_head_text_{dataset}_full_20260930_results.json")
        if sig["global_sample_keys_sha256"] != native_vip["signature"]["global_sample_keys_sha256"]:
            raise ValueError(f"VIP control sequence differs: {dataset}")
        controls[dataset] = {}
        for arm in ("Geometry", "Multiscale"):
            controls[dataset][arm] = (j["metrics"][arm]["confusion_matrix"]
                                     == historical["metrics"][dataset][arm]["confusion_matrix"])
        controls[dataset]["VIP20"] = (j["metrics"]["VIP20"]["confusion_matrix"]
                                      == native_vip["metrics"]["VIP_ImageNet"]["confusion_matrix"])
        if not all(controls[dataset].values()):
            raise ValueError(f"Historical controls failed exact reproduction: {dataset}")
        results[dataset] = j
    lines = ["# Geometry / VIP branch-selection feasibility: full VDD and Potsdam", "",
             "2026-09-30. Frozen image-only rules, same 20 aliases per class. Full VDD 80 images and",
             "Potsdam 504 RGB tiles from 14 parent scenes. No target-mask parameter selection.",
             "These repeatedly consulted datasets are exploratory development, not untouched test sets.", "",
             "The strong branch profiles intentionally differ: Geometry/Multiscale uses its existing",
             "512/128 three-view RS-template protocol; VIP20 uses pinned proxy/ImageNet/448/336/112",
             "with native scoring and no background threshold. This is not a head-only factorial.",
             "Both runtime copies load the same frozen weights; the copies preserve native precision.", "",
             "## Full mIoU", "", "| Method | VDD | Delta vs Multiscale | Potsdam | Delta vs Multiscale |",
             "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        values = []
        for dataset in TOTALS:
            metrics = results[dataset]["metrics"]
            score = metrics[arm]["mean_iou_percent"]
            delta = score - metrics["Multiscale"]["mean_iou_percent"]
            values += [f"{score:.4f}", f"{delta:+.4f}"]
        lines.append("| " + " | ".join((arm, *values)) + " |")
    lines += ["", "## Control reproduction", ""]
    for dataset in TOTALS:
        lines.append(f"- {dataset}: exact historical confusion equality {controls[dataset]}.")
    lines += ["", "## Complementarity and routing", "",
              "Pattern order: both wrong, Multiscale only correct, VIP only correct, both correct.",
              "The oracle is a GT-only pixel-accuracy audit, not a usable rule or an mIoU bound.", ""]
    for dataset, j in results.items():
        patterns = [sum(row[index] for row in j["audit"]["correctness_by_true_class"]) for index in range(4)]
        lines += [f"### {dataset}", "", f"Full-resolution pattern counts: {patterns}.",
                  f"GT-only any-correct accuracy: {j['audit']['oracle_any_correct_accuracy_percent']:.4f}%.", "",
                  "| Selector | Beneficial retention | Harmful rejection | XOR route accuracy | Fixes vs Multiscale | Harms vs Multiscale |",
                  "|---|---:|---:|---:|---:|---:|"]
        for arm, rec in j["audit"]["routing"].items():
            rates = ["n/a" if rec[field] is None else f"{100*rec[field]:.2f}%"
                     for field in ("beneficial_retention", "harmful_rejection", "xor_route_accuracy")]
            changes = j["audit"]["changes"]["Multiscale"][arm]
            lines.append("| " + " | ".join((arm, *rates, str(changes["beneficial"]), str(changes["harmful"]))) + " |")
        lines += ["", "| Method | IoU by class |", "|---|---|"]
        for arm in ARMS:
            text = "; ".join(f"{entry['name']}={entry['iou_percent']}" for entry in j["metrics"][arm]["per_class"])
            lines.append(f"| {arm} | {text} |")
        lines += ["", "| Method / class | Predicted area % | Precision % | Recall % |", "|---|---:|---:|---:|"]
        for arm in ("Multiscale", "VIP20", "MeanProb50", "GroundedLeaveFamilyOut"):
            for entry in j["metrics"][arm]["per_class"]:
                lines.append(f"| {arm} / {entry['name']} | {entry['predicted_area_percent']:.4f} | {entry['precision_percent']:.4f} | {entry['recall_percent']:.4f} |")
        lines += ["", f"Patch diagnostics: `{json.dumps(j['diagnostics'], sort_keys=True)}`.",
                  f"Parallel wall {j['parallel_wall_seconds']:.2f}s; aggregate GPU {j['aggregate_gpu_seconds']:.2f}s;",
                  f"peak allocated CUDA {j['peak_cuda_memory_mb']:.2f} MiB (combined evaluation, two frozen runtime copies).", ""]
    passed = all(results[d]["metrics"]["GroundedLeaveFamilyOut"]["mean_iou_percent"] > max(
        results[d]["metrics"][arm]["mean_iou_percent"] for arm in ("Multiscale", "VIP20", "MeanProb50", "MaxConfidence"))
        for d in TOTALS)
    lines += ["## Predeclared mechanism decision", "",
              "GroundedLeaveFamilyOut exceeds Multiscale, VIP20, fixed equal-probability fusion and",
              f"max-confidence selection on both development datasets: **{passed}**.", "",
              "This decision does not establish statistical significance, eight-dataset transfer or novelty.",
              "Failure does not justify retuning this selector on these masks or renaming it a successful final model.",
              "All counterfactual, area and per-class counts are audit-only; they did not enter prediction."]
    return "\n".join(lines) + "\n", results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text, results = report(args.research)
    if args.output.exists():
        raise ValueError(f"Refusing existing report: {args.output}")
    args.output.write_text(text, encoding="utf-8")
    print(json.dumps({d: {a: results[d]["metrics"][a]["mean_iou_percent"] for a in ARMS} for d in TOTALS}))
