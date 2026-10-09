#!/usr/bin/env python3
"""Refuse non-matched PCA-DINO source pairs, then aggregate their deltas."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

STEP = 2612
ARG_EXCEPTIONS = {"arm", "output_dir"}


def load_json(path):
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{path} is not an object")
    return obj


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def number(value, label):
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} is not finite: {value!r}")
    return float(value)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_self_resume(seed, epl_dir, epl_args, provenance_path):
    """Allow only a recorded self-resume, never a generic resume exemption."""
    if provenance_path is None:
        raise ValueError(f"seed {seed}: resume differs without approved provenance")
    provenance = load_json(Path(provenance_path))
    expected_resume = str(Path(epl_dir) / "last.pt")
    required = {
        "schema": "self_resume_continuity_v1",
        "seed": str(seed),
        "arm": "pca_epl",
        "resume_argument": expected_resume,
        "resumed_from_step": 1500,
        "first_resumed_step": 1501,
    }
    for key, expected in required.items():
        if provenance.get(key) != expected:
            raise ValueError(f"seed {seed}: invalid resume provenance field {key}")
    if epl_args.get("resume") != expected_resume:
        raise ValueError(f"seed {seed}: resume is not this EPL arm's own last.pt")
    log_path = Path(provenance.get("resume_log", ""))
    expected_hash = provenance.get("resume_log_sha256")
    if not log_path.is_file() or sha256(log_path) != expected_hash:
        raise ValueError(f"seed {seed}: resume log is absent or its hash changed")
    first = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith('{"status"'):
            first = json.loads(line)
            break
    if not isinstance(first, dict) or first.get("step") != 1501:
        raise ValueError(f"seed {seed}: resume log does not begin at step 1501")
    steps = (Path(epl_dir) / "steps.jsonl").read_text(encoding="utf-8", errors="replace")
    if '"step": 1500' not in steps or '"step": 1520' not in steps:
        raise ValueError(f"seed {seed}: persisted step history lacks the resume boundary")
    return {"approved": True, "provenance": str(provenance_path),
            "resumed_from_step": 1500, "first_resumed_step": 1501,
            "resume_log_sha256": expected_hash}


def load_arm(directory, expected):
    files = {name: directory / name for name in (
        "status.json", "training_config.json", "source_manifest.json", "best_inference.pt")}
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise ValueError(f"{directory}: missing {', '.join(missing)}")
    status = load_json(files["status.json"])
    config = load_json(files["training_config.json"])
    manifest = load_json(files["source_manifest.json"])
    args = config.get("args")
    if not isinstance(args, dict) or args.get("arm") != expected:
        found = args.get("arm") if isinstance(args, dict) else None
        raise ValueError(f"{directory}: expected arm={expected}, found {found}")
    validation = status.get("validation")
    if (status.get("status"), status.get("step"),
            validation.get("validation_step") if isinstance(validation, dict) else None) != ("paused_at_validation", STEP, STEP):
        raise ValueError(f"{directory}: lacks locked validation at step {STEP}")
    try:
        coco = number(validation["coco"]["mean_iou"], f"{directory} COCO")
        oem = number(validation["oem"]["mean_iou"], f"{directory} OEM")
        saved = number(validation["selection_score"], f"{directory} selection score")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{directory}: malformed source validation") from exc
    mean = (coco + oem) / 2
    if not math.isclose(mean, saved, abs_tol=1e-9):
        raise ValueError(f"{directory}: mean {mean} != saved score {saved}")
    return {"dir": str(directory), "args": args, "manifest": manifest,
            "world_size": config.get("world_size"),
            "max_global_batch": config.get("max_global_batch"),
            "base_checkpoint": config.get("base_checkpoint"),
            "coco_miou": coco, "oem_miou": oem, "source_mean": mean,
            "elapsed_seconds": status.get("elapsed_seconds")}


def verify(seed, parallel_dir, epl_dir, resume_provenance=None):
    parallel = load_arm(Path(parallel_dir), "parallel")
    epl = load_arm(Path(epl_dir), "pca_epl")
    if canon(parallel["manifest"]) != canon(epl["manifest"]):
        raise ValueError(f"seed {seed}: source manifests differ")
    for key in ("world_size", "max_global_batch", "base_checkpoint"):
        if parallel[key] != epl[key]:
            raise ValueError(f"seed {seed}: {key} differs")
    pa = {k: v for k, v in parallel["args"].items() if k not in ARG_EXCEPTIONS}
    ea = {k: v for k, v in epl["args"].items() if k not in ARG_EXCEPTIONS}
    changed = sorted(k for k in set(pa) | set(ea) if pa.get(k) != ea.get(k))
    resume_exception = None
    if changed:
        if changed != ["resume"]:
            raise ValueError(f"seed {seed}: non-arm arguments differ: {changed}")
        resume_exception = verify_self_resume(seed, epl_dir, epl["args"], resume_provenance)
    metrics = ("coco_miou", "oem_miou", "source_mean")
    return {"seed": str(seed),
            "parallel": {k: parallel[k] for k in ("dir", *metrics, "elapsed_seconds")},
            "pca_epl": {k: epl[k] for k in ("dir", *metrics, "elapsed_seconds")},
            "epl_minus_parallel": {k: epl[k] - parallel[k] for k in metrics},
            "protocol": {"validation_step": STEP, "world_size": parallel["world_size"],
                         "max_global_batch": parallel["max_global_batch"],
                         "manifest_match": True, "non_arm_arguments_match": not changed,
                         "resume_exception": resume_exception}}


def summarize_deltas(pairs):
    """Return paired-effect statistics without mixing resumed and clean runs."""
    metrics = ("coco_miou", "oem_miou", "source_mean")
    return {m: {"mean": statistics.fmean(row["epl_minus_parallel"][m] for row in pairs),
                "sample_std": statistics.stdev(row["epl_minus_parallel"][m] for row in pairs)
                if len(pairs) > 1 else None}
            for m in metrics}


def clean_seed_sensitivity(pairs):
    clean_pairs = [row for row in pairs if row["protocol"]["resume_exception"] is None]
    return {"clean_pair_count": len(clean_pairs),
            "clean_seeds": [row["seed"] for row in clean_pairs],
            "aggregate_epl_minus_parallel": summarize_deltas(clean_pairs) if clean_pairs else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=3, action="append", metavar=("SEED", "PARALLEL", "EPL"), required=True)
    parser.add_argument("--resume-provenance", nargs=2, action="append", metavar=("SEED", "JSON"), default=[],
                        help="approved self-resume evidence for one named seed only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-markdown", action="store_true")
    ns = parser.parse_args()
    if len({seed for seed, _, _ in ns.pair}) != len(ns.pair):
        raise ValueError("duplicate seed")
    provenance = dict(ns.resume_provenance)
    if len(provenance) != len(ns.resume_provenance):
        raise ValueError("duplicate resume-provenance seed")
    unknown = set(provenance) - {seed for seed, _, _ in ns.pair}
    if unknown:
        raise ValueError(f"resume provenance without pair: {sorted(unknown)}")
    pairs = sorted((verify(*row, resume_provenance=provenance.get(row[0])) for row in ns.pair),
                   key=lambda row: row["seed"])
    aggregate = summarize_deltas(pairs)
    clean_sensitivity = clean_seed_sensitivity(pairs)
    report = {"protocol": "strict_source_only_pca_dino_pair_aggregate_v1", "pairs": pairs,
              "aggregate_epl_minus_parallel": aggregate,
              "clean_seed_sensitivity": clean_sensitivity}
    ns.output.parent.mkdir(parents=True, exist_ok=True)
    ns.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if ns.print_markdown:
        print("| seed | parallel mean | EPL mean | delta |\n|---|---:|---:|---:|")
        for row in pairs:
            print(f"| {row['seed']} | {100*row['parallel']['source_mean']:.4f} | {100*row['pca_epl']['source_mean']:.4f} | {100*row['epl_minus_parallel']['source_mean']:+.4f} |")
        print(f"EPL−parallel source-mean: {100*aggregate['source_mean']['mean']:+.4f} pp")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"strict aggregation refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
