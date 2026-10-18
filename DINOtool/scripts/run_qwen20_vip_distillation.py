"""Run Qwen20 candidate banks through the VIP paper-rule selector and evaluator.

This controller is deliberately separate from the historical VIP controller:
all eight remote-sensing datasets use the supplied JSON bank, including the
three datasets for which the historical runner defaults to VIP's official
short-query files.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from itertools import zip_longest
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.vip_paper_distillation import select_from_scores
from eval_vip_official_eight import Confusion, digest
from eval_vip_paper_distillation import ARMS, CONFIG, CONVENTIONS, IMPLEMENTATION, save
from run_region_semantic_suite_a800 import BASE, PYTHON, THIRD, TOOL, idle


DATASETS = ("vdd", "potsdam", "udd5", "oem", "loveda", "vaihingen", "landcoverai", "flair1")
SHARDS = {"vdd": 1, "potsdam": 1, "udd5": 1, "oem": 1, "loveda": 2,
          "vaihingen": 1, "landcoverai": 1, "flair1": 4}
TOTALS = {"vdd": 80, "potsdam": 504, "udd5": 40, "oem": 384, "loveda": 1669,
          "vaihingen": 113, "landcoverai": 1602, "flair1": 15700}
REFS = {
    "vdd": TOOL / "results/vip_paper_distillation_finite_full_20261003/references/vdd.json",
    "potsdam": TOOL / "results/vip_paper_distillation_finite_full_20261003/references/potsdam.json",
    "vaihingen": TOOL / "results/vip_paper_distillation_finite_full_20261003/references/vaihingen.json",
    "udd5": TOOL / "results/vip_paper_distillation_finite_full_20261003/references/udd5.json",
    "oem": TOOL / "results/vip_paper_distillation_finite_full_20261003/references/oem.json",
    "loveda": TOOL / "results/vip_paper_distillation_finite_full_20261003/references/loveda.json",
    "landcoverai": TOOL / "results/vip_paper_distillation_finite_full_20261003/references/landcoverai.json",
    "flair1": TOOL / "results/vip_paper_distillation_finite_full_20261003/references/flair1.json",
}
DATA_ROOTS = {
    "vdd": "VDD Release Version/VDD", "potsdam": "Potsdam/preprocessed_RGB",
    "udd5": "UDD5/extracted/UDD/UDD5", "oem": "OpenEarthMap_wo_xBD",
    "loveda": "/data/test/cafe-efa/data/processed/loveda/val",
    "vaihingen": "Vaihingen/preprocessed_corrected_20261001",
    "landcoverai": "LandCoverAI_v1_target_20260922/preprocessed_official512_gear29",
    "flair1": "FLAIR1_target_20260922",
}


def read(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def alive(session: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0


def root_path(path: str) -> Path:
    return Path(path).resolve()


def command(root: Path, dataset: str, phase: str, shard: int, gpu: int, vocab_dir: Path,
            refs: Path | None = None) -> dict:
    if not idle(gpu):
        raise RuntimeError(f"GPU{gpu} occupied")
    output = root / phase / dataset / f"s{shard}"
    output.parent.mkdir(parents=True, exist_ok=True)
    session = f"q20vip_{phase}_{dataset}_s{shard}"
    log = output.with_suffix(".log")
    if output.exists() or log.exists() or alive(session):
        raise RuntimeError("existing output/session: " + session)
    args = [PYTHON, "-u", "scripts/eval_vip_paper_distillation.py", "--phase", phase,
            "--dataset", dataset, "--dinov3-repo", str(TOOL / "dinov3_hub"),
            "--checkpoint-dir", str(BASE / "ckpt/DINO"), "--data-root",
            str(Path("/data/test/datasets") / DATA_ROOTS[dataset]), "--vocabulary-config",
            str(vocab_dir / (dataset + ".json")), "--vocabulary-source", "json",
            "--custom-candidate", "--upstream-root", str(BASE / "third_party/VIP_official_5bd25ee"),
            "--output-dir", str(output), "--reference-json", str(root / "references" / (dataset + ".json")),
            "--num-shards", str(SHARDS[dataset]), "--shard-index", str(shard)]
    if phase == "evaluation":
        args += ["--source-dir", str(root / "selection" / dataset / f"s{shard}"),
                 "--selection-json", str(root / "selection" / dataset / "selected.json")]
    env = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
           "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2",
           "TOKENIZERS_PARALLELISM=false"]
    shell = "cd " + shlex.quote(str(TOOL)) + " && exec " + shlex.join(env + ["nice", "-n", "10", *args])
    shell += " > " + shlex.quote(str(log)) + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    return {"dataset": dataset, "phase": phase, "shard": shard, "gpu": gpu,
            "session": session, "output": str(output), "log": str(log)}


def complete_shards(root: Path, dataset: str, phase: str):
    paths = [root / phase / dataset / f"s{s}" / "results.json" for s in range(SHARDS[dataset])]
    rows = [read(p) for p in paths]
    if any(not r or r["status"] != "complete" or r["processed_images"] != r["total_images"] for r in rows):
        raise RuntimeError("incomplete shards: " + dataset + "/" + phase)
    first = rows[0]["signature"]
    excluded = {"sample_keys", "sample_keys_sha256", "sample_count", "shard_index"}
    source = {k: v for k, v in first.items() if k not in excluded}
    shard_keys = []
    for s, row in enumerate(rows):
        sig = row["signature"]
        if sig["shard_index"] != s or sig["num_shards"] != len(rows) or \
                source != {k: v for k, v in sig.items() if k not in excluded}:
            raise RuntimeError("selection signature differs: " + dataset)
        shard_keys.append(sig["sample_keys"])
    # Shards are index-strided (i, i+n, ...), so restore the original global
    # order before comparing with the locked reference sequence.
    keys = [key for group in zip_longest(*shard_keys) for key in group if key is not None]
    ref = read(root / "references" / (dataset + ".json"))
    if len(keys) != TOTALS[dataset] or len(set(keys)) != len(keys) or keys != ref["signature"]["sample_keys"]:
        raise RuntimeError("coverage differs: " + dataset)
    return rows, source, keys, ref


def freeze_selection(root: Path, dataset: str) -> Path:
    rows, source, keys, _ = complete_shards(root, dataset, "selection")
    if any(r["target_masks_loaded"] or not r["weights_frozen"] for r in rows):
        raise RuntimeError("selection mask/weight contract failed")
    banks = {}
    for p, meta in source["query_metadata"].items():
        state = {field: np.asarray([r["statistics"][p][field] for r in rows]).sum(0).tolist()
                 for field in ("vg_sum", "sc_sum", "observed_images")}
        bank = select_from_scores(state, meta["names"], meta["aliases"], meta["parents"],
                                  meta["cosine"], CONFIG)
        bank["statistics"] = state
        banks[p] = bank
    path = root / "selection" / dataset / "selected.json"
    if path.exists():
        raise RuntimeError("refusing existing selected bank: " + str(path))
    save(path, {"status": "complete", "coverage_verified": True,
        "processed_images": len(keys), "total_images": len(keys), "sample_keys": keys,
        "sample_keys_sha256": digest(keys), "source_signature": source, "banks": banks,
        "target_masks_loaded": False, "transductive": True,
        "selection_parallel_wall_seconds": max(r["wall_seconds"] for r in rows),
        "selection_peak_cuda_memory_mb": max(r["peak_cuda_memory_mb"] for r in rows)})
    return path


def merge_evaluation(root: Path, dataset: str) -> Path:
    rows, source, keys, ref = complete_shards(root, dataset, "evaluation")
    selected = read(root / "selection" / dataset / "selected.json")
    metrics = {}
    for p, names in source["scored_classes"].items():
        metrics[p] = {}
        for arm in ARMS:
            collector = Confusion(tuple(names), len(source["query_metadata"][p]["names"]))
            for s, row in enumerate(rows):
                metric = row["metrics"][p][arm]
                with np.load(root / "evaluation" / dataset / f"s{s}" / "per_image_confusions.npz",
                             allow_pickle=False) as arrays:
                    if arrays["sample_keys"].tolist() != row["signature"]["sample_keys"] or \
                            not np.array_equal(arrays[p + "__" + arm].sum(0), metric["confusion_matrix"]):
                        raise RuntimeError("per-image confusion verification failed")
                collector.matrix += np.asarray(metric["confusion_matrix"], dtype=np.int64)
                collector.ignored += metric["ignored_pixels"]
            metrics[p][arm] = collector.summary()
        baseline = metrics[p]["VIP_All20"]
        if not np.array_equal(np.asarray(baseline["confusion_matrix"]).sum(1),
                              np.asarray(ref["metrics"][p]["confusion_matrix"]).sum(1)):
            raise RuntimeError("scored target counts differ: " + dataset + "/" + p)
    path = root / "evaluation" / dataset / "merged.json"
    if path.exists():
        raise RuntimeError("refusing existing merged result: " + str(path))
    save(path, {"status": "complete", "coverage_verified": True, "processed_images": len(keys),
        "total_images": len(keys), "sample_keys": keys, "sample_keys_sha256": digest(keys),
        "source_signature": source, "selection": selected["banks"], "metrics": metrics,
        "per_image_confusions_verified": True,
        "evaluation_parallel_wall_seconds": max(r["wall_seconds"] for r in rows),
        "evaluation_peak_cuda_memory_mb": max(r["peak_cuda_memory_mb"] for r in rows)})
    return path


def write_protocol(root: Path, vocab_dir: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    (root / "references").mkdir()
    for dataset, path in REFS.items():
        subprocess.run(["cp", str(path), str(root / "references" / (dataset + ".json"))], check=True)
    save(root / "protocol.json", {"implementation": "qwen20-vip-paper-distillation-v1-20261018",
        "candidate_generation": "Qwen canonical plus 19 text-only proposals; no VIP official query file",
        "datasets": DATASETS, "shards": SHARDS, "totals": TOTALS, "candidate_count": 20,
        "fixed_vip_readout": True, "target_label_tuning": False,
        "use_all_unlabeled_evaluation_images": True, "custom_candidate": True,
        "config": asdict(CONFIG), "conventions": CONVENTIONS,
        "vocabulary_dir": str(vocab_dir)})


def main(args: argparse.Namespace) -> None:
    root, vocab_dir = root_path(args.root), root_path(args.vocabulary_dir)
    if root.exists():
        raise RuntimeError("refusing existing root: " + str(root))
    write_protocol(root, vocab_dir)
    gpus = tuple(range(args.first_gpu, args.first_gpu + args.gpu_count))
    active: dict[int, dict] = {}
    pending = [(d, "smoke", 0) for d in DATASETS]
    phase = "smoke"
    smoke_done: set[str] = set()
    selection_done: set[tuple[str, int]] = set()
    evaluation_done: set[tuple[str, int]] = set()
    failures = {}
    started = time.time()
    while active or pending:
        for gpu, job in list(active.items()):
            if alive(job["session"]):
                continue
            row = read(Path(job["output"]) / "results.json")
            try:
                if not row or row.get("status") != "complete":
                    raise RuntimeError("worker did not complete")
                dataset = job["dataset"]
                if job["phase"] == "smoke":
                    if row.get("target_masks_loaded") or not row.get("weights_unchanged") or not row.get("weights_frozen"):
                        raise RuntimeError("mask-free smoke contract failed")
                    smoke_done.add(dataset)
                elif job["phase"] == "selection":
                    selection_done.add((dataset, job["shard"]))
                    if all((dataset, s) in selection_done for s in range(SHARDS[dataset])):
                        freeze_selection(root, dataset)
                        pending.extend((dataset, "evaluation", s) for s in range(SHARDS[dataset]))
                else:
                    evaluation_done.add((dataset, job["shard"]))
                    if all((dataset, s) in evaluation_done for s in range(SHARDS[dataset])):
                        merge_evaluation(root, dataset)
                del active[gpu]
            except Exception as error:
                failures[job["session"]] = {"error": str(error),
                    "log": Path(job["log"]).read_text(errors="replace")[-6000:] if Path(job["log"]).exists() else "missing"}
                del active[gpu]
                pending.clear()
        if not active and phase == "smoke" and len(smoke_done) == len(DATASETS):
            phase = "selection"
            pending = [(d, "selection", s) for d in DATASETS for s in range(SHARDS[d])]
        for task in list(pending):
            free = next((g for g in gpus if g not in active and idle(g)), None)
            if free is None:
                break
            dataset, task_phase, shard = task
            active[free] = command(root, dataset, task_phase, shard, free, vocab_dir)
            pending.remove(task)
        save(root / "suite_status.json", {"status": "running" if active or pending else "failed" if failures else "complete",
            "phase": phase, "active": list(active.values()), "pending": pending,
            "smoke_completed": sorted(smoke_done), "selection_completed": sorted(selection_done),
            "evaluation_completed": sorted(evaluation_done), "failures": failures,
            "elapsed_seconds": time.time() - started})
        if active or pending:
            time.sleep(args.poll_seconds)
    save(root / "suite_results.json", {"status": "failed" if failures else "complete",
        "datasets": list(DATASETS), "completed": sorted(d for d in DATASETS if (root / "evaluation" / d / "merged.json").exists()),
        "failures": failures, "elapsed_seconds": time.time() - started})


def resume(root: Path, vocab_dir: Path, first_gpu: int, gpu_count: int, poll_seconds: int) -> None:
    """Repair a completed selection stage and continue evaluation.

    This is useful after a controller-only bookkeeping failure.  Existing
    selection caches and frozen banks are preserved; only missing evaluation
    shards are launched.
    """
    if not root.is_dir() or not (root / "protocol.json").is_file():
        raise RuntimeError("missing existing Qwen20 VIP root: " + str(root))
    for dataset in DATASETS:
        selected = root / "selection" / dataset / "selected.json"
        if not selected.exists():
            freeze_selection(root, dataset)
    pending = [(d, "evaluation", s) for d in DATASETS for s in range(SHARDS[d])
               if not (root / "evaluation" / d / f"s{s}" / "results.json").exists()]
    active: dict[int, dict] = {}
    failures = {}
    started = time.time()
    gpus = tuple(range(first_gpu, first_gpu + gpu_count))
    while active or pending:
        for gpu, job in list(active.items()):
            if alive(job["session"]):
                continue
            row = read(Path(job["output"]) / "results.json")
            try:
                if not row or row.get("status") != "complete":
                    raise RuntimeError("worker did not complete")
                dataset = job["dataset"]
                if all((root / "evaluation" / dataset / f"s{s}" / "results.json").is_file()
                       and read(root / "evaluation" / dataset / f"s{s}" / "results.json").get("status") == "complete"
                       for s in range(SHARDS[dataset])):
                    merge_evaluation(root, dataset)
                del active[gpu]
            except Exception as error:
                failures[job["session"]] = {"error": str(error),
                    "log": Path(job["log"]).read_text(errors="replace")[-6000:] if Path(job["log"]).exists() else "missing"}
                del active[gpu]
                pending.clear()
        for task in list(pending):
            free = next((g for g in gpus if g not in active and idle(g)), None)
            if free is None:
                break
            d, phase, shard = task
            active[free] = command(root, d, phase, shard, free, vocab_dir)
            pending.remove(task)
        save(root / "resume_status.json", {"status": "running" if active or pending else "failed" if failures else "complete",
            "active": list(active.values()), "pending": pending, "failures": failures,
            "elapsed_seconds": time.time() - started})
        if active or pending:
            time.sleep(poll_seconds)
    save(root / "suite_results.json", {"status": "failed" if failures else "complete",
        "datasets": list(DATASETS),
        "completed": sorted(d for d in DATASETS if (root / "evaluation" / d / "merged.json").exists()),
        "failures": failures, "elapsed_seconds": time.time() - started,
        "repaired_from_selection_cache": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--vocabulary-dir", required=True)
    parser.add_argument("--first-gpu", type=int, default=4)
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--resume", action="store_true",
                        help="freeze existing selection shards and continue missing evaluations")
    args = parser.parse_args()
    if args.resume:
        resume(root_path(args.root), root_path(args.vocabulary_dir), args.first_gpu,
               args.gpu_count, args.poll_seconds)
    else:
        main(args)
