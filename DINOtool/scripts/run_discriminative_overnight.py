#!/usr/bin/env python3
"""Run the fixed eight-dataset comparison sequentially on A800 GPUs 0-3."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from dinotool.discriminative_ownership import IMPLEMENTATION, METHODS, DiscriminativeOwnershipConfig
from dataclasses import asdict
from dinotool.prompts import load_class_specs
from eval_gear_ov import protocol, digest
from merge_gear_ov_shards import merge


BASE = Path("/data/test/code/ovss_cafe_ped_v2_20260919")
ORIGINAL_TOOL = BASE / "DINOtool_hero_external_20260926"
DATASETS = {
    "udd5": ("/data/test/datasets/UDD5/extracted/UDD/UDD5", "hero_udd5_vip20.json", 40),
    "oem": ("/data/test/datasets/OpenEarthMap_wo_xBD", "hero_oem_vip20.json", 384),
    "vdd": ("/data/test/datasets/VDD Release Version/VDD", "grounded_vdd_official20.json", 80),
    "potsdam": ("/data/test/datasets/Potsdam/preprocessed_RGB", "grounded_potsdam20.json", 504),
    "vaihingen": ("/data/test/datasets/Vaihingen/preprocessed_complete", "grounded_vaihingen20.json", 113),
    "landcoverai": ("/data/test/datasets/LandCoverAI_v1_target_20260922/preprocessed_official512_gear29",
                   "gear_landcoverai_v1_20.json", 1602),
    "loveda": ("/data/test/cafe-efa/data/processed/loveda/val", "gar_llm_raw20_loveda_v1.json", 1669),
    "flair1": ("/data/test/datasets/FLAIR1_target_20260922", "gear_flair1_main12_20.json", 15700),
}


def save(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def preflight(code: Path, baselines: Path) -> dict:
    checks = {}
    for dataset, (data, vocab, expected) in DATASETS.items():
        source = json.loads((baselines / f"gear_ov_v2_{dataset}_full_20260929_results.json").read_text())
        vocab_path = code / "configs" / vocab
        specs = load_class_specs(vocab_path)
        args = SimpleNamespace(dataset=dataset, data_root=data, sample_seed=20260923,
                               vdd_ontology="official")
        samples, protocols, load_image, load_mask = protocol(args, specs)
        keys = [sample.key for sample in samples]
        signature = source["signature"]
        if len(keys) != expected or len(set(keys)) != expected or digest(keys) != signature["global_sample_keys_sha256"]:
            raise ValueError(f"{dataset}: sample count/order differs from the locked full baseline")
        if hashlib.sha256(vocab_path.read_bytes()).hexdigest() != signature["vocabulary"]["sha256"]:
            raise ValueError(f"{dataset}: vocabulary differs from the locked full baseline")
        if any(len(item.synonyms) != 20 for group in protocols.values() for item in group):
            raise ValueError(f"{dataset}: expected exactly 20 candidates per class")
        image = load_image(samples[0].image_path) if dataset == "loveda" else load_image(samples[0])
        if image.ndim != 3 or image.shape[0] != 3 or not bool(image.isfinite().all()) \
                or float(image.min()) < 0 or float(image.max()) > 1:
            raise ValueError(f"{dataset}: invalid input tensor")
        for key in protocols:
            load_mask(samples[0], key, tuple(image.shape[-2:]))
        checks[dataset] = {"images": expected, "sample_keys_sha256": digest(keys),
                           "vocabulary_sha256": signature["vocabulary"]["sha256"],
                           "first_image_shape": list(image.shape), "verified": True}
    return checks


def gpu_idle(gpu: int) -> bool:
    query = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True).strip()
    memory, utilization = [int(value.strip()) for value in query.split(",")]
    processes = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid",
        "--format=csv,noheader"], text=True).strip()
    return not processes and memory <= 1024 and utilization <= 10


def write_report(root: Path, status: dict):
    lines = ["# Frozen discriminative ownership: eight datasets", "",
             f"Status: {status['status']}. GPUs: 0, 1, 2, 3 only.", "",
             "Same weights, 20-alias vocabularies and inference rule across all datasets.",
             "This is exploratory validation; no per-dataset parameter search or winner selection.", "",
             "| Dataset/protocol | Images | Geometry | Multiscale | COR_Fixed | COR_Discriminative | Delta vs Multiscale |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for dataset, item in status["datasets"].items():
        path = root / dataset / "merged.json"
        if item["status"] != "complete" or not path.exists():
            lines.append(f"| {dataset} ({item['status']}) | {DATASETS[dataset][2]} | — | — | — | — | — |")
            continue
        result = json.loads(path.read_text())
        for key, metrics in result["metrics"].items():
            values = [metrics[method]["mean_iou_percent"] for method in METHODS]
            lines.append(f"| {dataset}/{key} | {result['processed_images']} | "
                         + " | ".join(f"{value:.4f}" for value in values)
                         + f" | {values[-1] - values[1]:+.4f} |")
    lines += ["", "Vaihingen keeps the historical first-three-band input; this is not native RGB.",
              "OEM has 384 available validation images. Potsdam uses 504 prepared tiles.",
              "LandCover.ai substitutes for unlabeled iSAID. LoveDA P and D are both reported.",
              "Contrast regions are fixed spatial blocks. The dispersion margin is not a calibrated confidence interval."]
    if all(item["status"] == "complete" for item in status["datasets"].values()):
        deltas = []
        for dataset in DATASETS:
            result = json.loads((root / dataset / "merged.json").read_text())
            key = "D" if dataset == "loveda" else dataset
            metrics = result["metrics"][key]
            deltas.append(metrics["COR_Discriminative"]["mean_iou_percent"]
                          - metrics["Multiscale"]["mean_iou_percent"])
        lines += ["", "Eight-dataset aggregate (equal weights; LoveDA D is its main protocol):",
                  f"Mean delta {sum(deltas)/8:+.4f}; positive {sum(d > 0 for d in deltas)}/8; "
                  f"negative {sum(d < 0 for d in deltas)}/8; worst delta {min(deltas):+.4f}.",
                  "These are observed differences, not statistical significance claims."]
    for dataset, item in status["datasets"].items():
        if item["status"] == "complete":
            result = json.loads((root / dataset / "merged.json").read_text())
            lines += ["", f"## {dataset}", "",
                      f"Wall {result['parallel_wall_seconds']:.1f} s; summed shard runtime "
                      f"{result['aggregate_gpu_seconds']:.1f} s; peak {result['peak_cuda_memory_mb']:.1f} MiB."]
            for key, metrics in result["metrics"].items():
                lines += ["", f"Protocol {key}", "",
                          "| Class | Geometry | Multiscale | COR_Fixed | COR_Discriminative | Delta |",
                          "|---|---:|---:|---:|---:|---:|"]
                names = result["signature"]["classes"][key]
                for index, name in enumerate(names):
                    values = [metrics[m]["per_class"][index]["iou_percent"] for m in METHODS]
                    printed = ["n/a" if v is None else f"{v:.4f}" for v in values]
                    delta = "n/a" if values[-1] is None or values[1] is None else f"{values[-1]-values[1]:+.4f}"
                    lines.append(f"| {name} | " + " | ".join(printed) + f" | {delta} |")
                for method in METHODS:
                    for field in ("foreground_mean_iou_percent", "non_residual_mean_iou_percent"):
                        if field in metrics[method]:
                            lines.append(f"{method} {field}: {metrics[method][field]}")
            lines += ["", "Diagnostics (tile means):", "```json",
                      json.dumps(result["diagnostics"], indent=2), "```"]
        elif item["status"] == "failed":
            lines += ["", f"{dataset} failed: {item.get('error', '')}"]
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = args.run_root.resolve()
    code = Path(__file__).resolve().parents[1]
    root.mkdir(parents=True, exist_ok=True)
    checks = preflight(code, root / "baselines")
    save(root / "preflight.json", checks)
    if args.preflight_only:
        print(json.dumps(checks, indent=2), flush=True)
        return
    if (root / "queue_status.json").exists():
        raise ValueError("Existing queue state: do not overwrite or restart this run")
    status = {"status": "running", "implementation": IMPLEMENTATION,
              "configuration": asdict(DiscriminativeOwnershipConfig()), "gpus": [0, 1, 2, 3],
              "started": time.time(), "datasets": {name: {"status": "pending"} for name in DATASETS}}
    children = []
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for dataset, (data, vocab, expected) in DATASETS.items():
            item = status["datasets"][dataset]
            try:
                while not all(gpu_idle(gpu) for gpu in range(4)):
                    item["status"] = "waiting_for_idle_gpus"
                    save(root / "queue_status.json", status)
                    write_report(root, status)
                    time.sleep(30)
                run = root / dataset
                run.mkdir(exist_ok=True)
                if any((run / f"s{shard}").exists() for shard in range(4)):
                    raise ValueError("Existing shard output: refusing to overwrite")
                item.update(status="running", started=time.time())
                save(root / "queue_status.json", status)
                write_report(root, status)
                children = []
                for shard in range(4):
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(shard),
                           "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
                    command = [sys.executable, str(code / "scripts/eval_gear_ov.py"),
                               "--dataset", dataset, "--dinov3-repo", str(ORIGINAL_TOOL / "dinov3_hub"),
                               "--checkpoint-dir", str(BASE / "ckpt/DINO"), "--data-root", data,
                               "--vocabulary-config", str(code / "configs" / vocab),
                               "--output-dir", str(run / f"s{shard}"), "--num-shards", "4",
                               "--shard-index", str(shard), "--discriminative-ownership", "--progress-every", "2"]
                    with (run / f"s{shard}.log").open("x") as log:
                        children.append(subprocess.Popen(command, env=env, cwd=code,
                                                         stdout=log, stderr=subprocess.STDOUT,
                                                         start_new_session=True))
                codes = [child.wait() for child in children]
                children = []
                if any(code != 0 for code in codes):
                    raise RuntimeError(f"Shard exit codes {codes}; inspect {run}/sN.log")
                result = merge([str(run / f"s{n}") for n in range(4)], str(run / "merged.json"))
                signature = result["signature"]
                if (result["processed_images"] != expected or not result["coverage_verified"]
                        or signature["implementation"] != IMPLEMENTATION
                        or signature["global_sample_keys_sha256"] != checks[dataset]["sample_keys_sha256"]
                        or signature["vocabulary"]["sha256"] != checks[dataset]["vocabulary_sha256"]
                        or signature["competitive"] != asdict(DiscriminativeOwnershipConfig())
                        or any(count != 20 for counts in signature["vocabulary"]["alias_counts"].values()
                               for count in counts)):
                    raise ValueError("Merged output violates the frozen protocol")
                item.update(status="complete", completed=time.time(), images=expected)
            except Exception as error:
                for child in children:
                    if child.poll() is None:
                        child.terminate()
                for child in children:
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                children = []
                item.update(status="failed", error=str(error))
                print(f"FAILED {dataset}: {error}", flush=True)
            save(root / "queue_status.json", status)
            write_report(root, status)
        status["status"] = ("complete" if all(x["status"] == "complete"
                                             for x in status["datasets"].values()) else "complete_with_failures")
        status["finished"] = time.time()
    except KeyboardInterrupt:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for item in status["datasets"].values():
            if item["status"] in ("running", "waiting_for_idle_gpus"):
                item["status"] = "interrupted"
        status["status"] = "interrupted"
        raise
    finally:
        save(root / "queue_status.json", status)
        write_report(root, status)


if __name__ == "__main__":
    main()
