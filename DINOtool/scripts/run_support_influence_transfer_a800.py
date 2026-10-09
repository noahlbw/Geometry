#!/usr/bin/env python3
"""Run four fixed full-image transfer trials only on idle physical GPUs4-7."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from merge_geometry_semantic_innovation import main as merge


BASE = Path("/data/test/code/ovss_cafe_ped_v2_20260919")
TOOL = BASE/"DINOtool_hero_external_20260926"
PYTHON = "/data/miniconda3/envs/pfu/bin/python"
IMPLEMENTATION = "geometry-support-erasure-semantic-readout-v3-score-influence-20261001"
SETTINGS = (
    ("udd5", "UDD5/extracted/UDD/UDD5", "hero_udd5_vip20.json", 40),
    ("oem", "OpenEarthMap_wo_xBD", "hero_oem_vip20.json", 384),
    ("vdd", "VDD Release Version/VDD", "grounded_vdd_official20.json", 80),
    ("potsdam", "Potsdam/preprocessed_RGB", "grounded_potsdam20.json", 504),
)


def root_for(dataset):
    return TOOL/f"results/geometry_support_influence_v3_full_{dataset}_20261001"


def gpu_idle(gpu):
    used, utilization = map(int, subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True).strip().split(","))
    processes = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid",
                                        "--format=csv,noheader"], text=True).strip()
    return not processes and used <= 1024 and utilization <= 10


def start(setting, gpu):
    dataset, data, vocab, _ = setting
    root = root_for(dataset)
    session = f"gsi03_influence_full_{dataset}_s0"
    if not gpu_idle(gpu):
        return None
    if root.exists() or subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise RuntimeError(f"Existing output/session; refusing overwrite: {root}, {session}")
    root.mkdir(parents=True)
    command = [PYTHON, "scripts/eval_geometry_support_influence.py", "--dataset", dataset,
               "--dinov3-repo", str(TOOL/"dinov3_hub"), "--checkpoint-dir", str(BASE/"ckpt/DINO"),
               "--upstream-root", str(BASE/"third_party/VIP_official_5bd25ee"),
               "--data-root", str(Path("/data/test/datasets")/data),
               "--vocabulary-config", str(TOOL/"configs"/vocab),
               "--output-dir", str(root/"s0"), "--num-shards", "1", "--shard-index", "0"]
    env = (f"CUDA_VISIBLE_DEVICES={gpu} PYTHONPATH={TOOL}:{TOOL}/scripts:"
           f"{BASE}/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c "
           "OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1")
    shell = f"cd {shlex.quote(str(TOOL))} && exec env {env} nice -n 10 {shlex.join(command)} > {shlex.quote(str(root/'s0.log'))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    print(f"Started {session} physical GPU{gpu}", flush=True)
    return session


def verify(setting):
    dataset, _, _, total = setting
    root = root_for(dataset)
    merge(argparse.Namespace(inputs=[str(root/"s0")], output=str(root/"merged.json")))
    result = json.loads((root/"merged.json").read_text())
    baseline = json.loads((TOOL/f"results/geometry_semantic_innovation_v3_full_{dataset}_20261001/merged.json").read_text())
    if (not result["coverage_verified"] or result["processed_images"] != total
            or result["total_images"] != total or len(set(result["signature"]["sample_keys"])) != total
            or result["signature"]["implementation"] != IMPLEMENTATION):
        raise ValueError(f"Wrong full coverage/signature: {dataset}")
    for field in ("sample_keys", "global_sample_keys_sha256", "vocabulary", "checkpoints"):
        if result["signature"][field] != baseline["signature"][field]:
            raise ValueError(f"Matched baseline differs in {field}: {dataset}")
    if any(count != 20 for group in result["signature"]["vocabulary"]["counts"].values() for count in group):
        raise ValueError("Expected exactly20 aliases per class")
    for key, methods in result["metrics"].items():
        if not np.array_equal(methods["Geometry"]["confusion_matrix"], baseline["metrics"][key]["Geometry"]["confusion_matrix"]):
            raise ValueError(f"Original Geometry changed: {dataset}/{key}")
    print(f"Verified {dataset}: {total}, exact baseline Geometry/coverage/vocabulary/checkpoints.", flush=True)
    return result


def main():
    for dataset, *_ in SETTINGS:
        if root_for(dataset).exists():
            raise RuntimeError(f"Existing output: {root_for(dataset)}")
        reference = TOOL/f"results/geometry_semantic_innovation_v3_full_{dataset}_20261001/merged.json"
        if not reference.is_file():
            raise RuntimeError(f"Missing matched baseline: {reference}")
    pending, active, outcomes, failures = list(SETTINGS), {}, {}, {}
    while pending or active:
        for session, (setting, gpu) in list(active.items()):
            root = root_for(setting[0])
            path = root/"s0/results.json"
            row = None
            if path.exists():
                try:
                    row = json.loads(path.read_text())
                except json.JSONDecodeError:
                    pass
            if row is not None and row.get("status") == "complete" and row["processed_images"] == row["total_images"]:
                outcomes[setting[0]] = verify(setting)
                del active[session]
            elif subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode:
                failures[setting[0]] = (root/"s0.log").read_text()[-6000:]
                print(f"FAILED {session}\n{failures[setting[0]]}", flush=True)
                del active[session]
        for gpu in range(4, 8):
            if pending and not any(assigned == gpu for _, assigned in active.values()) and gpu_idle(gpu):
                session = start(pending[0], gpu)
                if session is not None:
                    active[session] = (pending.pop(0), gpu)
        if pending or active:
            time.sleep(30)
    passed = not failures and all(
        group["ScoreInfluence"]["mean_iou_percent"] >= group["Geometry"]["mean_iou_percent"]
        and row["transitions"][key]["ScoreInfluence"]["beneficial"] >= row["transitions"][key]["ScoreInfluence"]["harmful"]
        for row in outcomes.values() for key, group in row["metrics"].items())
    print(f"Full transfer nondegradation/net-correction gate: {passed}. No automatic eight-domain launch or domain switching.", flush=True)
    if failures:
        raise RuntimeError("Some trials failed; original logs and outputs preserved.")


if __name__ == "__main__":
    main()
