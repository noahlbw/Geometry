"""Launch a predeclared eight-domain regional-readout screen on idle GPUs0-7."""
import argparse
import json
from pathlib import Path
import random
import shlex
import subprocess
import time

import numpy as np

from merge_geometry_semantic_innovation import main as merge


BASE = Path("/data/test/code/ovss_cafe_ped_v2_20260919")
TOOL = BASE/"DINOtool_hero_external_20260926"
PYTHON = "/data/miniconda3/envs/pfu/bin/python"
SOURCE = BASE/"ckpt/SigLIP2_base256_region_20261001"
THIRD = BASE/"third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c"
SETTINGS = (
    ("vdd", "VDD Release Version/VDD", "grounded_vdd_official20.json", 8, 80),
    ("potsdam", "Potsdam/preprocessed_RGB", "grounded_potsdam20.json", 8, 504),
    ("udd5", "UDD5/extracted/UDD/UDD5", "hero_udd5_vip20.json", 40, 40),
    ("oem", "OpenEarthMap_wo_xBD", "hero_oem_vip20.json", 32, 384),
    ("loveda", "/data/test/cafe-efa/data/processed/loveda/val", "gar_llm_raw20_loveda_v1.json", 16, 1669),
    ("vaihingen", "Vaihingen/preprocessed_corrected_20261001", "grounded_vaihingen20.json", 16, 113),
    ("landcoverai", "LandCoverAI_v1_target_20260922/preprocessed_official512_gear29", "gear_landcoverai_v1_20.json", 32, 1602),
    ("flair1", "FLAIR1_target_20260922", "gear_flair1_main12_20.json", 64, 15700),
)


def reference(dataset):
    if dataset == "vaihingen":
        return TOOL/"results/geometry_semantic_innovation_v3_inputcorrected_full_vaihingen_20261001/merged.json"
    return TOOL/f"results/geometry_semantic_innovation_v3_full_{dataset}_20261001/merged.json"


def idle(gpu):
    status = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits"], text=True)
    used, utilization = map(int, status.strip().split(","))
    processes = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid",
                                        "--format=csv,noheader"], text=True).strip()
    return not processes and used <= 1024 and utilization <= 10


def launch(root, setting, gpu, phase):
    dataset, data, vocabulary, _, _ = setting
    output, log = root/dataset/"s0", root/f"{dataset}.log"
    session = f"grsr02_{phase}_{dataset}"
    if output.exists() or subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise RuntimeError(f"Existing run/output: {session}")
    if not idle(gpu):
        raise RuntimeError(f"GPU{gpu} occupied; no launch")
    command = [PYTHON, "-u", "scripts/eval_region_semantic_readout.py", "--dataset", dataset,
               "--dinov3-repo", str(TOOL/"dinov3_hub"), "--checkpoint-dir", str(BASE/"ckpt/DINO"),
               "--data-root", str(Path("/data/test/datasets")/data),
               "--vocabulary-config", str(TOOL/"configs"/vocabulary), "--semantic-source", str(SOURCE),
               "--reference-full", str(reference(dataset)), "--output-dir", str(output)]
    if phase == "screen":
        command += ["--sample-manifest", str(root/f"{dataset}_samples.json")]
    env = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
           "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "TOKENIZERS_PARALLELISM=false"]
    shell = f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(env)} nice -n 10 {shlex.join(command)} > {shlex.quote(str(log))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    print(f"Started {session}: physical GPU{gpu}", flush=True)
    return session


def initialize(root, phase):
    if root.exists():
        raise RuntimeError(f"Refusing existing suite root: {root}")
    if not (SOURCE/"region_source_manifest.json").is_file():
        raise RuntimeError("Frozen semantic checkpoint is not ready.")
    for gpu in range(8):
        if not idle(gpu):
            raise RuntimeError(f"GPU{gpu} occupied; suite not launched.")
    for dataset, *_ in SETTINGS:
        if not reference(dataset).is_file():
            raise RuntimeError(f"Missing verified reference: {dataset}")
    root.mkdir(parents=True)
    protocol = {"phase": phase, "seed": 20261001,
        "primary": "RegionJoint", "fixed_rule": True,
        "gate": "Every screened protocol must retain Geometry mIoU and positive net corrections; "
                "RegionJoint must beat RegionFusion on both VDD and Potsdam. No dataset winner switching.",
        "note": "Predeclared exploratory development screen. UDD5 is full40; all other screens are partial. "
                "Passing small screens does not establish SOTA or independent generalization.",
        "datasets": []}
    for dataset, _, _, count, total in SETTINGS:
        full = json.loads(reference(dataset).read_text())
        keys = full["signature"]["sample_keys"]
        if len(keys) != total or len(set(keys)) != total or not full["coverage_verified"]:
            raise RuntimeError(f"Wrong reference coverage: {dataset}")
        if dataset in ("vdd", "potsdam"):
            diagnostic = TOOL/f"results/geometry_readout_diagnostic_8_20261001_r2/{dataset}/results.json"
            keys = json.loads(diagnostic.read_text())["signature"]["samples"]
        else:
            chosen = set(random.Random(20261001).sample(keys, count))
            keys = [key for key in keys if key in chosen]
        if len(keys) != count:
            raise RuntimeError(f"Wrong screen count: {dataset}")
        (root/f"{dataset}_samples.json").write_text(json.dumps({"sample_keys": keys, "seed": 20261001}, indent=2)+"\n")
        protocol["datasets"].append({"dataset": dataset, "screen_images": count, "full_images": total})
    (root/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")


def verify(root, setting, phase):
    dataset, _, _, count, total = setting
    path = root/dataset/"merged.json"
    merge(argparse.Namespace(inputs=[str(root/dataset/"s0")], output=str(path)))
    result = json.loads(path.read_text())
    expected = count if phase == "screen" else total
    if (not result["coverage_verified"] or result["processed_images"] != expected
            or result["total_images"] != expected or len(set(result["signature"]["sample_keys"])) != expected):
        raise RuntimeError(f"Incomplete unique coverage: {dataset}")
    full = json.loads(reference(dataset).read_text())
    if result["signature"]["vocabulary"] != full["signature"]["vocabulary"]:
        raise RuntimeError(f"Vocabulary differs: {dataset}")
    if result["signature"]["checkpoints"] != full["signature"]["checkpoints"]:
        raise RuntimeError(f"Original Geometry checkpoints differ: {dataset}")
    previous = None
    if expected == total:
        previous = full["metrics"]
    elif dataset in ("vdd", "potsdam"):
        diagnostic = json.loads((TOOL/f"results/geometry_readout_diagnostic_8_20261001_r2/{dataset}/results.json").read_text())
        previous = {dataset: {"Geometry": diagnostic["stage_probe_metrics"]["Geometry_original"]}}
    if previous:
        for key in result["metrics"]:
            if not np.array_equal(result["metrics"][key]["Geometry"]["confusion_matrix"],
                                  previous[key]["Geometry"]["confusion_matrix"]):
                raise RuntimeError(f"Original Geometry baseline changed: {dataset}/{key}")
    shard = json.loads((root/dataset/"s0/results.json").read_text())
    result["support_observation_audit"] = shard["support_observation_audit"]
    result["trained_pool_replay_max_error"] = shard["trained_pool_replay_max_error"]
    result["exact_reference_geometry_confusion"] = previous is not None
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(f"Verified {dataset}: {expected} unique images; original Geometry checked={previous is not None}", flush=True)
    return result


def main(args):
    root = args.root
    initialize(root, args.phase)
    active = {launch(root, setting, gpu, args.phase): setting for gpu, setting in enumerate(SETTINGS)}
    outcomes, failures = {}, {}
    while active:
        for session, setting in list(active.items()):
            dataset = setting[0]
            source = root/dataset/"s0/results.json"
            row = json.loads(source.read_text()) if source.exists() else None
            alive = subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0
            if row and row["status"] == "complete":
                try:
                    outcomes[dataset] = verify(root, setting, args.phase)
                except Exception as error:
                    failures[dataset] = str(error)
                    print(f"VERIFY FAILURE {dataset}: {error}", flush=True)
                del active[session]
            elif not alive:
                text = (root/f"{dataset}.log").read_text()
                failures[dataset] = text[-6000:]
                print(f"RUN FAILURE {dataset}: {failures[dataset]}", flush=True)
                del active[session]
        if active:
            time.sleep(20)
    passed, comparisons = not failures, {}
    for dataset, result in outcomes.items():
        comparisons[dataset] = {}
        for key, metrics in result["metrics"].items():
            delta = metrics["RegionJoint"]["mean_iou_percent"]-metrics["Geometry"]["mean_iou_percent"]
            fusion_delta = metrics["RegionJoint"]["mean_iou_percent"]-metrics["RegionFusion"]["mean_iou_percent"]
            transition = result["transitions"][key]["RegionJoint"]
            net = transition["beneficial"]-transition["harmful"]
            comparisons[dataset][key] = {"delta_vs_Geometry": delta, "delta_vs_RegionFusion": fusion_delta,
                                        "net_corrected_pixels": net}
            passed = passed and delta >= 0 and net > 0
            if dataset in ("vdd", "potsdam"):
                passed = passed and fusion_delta > 0
    result = {"status": "complete" if not failures else "failed", "gate_passed": bool(passed),
              "comparisons": comparisons, "failures": failures,
              "datasets_complete": list(outcomes), "automatic_full_launch": False}
    (root/"suite_results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    print("Full-suite promotion requires the predeclared cross-domain gate; no per-domain tuning.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "full"), default="screen")
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args())
