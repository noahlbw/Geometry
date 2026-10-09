#!/usr/bin/env python3
"""Launch the fixed readout screen and corrected-input repeat on idle GPUs."""
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


def launch(candidate):
    jobs = []
    repaired = TOOL/"results/geometry_semantic_innovation_v3_inputcorrected_full_vaihingen_20261001"
    data = Path("/data/test/datasets")
    if candidate == "frozen-path":
        for index in range(4):
            jobs.append((index, f"gsi03_repaired_vaihingen_s{index}", repaired/f"s{index}",
                         "eval_geometry_semantic_innovation.py", "vaihingen",
                         data/"Vaihingen/preprocessed_corrected_20261001", "grounded_vaihingen20.json", 4, index, None))
    settings = [("udd5", "full", data/"UDD5/extracted/UDD/UDD5", "hero_udd5_vip20.json"),
                ("oem", "full", data/"OpenEarthMap_wo_xBD", "hero_oem_vip20.json"),
                ("vdd", "diagnostic", data/"VDD Release Version/VDD", "grounded_vdd_official20.json"),
                ("potsdam", "diagnostic", data/"Potsdam/preprocessed_RGB", "grounded_potsdam20.json")]
    families = {"frozen-path": ("frozen_semantic_path_v2", "fsp01", "eval_frozen_semantic_path.py"),
                "region-cls": ("geometry_region_cls_v1", "grc01", "eval_geometry_region_cls.py"),
                "attention-evidence": ("geometry_attention_evidence_v1", "gae01", "eval_geometry_attention_evidence.py")}
    family, session_prefix, evaluator = families[candidate]
    shards = 1
    first_gpu = 0 if candidate == "region-cls" else 4
    for dataset_index, (dataset, mode, root, vocab) in enumerate(settings):
        diagnostic = TOOL/f"results/geometry_readout_diagnostic_8_20261001_r2/{dataset}/results.json" if mode == "diagnostic" else None
        for index in range(shards):
            gpu = first_gpu+dataset_index*shards+index
            output = TOOL/f"results/{family}_{mode}_{dataset}_20261001/s{index}"
            jobs.append((gpu, f"{session_prefix}_{mode}_{dataset}_s{index}", output, evaluator,
                         dataset, root, vocab, shards, index, diagnostic))
    for gpu, session, output, *_ in jobs:
        if session.startswith("gsi03_repaired_") and output.exists():
            continue
        used, utilization = map(int, subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True).strip().split(","))
        processes = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if processes or used > 1024 or utilization > 10:
            raise RuntimeError(f"GPU {gpu} occupied; no jobs launched.")
        if output.exists() or subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
            raise RuntimeError(f"Existing output/session: {output}, {session}")
    for gpu, session, output, evaluator, dataset, data_root, vocab, shards, index, diagnostic in jobs:
        if session.startswith("gsi03_repaired_") and output.exists():
            print(f"Joining existing {session}; no relaunch or overwrite.", flush=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [PYTHON, f"scripts/{evaluator}", "--dataset", dataset,
                   "--dinov3-repo", str(TOOL/"dinov3_hub"), "--checkpoint-dir", str(BASE/"ckpt/DINO"),
                   "--upstream-root", str(BASE/"third_party/VIP_official_5bd25ee"),
                   "--data-root", str(data_root), "--vocabulary-config", str(TOOL/"configs"/vocab),
                   "--output-dir", str(output), "--num-shards", str(shards), "--shard-index", str(index)]
        if diagnostic:
            command.extend(["--source-diagnostic", str(diagnostic)])
        env = f"CUDA_VISIBLE_DEVICES={gpu} PYTHONPATH={TOOL}:{TOOL}/scripts:{BASE}/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1"
        shell = f"cd {shlex.quote(str(TOOL))} && exec env {env} nice -n 10 {shlex.join(command)} > {shlex.quote(str(output.parent/(output.name+'.log')))} 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
        print(f"Started {session} GPU {gpu}", flush=True)
    return jobs


def main(candidate):
    jobs = launch(candidate)
    pending = {job[1]: job[2] for job in jobs}
    while pending:
        for session, output in list(pending.items()):
            row = None
            path = output/"results.json"
            if path.exists():
                try:
                    row = json.loads(path.read_text())
                except json.JSONDecodeError:
                    pass
            if row is not None and row.get("status") == "complete" and row["processed_images"] == row["total_images"]:
                del pending[session]
            elif subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode:
                log = output.parent/(output.name+".log")
                raise RuntimeError(f"Exited without complete result: {session}\n{log.read_text()[-6000:]}")
        if pending:
            time.sleep(30)
    roots = list(dict.fromkeys(job[2].parent for job in jobs))
    passed = True
    for root in roots:
        inputs = sorted(root.glob("s[0-9]"))
        if not (root/"merged.json").exists():
            merge(argparse.Namespace(inputs=list(map(str, inputs)), output=str(root/"merged.json")))
        result = json.loads((root/"merged.json").read_text())
        dataset = result["signature"]["dataset"]
        expected = {"vaihingen": 113, "udd5": 40, "oem": 384, "vdd": 8, "potsdam": 8}[dataset]
        if result["processed_images"] != expected:
            raise ValueError("Wrong fixed coverage.")
        if dataset == "vaihingen":
            reference = TOOL/"results/gear_vaihingen_input_corrected_20261001/merged.json"
        elif dataset in ("vdd", "potsdam"):
            reference = TOOL/f"results/matched_counterfactual_v4_diagnostic_{dataset}_20261001/merged.json"
        else:
            reference = TOOL/f"results/geometry_semantic_innovation_v3_full_{dataset}_20261001/merged.json"
        baseline = json.loads(reference.read_text())
        if result["signature"]["sample_keys"] != baseline["signature"]["sample_keys"]:
            raise ValueError("Baseline sample sequence differs.")
        if result["signature"]["vocabulary"]["sha256"] != baseline["signature"]["vocabulary"]["sha256"]:
            raise ValueError("Baseline vocabulary differs.")
        if not np.array_equal(result["metrics"][dataset]["Geometry"]["confusion_matrix"],
                              baseline["metrics"][dataset]["Geometry"]["confusion_matrix"]):
            raise ValueError("Original Geometry control changed.")
        if dataset != "vaihingen":
            metrics = result["metrics"][dataset]
            primary = {"region-cls": "AnchoredRegion", "attention-evidence": "AnchoredEvidence",
                       "frozen-path": "FrozenPathGeometry"}[candidate]
            passed &= metrics[primary]["mean_iou_percent"] >= metrics["Geometry"]["mean_iou_percent"]
        print(f"Verified {dataset}: {expected}, original Geometry unchanged.", flush=True)
    print(f"{candidate} all-four nondegradation gate: {passed}. No automatic per-domain model switching.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("frozen-path", "region-cls", "attention-evidence"), default="frozen-path")
    main(parser.parse_args().candidate)
