"""Launch once or verify the existing fixed96 screen; no full-suite scheduler."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.geometry_semantic_budget import IMPLEMENTATION, METHODS, PRIMARY, BudgetConfig
from run_region_semantic_suite_a800 import BASE, TOOL, PYTHON, THIRD, SETTINGS, idle
from run_sat_geometry_transport_suite import reference, exact_miou, read_json, alive
from run_geometry_region_strong_suite import verify_dataset


SCREEN = TOOL/"results/sat_geometry_transport_screen_20261002"


def launch(root):
    root.resolve().relative_to((TOOL/"results").resolve())
    smoke = read_json(root/"smoke/results.json")
    if (not smoke or smoke["status"] != "complete" or smoke["implementation"] != IMPLEMENTATION
            or smoke["target_masks_loaded"] or not smoke["weights_frozen"] or not smoke["head_weights_unchanged"]):
        raise RuntimeError("Complete frozen mask-free smoke required.")
    if (root/"protocol.json").exists():
        raise RuntimeError("Refusing a previously launched screen.")
    if any(not idle(gpu) for gpu in range(8)):
        raise RuntimeError("All eight GPUs must be idle for this one-shot screen.")
    manifests = {}
    for dataset, *_ in SETTINGS:
        manifest = json.loads((SCREEN/f"{dataset}_samples.json").read_text())
        keys = manifest["signature"]["samples"]
        old = read_json(reference(dataset))
        if (len(keys) != (40 if dataset == "udd5" else 8) or len(keys) != len(set(keys))
                or not old["coverage_verified"] or not set(keys).issubset(old["signature"]["sample_keys"])):
            raise RuntimeError("Invalid existing fixed96 manifest: "+dataset)
        manifests[dataset] = manifest
    protocol = {"implementation": IMPLEMENTATION, "config": BudgetConfig().signature(),
        "methods": METHODS, "primary": PRIMARY, "physical_gpus": list(range(8)),
        "datasets": {key: len(row["signature"]["samples"]) for key, row in manifests.items()},
        "gate": "mean>Geometry/CSA_SameShell/Proxy_SameShell, positive VDD/Potsdam deltas, all protocol losses<=1pp",
        "note": "Fixed96 complete-image development screen only; no automatic full rollout."}
    (root/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    for gpu, setting in enumerate(SETTINGS):
        dataset, data, vocab, *_ = setting
        (root/f"{dataset}_samples.json").write_text(json.dumps(manifests[dataset], indent=2)+"\n")
        output, log, session = root/dataset/"s0", root/f"{dataset}.log", f"gsb04_screen_{dataset}"
        if output.exists() or alive(session):
            raise RuntimeError("Refusing existing worker/output: "+dataset)
        args = [PYTHON, "-u", "scripts/eval_geometry_semantic_budget.py", "--dataset", dataset,
            "--dinov3-repo", str(TOOL/"dinov3_hub"), "--checkpoint-dir", str(BASE/"ckpt/DINO"),
            "--data-root", str(Path('/data/test/datasets')/data), "--vocabulary-config", str(TOOL/"configs"/vocab),
            "--upstream-root", str(BASE/"third_party/VIP_official_5bd25ee"), "--output-dir", str(output),
            "--source-diagnostic", str(root/f"{dataset}_samples.json"), "--num-shards", "4", "--shard-index", "0"]
        env = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
               "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "TOKENIZERS_PARALLELISM=false"]
        shell = f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(env+args)} > {shlex.quote(str(log))} 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
        print(f"Started {session} on GPU{gpu}", flush=True)


def complete_shards(root):
    """Complete the launcher's existing four-way layout without replaying s0."""
    protocol = read_json(root/"protocol.json")
    if not protocol or protocol["config"] != BudgetConfig().signature():
        raise RuntimeError("Frozen protocol mismatch.")
    queue = []
    for setting in SETTINGS:
        dataset = setting[0]
        row = read_json(root/dataset/"s0/results.json")
        if not row or row["signature"]["num_shards"] != 4:
            raise RuntimeError("Expected existing four-way s0: "+dataset)
        for shard in (1, 2, 3):
            if (root/dataset/f"s{shard}").exists() or alive(f"gsb04_screen_{dataset}_s{shard}"):
                raise RuntimeError("Refusing existing queued shard.")
            queue.append((setting, shard))
    plan = root/"shard_completion_plan.json"
    if plan.exists():
        raise RuntimeError("Completion plan already exists.")
    plan.write_text(json.dumps({"num_shards": 4, "queued": [(s[0], n) for s, n in queue],
        "note": "Preserve existing s0, complete only never-launched shards1..3; no model changes."}, indent=2)+"\n")
    active = {}
    failures = {}
    while queue or active or any(alive(f"gsb04_screen_{setting[0]}") for setting in SETTINGS):
        for gpu, job in list(active.items()):
            dataset, shard, session = job
            if alive(session):
                continue
            row = read_json(root/dataset/f"s{shard}"/"results.json")
            if not row or row["status"] != "complete" or row["processed_images"] != row["total_images"]:
                failures[session] = (root/dataset/f"s{shard}.log").read_text()[-4000:]
            del active[gpu]
        if failures:
            queue.clear()
        for gpu in range(8):
            if not queue or gpu in active or not idle(gpu):
                continue
            setting, shard = queue.pop(0)
            dataset, data, vocab, *_ = setting
            output, log, session = root/dataset/f"s{shard}", root/dataset/f"s{shard}.log", f"gsb04_screen_{dataset}_s{shard}"
            if output.exists() or alive(session):
                raise RuntimeError("Duplicate output/session: "+session)
            log.parent.mkdir(parents=True, exist_ok=True)
            args = [PYTHON, "-u", "scripts/eval_geometry_semantic_budget.py", "--dataset", dataset,
                "--dinov3-repo", str(TOOL/"dinov3_hub"), "--checkpoint-dir", str(BASE/"ckpt/DINO"),
                "--data-root", str(Path('/data/test/datasets')/data), "--vocabulary-config", str(TOOL/"configs"/vocab),
                "--upstream-root", str(BASE/"third_party/VIP_official_5bd25ee"), "--output-dir", str(output),
                "--source-diagnostic", str(root/f"{dataset}_samples.json"), "--num-shards", "4", "--shard-index", str(shard)]
            env = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
                   "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "TOKENIZERS_PARALLELISM=false"]
            shell = f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(env+args)} > {shlex.quote(str(log))} 2>&1"
            subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
            active[gpu] = (dataset, shard, session)
        state = {"queued": [(s[0], n) for s, n in queue], "active": active, "failures": failures}
        (root/"completion_status.json").write_text(json.dumps(state, indent=2)+"\n")
        if queue or active:
            time.sleep(10)
    if failures:
        raise RuntimeError(json.dumps(failures))
    verify(root)


def verify(root):
    protocol = read_json(root/"protocol.json")
    if not protocol or protocol["config"] != BudgetConfig().signature():
        raise RuntimeError("Frozen protocol mismatch.")
    outcomes = {}
    for dataset, *_ in SETTINGS:
        for shard in range(4):
            output = root/dataset/f"s{shard}"
            row = read_json(output/"results.json")
            session = f"gsb04_screen_{dataset}" if shard == 0 else f"gsb04_screen_{dataset}_s{shard}"
            if (alive(session) or not row or row["status"] != "complete"
                    or row["processed_images"] != row["total_images"]
                    or not (output/"per_image_confusions.npz").exists()):
                raise RuntimeError("Complete worker required: "+session)
        merged = root/dataset/"merged.json"
        if not merged.exists():
            row = verify_dataset(root, dataset, "screen", shard_count=4,
                                 implementation=IMPLEMENTATION, semantic_source=None)
        else:
            row = read_json(merged)
        expected = {"geometry": read_json(reference(dataset))["signature"]["gear"]["geometry"],
                    "primary": PRIMARY, **BudgetConfig().signature()}
        if row["signature"]["gear"] != expected or not row["per_image_coverage_verified"]:
            raise RuntimeError("Readout configuration or coverage mismatch.")
        outcomes[dataset] = {key: {method: exact_miou(metric) for method, metric in group.items()}
                             for key, group in row["metrics"].items()}
    means = {method: float(np.mean([row['D' if dataset == 'loveda' else dataset][method]
                                   for dataset, row in outcomes.items()])) for method in METHODS}
    checks = {"mean_vs_"+method: means[PRIMARY] > means[method]
              for method in ("Geometry", "CSA_SameShell", "Proxy_SameShell")}
    for dataset, groups in outcomes.items():
        for key, group in groups.items():
            checks["retain_"+dataset+"_"+key] = group[PRIMARY]-group["Geometry"] >= -1.
        if dataset in ("vdd", "potsdam"):
            checks["focus_"+dataset] = groups[dataset][PRIMARY] > groups[dataset]["Geometry"]
    result = {"status": "complete", "implementation": IMPLEMENTATION, "images": 96,
              "values": outcomes, "mean_miou": means, "checks": checks, "passed": all(checks.values())}
    (root/"suite_results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "complete-shards", "verify"))
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    {"launch": launch, "complete-shards": complete_shards, "verify": verify}[args.action](args.root)
