"""Locked broader development panel for native-row Geometry composition."""
import argparse
import json
from pathlib import Path
import random
import shlex
import subprocess
import time

import numpy as np

from dinotool.geometry_row_composition import BASELINES, IMPLEMENTATION, METHODS, PRIMARY, settings
from report_matched_readout_publication import group_key
from run_region_semantic_suite_a800 import BASE, TOOL, PYTHON, THIRD, SETTINGS, idle
from run_sat_geometry_transport_suite import reference, exact_miou, read_json, alive
from run_geometry_region_strong_suite import verify_dataset


OLD_SCREEN = TOOL/"results/sat_geometry_transport_screen_20261002"
SEED = 20261005
MEAN_CONTROLS = (*BASELINES, "Geometry_FPRead", "Geometry_GlobalBudget")
GATE = {"mean_above": list(MEAN_CONTROLS), "focus_no_loss": ["vdd", "potsdam"],
        "maximum_protocol_loss_pp": 1., "maximum_latency_ratio": 1.10}


def panel(dataset, full_keys, old_keys):
    count = 40 if dataset == "udd5" else 32
    if len(full_keys) != len(set(full_keys)) or not set(old_keys).issubset(full_keys):
        raise ValueError("Duplicate full keys or invalid old panel.")
    if dataset == "udd5":
        chosen = set(full_keys)
    else:
        previous, groups = set(old_keys), {}
        for key in full_keys:
            if key not in previous:
                groups.setdefault(group_key(dataset, key), []).append(key)
        rng = random.Random(SEED)
        names = sorted(groups)
        rng.shuffle(names)
        for name in names:
            rng.shuffle(groups[name])
        chosen = set()
        while len(chosen) < count and any(groups.values()):
            for name in names:
                if groups[name]:
                    chosen.add(groups[name].pop())
                if len(chosen) == count:
                    break
    keys = [key for key in full_keys if key in chosen]
    if len(keys) != count or (dataset != "udd5" and set(keys).intersection(old_keys)):
        raise ValueError("Wrong count or failed prior-panel exclusion.")
    return {"signature": {"samples": keys}, "seed": SEED,
            "old_panel_exact_key_overlap": len(set(keys).intersection(old_keys)),
            "source_groups": len({group_key(dataset, key) for key in keys}),
            "selection": "label-free grouped round-robin; original reference ordering",
            "note": "UDD5 reused full40; other domains exclude old8; developed confirmation, not untouched validation"}


def shard_count(dataset):
    return 4 if dataset in ("udd5", "vdd") else 1


def launch(root, setting, shard, gpu):
    dataset, data, vocab, *_ = setting
    output, log = root/dataset/f"s{shard}", root/dataset/f"s{shard}.log"
    session = f"grow05_{dataset}_s{shard}"
    if output.exists() or log.exists() or alive(session):
        raise RuntimeError("Existing worker/output: "+session)
    if not idle(gpu):
        return None
    log.parent.mkdir(parents=True, exist_ok=True)
    args = [PYTHON, "-u", "scripts/eval_geometry_row_composition.py", "--dataset", dataset,
        "--dinov3-repo", str(TOOL/"dinov3_hub"), "--checkpoint-dir", str(BASE/"ckpt/DINO"),
        "--data-root", str(Path("/data/test/datasets")/data), "--vocabulary-config", str(TOOL/"configs"/vocab),
        "--upstream-root", str(BASE/"third_party/VIP_official_5bd25ee"), "--output-dir", str(output),
        "--source-diagnostic", str(root/f"{dataset}_samples.json"),
        "--num-shards", str(shard_count(dataset)), "--shard-index", str(shard)]
    env = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
           "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "TOKENIZERS_PARALLELISM=false"]
    shell = f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(env+args)} > {shlex.quote(str(log))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    print(f"Started {session} GPU{gpu}", flush=True)
    return {"dataset": dataset, "shard": shard, "gpu": gpu, "session": session, "output": str(output), "log": str(log)}


def verify_one(root, dataset):
    row = verify_dataset(root, dataset, "screen", shard_count=shard_count(dataset),
                         implementation=IMPLEMENTATION, semantic_source=None)
    old = read_json(reference(dataset))
    if row["signature"]["gear"] != {"geometry": old["signature"]["gear"]["geometry"], "primary": PRIMARY, **settings()}:
        raise RuntimeError("Frozen composition rule differs.")
    with np.load(root/dataset/"per_image_confusions.npz", allow_pickle=False) as current, np.load(
            reference(dataset).parent/"per_image_confusions.npz", allow_pickle=False) as previous:
        lookup = {key: i for i, key in enumerate(previous["sample_keys"].tolist())}
        order = [lookup[key] for key in row["signature"]["sample_keys"]]
        for key in row["metrics"]:
            name = key+"__Geometry_BlockPrefix"
            if not np.array_equal(current[name], previous[name][order]):
                raise RuntimeError("Historical BlockPrefix replay differs.")
    row["exact_reference_matched_baselines"].append("Geometry_BlockPrefix")
    (root/dataset/"merged.json").write_text(json.dumps(row, indent=2)+"\n")
    return row


def run(root):
    root.resolve().relative_to((TOOL/"results").resolve())
    smoke = read_json(root/"smoke/results.json")
    if (not smoke or smoke["status"] != "complete" or smoke["config"] != settings()
            or smoke["target_masks_loaded"] or not smoke["weights_frozen"] or not smoke["head_weights_unchanged"]):
        raise RuntimeError("Verified checkpoint smoke required.")
    if (root/"protocol.json").exists():
        raise RuntimeError("Refusing previously launched suite.")
    if any(not idle(gpu) for gpu in range(8)):
        raise RuntimeError("All eight GPUs must be idle at suite launch.")
    manifests = {}
    for dataset, *_, total in SETTINGS:
        old = read_json(reference(dataset))
        keys = old["signature"]["sample_keys"]
        if not old["coverage_verified"] or len(keys) != total:
            raise RuntimeError("Incomplete source reference: "+dataset)
        previous = read_json(OLD_SCREEN/f"{dataset}_samples.json")["signature"]["samples"]
        manifests[dataset] = panel(dataset, keys, previous)
    protocol = {"implementation": IMPLEMENTATION, "config": settings(), "methods": METHODS, "primary": PRIMARY,
        "physical_gpus": list(range(8)), "datasets": {key: len(row["signature"]["samples"]) for key, row in manifests.items()},
        "gate": GATE, "seed": SEED,
        "note": "Broader264 developed confirmation; no control-to-primary switching, label tuning or automatic full rollout."}
    (root/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    for dataset, manifest in manifests.items():
        (root/f"{dataset}_samples.json").write_text(json.dumps(manifest, indent=2)+"\n")
    by_name = {row[0]: row for row in SETTINGS}
    queue = [(by_name[dataset], shard) for dataset in ("vdd", "udd5", "potsdam", "oem", "loveda", "vaihingen", "landcoverai", "flair1")
             for shard in range(shard_count(dataset))]
    active, completed, outcomes, failures = {}, set(), {}, {}
    started = time.perf_counter()
    while queue or active:
        for gpu, job in list(active.items()):
            if alive(job["session"]):
                continue
            row = read_json(Path(job["output"])/"results.json")
            if (row and row["status"] == "complete" and row["processed_images"] == row["total_images"]
                    and (Path(job["output"])/"per_image_confusions.npz").exists()):
                completed.add((job["dataset"], job["shard"]))
            else:
                failures[job["session"]] = Path(job["log"]).read_text()[-6000:]
            del active[gpu]
        for dataset in by_name:
            if dataset not in outcomes and dataset not in failures and all((dataset, s) in completed for s in range(shard_count(dataset))):
                try:
                    outcomes[dataset] = verify_one(root, dataset)
                except Exception as error:
                    failures[dataset] = str(error)
                    print("Verification failure", dataset, error, flush=True)
        if failures:
            queue.clear()
        for gpu in range(8):
            if queue and gpu not in active and idle(gpu):
                setting, shard = queue[0]
                job = launch(root, setting, shard, gpu)
                if job is not None:
                    queue.pop(0)
                    active[gpu] = job
        state = {"active": active, "queued": [(row[0], shard) for row, shard in queue],
                 "completed": sorted(completed), "verified_datasets": list(outcomes), "failures": failures}
        temporary = root/"suite_status.tmp"
        temporary.write_text(json.dumps(state, indent=2)+"\n")
        temporary.replace(root/"suite_status.json")
        if queue or active:
            time.sleep(10)
    if failures or len(outcomes) != 8:
        result = {"status": "failed", "failures": failures}
    else:
        values = {dataset: {key: {method: exact_miou(metric) for method, metric in group.items()}
                           for key, group in row["metrics"].items()} for dataset, row in outcomes.items()}
        means = {method: float(np.mean([groups["D" if dataset == "loveda" else dataset][method]
                                       for dataset, groups in values.items()])) for method in METHODS}
        checks = {"mean_vs_"+method: means[PRIMARY] > means[method] for method in MEAN_CONTROLS}
        for dataset, groups in values.items():
            for key, group in groups.items():
                checks["retain_"+dataset+"_"+key] = group[PRIMARY]-group["Geometry"] >= -1.
            if dataset in ("vdd", "potsdam"):
                checks["focus_"+dataset] = groups[dataset][PRIMARY] >= groups[dataset]["Geometry"]
        ratio = smoke["deployed_window_cost"][PRIMARY]["window_median_seconds"]/smoke["deployed_window_cost"]["Geometry"]["window_median_seconds"]
        checks["latency_ratio_at_most_1_10"] = ratio <= 1.10
        result = {"status": "complete", "implementation": IMPLEMENTATION, "images": sum(protocol["datasets"].values()),
                  "values": values, "mean_miou": means, "checks": checks, "passed": all(checks.values()), "latency_ratio": ratio}
    result["wall_seconds"] = time.perf_counter()-started
    (root/"suite_results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
