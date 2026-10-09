"""Run the frozen soft-versus-hard alias study on idle physical A800 GPUs4-7."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.soft_competitive_alias import IMPLEMENTATION, METHODS, PAIRS, PRIMARY, SoftAliasConfig
from merge_geometry_semantic_innovation import main as merge
from run_bounded_alias_suite import GPUS, SOURCE_SCREEN, save
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, SETTINGS, idle
from run_sat_geometry_transport_suite import alive, exact_miou, read_json


CONFIG = asdict(SoftAliasConfig())
HARD_SOURCE = TOOL / "results/pair_conditional_alias_screen_20261002"
REPLAY_METHODS = ("Geometry", "BroadVIP", "MeanProb_VIP", "MeanLogit_VIP", "Anchored_VIP", "PairAlias_Local", "PairAlias_Coupled")
GATE = {"requested_goal": "positive equal-domain mean over matched hard deletion",
        "mechanism_requires": "positive mean over simple soft text and shuffled weights, mean loss <=0.5pp and worst loss <=1pp versus all20 coupling",
        "mean_protocol": "LoveDA D once and seven other domains once", "automatic_full_rollout": False}


def launch(root, setting, gpu, smoke=False):
    if gpu not in GPUS or not idle(gpu):
        raise RuntimeError("Authorized GPU not idle.")
    dataset, data, vocabulary, _, _ = setting
    output = root / "smoke" if smoke else root / dataset / "s0"
    log = root / "smoke.log" if smoke else root / dataset / "s0.log"
    session = "gsca02_smoke" if smoke else "gsca02_" + dataset
    if output.exists() or log.exists() or alive(session):
        raise RuntimeError("Existing output/log/session: " + session)
    log.parent.mkdir(parents=True, exist_ok=True)
    args = [PYTHON, "-u", "scripts/eval_soft_competitive_alias.py", "--dataset", dataset,
            "--dinov3-repo", str(TOOL / "dinov3_hub"), "--checkpoint-dir", str(BASE / "ckpt/DINO"),
            "--data-root", str(Path("/data/test/datasets") / data), "--vocabulary-config", str(TOOL / "configs" / vocabulary),
            "--upstream-root", str(BASE / "third_party/VIP_official_5bd25ee"), "--output-dir", str(output),
            "--num-shards", "1", "--shard-index", "0"]
    args += ["--smoke"] if smoke else ["--source-diagnostic", str(root / (dataset + "_samples.json"))]
    env = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
           "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "TOKENIZERS_PARALLELISM=false"]
    shell = f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(env)} nice -n 10 {shlex.join(args)} > {shlex.quote(str(log))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    print("Started", session, "physical GPU", gpu, flush=True)
    return {"dataset": dataset, "gpu": gpu, "session": session, "output": str(output), "log": str(log)}


def initialize(root):
    root.resolve().relative_to((TOOL / "results").resolve())
    if root.exists() or any(not idle(gpu) for gpu in GPUS):
        raise RuntimeError("Existing root or occupied authorized GPU.")
    manifests = {}
    for dataset, *_ in SETTINGS:
        keys = read_json(SOURCE_SCREEN / (dataset + "_samples.json"))["signature"]["samples"]
        hard = read_json(HARD_SOURCE / dataset / "merged.json")
        if (not hard["coverage_verified"] or len(keys) != (40 if dataset == "udd5" else 8)
                or len(keys) != len(set(keys)) or hard["signature"]["sample_keys"] != keys):
            raise RuntimeError("Matched hard-deletion sequence changed: " + dataset)
        manifests[dataset] = {"signature": {"samples": keys}}
    root.mkdir(parents=True)
    for dataset, manifest in manifests.items():
        save(root / (dataset + "_samples.json"), manifest)
    save(root / "protocol.json", {"implementation": IMPLEMENTATION, "config": CONFIG, "methods": METHODS,
         "primary": PRIMARY, "factorial_pairs": PAIRS, "gate": GATE, "physical_gpus": GPUS,
         "datasets": {key: len(value["signature"]["samples"]) for key, value in manifests.items()},
         "phase": "complete-image96 development screen", "target_label_fitting": False,
         "note": "All20 retained; exact leave-one-alias-out local and broad witnesses; Geometry top32 donor support. "
                 "Original Geometry anchor, half-uniform prior, bounded pair correction and original broad observer. "
                 "Simple soft-text and matched-weight-spectrum shuffled controls. No dataset-specific tuning."})


def verify(root, dataset):
    output = root / dataset / "merged.json"
    merge(argparse.Namespace(inputs=[str(root / dataset / "s0")], output=str(output)))
    row, old = read_json(output), read_json(HARD_SOURCE / dataset / "merged.json")
    source = read_json(root / dataset / "s0/results.json")
    expected = read_json(root / (dataset + "_samples.json"))["signature"]["samples"]
    if (not row["coverage_verified"] or row["signature"]["sample_keys"] != expected
            or row["processed_images"] != len(expected) or row["total_images"] != len(expected)
            or row["signature"]["implementation"] != IMPLEMENTATION or row["signature"]["gear"]["soft_alias"] != CONFIG):
        raise RuntimeError("Coverage or frozen configuration changed.")
    for field in ("vocabulary", "checkpoints"):
        if row["signature"][field] != old["signature"][field]:
            raise RuntimeError("Changed input identity: " + field)
    for field in ("geometry", "observation", "upstream_commit"):
        if row["signature"]["gear"][field] != old["signature"]["gear"][field]:
            raise RuntimeError("Changed source profile: " + field)
    with np.load(root / dataset / "s0/per_image_confusions.npz", allow_pickle=False) as current, np.load(
            HARD_SOURCE / dataset / "s0/per_image_confusions.npz", allow_pickle=False) as saved:
        if current["sample_keys"].tolist() != expected or saved["sample_keys"].tolist() != expected:
            raise RuntimeError("Per-image sequence changed.")
        for key, group in row["metrics"].items():
            for method in REPLAY_METHODS:
                if not np.array_equal(current[key + "__" + method], saved[key + "__" + method]):
                    raise RuntimeError("Historical baseline replay failed: " + key + "/" + method)
            for method, metric in group.items():
                if not np.array_equal(current[key + "__" + method].sum(0), metric["confusion_matrix"]):
                    raise RuntimeError("Per-image reconstruction failed.")
            for name, (base, candidate) in PAIRS.items():
                counts = np.asarray(source["factorial_transitions"][key][name]["counts"], np.int64)
                if (not np.array_equal(counts.sum(1).T, group[base]["confusion_matrix"])
                        or not np.array_equal(counts.sum(0).T, group[candidate]["confusion_matrix"])):
                    raise RuntimeError("Transition endpoints differ.")
            diag = row["diagnostics"][key]
            if diag["shuffled_weight_spectrum_error"] != 0 or diag["pair_partition_error"] > 1e-5:
                raise RuntimeError("Weight spectrum or pair partition changed.")
    row["factorial_transitions"] = source["factorial_transitions"]
    row["exact_seven_historical_per_image_replays"] = True
    save(output, row)
    print("Verified", dataset, len(expected), flush=True)
    return row


def decision(outcomes, failures):
    if failures or len(outcomes) != 8:
        return {"better_than_hard": False, "mechanism_supported": False, "reason": "Incomplete/failed screen."}
    values = {dataset: {key: {method: exact_miou(metric) for method, metric in group.items()}
                        for key, group in row["metrics"].items()} for dataset, row in outcomes.items()}
    means = {method: float(np.mean([group["D" if dataset == "loveda" else dataset][method]
             for dataset, group in values.items()])) for method in METHODS}
    worst = min(scores[PRIMARY] - scores["Anchored_VIP"] for group in values.values() for scores in group.values())
    better = means[PRIMARY] > means["PairAlias_Coupled"]
    supported = (better and means[PRIMARY] > means["SoftText_Coupled"]
                 and means[PRIMARY] > means["ShuffledCounterfactual_Coupled"]
                 and means[PRIMARY] - means["Anchored_VIP"] >= -.5 and worst >= -1)
    return {"better_than_hard": better, "mechanism_supported": supported, "worst_delta_vs_all20_coupling": worst,
            "mean_miou": means, "values": values, "gate": GATE}


def run(root):
    initialize(root)
    started = time.time()
    smoke = launch(root, next(setting for setting in SETTINGS if setting[0] == "oem"), 4, True)
    while alive(smoke["session"]):
        time.sleep(5)
    row = read_json(Path(smoke["output"]) / "results.json")
    if (not row or row["status"] != "complete" or row["implementation"] != IMPLEMENTATION
            or row["config"] != CONFIG or any(row["errors"].values()) or row["target_masks_loaded"]
            or not row["head_weights_unchanged"] or not row["weights_frozen"]):
        error = Path(smoke["log"]).read_text()[-6000:]
        save(root / "suite_results.json", {"status": "failed", "failures": {"smoke": error}})
        raise RuntimeError(error)
    queue = [next(setting for setting in SETTINGS if setting[0] == dataset)
             for dataset in ("udd5", "vdd", "potsdam", "oem", "loveda", "flair1", "vaihingen", "landcoverai")]
    active, outcomes, failures = {}, {}, {}
    while queue or active:
        for gpu, job in list(active.items()):
            if alive(job["session"]):
                continue
            try:
                result = read_json(Path(job["output"]) / "results.json")
                if not result or result.get("status") != "complete" or result["processed_images"] != result["total_images"]:
                    raise RuntimeError("Worker exited before completion.")
                outcomes[job["dataset"]] = verify(root, job["dataset"])
            except Exception as error:
                failures[job["dataset"]] = {"error": str(error), "log_tail": Path(job["log"]).read_text()[-6000:]}
            del active[gpu]
        if failures:
            queue.clear()
        for gpu in GPUS:
            if queue and gpu not in active and idle(gpu):
                active[gpu] = launch(root, queue.pop(0), gpu)
        state = {"status": "running" if queue or active else "failed" if failures else "complete",
                 "implementation": IMPLEMENTATION, "physical_gpus": GPUS, "active": list(active.values()),
                 "queued": [setting[0] for setting in queue], "completed": list(outcomes), "failures": failures,
                 "elapsed_seconds": time.time() - started}
        save(root / "suite_status.json", state)
        if queue or active:
            time.sleep(10)
    state.update(decision=decision(outcomes, failures), full_evaluation_launched=False)
    save(root / "suite_results.json", state)
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
