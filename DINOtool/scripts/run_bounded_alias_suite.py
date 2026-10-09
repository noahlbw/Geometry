"""Run and verify one frozen all20 alias factorial on idle physical GPUs4-7."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.bounded_alias_readout import CONFIG, FACTORIAL_PAIRS, IMPLEMENTATION, METHODS, PRIMARY
from merge_geometry_semantic_innovation import main as merge
from run_region_semantic_suite_a800 import BASE, TOOL, PYTHON, THIRD, SETTINGS, idle
from run_sat_geometry_transport_suite import alive, exact_miou, read_json, reference


GPUS = (4, 5, 6, 7)
SOURCE_SCREEN = TOOL / "results/sat_geometry_transport_screen_20261002"
MATCHED_OBSERVER = TOOL / "results/geometry_paired_context_screen_20261002"
ORIGINAL_METHODS = ("Geometry", "BroadVIP", "MeanProb_VIP", "MeanLogit_VIP", "Anchored_VIP")
GATE = {"mean_comparators": ("Geometry", "Anchored_VIP", "Bounded_MeanLogit", "MeanLogit_VIP", "MeanProb_VIP"),
        "maximum_anchored_protocol_loss_pp": 1.0, "focus_datasets": ("vdd", "potsdam"),
        "focus_comparator": "Anchored_VIP", "mean_protocol": "LoveDA D once; seven other domains once"}


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def launch(root, setting, gpu, smoke=False):
    if gpu not in GPUS or not idle(gpu):
        raise RuntimeError(f"Physical GPU{gpu} unavailable.")
    dataset, data, vocabulary, _, _ = setting
    output = root / "smoke" if smoke else root / dataset / "s0"
    log = root / "smoke.log" if smoke else root / dataset / "s0.log"
    session = "gba02_smoke" if smoke else "gba02_" + dataset
    if output.exists() or log.exists() or alive(session):
        raise RuntimeError("Existing output/log/session: " + session)
    log.parent.mkdir(parents=True, exist_ok=True)
    args = [PYTHON, "-u", "scripts/eval_bounded_alias_anchored.py", "--dataset", dataset,
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
    if root.exists():
        raise RuntimeError("Existing suite root.")
    manifests = {}
    for dataset, *_ in SETTINGS:
        full = read_json(reference(dataset))
        chosen = read_json(SOURCE_SCREEN / (dataset + "_samples.json"))
        keys = chosen["signature"]["samples"]
        observer = read_json(MATCHED_OBSERVER / dataset / "merged.json")
        if (not full["coverage_verified"] or len(keys) != (40 if dataset == "udd5" else 8)
                or len(keys) != len(set(keys)) or not set(keys).issubset(full["signature"]["sample_keys"])
                or not observer["coverage_verified"] or observer["signature"]["sample_keys"] != keys):
            raise RuntimeError("Changed fixed full-image sequence: " + dataset)
        manifests[dataset] = {"signature": {"samples": keys}}
    if any(not idle(gpu) for gpu in GPUS):
        raise RuntimeError("Authorized GPUs not idle.")
    root.mkdir(parents=True)
    for dataset, manifest in manifests.items():
        save(root / (dataset + "_samples.json"), manifest)
    save(root / "protocol.json", {"implementation": IMPLEMENTATION, "config": CONFIG, "methods": METHODS,
         "primary": PRIMARY, "factorial_pairs": FACTORIAL_PAIRS, "gate": GATE, "physical_gpus": GPUS,
         "datasets": {key: len(value["signature"]["samples"]) for key, value in manifests.items()},
         "phase": "complete-image96 development screen", "target_label_fitting": False,
         "note": "Frozen rho4; all20 retained; VIP salience/template/count offset unchanged; corrected IRRG Vaihingen."})


def verify(root, dataset):
    output = root / dataset / "merged.json"
    merge(argparse.Namespace(inputs=[str(root / dataset / "s0")], output=str(output)))
    row, old = read_json(output), read_json(reference(dataset))
    source = read_json(root / dataset / "s0/results.json")
    expected = read_json(root / (dataset + "_samples.json"))["signature"]["samples"]
    if (not row["coverage_verified"] or row["signature"]["sample_keys"] != expected
            or row["processed_images"] != len(expected) or row["total_images"] != len(expected)
            or row["signature"]["implementation"] != IMPLEMENTATION
            or row["signature"]["gear"]["bounded_alias"] != CONFIG):
        raise RuntimeError("Changed complete coverage/configuration.")
    for field in ("vocabulary", "checkpoints"):
        if row["signature"][field] != old["signature"][field]:
            raise RuntimeError("Input identity changed: " + field)
    if row["signature"]["gear"]["geometry"] != old["signature"]["gear"]["geometry"]:
        raise RuntimeError("Geometry changed.")
    with np.load(root / dataset / "s0/per_image_confusions.npz", allow_pickle=False) as current, np.load(
            MATCHED_OBSERVER / dataset / "s0/per_image_confusions.npz", allow_pickle=False) as observer:
        if current["sample_keys"].tolist() != expected or observer["sample_keys"].tolist() != expected:
            raise RuntimeError("Per-image sequence changed.")
        for key, group in row["metrics"].items():
            for method in ORIGINAL_METHODS:
                if not np.array_equal(current[key + "__" + method], observer[key + "__" + method]):
                    raise RuntimeError("Exact historical baseline replay failed: " + key + "/" + method)
            for method, metric in group.items():
                if not np.array_equal(current[key + "__" + method].sum(0), metric["confusion_matrix"]):
                    raise RuntimeError("Per-image reconstruction failed.")
            for name, (base, candidate) in FACTORIAL_PAIRS.items():
                counts = np.asarray(source["factorial_transitions"][key][name]["counts"], np.int64)
                if (not np.array_equal(counts.sum(1).T, group[base]["confusion_matrix"])
                        or not np.array_equal(counts.sum(0).T, group[candidate]["confusion_matrix"])):
                    raise RuntimeError("Factorial transition endpoints differ.")
    row["factorial_transitions"] = source["factorial_transitions"]
    row["exact_five_historical_per_image_replays"] = True
    save(output, row)
    print("Verified", dataset, len(expected), flush=True)
    return row


def decision(outcomes, failures):
    if failures or len(outcomes) != 8:
        return {"passed": False, "reason": "Incomplete/failed screen.", "gate": GATE}
    values = {dataset: {key: {method: exact_miou(metric) for method, metric in group.items()}
                        for key, group in row["metrics"].items()} for dataset, row in outcomes.items()}
    means = {method: float(np.mean([group["D" if dataset == "loveda" else dataset][method]
             for dataset, group in values.items()])) for method in METHODS}
    checks = {"mean_vs_" + method: means[PRIMARY] > means[method] for method in GATE["mean_comparators"]}
    for dataset, protocols in values.items():
        for key, scores in protocols.items():
            checks["retain_anchored_" + dataset + "_" + key] = scores[PRIMARY] - scores["Anchored_VIP"] >= -1.0
            if dataset in GATE["focus_datasets"]:
                checks["focus_" + dataset] = scores[PRIMARY] >= scores["Anchored_VIP"]
    return {"passed": all(checks.values()), "mean_miou": means, "values": values, "checks": checks, "gate": GATE}


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
