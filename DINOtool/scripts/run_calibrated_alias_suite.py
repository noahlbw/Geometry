"""Frozen ten-arm calibrated alias development screen on physical A800 GPUs0-7."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.calibrated_competitive_alias import (IMPLEMENTATION, METHODS, PAIRS, PRIMARY, REPLAY, SHUFFLED, CalibratedAliasConfig)
from dinotool.stratified_soft_alias import StratifiedAliasConfig
from merge_geometry_semantic_innovation import main as merge
from run_bounded_alias_suite import save
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, SETTINGS, idle
from run_sat_geometry_transport_suite import alive, exact_miou, read_json


CONFIG, ACTION = asdict(StratifiedAliasConfig()), asdict(CalibratedAliasConfig())
SOURCE = TOOL / "results/stratified_soft_alias_screen_20261002"
ORDER = ("udd5", "vdd", "potsdam", "oem", "loveda", "vaihingen", "landcoverai", "flair1")
GATE = {"minimum_mean_gain_vs_all20_pp": .1, "minimum_winning_domains": 5,
        "maximum_protocol_loss_vs_all20_pp": 1., "must_beat_text_and_all_three_shuffled_means": True,
        "mean_protocol": "LoveDA D once and seven other domains once", "automatic_full_rollout": False}


def launch(root, dataset, gpu, smoke=False):
    if gpu not in range(8) or not idle(gpu):
        raise RuntimeError("Authorized GPU not idle.")
    _, data, vocabulary, _, _ = next(row for row in SETTINGS if row[0] == dataset)
    output, log = (root / "smoke", root / "smoke.log") if smoke else (root / dataset / "s0", root / dataset / "s0.log")
    session = "gcca02_smoke" if smoke else "gcca02_" + dataset
    if output.exists() or log.exists() or alive(session):
        raise RuntimeError("Existing output/log/session: " + session)
    log.parent.mkdir(parents=True, exist_ok=True)
    args = [PYTHON, "-u", "scripts/eval_calibrated_competitive_alias.py", "--dataset", dataset,
            "--dinov3-repo", str(TOOL / "dinov3_hub"), "--checkpoint-dir", str(BASE / "ckpt/DINO"),
            "--data-root", str(Path("/data/test/datasets") / data), "--vocabulary-config", str(TOOL / "configs" / vocabulary),
            "--upstream-root", str(BASE / "third_party/VIP_official_5bd25ee"), "--output-dir", str(output), "--num-shards", "1", "--shard-index", "0"]
    args += ["--smoke"] if smoke else ["--source-diagnostic", str(root / (dataset + "_samples.json"))]
    env = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
           "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "TOKENIZERS_PARALLELISM=false"]
    shell = f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(env)} nice -n 10 {shlex.join(args)} > {shlex.quote(str(log))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    print("Started", session, "GPU", gpu, flush=True)
    return {"dataset": dataset, "gpu": gpu, "session": session, "output": str(output), "log": str(log)}


def initialize(root):
    root.resolve().relative_to((TOOL / "results").resolve())
    if root.exists() or any(not idle(gpu) for gpu in range(8)):
        raise RuntimeError("Existing root or occupied GPU.")
    manifests = {}
    for dataset in ORDER:
        keys = read_json(SOURCE / (dataset + "_samples.json"))["signature"]["samples"]
        row = read_json(SOURCE / dataset / "merged.json")
        if not row["coverage_verified"] or len(keys) != (40 if dataset == "udd5" else 8) or len(set(keys)) != len(keys) or row["signature"]["sample_keys"] != keys:
            raise RuntimeError("Historical sequence changed: " + dataset)
        manifests[dataset] = {"signature": {"samples": keys}}
    root.mkdir(parents=True)
    for dataset, manifest in manifests.items():
        save(root / (dataset + "_samples.json"), manifest)
    save(root / "protocol.json", {"implementation": IMPLEMENTATION, "config": CONFIG, "action_config": ACTION,
         "methods": METHODS, "primary": PRIMARY, "factorial_pairs": PAIRS, "gate": GATE, "physical_gpus": list(range(8)),
         "datasets": {key: len(value["signature"]["samples"]) for key, value in manifests.items()},
         "phase": "96 complete-image development screen", "target_label_fitting": False,
         "note": "Original Geometry and fixed20 aliases. Writer-only control retains exact v2 weights. Primary excludes tested alias plus two closest "
                 "text neighbors across parents; Geometry-retrieved witnesses supply exact calibrated half-attenuation marginals. "
                 "Reference agreement is not ground truth. Three matched-spectrum shuffles and a simple canonical-text prior. No per-domain tuning."})


def verify(root, dataset):
    output = root / dataset / "merged.json"
    merge(argparse.Namespace(inputs=[str(root / dataset / "s0")], output=str(output)))
    row, old = read_json(output), read_json(SOURCE / dataset / "merged.json")
    source = read_json(root / dataset / "s0/results.json")
    expected = read_json(root / (dataset + "_samples.json"))["signature"]["samples"]
    if (not row["coverage_verified"] or row["signature"]["sample_keys"] != expected or row["processed_images"] != len(expected)
            or row["total_images"] != len(expected) or row["signature"]["implementation"] != IMPLEMENTATION
            or row["signature"]["gear"]["stratified_alias"] != CONFIG or row["signature"]["gear"]["calibrated_action"] != ACTION):
        raise RuntimeError("Coverage or configuration mismatch.")
    for field in ("vocabulary", "checkpoints"):
        if row["signature"][field] != old["signature"][field]:
            raise RuntimeError("Input identity changed: " + field)
    for field in ("geometry", "observation", "upstream_commit"):
        if row["signature"]["gear"][field] != old["signature"]["gear"][field]:
            raise RuntimeError("Source profile changed: " + field)
    with np.load(root / dataset / "s0/per_image_confusions.npz", allow_pickle=False) as current, np.load(
            SOURCE / dataset / "s0/per_image_confusions.npz", allow_pickle=False) as saved:
        if current["sample_keys"].tolist() != expected or saved["sample_keys"].tolist() != expected:
            raise RuntimeError("Per-image sequence mismatch.")
        for key, group in row["metrics"].items():
            for method in REPLAY:
                if not np.array_equal(current[key + "__" + method], saved[key + "__" + method]):
                    raise RuntimeError("Historical replay failed: " + key + "/" + method)
            for method, metric in group.items():
                if not np.array_equal(current[key + "__" + method].sum(0), metric["confusion_matrix"]):
                    raise RuntimeError("Per-image confusion sum failed.")
            for name, (before, after) in PAIRS.items():
                counts = np.asarray(source["factorial_transitions"][key][name]["counts"], np.int64)
                if not np.array_equal(counts.sum(1).T, group[before]["confusion_matrix"]) or not np.array_equal(counts.sum(0).T, group[after]["confusion_matrix"]):
                    raise RuntimeError("Transition endpoints changed.")
            if any(value > 1e-5 for field, value in row["diagnostics"][key].items() if field.endswith("_error")):
                raise RuntimeError("Pair partition or shuffled spectrum failed.")
    row["factorial_transitions"] = source["factorial_transitions"]
    row["exact_four_historical_per_image_replays"] = True
    save(output, row)
    print("Verified", dataset, len(expected), flush=True)
    return row


def decision(outcomes, failures):
    if failures or len(outcomes) != 8:
        return {"promising": False, "reason": "Incomplete/failed screen."}
    values = {dataset: {key: {method: exact_miou(metric) for method, metric in group.items()} for key, group in row["metrics"].items()}
              for dataset, row in outcomes.items()}
    domain = {dataset: group["D" if dataset == "loveda" else dataset] for dataset, group in values.items()}
    means = {method: float(np.mean([group[method] for group in domain.values()])) for method in METHODS}
    wins = sum(group[PRIMARY] > group["Anchored_VIP"] for group in domain.values())
    worst = min(scores[PRIMARY] - scores["Anchored_VIP"] for group in values.values() for scores in group.values())
    meaningful = means[PRIMARY] > max(means[method] for method in (*SHUFFLED, "TextPrior_Coupled"))
    promising = means[PRIMARY] - means["Anchored_VIP"] >= .1 and wins >= 5 and worst >= -1 and meaningful
    return {"promising": promising, "winning_domains": wins, "worst_delta_vs_all20_coupling": worst,
            "beats_text_and_three_shuffled_means": meaningful, "mean_miou": means, "values": values, "gate": GATE}


def run(root):
    initialize(root)
    started = time.time()
    smoke = launch(root, "oem", 0, True)
    while alive(smoke["session"]):
        time.sleep(5)
    row = read_json(Path(smoke["output"]) / "results.json")
    if not row or row["status"] != "complete" or row["config"] != CONFIG or row["action_config"] != ACTION or row["target_masks_loaded"] or not row["weights_frozen"] or not row["head_weights_unchanged"]:
        error = Path(smoke["log"]).read_text()[-6000:]
        save(root / "suite_results.json", {"status": "failed", "failures": {"smoke": error}})
        raise RuntimeError(error)
    pending, active, outcomes, failures = dict(enumerate(ORDER)), {}, {}, {}
    while pending or active:
        for gpu, job in list(active.items()):
            if alive(job["session"]):
                continue
            try:
                row = read_json(Path(job["output"]) / "results.json")
                if not row or row["status"] != "complete" or row["processed_images"] != row["total_images"]:
                    raise RuntimeError("Worker exited without complete results.")
                outcomes[job["dataset"]] = verify(root, job["dataset"])
            except Exception as error:
                failures[job["dataset"]] = {"error": str(error), "log_tail": Path(job["log"]).read_text()[-6000:]}
            del active[gpu]
        if failures:
            pending.clear()
        for gpu, dataset in list(pending.items()):
            if idle(gpu):
                active[gpu] = launch(root, dataset, gpu)
                del pending[gpu]
        state = {"status": "running" if pending or active else "failed" if failures else "complete",
                 "implementation": IMPLEMENTATION, "active": list(active.values()), "queued": list(pending.values()),
                 "completed": list(outcomes), "failures": failures, "elapsed_seconds": time.time() - started}
        save(root / "suite_status.json", state)
        if pending or active:
            time.sleep(10)
    state.update(decision=decision(outcomes, failures), full_evaluation_launched=False)
    save(root / "suite_results.json", state)
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
