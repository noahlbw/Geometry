"""Frozen contextual-alias rejection screen on idle physical GPUs0-7."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.excess_alias_rejection import (IMPLEMENTATION, METHODS, PRIMARY, REPLAY, SHUFFLED, ExcessRejectConfig)
from run_bounded_alias_suite import save
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, SETTINGS, idle
from run_sat_geometry_transport_suite import alive, exact_miou, read_json


SOURCE = TOOL / "results/calibrated_group_alias_screen_20261002"
ORDER = ("udd5", "vdd", "potsdam", "oem", "loveda", "vaihingen", "landcoverai", "flair1")
CONFIG = asdict(ExcessRejectConfig())
GATE = {"clean_gain_route": "mean>=+0.1pp, wins>=5/8, worst>=-1pp, beat text and each shuffle",
        "robustness_route": "clean mean>=-0.1pp and worst>=-0.5pp; wrong-parent gain>=0.5pp and wins>=5/8; beat text/hard/each shuffle; paraphrase mean>=-0.1pp and worst>=-0.5pp",
        "stress_scope": "first8 complete images per domain, contextual vocabulary only; local20 fixed",
        "automatic_full_rollout": False, "independent_validation": False}


def launch(root, dataset, gpu, smoke=False, evaluator="scripts/eval_excess_alias_rejection.py", session_prefix="gear03"):
    if gpu not in range(8) or not idle(gpu):
        raise RuntimeError("Authorized GPU not idle.")
    _, data, vocabulary, _, _ = next(row for row in SETTINGS if row[0] == dataset)
    output, log = (root / "smoke", root / "smoke.log") if smoke else (root / dataset / "s0", root / dataset / "s0.log")
    session = session_prefix + ("_smoke" if smoke else "_" + dataset)
    if output.exists() or log.exists() or alive(session):
        raise RuntimeError("Existing output/log/session: " + session)
    log.parent.mkdir(parents=True, exist_ok=True)
    args = [PYTHON, "-u", evaluator, "--dataset", dataset,
            "--dinov3-repo", str(TOOL / "dinov3_hub"), "--checkpoint-dir", str(BASE / "ckpt/DINO"),
            "--data-root", str(Path("/data/test/datasets") / data), "--vocabulary-config", str(TOOL / "configs" / vocabulary),
            "--upstream-root", str(BASE / "third_party/VIP_official_5bd25ee"), "--output-dir", str(output),
            "--num-shards", "1", "--shard-index", "0"]
    args += ["--smoke"] if smoke else ["--source-diagnostic", str(root / (dataset + "_samples.json"))]
    env = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
           "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "TOKENIZERS_PARALLELISM=false"]
    shell = f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(env)} nice -n 10 {shlex.join(args)} > {shlex.quote(str(log))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    return {"dataset": dataset, "gpu": gpu, "session": session, "output": str(output), "log": str(log)}


def verify(root, dataset, source=SOURCE, implementation=IMPLEMENTATION, replay=REPLAY, replay_stress=False):
    directory = root / dataset / "s0"
    row = read_json(directory / "results.json")
    previous = read_json(source / dataset / "merged.json")
    expected = read_json(root / (dataset + "_samples.json"))["signature"]["samples"]
    if (not row or row["status"] != "complete" or row["processed_images"] != len(expected)
            or row["total_images"] != len(expected) or row["signature"]["sample_keys"] != expected
            or len(expected) != len(set(expected)) or row["stress_sample_keys"] != expected[:8]
            or row["signature"]["implementation"] != implementation or row["signature"]["gear"]["excess_reject"] != CONFIG):
        raise RuntimeError("Incomplete coverage or changed configuration.")
    for field in ("vocabulary", "checkpoints"):
        if row["signature"][field] != previous["signature"][field]:
            raise RuntimeError("Changed input identity: " + field)
    for field in ("geometry", "observation", "upstream_commit"):
        if row["signature"]["gear"][field] != previous["signature"]["gear"][field]:
            raise RuntimeError("Changed profile: " + field)
    with np.load(directory / "per_image_confusions.npz", allow_pickle=False) as current, np.load(
            source / dataset / "s0/per_image_confusions.npz", allow_pickle=False) as old, np.load(
            directory / "stress_per_image_confusions.npz", allow_pickle=False) as stress:
        if current["sample_keys"].tolist() != expected or stress["sample_keys"].tolist() != expected[:8]:
            raise RuntimeError("Per-image sequence mismatch.")
        row["stress_clean_metrics"] = {}
        for key, group in row["metrics"].items():
            row["stress_clean_metrics"][key] = {}
            for method in replay:
                if not np.array_equal(current[key + "__" + method], old[key + "__" + method]):
                    raise RuntimeError("Historical control replay failed: " + method)
            for method, metric in group.items():
                if not np.array_equal(current[key + "__" + method].sum(0), metric["confusion_matrix"]):
                    raise RuntimeError("Confusion sum mismatch.")
                if not np.array_equal(stress["clean__" + key + "__" + method], current[key + "__" + method][:8]):
                    raise RuntimeError("Clean stress slice mismatch.")
                row["stress_clean_metrics"][key][method] = {"confusion_matrix": stress["clean__" + key + "__" + method].sum(0).tolist()}
                if method != "Anchored_VIP":
                    counts = np.asarray(row["transitions"]["clean"][key][method]["counts"], np.int64)
                    if not np.array_equal(counts.sum(1).T, group["Anchored_VIP"]["confusion_matrix"]) or not np.array_equal(counts.sum(0).T, metric["confusion_matrix"]):
                        raise RuntimeError("Clean transition endpoint mismatch.")
            for scenario, protocols in row["stress_metrics"].items():
                for method, metric in protocols[key].items():
                    cm = stress[scenario + "__" + key + "__" + method].sum(0)
                    if not np.array_equal(cm, metric["confusion_matrix"]):
                        raise RuntimeError("Stress confusion sum mismatch.")
                    if method == "Geometry" and not np.array_equal(stress[scenario + "__" + key + "__" + method], stress["clean__" + key + "__" + method]):
                        raise RuntimeError("Context stress changed local Geometry.")
                    if method != "Anchored_VIP":
                        counts = np.asarray(row["transitions"][scenario][key][method]["counts"], np.int64)
                        if not np.array_equal(counts.sum(1).T, protocols[key]["Anchored_VIP"]["confusion_matrix"]) or not np.array_equal(counts.sum(0).T, cm):
                            raise RuntimeError("Stress transition endpoint mismatch.")
        if replay_stress:
            if row["vocabulary_stress"] != previous["vocabulary_stress"]:
                raise RuntimeError("Changed stress vocabulary.")
            with np.load(source / dataset / "s0/stress_per_image_confusions.npz", allow_pickle=False) as history:
                if history["sample_keys"].tolist() != expected[:8]:
                    raise RuntimeError("Historical stress sequence mismatch.")
                for scenario in ("clean", "wrong_parent", "paraphrase"):
                    for key in row["metrics"]:
                        for method in replay:
                            name = scenario + "__" + key + "__" + method
                            if not np.array_equal(stress[name], history[name]):
                                raise RuntimeError("Historical stress replay failed: " + name)
    for group in row["diagnostics"].values():
        if any(values["shuffle_spectrum_error"] != 0 for values in group.values()):
            raise RuntimeError("Unmatched shuffled retention spectrum.")
    row["coverage_verified"] = True
    row["historical_per_image_replays"] = list(replay)
    row["historical_stress_replays_verified"] = replay_stress
    if len(replay) == 3:
        row["exact_three_historical_per_image_replays"] = True
    save(root / dataset / "merged.json", row)
    print("Verified", dataset, len(expected), flush=True)
    return row


def decision(rows, methods=METHODS, primary=PRIMARY, shuffled=SHUFFLED):
    means, values = {}, {}
    for scenario in ("clean", "wrong_parent", "paraphrase"):
        values[scenario] = {}
        for dataset, row in rows.items():
            protocols = row["metrics"] if scenario == "clean" else row["stress_metrics"][scenario]
            values[scenario][dataset] = {key: {method: exact_miou(metric) for method, metric in group.items()}
                                         for key, group in protocols.items()}
        means[scenario] = {method: float(np.mean([protocols["D" if dataset == "loveda" else dataset][method]
            for dataset, protocols in values[scenario].items()])) for method in methods}
    clean, corrupt, benign = means["clean"], means["wrong_parent"], means["paraphrase"]
    def deltas(scenario):
        return [group[primary] - group["Anchored_VIP"] for protocols in values[scenario].values() for group in protocols.values()]
    def wins(scenario):
        return sum(protocols["D" if dataset == "loveda" else dataset][primary] > protocols["D" if dataset == "loveda" else dataset]["Anchored_VIP"]
                   for dataset, protocols in values[scenario].items())
    clean_gain = (clean[primary] - clean["Anchored_VIP"] >= .1 and wins("clean") >= 5 and min(deltas("clean")) >= -1
                  and clean[primary] > max(clean[m] for m in (*shuffled, "TextOnlyReject_Coupled")))
    robust = (clean[primary] - clean["Anchored_VIP"] >= -.1 and min(deltas("clean")) >= -.5
              and corrupt[primary] - corrupt["Anchored_VIP"] >= .5 and wins("wrong_parent") >= 5
              and corrupt[primary] > max(corrupt[m] for m in (*shuffled, "TextOnlyReject_Coupled", "HardDelete_Coupled"))
              and benign[primary] - benign["Anchored_VIP"] >= -.1 and min(deltas("paraphrase")) >= -.5)
    return {"promising": bool(clean_gain or robust), "clean_gain_route": bool(clean_gain), "robustness_route": bool(robust),
            "mean_miou": means, "domain_values": values, "wins": {s: wins(s) for s in values},
            "worst_protocol_delta": {s: min(deltas(s)) for s in values}, "gate": GATE}


def run(root, source=SOURCE, implementation=IMPLEMENTATION, methods=METHODS, primary=PRIMARY,
        shuffled=SHUFFLED, replay=REPLAY, replay_stress=False,
        evaluator="scripts/eval_excess_alias_rejection.py", session_prefix="gear03"):
    root.resolve().relative_to((TOOL / "results").resolve())
    if root.exists() or any(not idle(gpu) for gpu in range(8)):
        raise RuntimeError("Existing root or occupied GPUs.")
    root.mkdir(parents=True)
    for dataset in ORDER:
        keys = read_json(source / (dataset + "_samples.json"))["signature"]["samples"]
        if len(keys) != (40 if dataset == "udd5" else 8) or len(keys) != len(set(keys)):
            raise RuntimeError("Historical sequence mismatch.")
        save(root / (dataset + "_samples.json"), {"signature": {"samples": keys}})
    save(root / "protocol.json", {"implementation": implementation, "config": CONFIG, "methods": methods,
         "primary": primary, "gate": GATE, "physical_gpus": list(range(8)), "source": str(source),
         "historical_controls": list(replay), "historical_stress_replay": replay_stress,
         "clean_images": 96, "stress_images_per_scenario": 64, "target_label_fitting": False,
         "note": "Original local20 anchor fixed. Context20 changes in stress arms by public class-name rules only. No fresh LLM generation or target-label selector fitting."})
    started = time.time()
    smoke = launch(root, "oem", 0, True, evaluator, session_prefix)
    while alive(smoke["session"]):
        time.sleep(5)
    row = read_json(Path(smoke["output"]) / "results.json")
    if not row or row["status"] != "complete" or row["config"] != CONFIG or row["target_masks_loaded"] or not row["weights_frozen"] or not row["head_weights_unchanged"]:
        save(root / "suite_results.json", {"status": "failed", "failures": {"smoke": Path(smoke["log"]).read_text()[-6000:]}})
        raise RuntimeError("Smoke failed.")
    pending, active, rows, failures = dict(enumerate(ORDER)), {}, {}, {}
    while pending or active:
        for gpu, job in list(active.items()):
            if alive(job["session"]):
                continue
            try:
                rows[job["dataset"]] = verify(root, job["dataset"], source, implementation, replay, replay_stress)
            except Exception as error:
                failures[job["dataset"]] = {"error": str(error), "log_tail": Path(job["log"]).read_text()[-6000:]}
            del active[gpu]
        if failures:
            pending.clear()
        for gpu, dataset in list(pending.items()):
            if idle(gpu):
                active[gpu] = launch(root, dataset, gpu, evaluator=evaluator, session_prefix=session_prefix)
                del pending[gpu]
        state = {"status": "running" if pending or active else "failed" if failures else "complete",
                 "implementation": implementation, "active": list(active.values()), "queued": list(pending.values()),
                 "completed": list(rows), "failures": failures, "elapsed_seconds": time.time()-started}
        save(root / "suite_status.json", state)
        if pending or active:
            time.sleep(10)
    state.update(decision=decision(rows, methods, primary, shuffled) if not failures and len(rows) == 8 else {"promising": False}, full_evaluation_launched=False)
    save(root / "suite_results.json", state)
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
