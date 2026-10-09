"""Fixed cached-observation likelihood screen on physical A800 GPUs4-7."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.geometry_localized_likelihood import IMPLEMENTATION, PRIMARY, METHODS, LocalLikelihoodConfig
from run_region_semantic_suite_a800 import TOOL, PYTHON, THIRD, SETTINGS, idle
from run_sat_geometry_transport_suite import read_json, alive, exact_miou


GPUS = (4, 5, 6, 7)
SOURCE = TOOL/"results/geometry_grounding_observation_20261002"


def launch(root, setting, gpu):
    if gpu not in GPUS or not idle(gpu):
        return None
    dataset, data, vocab, _, _ = setting
    output, log = root/dataset, root/f"{dataset}.log"
    session = "gll02_"+dataset
    if output.exists() or alive(session):
        raise RuntimeError("Refusing existing result/session.")
    args = [PYTHON, "-u", "scripts/eval_cached_localized_likelihood.py", "--dataset", dataset,
        "--data-root", str(Path("/data/test/datasets")/data), "--vocabulary-config", str(TOOL/"configs"/vocab),
        "--source", str(SOURCE/dataset), "--output-dir", str(output)]
    env = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
           "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2"]
    shell = f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(env)} nice -n 10 {shlex.join(args)} > {shlex.quote(str(log))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    print(f"Started {session}, physical GPU{gpu}", flush=True)
    return {"dataset": dataset, "gpu": gpu, "output": str(output), "log": str(log), "session": session}


def verify(root, dataset):
    row = read_json(root/dataset/"results.json")
    status = read_json(root/dataset/"collection_status.json")
    source = read_json(SOURCE/dataset/"verified.json")
    if (not row or row["status"] != "complete" or row["implementation"] != IMPLEMENTATION
            or not row["coverage_verified"] or not row["original_geometry_exact"]
            or row["signature"]["sample_keys"] != source["signature"]["sample_keys"]
            or row["signature"]["config"] != LocalLikelihoodConfig().signature()
            or not status or status["status"] != "collected" or status["target_masks_loaded"]
            or not status["identity_control_exact"]):
        raise RuntimeError("Incomplete, changed or contaminated likelihood screen.")
    for key, group in row["metrics"].items():
        if not np.array_equal(group["Geometry"]["confusion_matrix"], source["metrics"][key]["Geometry"]["confusion_matrix"]):
            raise RuntimeError("Original cached Geometry confusion mismatch.")
        for method in METHODS[1:]:
            changes = row["transitions"][key][method]
            counts = np.asarray(changes["counts"], np.int64)
            if (not np.array_equal(counts.sum(1).T, group["Geometry"]["confusion_matrix"])
                    or not np.array_equal(counts.sum(0).T, group[method]["confusion_matrix"])):
                raise RuntimeError("Invalid transitions.")
    row["observation_cost"] = status
    (root/dataset/"verified.json").write_text(json.dumps(row, indent=2)+"\n")
    return row


def gate(outcomes, failures):
    if failures or len(outcomes) != 8:
        return {"passed": False, "reason": "Incomplete or failed verified screen."}
    values = {dataset: {key: {method: exact_miou(metric) for method, metric in group.items()}
                        for key, group in row["metrics"].items()} for dataset, row in outcomes.items()}
    means = {method: float(np.mean([group["D" if dataset == "loveda" else dataset][method]
                                   for dataset, group in values.items()])) for method in METHODS}
    checks = {"mean_vs_"+method: means[PRIMARY] > means[method] for method in ("BoxLocalLikelihood", "MeanProb_BoxLikelihood")}
    for dataset, groups in values.items():
        for key, group in groups.items():
            checks[f"retain_{dataset}_{key}"] = group[PRIMARY] >= group["Geometry"]
            changes = outcomes[dataset]["transitions"][key][PRIMARY]
            checks[f"net_corrections_{dataset}_{key}"] = changes["beneficial"] > changes["harmful"]
            if dataset in ("vdd", "potsdam"):
                checks[f"coupling_{dataset}_{key}"] = group[PRIMARY] > group["BoxLocalLikelihood"]
    return {"passed": all(checks.values()), "checks": checks, "window_mean_miou": means,
            "scope": "Cached-window-only development evidence, not full-image results or official-VIP superiority"}


def main(root):
    root.resolve().relative_to((TOOL/"results").resolve())
    if root.exists():
        raise RuntimeError("Refusing existing root.")
    root.mkdir(parents=True)
    (root/"protocol.json").write_text(json.dumps({"implementation": IMPLEMENTATION, "physical_gpus": GPUS,
        "methods": METHODS, "primary": PRIMARY, "config": LocalLikelihoodConfig().signature(),
        "source": str(SOURCE), "full_launch": False, "scope": "Cached windows from96 development images",
        "gate": "retain Geometry/net corrections every protocol; primary mean>box/blend; primary>box VDD/Potsdam"}, indent=2)+"\n")
    queue, active, outcomes, failures = list(SETTINGS), {}, {}, {}
    started = time.time()
    while queue or active:
        for gpu, job in list(active.items()):
            if alive(job["session"]):
                continue
            try:
                outcomes[job["dataset"]] = verify(root, job["dataset"])
            except Exception as error:
                failures[job["dataset"]] = {"error": str(error), "log_tail": Path(job["log"]).read_text()[-6000:]}
            del active[gpu]
        if failures:
            queue.clear()
        for gpu in GPUS:
            if queue and gpu not in active and idle(gpu):
                job = launch(root, queue[0], gpu)
                if job:
                    queue.pop(0)
                    active[gpu] = job
        state = {"status": "running" if queue or active else "failed" if failures else "complete",
            "implementation": IMPLEMENTATION, "physical_gpus": GPUS, "active": list(active.values()),
            "queued": [row[0] for row in queue], "completed": list(outcomes), "failures": failures,
            "outcomes": {dataset: {key: {method: exact_miou(metric) for method, metric in group.items()}
                for key, group in row["metrics"].items()} for dataset, row in outcomes.items()},
            "elapsed_seconds": time.time()-started}
        temporary = root/"suite_status.tmp"
        temporary.write_text(json.dumps(state, indent=2)+"\n")
        temporary.replace(root/"suite_status.json")
        if queue or active:
            time.sleep(20)
    state["decision"] = gate(outcomes, failures)
    state["full_evaluation_launched"] = False
    (root/"suite_results.json").write_text(json.dumps(state, indent=2)+"\n")
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
