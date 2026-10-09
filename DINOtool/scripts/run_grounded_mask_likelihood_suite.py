"""One complete fixed pixel-support candidate, on physical GPUs4-7 only."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.grounded_mask_observer import IMPLEMENTATION, PRIMARY, METHODS, MaskObservationConfig
from run_region_semantic_suite_a800 import BASE, TOOL, PYTHON, THIRD, SETTINGS, idle
from run_sat_geometry_transport_suite import read_json, alive, exact_miou


GPUS = (4, 5, 6, 7)
SOURCE = TOOL/"results/geometry_grounding_observation_20261002"
BOX_SOURCE = TOOL/"results/geometry_localized_likelihood_cached_r2_20261002"
MASK_SOURCE = BASE/"ckpt/SAM21_tiny_de431c4"


def launch(root, setting, gpu, smoke=False):
    if gpu not in GPUS or not idle(gpu):
        return None
    dataset, data, vocab, _, _ = setting
    output, log = root/dataset, root/f"{dataset}.log"
    session = "gmask02_smoke" if smoke else "gmask02_"+dataset
    if output.exists() or alive(session):
        raise RuntimeError("Refusing existing mask result/session.")
    args = [PYTHON, "-u", "scripts/eval_grounded_mask_likelihood.py", "--dataset", dataset,
        "--data-root", str(Path("/data/test/datasets")/data), "--vocabulary-config", str(TOOL/"configs"/vocab),
        "--source", str(SOURCE/dataset), "--box-source", str(BOX_SOURCE/dataset),
        "--mask-source", str(MASK_SOURCE), "--output-dir", str(output)]
    if smoke:
        args.append("--smoke")
    environment = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
        "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "TOKENIZERS_PARALLELISM=false"]
    shell = f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(environment)} nice -n 10 {shlex.join(args)} > {shlex.quote(str(log))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    print(f"Started {session}, physical GPU{gpu}", flush=True)
    return {"dataset": dataset, "gpu": gpu, "output": str(output), "log": str(log), "session": session}


def verify(root, dataset):
    path = root/dataset
    row, status = read_json(path/"results.json"), read_json(path/"collection_status.json")
    source = read_json(SOURCE/dataset/"verified.json")
    expected = source["signature"]["sample_keys"]
    if (not row or row["status"] != "complete" or row["implementation"] != IMPLEMENTATION
            or row["primary"] != PRIMARY or not row["coverage_verified"] or not row["source_predictions_exact"]
            or row["processed_images"] != len(expected) or row["total_images"] != len(expected)
            or len(expected) != len(set(expected)) or row["signature"]["sample_keys"] != expected
            or row["signature"]["mask_config"] != MaskObservationConfig().signature()
            or not row["signature"]["weights_frozen"] or row["signature"]["candidate_scores_loaded_masks"]
            or not status or status["status"] != "collected" or status["target_masks_loaded"]
            or not status["identity_control_exact"] or not status["original_source_scores_exact"]):
        raise RuntimeError("Incomplete or changed mask observation/coverage.")
    manifests = read_json(path/"window_manifest.json")
    if len(manifests) != status["total_windows"] or not all(
            (path/window["mask_file"]).is_file() and (path/window["scores_file"]).is_file() for window in manifests):
        raise RuntimeError("Missing image-only mask/score caches.")
    for key, group in row["metrics"].items():
        for method in METHODS[1:]:
            counts = np.asarray(row["transitions"][key][method]["counts"], np.int64)
            if (not np.array_equal(counts.sum(1).T, group["Geometry"]["confusion_matrix"])
                    or not np.array_equal(counts.sum(0).T, group[method]["confusion_matrix"])):
                raise RuntimeError("Invalid original/proposal transition reconstruction.")
    row["observation_cost"] = status
    (path/"verified.json").write_text(json.dumps(row, indent=2)+"\n")
    return row


def gate(outcomes, failures):
    if failures or len(outcomes) != 8:
        return {"passed": False, "reason": "Incomplete or failed verified screen."}
    values = {dataset: {key: {method: exact_miou(metric) for method, metric in group.items()}
                        for key, group in row["metrics"].items()} for dataset, row in outcomes.items()}
    means = {method: float(np.mean([group["D" if dataset == "loveda" else dataset][method]
                                   for dataset, group in values.items()])) for method in METHODS}
    checks = {"mean_vs_"+method: means[PRIMARY] > means[method]
              for method in ("MaskLocalLikelihood", "MeanProb_MaskLikelihood")}
    for dataset, groups in values.items():
        for key, group in groups.items():
            checks[f"retain_{dataset}_{key}"] = group[PRIMARY] >= group["Geometry"]
            if dataset in ("vdd", "potsdam"):
                for method in ("MaskLocalLikelihood", "MeanProb_MaskLikelihood"):
                    checks[f"coupling_{dataset}_{key}_vs_{method}"] = group[PRIMARY] > group[method]
    return {"passed": all(checks.values()), "checks": checks, "window_mean_miou": means,
            "scope": "Fixed-window development evidence; not full-image results or official-VIP superiority"}


def main(root, smoke_result):
    root.resolve().relative_to((TOOL/"results").resolve())
    if root.exists():
        raise RuntimeError("Refusing existing suite root.")
    smoke = read_json(smoke_result)
    signature = read_json(smoke_result.parent/"signature.json")
    if (not smoke or smoke["status"] != "collected" or smoke["target_masks_loaded"]
            or not smoke["identity_control_exact"] or not smoke["original_source_scores_exact"]
            or not signature or signature["implementation"] != IMPLEMENTATION
            or not signature["weights_frozen"] or signature["candidate_scores_loaded_masks"]
            or signature["mask_config"] != MaskObservationConfig().signature()):
        raise RuntimeError("Actual-checkpoint image-only smoke required.")
    root.mkdir(parents=True)
    (root/"protocol.json").write_text(json.dumps({"implementation": IMPLEMENTATION, "physical_gpus": GPUS,
        "methods": METHODS, "primary": PRIMARY, "mask_config": MaskObservationConfig().signature(),
        "source": str(SOURCE), "full_launch": False, "scope": "Cached windows from96 development images",
        "gate": "retain Geometry every protocol; primary mean>mask/blend; primary>mask/blend VDD/Potsdam"}, indent=2)+"\n")
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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-result", type=Path)
    args = parser.parse_args()
    if args.smoke:
        args.root.resolve().relative_to((TOOL/"results").resolve())
        if args.root.exists() or not idle(4):
            raise RuntimeError("Existing smoke root or occupied GPU4.")
        args.root.mkdir(parents=True)
        launch(args.root, SETTINGS[0], 4, True)
    else:
        main(args.root, args.smoke_result)
