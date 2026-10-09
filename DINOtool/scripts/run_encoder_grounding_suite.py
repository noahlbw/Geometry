"""Run a fixed encoder-position observation screen on idle physical GPUs0-7."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.encoder_grounding_observer import IMPLEMENTATION, PRIMARY, METHODS, EncoderObservationConfig
from eval_cached_localized_likelihood import save_json
from run_region_semantic_suite_a800 import BASE, TOOL, PYTHON, THIRD, SETTINGS, idle
from run_sat_geometry_transport_suite import read_json, alive, exact_miou


GPUS = tuple(range(8))
SOURCE = TOOL/"results/geometry_grounding_observation_20261002"
ENCODER_SOURCE = BASE/"ckpt/GroundingDINO_tiny_a2bb814"


def launch(root, setting, gpu, smoke=False):
    if gpu not in GPUS or not idle(gpu):
        return None
    dataset, data, vocabulary, _, _ = setting
    output, log = root/dataset, root/f"{dataset}.log"
    session = "genc02_smoke" if smoke else "genc02_"+dataset
    if output.exists() or alive(session):
        raise RuntimeError("Refusing existing encoder result/session.")
    args = [PYTHON, "-u", "scripts/eval_encoder_grounding.py", "--dataset", dataset,
        "--data-root", str(Path("/data/test/datasets")/data), "--vocabulary-config", str(TOOL/"configs"/vocabulary),
        "--source", str(SOURCE/dataset), "--encoder-source", str(ENCODER_SOURCE), "--output-dir", str(output)]
    if smoke:
        args.append("--smoke")
    environment = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
        "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "TOKENIZERS_PARALLELISM=false"]
    shell = f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(environment)} nice -n 10 {shlex.join(args)} > {shlex.quote(str(log))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    print(f"Started {session}, physical GPU{gpu}", flush=True)
    return {"dataset": dataset, "gpu": gpu, "output": str(output), "log": str(log), "session": session}


def validate_signature(signature):
    if (signature["implementation"] != IMPLEMENTATION or not signature["weights_frozen"]
            or signature["candidate_scores_loaded_masks"]
            or signature["encoder_config"] != EncoderObservationConfig().signature()
            or signature["encoder_source"] != signature["source_signature"]["source"]
            or signature["captions"] != signature["source_signature"]["captions"]
            or any(count != 20 for bank in signature["source_signature"]["vocabularies"].values() for count in bank["counts"])):
        raise RuntimeError("Changed frozen encoder config, source or aliases.")


def verify(root, dataset):
    path = root/dataset
    row, status = read_json(path/"results.json"), read_json(path/"collection_status.json")
    source = read_json(SOURCE/dataset/"verified.json")
    expected = source["signature"]["sample_keys"]
    if (not row or row["status"] != "complete" or row["primary"] != PRIMARY
            or not row["coverage_verified"] or not row["original_geometry_exact"]
            or row["processed_images"] != len(expected) or row["total_images"] != len(expected)
            or len(expected) != len(set(expected)) or row["signature"]["sample_keys"] != expected
            or not status or status["status"] != "collected" or status["target_masks_loaded"]
            or not status["identity_control_exact"] or not status["original_geometry_scores_exact"]):
        raise RuntimeError("Incomplete or changed observation/coverage.")
    validate_signature(row["signature"])
    manifests = read_json(path/"window_manifest.json")
    originals = read_json(SOURCE/dataset/"raw_manifest.json")
    identities = lambda rows: [(r["sample_key"], r["protocol"], r["top"], r["left"], r["image_shape"], r["file"]) for r in rows]
    if (identities(manifests) != identities(originals) or len(manifests) != status["total_windows"]
            or not all((path/window["observation_file"]).is_file() and
                       (path/window["scores_file"]).is_file() for window in manifests)):
        raise RuntimeError("Original window identities or image-only caches changed.")
    for key, group in row["metrics"].items():
        if not np.array_equal(group["Geometry"]["confusion_matrix"], source["metrics"][key]["Geometry"]["confusion_matrix"]):
            raise RuntimeError("Original Geometry confusion changed.")
        for method in METHODS[1:]:
            counts = np.asarray(row["transitions"][key][method]["counts"], np.int64)
            if (not np.array_equal(counts.sum(1).T, group["Geometry"]["confusion_matrix"])
                    or not np.array_equal(counts.sum(0).T, group[method]["confusion_matrix"])):
                raise RuntimeError("Invalid transition reconstruction.")
    row["observation_cost"] = status
    save_json(path/"verified.json", row)
    return row


def gate(outcomes, failures):
    if failures or len(outcomes) != 8:
        return {"passed": False, "reason": "Incomplete or failed verified screen."}
    values = {dataset: {key: {method: exact_miou(metric) for method, metric in group.items()}
                        for key, group in row["metrics"].items()} for dataset, row in outcomes.items()}
    means = {method: float(np.mean([group["D" if dataset == "loveda" else dataset][method]
                                   for dataset, group in values.items()])) for method in METHODS}
    controls = ("MeanLogit_Encoder", "MeanProb_Encoder")
    checks = {"mean_vs_"+method: means[PRIMARY] > means[method] for method in controls}
    for dataset, groups in values.items():
        for key, group in groups.items():
            checks[f"retain_{dataset}_{key}"] = group[PRIMARY] >= group["Geometry"]
            if dataset in ("vdd", "potsdam"):
                for method in controls:
                    checks[f"coupling_{dataset}_{key}_vs_{method}"] = group[PRIMARY] > group[method]
    return {"passed": all(checks.values()), "checks": checks, "window_mean_miou": means,
            "scope": "Fixed-window development screen; not full-image results or official-VIP superiority"}


def main(root, smoke_result):
    root.resolve().relative_to((TOOL/"results").resolve())
    if root.exists():
        raise RuntimeError("Refusing existing suite root.")
    smoke = read_json(smoke_result)
    signature = read_json(smoke_result.parent/"signature.json")
    if (not smoke or smoke["status"] != "collected" or smoke["target_masks_loaded"]
            or not smoke["identity_control_exact"] or not smoke["original_geometry_scores_exact"]):
        raise RuntimeError("Actual-checkpoint image-only smoke required.")
    validate_signature(signature)
    for gpu in GPUS:
        if not idle(gpu):
            raise RuntimeError(f"GPU{gpu} occupied; eight-card suite not launched.")
    root.mkdir(parents=True)
    save_json(root/"protocol.json", {"implementation": IMPLEMENTATION, "physical_gpus": GPUS,
        "methods": METHODS, "primary": PRIMARY, "encoder_config": EncoderObservationConfig().signature(),
        "source": str(SOURCE), "full_launch": False, "scope": "Cached windows from96 development images",
        "gate": "retain Geometry every protocol; primary mean>both simple fusions; primary>both on VDD/Potsdam"})
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
            "elapsed_seconds": time.time()-started}
        save_json(root/"suite_status.json", state)
        if queue or active:
            time.sleep(20)
    state["decision"] = gate(outcomes, failures)
    state["full_evaluation_launched"] = False
    save_json(root/"suite_results.json", state)
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-result", type=Path)
    args = parser.parse_args()
    if args.smoke:
        args.root.resolve().relative_to((TOOL/"results").resolve())
        if args.root.exists():
            raise RuntimeError("Existing smoke root.")
        available = next((gpu for gpu in GPUS if idle(gpu)), None)
        if available is None:
            raise RuntimeError("No idle authorized GPU.")
        args.root.mkdir(parents=True)
        launch(args.root, SETTINGS[0], available, True)
    else:
        main(args.root, args.smoke_result)
