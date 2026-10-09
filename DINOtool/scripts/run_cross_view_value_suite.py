"""Run one fixed cross-view Value candidate on idle physical A800 GPUs0-7."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.cross_view_value_innovation import IMPLEMENTATION, PRIMARY, METHODS, ValueInnovationConfig
from eval_cached_localized_likelihood import save_json
from merge_geometry_semantic_innovation import main as merge
from run_region_semantic_suite_a800 import BASE, TOOL, PYTHON, THIRD, SETTINGS, idle
from run_sat_geometry_transport_suite import alive, exact_miou, read_json, reference


GPUS = tuple(range(8))
SOURCE_SCREEN = TOOL/"results/sat_geometry_transport_screen_20261002"
GATE = {"mean_comparators": ("Geometry", "SCLIP_Two", "VIPProxy_Two", "MeanLogit_ContextGeometry", "Shuffled_ValueInnovation"),
        "maximum_geometry_protocol_loss_pp": 0., "focus_datasets": ("vdd", "potsdam"),
        "focus_comparators": ("Geometry", "MeanLogit_ContextGeometry"), "mean_protocol": "LoveDA D once",
        "full_launch": "not automatic; a complete verified pass permits follow-up under unchanged rule"}


def launch(root, setting, gpu, smoke=False):
    if gpu not in GPUS or not idle(gpu):
        raise RuntimeError(f"Authorized GPU{gpu} not idle.")
    dataset, data, vocabulary, _, _ = setting
    output = root/"smoke" if smoke else root/dataset/"s0"
    log = root/"smoke.log" if smoke else root/dataset/"s0.log"
    session = "gcvv02_smoke" if smoke else "gcvv02_"+dataset
    if output.exists() or log.exists() or alive(session):
        raise RuntimeError("Refusing existing result/log/session: "+session)
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, "-u", "scripts/eval_cross_view_value_innovation.py", "--dataset", dataset,
        "--dinov3-repo", str(TOOL/"dinov3_hub"), "--checkpoint-dir", str(BASE/"ckpt/DINO"),
        "--data-root", str(Path("/data/test/datasets")/data), "--vocabulary-config", str(TOOL/"configs"/vocabulary),
        "--output-dir", str(output), "--num-shards", "1", "--shard-index", "0"]
    command += ["--smoke"] if smoke else ["--source-diagnostic", str(root/(dataset+"_samples.json"))]
    env = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
           "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "TOKENIZERS_PARALLELISM=false"]
    shell = f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(env)} nice -n 10 {shlex.join(command)} > {shlex.quote(str(log))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    print("Started", session, "physical GPU", gpu, flush=True)
    return {"dataset": dataset, "gpu": gpu, "session": session, "output": str(output), "log": str(log)}


def initialize(root):
    root.resolve().relative_to((TOOL/"results").resolve())
    if root.exists():
        raise RuntimeError("Existing suite root.")
    manifests = {}
    for dataset, *_ in SETTINGS:
        full = read_json(reference(dataset))
        keys = read_json(SOURCE_SCREEN/(dataset+"_samples.json"))["signature"]["samples"]
        expected = 40 if dataset == "udd5" else 8
        if (not full["coverage_verified"] or len(keys) != expected or len(set(keys)) != expected
                or not set(keys).issubset(full["signature"]["sample_keys"])):
            raise RuntimeError("Wrong unchanged development sequence: "+dataset)
        manifests[dataset] = {"signature": {"samples": keys}}
    for gpu in GPUS:
        if not idle(gpu):
            raise RuntimeError(f"GPU{gpu} occupied; suite not launched.")
    root.mkdir(parents=True)
    for dataset, manifest in manifests.items():
        save_json(root/(dataset+"_samples.json"), manifest)
    save_json(root/"protocol.json", {"implementation": IMPLEMENTATION, "config": ValueInnovationConfig().signature(),
        "primary": PRIMARY, "methods": METHODS, "gate": GATE, "physical_gpus": GPUS,
        "datasets": {key: len(row["signature"]["samples"]) for key, row in manifests.items()},
        "phase": "complete-image96 development screen", "target_label_fitting": False,
        "note": "Frozen all20; corrected IRRG Vaihingen; primary has no VIP or external teacher."})


def verify(root, dataset):
    output = root/dataset/"merged.json"
    merge(argparse.Namespace(inputs=[str(root/dataset/"s0")], output=str(output)))
    row, old = read_json(output), read_json(reference(dataset))
    expected = read_json(root/(dataset+"_samples.json"))["signature"]["samples"]
    if (not row["coverage_verified"] or row["signature"]["sample_keys"] != expected
            or row["processed_images"] != len(expected) or row["total_images"] != len(expected)
            or row["signature"]["implementation"] != IMPLEMENTATION
            or row["signature"]["gear"] != {"geometry": old["signature"]["gear"]["geometry"],
                                           "primary": PRIMARY, **ValueInnovationConfig().signature()}):
        raise RuntimeError("Changed unique full coverage or frozen model signature.")
    for field in ("vocabulary", "checkpoints"):
        if row["signature"][field] != old["signature"][field]:
            raise RuntimeError("Input identity changed: "+field)
    with np.load(root/dataset/"s0/per_image_confusions.npz", allow_pickle=False) as current, np.load(
            reference(dataset).parent/"per_image_confusions.npz", allow_pickle=False) as historical:
        lookup = {key: index for index, key in enumerate(historical["sample_keys"].tolist())}
        order = [lookup[key] for key in expected]
        if current["sample_keys"].tolist() != expected:
            raise RuntimeError("Per-image unique sequence changed.")
        for key, group in row["metrics"].items():
            for method in ("Geometry", "SCLIP_Two", "VIPProxy_Two"):
                if not np.array_equal(current[key+"__"+method], historical[key+"__"+method][order]):
                    raise RuntimeError("Original per-image control changed: "+method)
            for method, metric in group.items():
                if not np.array_equal(current[key+"__"+method].sum(0), metric["confusion_matrix"]):
                    raise RuntimeError("Per-image confusion reconstruction failed.")
                if method != "Geometry":
                    counts = np.asarray(row["transitions"][key][method]["counts"], np.int64)
                    if (not np.array_equal(counts.sum(1).T, group["Geometry"]["confusion_matrix"])
                            or not np.array_equal(counts.sum(0).T, metric["confusion_matrix"])):
                        raise RuntimeError("Transition endpoint reconstruction failed.")
    if any(group["context_geometry_replay_max_error"] != 0. or group["prefix_displacement_max_error"] != 0.
           or group["invalid_query_displacement_max_error"] != 0. for group in row["diagnostics"].values()):
        raise RuntimeError("Internal original Geometry/prefix/validity identity failed.")
    row["exact_geometry_and_nearest_per_image_replay"] = True
    save_json(output, row)
    print("Verified", dataset, len(expected), flush=True)
    return row


def decision(outcomes, failures):
    if failures or len(outcomes) != 8:
        return {"passed": False, "reason": "Incomplete/failed screen.", "gate": GATE}
    values = {dataset: {key: {method: exact_miou(metric) for method, metric in group.items()}
        for key, group in row["metrics"].items()} for dataset, row in outcomes.items()}
    means = {method: float(np.mean([group["D" if dataset == "loveda" else dataset][method]
        for dataset, group in values.items()])) for method in METHODS}
    checks = {"mean_vs_"+method: means[PRIMARY] > means[method] for method in GATE["mean_comparators"]}
    for dataset, protocols in values.items():
        for key, scores in protocols.items():
            checks["retain_"+dataset+"_"+key] = scores[PRIMARY] >= scores["Geometry"]
            if dataset in GATE["focus_datasets"]:
                for method in GATE["focus_comparators"]:
                    checks[dataset+"_vs_"+method] = scores[PRIMARY] > scores[method]
    return {"passed": all(checks.values()), "mean_miou": means, "values": values, "checks": checks, "gate": GATE}


def run(root, smoke_result):
    smoke = read_json(smoke_result)
    if (not smoke or smoke["status"] != "complete" or smoke["implementation"] != IMPLEMENTATION
            or smoke["target_masks_loaded"] or not smoke["methods_finite"] or not smoke["weights_frozen"]
            or not smoke["head_weights_unchanged"] or any(smoke["errors"].values())
            or smoke["config"] != ValueInnovationConfig().signature()
            or smoke["diagnostics"]["context_geometry_replay_max_error"] != 0.):
        raise RuntimeError("Verified mask-free actual-checkpoint smoke required.")
    initialize(root)
    started = time.time()
    queue, active, outcomes, failures = list(SETTINGS), {}, {}, {}
    while queue or active:
        for gpu, job in list(active.items()):
            if alive(job["session"]):
                continue
            try:
                result = read_json(Path(job["output"])/"results.json")
                if not result or result.get("status") != "complete" or result["processed_images"] != result["total_images"]:
                    raise RuntimeError("Worker exited before complete predictions.")
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
            "elapsed_seconds": time.time()-started}
        save_json(root/"suite_status.json", state)
        if queue or active:
            time.sleep(15)
    state.update(decision=decision(outcomes, failures), full_evaluation_launched=False)
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
        if args.root.exists() or not idle(0):
            raise RuntimeError("Existing smoke root or occupied GPU0.")
        args.root.mkdir(parents=True)
        launch(args.root, next(setting for setting in SETTINGS if setting[0] == "oem"), 0, True)
    else:
        run(args.root, args.smoke_result)
