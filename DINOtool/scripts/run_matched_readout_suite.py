"""Schedule the fixed full-set operator comparison on idle A800 GPUs."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.matched_readout_controls import IMPLEMENTATION, METHODS, OPERATOR_NOTES
from merge_geometry_semantic_innovation import main as merge
from run_region_semantic_suite_a800 import BASE, TOOL, PYTHON, THIRD, SETTINGS, reference, idle


SHARDS = {"vdd": 2, "potsdam": 2, "udd5": 1, "oem": 2, "loveda": 4,
          "vaihingen": 1, "landcoverai": 4, "flair1": 8}
CONTROL_GROUP = "core"


def command(setting, output, shard=0, benchmark=False):
    dataset, data, vocab, _, _ = setting
    script = ("benchmark_matched_readout.py" if benchmark else
              "eval_matched_neighborhood_controls.py" if CONTROL_GROUP == "neighborhood" else
              "eval_matched_readout_controls.py")
    return [PYTHON, "-u", "scripts/"+script, "--dataset", dataset,
            "--dinov3-repo", str(TOOL/"dinov3_hub"), "--checkpoint-dir", str(BASE/"ckpt/DINO"),
            "--data-root", str(Path("/data/test/datasets")/data),
            "--vocabulary-config", str(TOOL/"configs"/vocab),
            "--upstream-root", str(BASE/"third_party/VIP_official_5bd25ee"),
            "--output-dir", str(output), "--num-shards", str(1 if benchmark else SHARDS[dataset]),
            "--shard-index", str(shard)]


def launch(root, setting, shard, gpu, benchmark=False):
    dataset = setting[0]
    output = root/"benchmark" if benchmark else root/dataset/f"s{shard}"
    log = root/"benchmark.log" if benchmark else root/dataset/f"s{shard}.log"
    prefix = "gmrc02" if CONTROL_GROUP == "neighborhood" else "gmrc01"
    session = "gmrc01_benchmark" if benchmark else f"{prefix}_{dataset}_s{shard}"
    if output.exists() or subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise RuntimeError(f"Existing output/session: {session}")
    if not idle(gpu):
        return None
    log.parent.mkdir(parents=True, exist_ok=True)
    env = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
           "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2"]
    shell = (f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(env)} nice -n 10 "
             f"{shlex.join(command(setting, output, shard, benchmark))} > {shlex.quote(str(log))} 2>&1")
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    print(f"Started {session} on physical GPU{gpu}", flush=True)
    return {"dataset": dataset, "shard": shard, "gpu": gpu, "session": session,
            "output": str(output), "log": str(log)}


def verify_dataset(root, dataset):
    output = root/dataset/"merged.json"
    inputs = [str(root/dataset/f"s{i}") for i in range(SHARDS[dataset])]
    merge(argparse.Namespace(inputs=inputs, output=str(output)))
    result = json.loads(output.read_text())
    old = json.loads(reference(dataset).read_text())
    for field in ("global_sample_keys_sha256", "vocabulary", "checkpoints"):
        if result["signature"][field] != old["signature"][field]:
            raise RuntimeError(f"Matched reference differs: {dataset}/{field}")
    if result["signature"]["implementation"] != IMPLEMENTATION:
        raise RuntimeError("Wrong matched implementation")
    keys, arrays = [], {}
    for path in inputs:
        with np.load(Path(path)/"per_image_confusions.npz", allow_pickle=False) as data:
            keys.extend(data["sample_keys"].tolist())
            for field in data.files:
                if field != "sample_keys":
                    arrays.setdefault(field, []).append(data[field])
    order = {key: index for index, key in enumerate(keys)}
    expected = result["signature"]["sample_keys"]
    if set(keys) != set(expected) or len(keys) != len(set(keys)):
        raise RuntimeError("Per-image key coverage differs")
    index = np.asarray([order[key] for key in expected])
    merged_arrays = {field: np.concatenate(values)[index] for field, values in arrays.items()}
    for key, methods in result["metrics"].items():
        baseline = methods["Geometry"]["confusion_matrix"]
        if not np.array_equal(baseline, old["metrics"][key]["Geometry"]["confusion_matrix"]):
            raise RuntimeError(f"Original Geometry changed: {dataset}/{key}")
        for method, metric in methods.items():
            if not np.array_equal(merged_arrays[key+"__"+method].sum(0), metric["confusion_matrix"]):
                raise RuntimeError("Per-image matrices do not reproduce full metrics")
    np.savez_compressed(root/dataset/"per_image_confusions.npz", sample_keys=np.asarray(expected), **merged_arrays)
    result["exact_reference_geometry_confusion"] = True
    result["per_image_coverage_verified"] = True
    output.write_text(json.dumps(result, indent=2)+"\n")
    print("Verified", dataset, result["processed_images"], flush=True)
    return {key: {method: metric["mean_iou_percent"] for method, metric in group.items()}
            for key, group in result["metrics"].items()}


def main(args):
    global CONTROL_GROUP, IMPLEMENTATION, METHODS, OPERATOR_NOTES
    CONTROL_GROUP = args.control_group
    if CONTROL_GROUP == "neighborhood":
        from dinotool import matched_neighborhood_controls as controls
        IMPLEMENTATION, METHODS, OPERATOR_NOTES = controls.IMPLEMENTATION, controls.METHODS, controls.OPERATOR_NOTES
    root = args.root
    settings = {setting[0]: setting for setting in SETTINGS}
    if args.mode == "benchmark":
        root.mkdir(parents=True, exist_ok=True)
        if not launch(root, settings["loveda"], 0, args.gpu, True):
            raise RuntimeError(f"GPU{args.gpu} occupied")
        return
    if (root/"protocol.json").exists():
        raise RuntimeError("Refusing an existing full suite")
    for dataset, setting in settings.items():
        old = json.loads(reference(dataset).read_text())
        if not old["coverage_verified"] or old["processed_images"] != setting[4]:
            raise RuntimeError(f"Incomplete reference: {dataset}")
    root.mkdir(parents=True, exist_ok=True)
    protocol = {"implementation": IMPLEMENTATION, "methods": METHODS, "operators": OPERATOR_NOTES,
                "shards": SHARDS, "datasets": {key: value[4] for key, value in settings.items()},
                "note": "Full eight-domain mechanism comparison. Shared fixed20 text, view, aggregation and precision. "
                        "DINO.text operator adaptations, not official complete CLIP baselines. Exploratory development. "
                        "No per-dataset parameters or winner switching. Corrected Vaihingen IRRG inputs."}
    (root/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    first = [("vdd", 0), ("potsdam", 0), ("udd5", 0), ("oem", 0),
             ("flair1", 0), ("flair1", 1), ("loveda", 0), ("landcoverai", 0)]
    queue = first+[(dataset, shard) for dataset in
                   ("flair1", "loveda", "landcoverai", "vdd", "potsdam", "oem", "vaihingen")
                   for shard in range(SHARDS[dataset]) if (dataset, shard) not in first]
    active, completed, failures, outcomes = {}, set(), {}, {}
    while queue or active:
        for gpu, job in list(active.items()):
            path = Path(job["output"])/"results.json"
            row = json.loads(path.read_text()) if path.exists() else None
            alive = subprocess.run(["tmux", "has-session", "-t", job["session"]], capture_output=True).returncode == 0
            if not alive:
                if row and row["status"] == "complete" and (Path(job["output"])/"per_image_confusions.npz").exists():
                    completed.add((job["dataset"], job["shard"]))
                else:
                    failures[job["session"]] = Path(job["log"]).read_text()[-6000:]
                    print("FAILURE", job["session"], failures[job["session"]], flush=True)
                del active[gpu]
        for dataset in settings:
            if dataset not in outcomes and all((dataset, shard) in completed for shard in range(SHARDS[dataset])):
                try:
                    outcomes[dataset] = verify_dataset(root, dataset)
                except Exception as error:
                    failures[dataset] = str(error)
                    outcomes[dataset] = {"verification_error": str(error)}
                    print("VERIFY FAILURE", dataset, str(error), flush=True)
        for gpu in range(8):
            if queue and gpu not in active and idle(gpu):
                dataset, shard = queue.pop(0)
                active[gpu] = launch(root, settings[dataset], shard, gpu)
        state = {"status": "running" if queue or active else "failed" if failures else "complete",
                 "active": list(active.values()), "queued": queue, "completed_shards": sorted(completed),
                 "outcomes": outcomes, "failures": failures}
        (root/"suite_status.json").write_text(json.dumps(state, indent=2)+"\n")
        if queue or active:
            time.sleep(20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("benchmark", "full"), default="full")
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--control-group", choices=("core", "neighborhood"), default="core")
    main(parser.parse_args())
