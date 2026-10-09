"""Run the fixed complete candidate on the four authorized A800 GPUs."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.geometry_reacquisition import IMPLEMENTATION, METHODS, PRIMARY, ReacquisitionConfig
from merge_geometry_semantic_innovation import main as merge
from run_region_semantic_suite_a800 import BASE, TOOL, PYTHON, THIRD, SETTINGS, idle


SHARDS = {"udd5": 1, "vdd": 2, "potsdam": 2, "oem": 2, "loveda": 4,
          "vaihingen": 1, "landcoverai": 4, "flair1": 4}
GPUS = (4, 5, 6, 7)
BASELINE_METHODS = ("Geometry", "SCLIP_Two", "VIPProxy_Two")


def reference(dataset):
    return TOOL / "results/geometry_matched_readout_full_20261001" / dataset / "merged.json"


def read_json(path):
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except json.JSONDecodeError:
        return None


def alive(session):
    return subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0


def command(setting, output, shard=0, smoke=False):
    dataset, data, vocab, _, _ = setting
    return [PYTHON, "-u", "scripts/" + ("smoke_geometry_reacquisition.py" if smoke else "eval_geometry_reacquisition.py"),
            "--dataset", dataset, "--dinov3-repo", str(TOOL/"dinov3_hub"),
            "--checkpoint-dir", str(BASE/"ckpt/DINO"),
            "--data-root", str(Path("/data/test/datasets")/data),
            "--vocabulary-config", str(TOOL/"configs"/vocab),
            "--upstream-root", str(BASE/"third_party/VIP_official_5bd25ee"),
            "--output-dir", str(output), "--num-shards", str(1 if smoke else SHARDS[dataset]),
            "--shard-index", str(shard)]


def launch(root, setting, shard, gpu, smoke=False):
    if gpu not in GPUS:
        raise ValueError("Only physical GPUs4-7 are authorized for this suite.")
    dataset = setting[0]
    output = root/"smoke" if smoke else root/dataset/f"s{shard}"
    log = root/"smoke.log" if smoke else root/dataset/f"s{shard}.log"
    session = "grq02_smoke" if smoke else f"grq02_{dataset}_s{shard}"
    if output.exists() or alive(session):
        raise RuntimeError(f"Existing output/session: {session}")
    if not idle(gpu):
        return None
    log.parent.mkdir(parents=True, exist_ok=True)
    env = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
           "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2"]
    shell = (f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(env)} nice -n 10 "
             f"{shlex.join(command(setting, output, shard, smoke))} > {shlex.quote(str(log))} 2>&1")
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    print(f"Started {session} on physical GPU{gpu}", flush=True)
    return {"dataset": dataset, "shard": shard, "gpu": gpu, "session": session,
            "output": str(output), "log": str(log)}


def verify_dataset(root, dataset):
    output = root/dataset/"merged.json"
    inputs = [root/dataset/f"s{i}" for i in range(SHARDS[dataset])]
    if not output.exists():
        merge(argparse.Namespace(inputs=list(map(str, inputs)), output=str(output)))
    result = json.loads(output.read_text())
    old = json.loads(reference(dataset).read_text())
    if not result["coverage_verified"] or result["processed_images"] != old["processed_images"]:
        raise RuntimeError(f"Incomplete merged coverage: {dataset}")
    for field in ("global_sample_keys_sha256", "vocabulary", "checkpoints", "sample_keys"):
        if result["signature"][field] != old["signature"][field]:
            raise RuntimeError(f"Matched reference differs: {dataset}/{field}")
    if result["signature"]["implementation"] != IMPLEMENTATION:
        raise RuntimeError("Wrong candidate implementation")
    if any(count != 20 for counts in result["signature"]["vocabulary"]["counts"].values() for count in counts):
        raise RuntimeError("Candidate vocabulary is not exactly20 per class")
    keys, arrays = [], {}
    for path in inputs:
        with np.load(path/"per_image_confusions.npz", allow_pickle=False) as data:
            keys.extend(data["sample_keys"].tolist())
            for field in data.files:
                if field != "sample_keys":
                    arrays.setdefault(field, []).append(data[field])
    expected = result["signature"]["sample_keys"]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise RuntimeError("Per-image coverage differs")
    lookup = {key: index for index, key in enumerate(keys)}
    order = np.asarray([lookup[key] for key in expected])
    merged_arrays = {field: np.concatenate(values)[order] for field, values in arrays.items()}
    for protocol, metrics in result["metrics"].items():
        for method, metric in metrics.items():
            if not np.array_equal(merged_arrays[protocol+"__"+method].sum(0), metric["confusion_matrix"]):
                raise RuntimeError("Per-image sums do not reproduce metrics")
        for method in BASELINE_METHODS:
            if not np.array_equal(metrics[method]["confusion_matrix"], old["metrics"][protocol][method]["confusion_matrix"]):
                raise RuntimeError(f"Original matched baseline changed: {dataset}/{protocol}/{method}")
    np.savez_compressed(root/dataset/"per_image_confusions.npz", sample_keys=np.asarray(expected), **merged_arrays)
    result["exact_reference_geometry_confusion"] = True
    result["exact_reference_matched_baselines"] = list(BASELINE_METHODS)
    result["per_image_coverage_verified"] = True
    output.write_text(json.dumps(result, indent=2)+"\n")
    print("Verified", dataset, result["processed_images"], flush=True)
    return {protocol: {method: row["mean_iou_percent"] for method, row in metrics.items()}
            for protocol, metrics in result["metrics"].items()}


def main(args):
    root = args.root.resolve()
    root.relative_to((TOOL/"results").resolve())
    settings = {setting[0]: setting for setting in SETTINGS}
    if args.mode == "smoke":
        if root.exists():
            raise RuntimeError("Refusing an existing smoke root")
        if not idle(4):
            raise RuntimeError("GPU4 occupied; no smoke launched")
        root.mkdir(parents=True)
        launch(root, settings["udd5"], 0, 4, True)
        return
    smoke = read_json(args.smoke_result) if args.smoke_result else None
    if (not smoke or smoke.get("status") != "complete" or smoke["implementation"] != IMPLEMENTATION
            or smoke["capture_geometry_max_error"] != 0.
            or any(row["uniform_backbone_max_error"] != 0. or row["uniform_geometry_max_error"] != 0.
                   for row in smoke["checks"].values())):
        raise RuntimeError("Real-checkpoint FP32/BF16 smoke is not verified")
    if root.exists():
        raise RuntimeError("Refusing an existing full-suite root")
    for dataset, setting in settings.items():
        old = json.loads(reference(dataset).read_text())
        if not old["coverage_verified"] or old["processed_images"] != setting[4]:
            raise RuntimeError(f"Incomplete matched reference: {dataset}")
    root.mkdir(parents=True)
    protocol = {"implementation": IMPLEMENTATION, "primary": PRIMARY, "methods": METHODS,
                "config": ReacquisitionConfig().signature(), "physical_gpus": GPUS,
                "shards": SHARDS, "datasets": {dataset: setting[4] for dataset, setting in settings.items()},
                "smoke_result": str(args.smoke_result),
                "note": "Complete eight-domain candidate trial. Fixed20 vocabularies and one common rule. "
                "Corrected Vaihingen inputs. No per-domain model or coefficient selection. "
                "All domains are development data; matched VIP proxy is not the official complete system."}
    (root/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    first = [("udd5", 0), ("vdd", 0), ("potsdam", 0), ("oem", 0)]
    queue = first+[(dataset, shard) for dataset in ("flair1", "loveda", "landcoverai", "vdd", "potsdam", "oem", "vaihingen")
                   for shard in range(SHARDS[dataset]) if (dataset, shard) not in first]
    active, completed, failures, outcomes = {}, set(), {}, {}
    started = time.time()
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
                print("FAILURE", job["session"], failures[job["session"]], flush=True)
            del active[gpu]
        for dataset in settings:
            if dataset not in outcomes and all((dataset, shard) in completed for shard in range(SHARDS[dataset])):
                try:
                    outcomes[dataset] = verify_dataset(root, dataset)
                except Exception as error:
                    failures[dataset] = str(error)
                    outcomes[dataset] = {"verification_error": str(error)}
                    print("VERIFY FAILURE", dataset, error, flush=True)
        for gpu in GPUS:
            if queue and gpu not in active and idle(gpu):
                dataset, shard = queue[0]
                job = launch(root, settings[dataset], shard, gpu)
                if job is not None:
                    active[gpu] = job
                    queue.pop(0)
        state = {"status": "running" if queue or active else "failed" if failures else "complete",
                 "implementation": IMPLEMENTATION, "physical_gpus": GPUS,
                 "active": list(active.values()), "queued": queue, "completed_shards": sorted(completed),
                 "outcomes": outcomes, "failures": failures, "suite_elapsed_seconds": time.time()-started}
        temporary = root/"suite_status.tmp"
        temporary.write_text(json.dumps(state, indent=2)+"\n")
        temporary.replace(root/"suite_status.json")
        if queue or active:
            time.sleep(20)
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--smoke-result", type=Path)
    main(parser.parse_args())
