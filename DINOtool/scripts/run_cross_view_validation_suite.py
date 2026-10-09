"""Queue the fixed96-image native-wide candidate on idle physical GPUs0-3."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.cross_view_validation import CONFIG, IMPLEMENTATION, METHODS, PRIMARY
from merge_geometry_semantic_innovation import main as merge
from run_region_semantic_suite_a800 import BASE, TOOL, PYTHON, THIRD, SETTINGS, idle
from run_sat_geometry_transport_suite import alive, exact_miou, read_json, reference


GPUS = (0, 1, 2, 3)
SOURCE_SCREEN = TOOL/"results/sat_geometry_transport_screen_20261002"


def launch(root, setting, gpu, smoke=False):
    if gpu not in GPUS or not idle(gpu):
        raise RuntimeError(f"GPU{gpu} is not available.")
    dataset, data, vocabulary, _, _ = setting
    output = root/"smoke" if smoke else root/dataset/"s0"
    log = root/"smoke.log" if smoke else root/dataset/"s0.log"
    session = "gcv04_smoke" if smoke else "gcv04_"+dataset
    if output.exists() or log.exists() or alive(session):
        raise RuntimeError("Existing run/output: "+session)
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, "-u", "scripts/eval_cross_view_validation.py", "--dataset", dataset,
               "--dinov3-repo", str(TOOL/"dinov3_hub"), "--checkpoint-dir", str(BASE/"ckpt/DINO"),
               "--data-root", str(Path("/data/test/datasets")/data),
               "--vocabulary-config", str(TOOL/"configs"/vocabulary),
               "--upstream-root", str(BASE/"third_party/VIP_official_5bd25ee"),
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
        chosen = read_json(SOURCE_SCREEN/(dataset+"_samples.json"))
        keys = chosen["signature"]["samples"]
        count = 40 if dataset == "udd5" else 8
        if not full["coverage_verified"] or len(keys) != count or len(set(keys)) != count or not set(keys).issubset(full["signature"]["sample_keys"]):
            raise RuntimeError("Wrong fixed development sequence: "+dataset)
        manifests[dataset] = {"signature": {"samples": keys}}
    root.mkdir(parents=True)
    for dataset, manifest in manifests.items():
        (root/(dataset+"_samples.json")).write_text(json.dumps(manifest, indent=2)+"\n")
    (root/"protocol.json").write_text(json.dumps({"implementation": IMPLEMENTATION, "config": CONFIG,
        "primary": PRIMARY, "methods": METHODS, "physical_gpus": GPUS,
        "datasets": {key: len(value["signature"]["samples"]) for key, value in manifests.items()},
        "phase": "fixed96 complete-image exploratory development screen",
        "decision": "mean above Geometry, same-source mean and ungated coupling; VDD/Potsdam above Geometry and ungated; no protocol loss above1pp",
        "note": "All20; no label fitting; corrected IRRG Vaihingen; LandCover.ai substitutes for unlabeled iSAID."}, indent=2)+"\n")


def verify(root, dataset):
    path = root/dataset/"merged.json"
    merge(argparse.Namespace(inputs=[str(root/dataset/"s0")], output=str(path)))
    row, old = read_json(path), read_json(reference(dataset))
    expected = read_json(root/(dataset+"_samples.json"))["signature"]["samples"]
    if not row["coverage_verified"] or row["signature"]["sample_keys"] != expected or row["processed_images"] != len(expected) or row["total_images"] != len(expected) or row["signature"]["implementation"] != IMPLEMENTATION:
        raise RuntimeError("Wrong complete coverage/signature.")
    for field in ("vocabulary", "checkpoints"):
        if row["signature"][field] != old["signature"][field]:
            raise RuntimeError("Input identity changed: "+field)
    if row["signature"]["gear"]["geometry"] != old["signature"]["gear"]["geometry"]:
        raise RuntimeError("Geometry configuration changed.")
    with np.load(root/dataset/"s0/per_image_confusions.npz", allow_pickle=False) as current, np.load(reference(dataset).parent/"per_image_confusions.npz", allow_pickle=False) as historical:
        lookup = {key: index for index, key in enumerate(historical["sample_keys"].tolist())}
        order = [lookup[key] for key in expected]
        if current["sample_keys"].tolist() != expected:
            raise RuntimeError("Per-image coverage changed.")
        for key, group in row["metrics"].items():
            if not np.array_equal(current[key+"__Geometry"], historical[key+"__Geometry"][order]):
                raise RuntimeError("Original Geometry per-image confusions changed.")
            for method, metric in group.items():
                if not np.array_equal(current[key+"__"+method].sum(0), metric["confusion_matrix"]):
                    raise RuntimeError("Per-image confusion sum failed.")
    row["exact_geometry_per_image_replay"] = True
    path.write_text(json.dumps(row, indent=2)+"\n")
    return row


def decision(outcomes, failures):
    if failures or len(outcomes) != 8:
        return {"passed": False, "reason": "Incomplete/failed screen."}
    values = {dataset: {key: {method: exact_miou(metric) for method, metric in group.items()}
                           for key, group in row["metrics"].items()} for dataset, row in outcomes.items()}
    means = {method: float(np.mean([group["D" if dataset == "loveda" else dataset][method]
                                   for dataset, group in values.items()])) for method in METHODS}
    checks = {"mean_vs_"+method: means[PRIMARY] > means[method]
              for method in ("Geometry", "MeanLogit_Native", "Anchored_Native")}
    for dataset, protocols in values.items():
        for key, scores in protocols.items():
            checks["retain_"+dataset+"_"+key] = scores[PRIMARY]-scores["Geometry"] >= -1.
            if dataset in ("vdd", "potsdam"):
                for method in ("Geometry", "Anchored_Native"):
                    checks[dataset+"_vs_"+method] = scores[PRIMARY] > scores[method]
    return {"passed": all(checks.values()), "mean_miou": means, "values": values, "checks": checks}


def run(root):
    initialize(root)
    started = time.time()
    free = next((gpu for gpu in GPUS if idle(gpu)), None)
    if free is None:
        raise RuntimeError("No authorized idle GPU for initial smoke.")
    smoke = launch(root, next(setting for setting in SETTINGS if setting[0] == "oem"), free, True)
    while alive(smoke["session"]):
        time.sleep(5)
    smoke_result = read_json(Path(smoke["output"])/"results.json")
    if not smoke_result or smoke_result.get("status") != "complete":
        error = Path(smoke["log"]).read_text()[-6000:]
        (root/"suite_results.json").write_text(json.dumps({"status": "failed", "failures": {"smoke": error}}, indent=2)+"\n")
        raise RuntimeError(error)
    queue = [next(setting for setting in SETTINGS if setting[0] == dataset)
             for dataset in ("udd5", "vdd", "potsdam", "oem", "loveda", "flair1", "vaihingen", "landcoverai")]
    active, outcomes, failures = {}, {}, {}
    while queue or active:
        for gpu, job in list(active.items()):
            if alive(job["session"]):
                continue
            result = read_json(Path(job["output"])/"results.json")
            try:
                if not result or result.get("status") != "complete" or result["processed_images"] != result["total_images"]:
                    raise RuntimeError(Path(job["log"]).read_text()[-6000:])
                outcomes[job["dataset"]] = verify(root, job["dataset"])
            except Exception as error:
                failures[job["dataset"]] = str(error)
                print("FAILURE", job["dataset"], error, flush=True)
            del active[gpu]
        if failures:
            queue.clear()
        for gpu in GPUS:
            if queue and gpu not in active and idle(gpu):
                active[gpu] = launch(root, queue.pop(0), gpu)
        state = {"status": "running" if queue or active else "failed" if failures else "complete",
                 "implementation": IMPLEMENTATION, "active": list(active.values()), "queued": [s[0] for s in queue],
                 "completed": list(outcomes), "failures": failures, "elapsed_seconds": time.time()-started}
        temp = root/"suite_status.tmp"
        temp.write_text(json.dumps(state, indent=2)+"\n")
        temp.replace(root/"suite_status.json")
        if queue or active:
            time.sleep(10)
    final = {**state, "decision": decision(outcomes, failures)}
    (root/"suite_results.json").write_text(json.dumps(final, indent=2)+"\n")
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=TOOL/"results/geometry_native_cross_view_screen_20261004")
    run(parser.parse_args().root)
