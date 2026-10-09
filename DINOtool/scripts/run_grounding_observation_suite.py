"""Frozen localized-observation diagnostic on physical A800 GPUs4-7 only."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from dinotool.grounded_local_observer import IMPLEMENTATION, GroundingConfig
from run_region_semantic_suite_a800 import BASE, TOOL, PYTHON, THIRD, SETTINGS, idle
from run_sat_geometry_transport_suite import read_json, alive


GPUS = (4, 5, 6, 7)
SOURCE = BASE/"ckpt/GroundingDINO_tiny_a2bb814"
SAMPLES = TOOL/"results/sat_geometry_transport_screen_20261002"


def launch(root, setting, gpu, smoke=False):
    if gpu not in GPUS or not idle(gpu):
        return None
    dataset, data, vocab, _, _ = setting
    output = root/"smoke" if smoke else root/dataset
    session = "ggd02_smoke" if smoke else "ggd02_"+dataset
    log = root/"smoke.log" if smoke else root/f"{dataset}.log"
    if output.exists() or alive(session):
        raise RuntimeError("Refusing existing diagnostic output/session.")
    args = [PYTHON, "-u", "scripts/diagnose_grounding_observations.py", "--dataset", dataset,
        "--dinov3-repo", str(TOOL/"dinov3_hub"), "--checkpoint-dir", str(BASE/"ckpt/DINO"),
        "--data-root", str(Path("/data/test/datasets")/data), "--vocabulary-config", str(TOOL/"configs"/vocab),
        "--grounding-source", str(SOURCE), "--source-diagnostic", str(SAMPLES/f"{dataset}_samples.json"),
        "--output-dir", str(output)]
    if smoke:
        args.append("--smoke")
    environment = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
        "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "TOKENIZERS_PARALLELISM=false"]
    shell = f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(environment)} nice -n 10 {shlex.join(args)} > {shlex.quote(str(log))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    print(f"Started {session} on physical GPU{gpu}", flush=True)
    return {"dataset": dataset, "gpu": gpu, "session": session, "output": str(output), "log": str(log)}


def verify(output, dataset):
    result = read_json(output/"results.json")
    status = read_json(output/"observation_status.json")
    fixed = read_json(SAMPLES/f"{dataset}_samples.json")["signature"]["samples"]
    if (not result or result["status"] != "complete" or result["processed_images"] != len(fixed)
            or result["total_images"] != len(fixed) or result["implementation"] != IMPLEMENTATION
            or not status or status["status"] != "collected" or status["target_masks_loaded"]):
        raise RuntimeError("Incomplete or label-contaminated observation diagnostic.")
    signature = result["signature"]
    if (signature["sample_keys"] != fixed or len(fixed) != len(set(fixed))
            or not signature["weights_frozen"] or signature["target_masks_loaded_during_observation"]
            or signature["grounding"] != GroundingConfig().signature()
            or any(count != 20 for bank in signature["vocabularies"].values() for count in bank["counts"])):
        raise RuntimeError("Fixed sample/weight/alias/source rule changed.")
    manifest = read_json(output/"raw_manifest.json")
    if set(row["sample_key"] for row in manifest) != set(fixed) or not all((output/row["file"]).is_file() for row in manifest):
        raise RuntimeError("Incomplete raw observation coverage.")
    result["coverage_verified"] = True
    result["observation_cost"] = status
    (output/"verified.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def run(root, smoke_result):
    signature = read_json(smoke_result.parent/"signature.json")
    smoke = read_json(smoke_result)
    if (not smoke or smoke["status"] != "collected" or smoke["target_masks_loaded"]
            or not signature or not signature["weights_frozen"] or signature["grounding"] != GroundingConfig().signature()):
        raise RuntimeError("Actual-checkpoint mask-free smoke required.")
    root.mkdir(parents=True)
    (root/"protocol.json").write_text(json.dumps({"implementation": IMPLEMENTATION,
        "physical_gpus": GPUS, "grounding": GroundingConfig().signature(), "smoke_result": str(smoke_result),
        "source_gate": "BoxMean beneficial>harmful on every protocol; no retrospective threshold/score selection",
        "full_launch": False, "note": "Diagnostic observations only, not full-image dataset mIoU"}, indent=2)+"\n")
    queue = list(SETTINGS)
    active, outcomes, failures = {}, {}, {}
    started = time.time()
    while queue or active:
        for gpu, job in list(active.items()):
            if alive(job["session"]):
                continue
            try:
                outcomes[job["dataset"]] = verify(Path(job["output"]), job["dataset"])
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
            "outcomes": {dataset: {"source_gate_passed": row["source_gate_passed"],
                "window_miou": {key: {method: metric["mean_iou_percent"] for method, metric in group.items()}
                                 for key, group in row["metrics"].items()}}
                         for dataset, row in outcomes.items()}, "elapsed_seconds": time.time()-started}
        temp = root/"suite_status.tmp"
        temp.write_text(json.dumps(state, indent=2)+"\n")
        temp.replace(root/"suite_status.json")
        if queue or active:
            time.sleep(20)
    state["source_gate_passed"] = not failures and len(outcomes) == 8 and all(row["source_gate_passed"] for row in outcomes.values())
    state["full_evaluation_launched"] = False
    (root/"suite_results.json").write_text(json.dumps(state, indent=2)+"\n")
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-result", type=Path)
    args = parser.parse_args()
    args.root.resolve().relative_to((TOOL/"results").resolve())
    if args.root.exists():
        raise RuntimeError("Refusing existing suite root.")
    if args.smoke:
        if not idle(4):
            raise RuntimeError("GPU4 occupied, no launch.")
        args.root.mkdir(parents=True)
        launch(args.root, SETTINGS[0], 4, True)
    else:
        run(args.root, args.smoke_result)
