"""Frozen Geometry/native-donor coupling screen and gated full rollout."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.geometry_donor_budget import IMPLEMENTATION, METHODS, PRIMARY, DonorBudgetConfig
from run_region_semantic_suite_a800 import BASE, TOOL, PYTHON, THIRD, SETTINGS, idle
from run_sat_geometry_transport_suite import reference, exact_miou, read_json, alive
from run_geometry_region_strong_suite import verify_dataset


GPUS = (4, 5, 6, 7)
SCREEN = TOOL/"results/sat_geometry_transport_screen_20261002"
FULL_SHARDS = {"udd5": 4, "vdd": 4, "potsdam": 4, "oem": 4,
               "loveda": 4, "vaihingen": 1, "landcoverai": 4, "flair1": 4}
GATE = {"mean_comparators": ("Geometry", "VIPProxy_Two", "UniformDonorBudget", "MeanLogit_DonorBudget"),
        "maximum_geometry_protocol_loss_pp": 1., "focus_datasets": ("vdd", "potsdam"),
        "focus_comparators": ("Geometry", "MeanLogit_DonorBudget"),
        "mean_protocol": "LoveDA D once; seven other domains once"}


def counts(dataset, phase):
    return FULL_SHARDS[dataset] if phase == "full" else 4 if dataset == "udd5" else 1


def promotion_gate(outcomes, failures):
    if failures or set(outcomes) != {row[0] for row in SETTINGS}:
        return {"passed": False, "reason": "Incomplete/failed verified eight-domain screen.", "gate": GATE}
    values = {dataset: {key: {method: exact_miou(metric) for method, metric in group.items()}
                        for key, group in row["metrics"].items()} for dataset, row in outcomes.items()}
    means = {method: float(np.mean([group["D" if dataset == "loveda" else dataset][method]
                                   for dataset, group in values.items()])) for method in METHODS}
    checks = {"mean_vs_"+method: means[PRIMARY] > means[method] for method in GATE["mean_comparators"]}
    deltas = {}
    for dataset, protocols in values.items():
        deltas[dataset] = {}
        for key, group in protocols.items():
            deltas[dataset][key] = {method: group[PRIMARY]-value for method, value in group.items()}
            checks[f"retain_{dataset}_{key}"] = group[PRIMARY]-group["Geometry"] >= -1.
            if dataset in GATE["focus_datasets"]:
                for method in GATE["focus_comparators"]:
                    checks[f"focus_{dataset}_vs_{method}"] = group[PRIMARY] > group[method]
    return {"passed": all(checks.values()), "checks": checks, "mean_miou": means,
            "primary_deltas": deltas, "gate": GATE}


def verify_smoke(path):
    row = read_json(path)
    if not row or row.get("status") != "complete" or row["implementation"] != IMPLEMENTATION or not row["weights_frozen"]:
        raise RuntimeError("Complete actual-checkpoint frozen smoke required.")
    for check in row["checks"].values():
        if check["geometry_identity_max_error"] != 0. or check["native_cache_max_error"] != 0. or not check["finite_features"]:
            raise RuntimeError("Actual-checkpoint Geometry identity failed.")
    return row


def launch(root, setting, shard, gpu, phase):
    if gpu not in GPUS:
        raise ValueError("Only physical GPUs4-7 are authorized.")
    dataset, data, vocab, _, _ = setting
    output = root/"smoke" if phase == "smoke" else root/dataset/f"s{shard}"
    log = root/"smoke.log" if phase == "smoke" else root/dataset/f"s{shard}.log"
    session = "gdb02_smoke" if phase == "smoke" else f"gdb02_{phase}_{dataset}_s{shard}"
    if output.exists() or alive(session):
        raise RuntimeError(f"Existing output/session: {session}")
    if not idle(gpu):
        return None
    script = "smoke_geometry_donor_budget.py" if phase == "smoke" else "eval_geometry_donor_budget.py"
    args = [PYTHON, "-u", "scripts/"+script, "--dataset", dataset,
            "--dinov3-repo", str(TOOL/"dinov3_hub"), "--checkpoint-dir", str(BASE/"ckpt/DINO"),
            "--data-root", str(Path("/data/test/datasets")/data), "--vocabulary-config", str(TOOL/"configs"/vocab),
            "--upstream-root", str(BASE/"third_party/VIP_official_5bd25ee"), "--output-dir", str(output),
            "--num-shards", str(1 if phase == "smoke" else counts(dataset, phase)), "--shard-index", str(shard)]
    if phase == "screen":
        args += ["--source-diagnostic", str(root/f"{dataset}_samples.json")]
    log.parent.mkdir(parents=True, exist_ok=True)
    env = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
           "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2"]
    shell = f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(env)} nice -n 10 {shlex.join(args)} > {shlex.quote(str(log))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    print(f"Started {session}: physical GPU{gpu}", flush=True)
    return {"dataset": dataset, "shard": shard, "gpu": gpu, "session": session,
            "output": str(output), "log": str(log)}


def initialize(root, phase, smoke):
    root.resolve().relative_to((TOOL/"results").resolve())
    if root.exists():
        raise RuntimeError("Refusing existing suite root.")
    verify_smoke(smoke)
    manifests, rows = {}, []
    for dataset, _, _, _, total in SETTINGS:
        old = json.loads(reference(dataset).read_text())
        keys = old["signature"]["sample_keys"]
        if not old["coverage_verified"] or len(keys) != total or len(set(keys)) != total:
            raise RuntimeError("Incomplete authoritative reference.")
        if phase == "screen":
            keys = json.loads((SCREEN/f"{dataset}_samples.json").read_text())["signature"]["samples"]
        manifests[dataset] = {"signature": {"samples": keys}, "seed": 20261002}
        rows.append({"dataset": dataset, "images": len(keys), "full_images": total, "shards": counts(dataset, phase)})
    root.mkdir(parents=True)
    for dataset, manifest in manifests.items():
        (root/f"{dataset}_samples.json").write_text(json.dumps(manifest, indent=2)+"\n")
    protocol = {"implementation": IMPLEMENTATION, "primary": PRIMARY, "methods": METHODS,
                "config": DonorBudgetConfig().signature(), "phase": phase, "physical_gpus": GPUS,
                "gate": GATE, "datasets": rows, "smoke_result": str(smoke),
                "note": "Frozen complete single-encoder exploratory coupling, no target-label fitting; "
                        "native visual donor marginals, not class quotas. Matched VIP is a proxy adaptation."}
    (root/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")


def run_suite(root, phase, smoke):
    initialize(root, phase, smoke)
    settings = {row[0]: row for row in SETTINGS}
    queue = [(dataset, shard) for dataset in ("vdd", "potsdam", "oem", "loveda", "flair1", "landcoverai", "vaihingen", "udd5")
             for shard in range(counts(dataset, phase))]
    active, completed, outcomes, failures = {}, set(), {}, {}
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
            if dataset not in outcomes and all((dataset, shard) in completed for shard in range(counts(dataset, phase))):
                try:
                    row = verify_dataset(root, dataset, phase, shard_count=counts(dataset, phase),
                                         implementation=IMPLEMENTATION, semantic_source=None)
                    expected = {"geometry": json.loads(reference(dataset).read_text())["signature"]["gear"]["geometry"],
                                "primary": PRIMARY, **DonorBudgetConfig().signature()}
                    if row["signature"]["gear"] != expected:
                        raise RuntimeError("Frozen donor-budget rule changed.")
                    outcomes[dataset] = row
                except Exception as error:
                    failures[dataset] = str(error)
                    outcomes[dataset] = {"verification_error": str(error)}
                    print("VERIFY FAILURE", dataset, error, flush=True)
        if failures:
            queue.clear()
        for gpu in GPUS:
            if queue and gpu not in active and idle(gpu):
                dataset, shard = queue[0]
                job = launch(root, settings[dataset], shard, gpu, phase)
                if job:
                    active[gpu] = job; queue.pop(0)
        state = {"status": "running" if queue or active else "failed" if failures else "complete",
                 "implementation": IMPLEMENTATION, "phase": phase, "physical_gpus": GPUS,
                 "active": list(active.values()), "queued": queue, "completed_shards": sorted(completed),
                 "outcomes": {dataset: {key: {method: metric["mean_iou_percent"] for method, metric in group.items()}
                                            for key, group in row["metrics"].items()} if "metrics" in row else row
                              for dataset, row in outcomes.items()},
                 "failures": failures, "suite_elapsed_seconds": time.time()-started}
        temp = root/"suite_status.tmp"; temp.write_text(json.dumps(state, indent=2)+"\n"); temp.replace(root/"suite_status.json")
        if queue or active:
            time.sleep(20)
    result = {"status": state["status"], "implementation": IMPLEMENTATION, "phase": phase,
              "decision": promotion_gate(outcomes, failures), "failures": failures,
              "suite_elapsed_seconds": time.time()-started}
    (root/"suite_results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    return result


def main(args):
    root = args.root.resolve()
    if args.phase == "smoke":
        root.relative_to((TOOL/"results").resolve())
        if root.exists() or not idle(args.gpu):
            raise RuntimeError("Smoke root exists or GPU occupied.")
        root.mkdir(parents=True)
        launch(root, next(row for row in SETTINGS if row[0] == "udd5"), 0, args.gpu, "smoke")
        return
    if args.phase == "full":
        prior = read_json(args.screen_result) if args.screen_result else None
        if not prior or prior["implementation"] != IMPLEMENTATION or not prior["decision"]["passed"]:
            raise RuntimeError("Unchanged candidate must pass its predeclared gate before full rollout.")
    result = run_suite(root, args.phase, args.smoke_result)
    if args.promote_full_root and args.phase == "screen":
        if result["decision"]["passed"]:
            run_suite(args.promote_full_root.resolve(), "full", args.smoke_result)
        else:
            print("Gate failed. No full rollout or target-label tuning.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "screen", "full"), default="screen")
    parser.add_argument("--gpu", type=int, choices=GPUS, default=6)
    parser.add_argument("--smoke-result", type=Path)
    parser.add_argument("--screen-result", type=Path)
    parser.add_argument("--promote-full-root", type=Path)
    main(parser.parse_args())
