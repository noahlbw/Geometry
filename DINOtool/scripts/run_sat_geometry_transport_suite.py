"""Fixed eight-domain SAT/Geometry candidate, restricted to physical GPUs4-7."""
import argparse
import json
from pathlib import Path
import random
import shlex
import subprocess
import time

import numpy as np

from dinotool.sat_geometry_transport import IMPLEMENTATION, METHODS, PRIMARY, SATTransportConfig
from merge_geometry_semantic_innovation import main as merge
from run_region_semantic_suite_a800 import BASE, TOOL, PYTHON, THIRD, SETTINGS, idle


GPUS = (4, 5, 6, 7)
SEED = 20261002
SHARDS = {"udd5": 1, "vdd": 2, "potsdam": 2, "oem": 2, "loveda": 4,
          "vaihingen": 1, "landcoverai": 4, "flair1": 4}
BASELINES = ("Geometry", "SCLIP_Two", "VIPProxy_Two")
GATE = {"mean_comparators": ("Geometry", "VIPProxy_Two", "SAT_UniformTransport", "MeanLogit_SATTransport"),
        "maximum_geometry_protocol_loss_pp": 1.,
        "focus_datasets": ("vdd", "potsdam"),
        "focus_comparators": ("Geometry", "MeanLogit_SATTransport"),
        "mean_protocol": "LoveDA D once; all seven other datasets once"}


def reference(dataset):
    return TOOL/"results/geometry_matched_readout_full_20261001"/dataset/"merged.json"


def read_json(path):
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except json.JSONDecodeError:
        return None


def alive(session):
    return subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0


def exact_miou(metric):
    matrix = np.asarray(metric["confusion_matrix"], np.float64)
    union = matrix.sum(0)+matrix.sum(1)-matrix.diagonal()
    return float(100*np.mean(matrix.diagonal()[union > 0]/union[union > 0]))


def promotion_gate(outcomes, failures):
    if failures or set(outcomes) != {setting[0] for setting in SETTINGS}:
        return {"passed": False, "reason": "Incomplete or failed verified eight-domain screen.", "gate": GATE}
    values = {dataset: {protocol: {method: exact_miou(metric) for method, metric in metrics.items()}
                        for protocol, metrics in result["metrics"].items()}
              for dataset, result in outcomes.items()}
    means = {method: float(np.mean([group["D" if dataset == "loveda" else dataset][method]
                                   for dataset, group in values.items()])) for method in METHODS}
    checks = {"mean_vs_"+method: means[PRIMARY] > means[method] for method in GATE["mean_comparators"]}
    deltas = {}
    for dataset, protocols in values.items():
        deltas[dataset] = {}
        for protocol, metrics in protocols.items():
            delta = metrics[PRIMARY]-metrics["Geometry"]
            deltas[dataset][protocol] = {method: metrics[PRIMARY]-value for method, value in metrics.items()}
            checks[f"retain_{dataset}_{protocol}"] = delta >= -GATE["maximum_geometry_protocol_loss_pp"]
            if dataset in GATE["focus_datasets"]:
                for method in GATE["focus_comparators"]:
                    checks[f"focus_{dataset}_vs_{method}"] = metrics[PRIMARY] > metrics[method]
    return {"passed": all(checks.values()), "checks": checks, "mean_miou": means,
            "primary_deltas": deltas, "gate": GATE}


def command(setting, output, shard, phase, root):
    dataset, data, vocab, _, _ = setting
    script = {"smoke": "smoke_sat_geometry_transport.py",
              "benchmark": "benchmark_sat_geometry_transport.py"}.get(phase, "eval_sat_geometry_transport.py")
    args = [PYTHON, "-u", "scripts/"+script, "--dataset", dataset,
            "--dinov3-repo", str(TOOL/"dinov3_hub"), "--checkpoint-dir", str(BASE/"ckpt/DINO"),
            "--data-root", str(Path("/data/test/datasets")/data),
            "--vocabulary-config", str(TOOL/"configs"/vocab),
            "--upstream-root", str(BASE/"third_party/VIP_official_5bd25ee"),
            "--output-dir", str(output), "--num-shards", str(SHARDS[dataset] if phase == "full" else 1),
            "--shard-index", str(shard)]
    if phase == "screen":
        args += ["--source-diagnostic", str(root/f"{dataset}_samples.json")]
    return args


def launch(root, setting, shard, gpu, phase):
    if gpu not in GPUS:
        raise ValueError("Only physical GPUs4-7 are authorized.")
    dataset = setting[0]
    diagnostic = phase in ("smoke", "benchmark")
    output = root/phase if diagnostic else root/dataset/f"s{shard}"
    log = root/(phase+".log") if diagnostic else root/dataset/f"s{shard}.log"
    session = "gst02_"+phase if diagnostic else f"gst02_{phase}_{dataset}_s{shard}"
    if output.exists() or alive(session):
        raise RuntimeError(f"Refusing existing output/session: {session}")
    if not idle(gpu):
        return None
    log.parent.mkdir(parents=True, exist_ok=True)
    env = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
           "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2"]
    shell = (f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(env)} nice -n 10 "
             f"{shlex.join(command(setting, output, shard, phase, root))} > {shlex.quote(str(log))} 2>&1")
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    print(f"Started {session}: physical GPU{gpu}", flush=True)
    return {"dataset": dataset, "shard": shard, "gpu": gpu, "session": session,
            "output": str(output), "log": str(log)}


def verify_smoke(path):
    row = read_json(path)
    if not row or row.get("status") != "complete" or row["implementation"] != IMPLEMENTATION:
        raise RuntimeError("Complete actual-checkpoint smoke is required.")
    for mode in ("fp32", "bf16"):
        check = row["checks"][mode]
        if (not check["finite_features"] or any(check[field] != 0. for field in (
                "same_source_head_max_error", "matched_baseline_max_error", "cached_backbone_max_error",
                "geometry_repeat_max_error", "padding_state_max_error"))):
            raise RuntimeError(f"Actual-checkpoint smoke invariants failed: {mode}")


def initialize(root, phase, smoke_result):
    root.resolve().relative_to((TOOL/"results").resolve())
    if root.exists():
        raise RuntimeError("Refusing existing suite root.")
    verify_smoke(smoke_result)
    manifests, rows = {}, []
    for dataset, _, _, _, total in SETTINGS:
        old = json.loads(reference(dataset).read_text())
        keys = old["signature"]["sample_keys"]
        if (not old["coverage_verified"] or len(keys) != total or len(set(keys)) != total
                or not (reference(dataset).parent/"per_image_confusions.npz").is_file()):
            raise RuntimeError(f"Incomplete matched reference: {dataset}")
        if phase == "screen":
            count = 40 if dataset == "udd5" else 8
            chosen = set(random.Random(SEED).sample(keys, count))
            keys = [key for key in keys if key in chosen]
        manifests[dataset] = {"signature": {"samples": keys}, "seed": SEED}
        rows.append({"dataset": dataset, "images": len(keys), "full_images": total})
    root.mkdir(parents=True)
    for dataset, manifest in manifests.items():
        (root/f"{dataset}_samples.json").write_text(json.dumps(manifest, indent=2)+"\n")
    protocol = {"implementation": IMPLEMENTATION, "primary": PRIMARY, "methods": METHODS,
                "config": SATTransportConfig().signature(), "phase": phase, "seed": SEED,
                "physical_gpus": GPUS, "gate": GATE, "datasets": rows,
                "smoke_result": str(smoke_result),
                "note": "Frozen fixed20 eight-domain exploratory development; no target-label fitting. "
                        "SAT/LVD rotations are image-only inference states. Matched VIP proxy is not the official system."}
    (root/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")


def verify_dataset(root, dataset, phase):
    count = SHARDS[dataset] if phase == "full" else 1
    inputs = [root/dataset/f"s{i}" for i in range(count)]
    output = root/dataset/"merged.json"
    merge(argparse.Namespace(inputs=list(map(str, inputs)), output=str(output)))
    result, old = json.loads(output.read_text()), json.loads(reference(dataset).read_text())
    expected = json.loads((root/f"{dataset}_samples.json").read_text())["signature"]["samples"]
    if (not result["coverage_verified"] or result["signature"]["sample_keys"] != expected
            or result["processed_images"] != len(expected) or result["total_images"] != len(expected)
            or result["signature"]["implementation"] != IMPLEMENTATION):
        raise RuntimeError(f"Coverage/signature differs: {dataset}")
    for field in ("vocabulary", "checkpoints"):
        if result["signature"][field] != old["signature"][field]:
            raise RuntimeError(f"Matched input differs: {dataset}/{field}")
    if result["signature"]["gear"] != {"geometry": old["signature"]["gear"]["geometry"],
                                           "primary": PRIMARY, **SATTransportConfig().signature()}:
        raise RuntimeError(f"Changed Geometry/SAT config: {dataset}")
    if any(count != 20 for counts in result["signature"]["vocabulary"]["counts"].values() for count in counts):
        raise RuntimeError("Vocabulary must have exactly20 aliases per class.")
    keys, arrays = [], {}
    for path in inputs:
        with np.load(path/"per_image_confusions.npz", allow_pickle=False) as data:
            keys.extend(data["sample_keys"].tolist())
            for field in data.files:
                if field != "sample_keys":
                    arrays.setdefault(field, []).append(data[field])
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise RuntimeError("Incomplete per-image coverage.")
    lookup = {key: index for index, key in enumerate(keys)}
    order = np.asarray([lookup[key] for key in expected])
    arrays = {field: np.concatenate(values)[order] for field, values in arrays.items()}
    with np.load(reference(dataset).parent/"per_image_confusions.npz", allow_pickle=False) as ref:
        ref_lookup = {key: index for index, key in enumerate(ref["sample_keys"].tolist())}
        ref_order = np.asarray([ref_lookup[key] for key in expected])
        for protocol, metrics in result["metrics"].items():
            for method, metric in metrics.items():
                if not np.array_equal(arrays[protocol+"__"+method].sum(0), metric["confusion_matrix"]):
                    raise RuntimeError("Per-image sums do not reproduce metrics.")
            for method in BASELINES:
                if not np.array_equal(arrays[protocol+"__"+method], ref[protocol+"__"+method][ref_order]):
                    raise RuntimeError(f"Original per-image baseline changed: {dataset}/{protocol}/{method}")
    np.savez_compressed(root/dataset/"per_image_confusions.npz", sample_keys=np.asarray(expected), **arrays)
    result["exact_reference_matched_baselines"] = list(BASELINES)
    result["per_image_coverage_verified"] = True
    output.write_text(json.dumps(result, indent=2)+"\n")
    print("Verified", dataset, len(expected), flush=True)
    return result


def run_suite(root, phase, smoke_result):
    initialize(root, phase, smoke_result)
    settings = {setting[0]: setting for setting in SETTINGS}
    first = [("udd5", 0), ("vdd", 0), ("potsdam", 0), ("oem", 0)]
    queue = first+[(dataset, shard) for dataset in ("flair1", "loveda", "landcoverai", "vdd", "potsdam", "oem", "vaihingen")
                   for shard in range(SHARDS[dataset] if phase == "full" else 1) if (dataset, shard) not in first]
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
            count = SHARDS[dataset] if phase == "full" else 1
            if dataset not in outcomes and all((dataset, shard) in completed for shard in range(count)):
                try:
                    outcomes[dataset] = verify_dataset(root, dataset, phase)
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
                if job is not None:
                    active[gpu] = job
                    queue.pop(0)
        state = {"status": "running" if queue or active else "failed" if failures else "complete",
                 "implementation": IMPLEMENTATION, "phase": phase, "physical_gpus": GPUS,
                 "active": list(active.values()), "queued": queue, "completed_shards": sorted(completed),
                 "outcomes": {dataset: {protocol: {method: metric["mean_iou_percent"] for method, metric in group.items()}
                                           for protocol, group in result["metrics"].items()}
                              if "metrics" in result else result for dataset, result in outcomes.items()},
                 "failures": failures, "suite_elapsed_seconds": time.time()-started}
        temporary = root/"suite_status.tmp"
        temporary.write_text(json.dumps(state, indent=2)+"\n")
        temporary.replace(root/"suite_status.json")
        if queue or active:
            time.sleep(20)
    decision = promotion_gate(outcomes, failures)
    result = {"status": state["status"], "phase": phase, "implementation": IMPLEMENTATION,
              "decision": decision, "failures": failures, "suite_elapsed_seconds": time.time()-started}
    (root/"suite_results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    return result


def main(args):
    settings = {setting[0]: setting for setting in SETTINGS}
    if args.phase in ("smoke", "benchmark"):
        args.root.resolve().relative_to((TOOL/"results").resolve())
        if args.root.exists() or not idle(args.gpu):
            raise RuntimeError(f"Diagnostic root exists or physical GPU{args.gpu} occupied.")
        if args.phase == "benchmark":
            verify_smoke(args.smoke_result)
        args.root.mkdir(parents=True)
        launch(args.root, settings["udd5" if args.phase == "smoke" else "loveda"], 0, args.gpu, args.phase)
        return
    if args.phase == "full":
        prior = read_json(args.screen_result) if args.screen_result else None
        if not prior or prior["implementation"] != IMPLEMENTATION or not prior["decision"]["passed"]:
            raise RuntimeError("The unchanged candidate must pass the predeclared screen before full rollout.")
    result = run_suite(args.root, args.phase, args.smoke_result)
    if args.promote_full_root:
        if args.phase != "screen":
            raise ValueError("Automatic promotion is only valid for the screen phase.")
        if result["decision"]["passed"]:
            print("Predeclared gate passed. Starting unchanged full eight-domain suite.", flush=True)
            run_suite(args.promote_full_root, "full", args.smoke_result)
        else:
            print("Predeclared gate failed. No full rollout or parameter changes.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "benchmark", "screen", "full"), default="screen")
    parser.add_argument("--gpu", type=int, choices=GPUS, default=4)
    parser.add_argument("--smoke-result", type=Path)
    parser.add_argument("--screen-result", type=Path)
    parser.add_argument("--promote-full-root", type=Path)
    main(parser.parse_args())
