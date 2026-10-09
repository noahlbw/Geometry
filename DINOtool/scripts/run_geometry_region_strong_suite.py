"""One frozen complete candidate across eight domains, on physical GPUs4-7 only."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.geometry_region_strong import BASELINES, IMPLEMENTATION, METHODS, PRIMARY, SOURCE, StrongRegionConfig
from merge_geometry_semantic_innovation import main as merge
from run_region_semantic_suite_a800 import BASE, TOOL, PYTHON, THIRD, SETTINGS, idle
from run_sat_geometry_transport_suite import SHARDS, reference, exact_miou, read_json, alive


GPUS = (4, 5, 6, 7)
SCREEN = TOOL/"results/sat_geometry_transport_screen_20261002"
GATE = {"mean_comparators": ("Geometry", "VIPProxy_Two", "RegionFusion"),
        "maximum_geometry_protocol_loss_pp": 1., "focus_datasets": ("vdd", "potsdam"),
        "focus_comparators": ("Geometry", "RegionFusion"),
        "mean_protocol": "LoveDA D once; other seven domains once"}


def promotion_gate(outcomes, failures):
    if failures or set(outcomes) != {row[0] for row in SETTINGS}:
        return {"passed": False, "reason": "Incomplete/failed verified eight-domain screen.", "gate": GATE}
    values = {dataset: {key: {method: exact_miou(metric) for method, metric in group.items()}
                        for key, group in result["metrics"].items()} for dataset, result in outcomes.items()}
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
    if not row or row.get("status") != "complete" or row["implementation"] != IMPLEMENTATION:
        raise RuntimeError("Complete actual-checkpoint smoke is required.")
    checks = row["checks"]
    if (checks["trained_pool_replay_max_error"] > 3e-3 or checks["geometry_baseline_max_error"] != 0
            or checks["native_cache_max_error"] != 0 or not checks["finite_scores"] or not checks["weights_frozen"]
            or checks["observer_token_count"] != 729):
        raise RuntimeError("Actual-checkpoint invariants failed.")
    return row


def arguments(setting, output, shard, phase, root):
    dataset, data, vocab, _, _ = setting
    script = "smoke_geometry_region_strong.py" if phase == "smoke" else "eval_geometry_region_strong.py"
    args = [PYTHON, "-u", "scripts/"+script, "--dataset", dataset,
            "--dinov3-repo", str(TOOL/"dinov3_hub"), "--checkpoint-dir", str(BASE/"ckpt/DINO"),
            "--data-root", str(Path("/data/test/datasets")/data), "--vocabulary-config", str(TOOL/"configs"/vocab),
            "--upstream-root", str(BASE/"third_party/VIP_official_5bd25ee"), "--output-dir", str(output),
            "--num-shards", str(SHARDS[dataset] if phase == "full" else 1), "--shard-index", str(shard)]
    if phase == "screen":
        args += ["--source-diagnostic", str(root/f"{dataset}_samples.json")]
    return args


def launch(root, setting, shard, gpu, phase):
    if gpu not in GPUS:
        raise ValueError("Only physical GPUs4-7 are authorized.")
    dataset = setting[0]
    output = root/"smoke" if phase == "smoke" else root/dataset/f"s{shard}"
    log = root/"smoke.log" if phase == "smoke" else root/dataset/f"s{shard}.log"
    session = "grs02_smoke" if phase == "smoke" else f"grs02_{phase}_{dataset}_s{shard}"
    if output.exists() or alive(session):
        raise RuntimeError(f"Refusing existing output/session: {session}")
    if not idle(gpu):
        return None
    log.parent.mkdir(parents=True, exist_ok=True)
    env = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
           "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "TOKENIZERS_PARALLELISM=false"]
    shell = (f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(env)} nice -n 10 "
             f"{shlex.join(arguments(setting, output, shard, phase, root))} > {shlex.quote(str(log))} 2>&1")
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    print(f"Started {session}: physical GPU{gpu}", flush=True)
    return {"dataset": dataset, "shard": shard, "gpu": gpu, "session": session,
            "output": str(output), "log": str(log)}


def initialize(root, phase, smoke):
    root.resolve().relative_to((TOOL/"results").resolve())
    if root.exists():
        raise RuntimeError("Refusing existing suite root.")
    source = json.loads((SOURCE/"region_source_manifest.json").read_text())
    if verify_smoke(smoke)["source"] != source:
        raise RuntimeError("Source checkpoint differs from actual smoke.")
    manifests, settings = {}, []
    for dataset, _, _, _, total in SETTINGS:
        old = json.loads(reference(dataset).read_text())
        keys = old["signature"]["sample_keys"]
        if not old["coverage_verified"] or len(set(keys)) != total:
            raise RuntimeError(f"Incomplete matched reference: {dataset}")
        if phase == "screen":
            keys = json.loads((SCREEN/f"{dataset}_samples.json").read_text())["signature"]["samples"]
        if len(set(keys)) != len(keys) or not set(keys).issubset(old["signature"]["sample_keys"]):
            raise RuntimeError("Wrong fixed sample coverage.")
        manifests[dataset] = {"signature": {"samples": keys}, "seed": 20261002}
        settings.append({"dataset": dataset, "images": len(keys), "full_images": total})
    root.mkdir(parents=True)
    for dataset, manifest in manifests.items():
        (root/f"{dataset}_samples.json").write_text(json.dumps(manifest, indent=2)+"\n")
    protocol = {"implementation": IMPLEMENTATION, "primary": PRIMARY, "methods": METHODS,
                "config": StrongRegionConfig().signature(), "source": source, "phase": phase,
                "physical_gpus": GPUS, "gate": GATE, "datasets": settings, "smoke_result": str(smoke),
                "note": "Frozen two-encoder exploratory development. No target-label fitting. "
                        "Matched VIP proxy is not official full VIP; same-information fusion is included."}
    (root/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")


def verify_dataset(root, dataset, phase, *, shard_count=None, implementation=IMPLEMENTATION, semantic_source=SOURCE):
    count = shard_count if shard_count is not None else SHARDS[dataset] if phase == "full" else 1
    inputs = [root/dataset/f"s{index}" for index in range(count)]
    output = root/dataset/"merged.json"
    merge(argparse.Namespace(inputs=list(map(str, inputs)), output=str(output)))
    row, old = json.loads(output.read_text()), json.loads(reference(dataset).read_text())
    expected = json.loads((root/f"{dataset}_samples.json").read_text())["signature"]["samples"]
    if (not row["coverage_verified"] or row["signature"]["sample_keys"] != expected
            or row["processed_images"] != len(expected) or row["total_images"] != len(expected)
            or row["signature"]["implementation"] != implementation):
        raise RuntimeError("Incomplete coverage or signature mismatch.")
    for field in ("vocabulary", "checkpoints"):
        if row["signature"][field] != old["signature"][field]:
            raise RuntimeError(f"Matched input differs: {dataset}/{field}")
    if row["signature"]["gear"]["geometry"] != old["signature"]["gear"]["geometry"]:
        raise RuntimeError("Original Geometry configuration changed.")
    if semantic_source is not None and row["signature"]["gear"]["semantic_source"] != json.loads((semantic_source/"region_source_manifest.json").read_text()):
        raise RuntimeError("Semantic source checkpoint changed.")
    keys, arrays = [], {}
    for path in inputs:
        with np.load(path/"per_image_confusions.npz", allow_pickle=False) as data:
            keys.extend(data["sample_keys"].tolist())
            for field in data.files:
                if field != "sample_keys":
                    arrays.setdefault(field, []).append(data[field])
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise RuntimeError("Incomplete unique per-image keys.")
    lookup = {key: index for index, key in enumerate(keys)}
    order = np.asarray([lookup[key] for key in expected])
    arrays = {field: np.concatenate(parts)[order] for field, parts in arrays.items()}
    with np.load(reference(dataset).parent/"per_image_confusions.npz", allow_pickle=False) as ref:
        ref_lookup = {key: index for index, key in enumerate(ref["sample_keys"].tolist())}
        ref_order = np.asarray([ref_lookup[key] for key in expected])
        for key, group in row["metrics"].items():
            for method, metric in group.items():
                if not np.array_equal(arrays[key+"__"+method].sum(0), metric["confusion_matrix"]):
                    raise RuntimeError("Per-image matrices do not reproduce merged metrics.")
            for method in BASELINES:
                if not np.array_equal(arrays[key+"__"+method], ref[key+"__"+method][ref_order]):
                    raise RuntimeError(f"Matched baseline changed: {dataset}/{key}/{method}")
    np.savez_compressed(root/dataset/"per_image_confusions.npz", sample_keys=np.asarray(expected), **arrays)
    row["exact_reference_matched_baselines"] = list(BASELINES)
    row["per_image_coverage_verified"] = True
    output.write_text(json.dumps(row, indent=2)+"\n")
    print("Verified", dataset, len(expected), flush=True)
    return row


def run_suite(root, phase, smoke):
    initialize(root, phase, smoke)
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
                    and (Path(job["output"])/"per_image_confusions.npz").is_file()):
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
                if job:
                    active[gpu] = job
                    queue.pop(0)
        state = {"status": "running" if queue or active else "failed" if failures else "complete",
                 "implementation": IMPLEMENTATION, "phase": phase, "physical_gpus": GPUS,
                 "active": list(active.values()), "queued": queue, "completed_shards": sorted(completed),
                 "outcomes": {dataset: {key: {method: metric["mean_iou_percent"] for method, metric in group.items()}
                                            for key, group in row["metrics"].items()} if "metrics" in row else row
                              for dataset, row in outcomes.items()},
                 "failures": failures, "suite_elapsed_seconds": time.time()-started}
        temporary = root/"suite_status.tmp"
        temporary.write_text(json.dumps(state, indent=2)+"\n")
        temporary.replace(root/"suite_status.json")
        if queue or active:
            time.sleep(20)
    result = {"status": state["status"], "implementation": IMPLEMENTATION, "phase": phase,
              "decision": promotion_gate(outcomes, failures), "failures": failures,
              "suite_elapsed_seconds": time.time()-started}
    (root/"suite_results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    return result


def main(args):
    if args.phase == "smoke":
        args.root.resolve().relative_to((TOOL/"results").resolve())
        if args.root.exists() or not idle(args.gpu):
            raise RuntimeError("Smoke output exists or requested physical GPU is occupied.")
        args.root.mkdir(parents=True)
        launch(args.root, next(setting for setting in SETTINGS if setting[0] == "udd5"), 0, args.gpu, "smoke")
        return
    if args.phase == "full":
        prior = read_json(args.screen_result) if args.screen_result else None
        if not prior or prior["implementation"] != IMPLEMENTATION or not prior["decision"]["passed"]:
            raise RuntimeError("Unchanged candidate must pass the predeclared screen.")
    result = run_suite(args.root, args.phase, args.smoke_result)
    if args.promote_full_root:
        if args.phase != "screen":
            raise ValueError("Automatic promotion applies only to screen phase.")
        if result["decision"]["passed"]:
            print("Gate passed; starting unchanged full eight-domain suite.", flush=True)
            run_suite(args.promote_full_root, "full", args.smoke_result)
        else:
            print("Gate failed; no full rollout or parameter changes.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "screen", "full"), default="screen")
    parser.add_argument("--gpu", type=int, choices=GPUS, default=4)
    parser.add_argument("--smoke-result", type=Path)
    parser.add_argument("--screen-result", type=Path)
    parser.add_argument("--promote-full-root", type=Path)
    main(parser.parse_args())
