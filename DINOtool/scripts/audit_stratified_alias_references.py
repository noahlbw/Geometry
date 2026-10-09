"""Post-selection labeled audit of frozen reference pools, never selector tuning."""
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
import torch

from dinotool.prompts import load_class_specs
from dinotool.stratified_soft_alias import IMPLEMENTATION
from eval_bounded_alias_anchored import make_models
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import parse_args
from eval_stratified_soft_alias import CONFIG, image_references, prepare_wide
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, SETTINGS, idle
from run_sat_geometry_transport_suite import alive


@torch.inference_mode()
def audit(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Existing audit output.")
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    fixed = json.loads(args.source_diagnostic.read_text())["signature"]["samples"]
    lookup = {sample.key: sample for sample in samples}
    geometry, banks, vip, queries, _ = make_models(args, specs)
    counts = {key: np.zeros((bank.class_count,) * 2, np.int64) for key, bank in banks.items()}
    unique_counts = {key: np.zeros_like(value) for key, value in counts.items()}
    weighted = {key: np.zeros((bank.class_count,) * 2, np.float64) for key, bank in banks.items()}
    alias_counts = {key: np.zeros((len(bank.alias_names), bank.class_count, bank.class_count), np.int64) for key, bank in banks.items()}
    per_image = []
    started = time.perf_counter()
    for sample_key in fixed:
        sample = lookup[sample_key]
        image = load_image(sample.image_path if args.dataset == "loveda" else sample)
        broad, without, _, _ = prepare_wide(image, vip, queries)
        references, _, _ = image_references(image, geometry, banks, broad, without)
        image_row = {"sample_key": sample_key, "protocols": {}}
        # Masks are loaded only after the unchanged image-only selection is complete.
        for key, bank in banks.items():
            target = load_mask(sample, key, tuple(image.shape[-2:]))
            ref = references[key]
            coordinates = ref.coordinates.long().cpu().numpy()
            truth = target[coordinates[:, 0], coordinates[:, 1]]
            class_count = bank.class_count
            local_counts = np.zeros_like(counts[key])
            indices, present = ref.pool_indices.cpu().numpy(), ref.pool_valid.cpu().numpy()
            quality = ref.quality.cpu().numpy()
            for c in range(class_count):
                selected = np.unique(indices[:, c][present[:, c]])
                selected_truth = truth[selected]
                selected_truth = selected_truth[(selected_truth >= 0) & (selected_truth < class_count)]
                unique_counts[key][c] += np.bincount(selected_truth, minlength=class_count)
                for a in range(len(bank.alias_names)):
                    selected = indices[a, c, present[a, c]]
                    selected_truth = truth[selected]
                    valid = (selected_truth >= 0) & (selected_truth < class_count)
                    values = np.bincount(selected_truth[valid], minlength=class_count)
                    local_counts[c] += values
                    alias_counts[key][a, c] += values
                    weighted[key][c] += np.bincount(selected_truth[valid], weights=quality[selected[valid], a], minlength=class_count)
            counts[key] += local_counts
            image_row["protocols"][key] = {"reference_candidates": ref.candidate_count,
                "selected_tokens": len(ref.features), "pool_instances_by_declared_and_true_class": local_counts.tolist()}
        per_image.append(image_row)
    result = {"status": "complete", "implementation": IMPLEMENTATION, "dataset": args.dataset,
              "config": asdict(CONFIG), "sample_keys": fixed, "processed_images": len(fixed),
              "audit_only": True, "target_masks_loaded_only_after_reference_construction": True,
              "used_to_change_config": False, "elapsed_seconds": time.perf_counter() - started,
              "label_definition": "semantic label at patch center; not majority-patch or independent-object truth",
              "counts_note": "pool instances repeat tokens across aliases; unique counts de-duplicate tokens within each declared class/image",
              "protocols": {}, "per_image": per_image}
    for key, bank in banks.items():
        entries = []
        for c, name in enumerate(bank.class_names):
            entries.append({"name": name, "pool_instances": int(counts[key][c].sum()),
                "pool_purity_percent": 100 * int(counts[key][c, c]) / max(int(counts[key][c].sum()), 1),
                "confidence_weighted_purity_percent": 100 * float(weighted[key][c, c]) / max(float(weighted[key][c].sum()), 1e-12),
                "unique_witnesses": int(unique_counts[key][c].sum()),
                "unique_witness_purity_percent": 100 * int(unique_counts[key][c, c]) / max(int(unique_counts[key][c].sum()), 1)})
        result["protocols"][key] = {"class_names": bank.class_names, "per_class": entries,
            "pool_instances_by_declared_and_true_class": counts[key].tolist(),
            "unique_witnesses_by_declared_and_true_class": unique_counts[key].tolist(),
            "confidence_mass_by_declared_and_true_class": weighted[key].tolist(),
            "per_alias_class": [{"alias": alias, "parent": bank.class_names[int(bank.parent_indices[a])],
                                 "instances_by_reference_class_and_true_class": alias_counts[key][a].tolist()}
                                for a, alias in enumerate(bank.alias_names)]}
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"dataset": args.dataset, "per_class": {key: value["per_class"] for key, value in result["protocols"].items()}}), flush=True)


def launch(root):
    jobs = []
    audit_root = root / "reference_audits"
    if audit_root.exists() or any(not idle(gpu) for gpu in (2, 3)):
        raise RuntimeError("Existing audit root or occupied GPUs2-3.")
    audit_root.mkdir()
    for gpu, dataset in ((2, "potsdam"), (3, "oem")):
        _, data, vocabulary, _, _ = next(row for row in SETTINGS if row[0] == dataset)
        session = "gssa02_reference_audit_" + dataset
        if alive(session):
            raise RuntimeError("Existing audit session.")
        output, log = audit_root / dataset, audit_root / (dataset + ".log")
        args = [PYTHON, "-u", "scripts/audit_stratified_alias_references.py", "--dataset", dataset,
            "--dinov3-repo", str(TOOL / "dinov3_hub"), "--checkpoint-dir", str(BASE / "ckpt/DINO"),
            "--data-root", str(Path("/data/test/datasets") / data), "--vocabulary-config", str(TOOL / "configs" / vocabulary),
            "--upstream-root", str(BASE / "third_party/VIP_official_5bd25ee"), "--output-dir", str(output),
            "--source-diagnostic", str(root / (dataset + "_samples.json")), "--num-shards", "1", "--shard-index", "0"]
        env = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
               "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "TOKENIZERS_PARALLELISM=false"]
        shell = f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(env)} nice -n 10 {shlex.join(args)} > {shlex.quote(str(log))} 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
        jobs.append({"dataset": dataset, "gpu": gpu, "session": session, "output": str(output), "log": str(log)})
    (audit_root / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    print(json.dumps(jobs), flush=True)


if __name__ == "__main__":
    if "--launch-root" in sys.argv:
        launch(Path(sys.argv[sys.argv.index("--launch-root") + 1]))
    else:
        audit(parse_args())
