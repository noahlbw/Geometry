"""Matched post-selection reference purity audit; never selector fitting."""
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
import torch

from dinotool.calibrated_competitive_alias import IMPLEMENTATION, CalibratedCompetitiveAliases
from dinotool.prompts import load_class_specs
from eval_bounded_alias_anchored import make_models
from eval_calibrated_competitive_alias import ACTION, CONFIG, prepare_wide, references_for_image
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import parse_args
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
    readers = {key: CalibratedCompetitiveAliases(bank, CONFIG, ACTION) for key, bank in banks.items()}
    matrices = {key: {name: {field: np.zeros((bank.class_count,) * 2, np.float64 if field == "weighted" else np.int64)
                 for field in ("instances", "unique", "weighted")} for name in ("old", "group")}
                 for key, bank in banks.items()}
    started = time.perf_counter()
    for sample_key in fixed:
        sample = lookup[sample_key]
        image = load_image(sample.image_path if args.dataset == "loveda" else sample)
        broad, without, crops, count = prepare_wide(image, vip, queries)
        references, _, _ = references_for_image(image, geometry, banks, broad, without, crops, count, readers)
        # Image-only references are complete before masks are read.
        for key, bank in banks.items():
            target = load_mask(sample, key, tuple(image.shape[-2:]))
            for name, reference in zip(("old", "group"), references[key]):
                coordinates = reference.coordinates.long().cpu().numpy()
                truth = target[coordinates[:, 0], coordinates[:, 1]]
                indices, present = reference.pool_indices.cpu().numpy(), reference.pool_valid.cpu().numpy()
                quality = reference.quality.cpu().numpy()
                totals = matrices[key][name]
                for c in range(bank.class_count):
                    selected = np.unique(indices[:, c][present[:, c]])
                    labels = truth[selected]
                    valid = (labels >= 0) & (labels < bank.class_count)
                    totals["unique"][c] += np.bincount(labels[valid], minlength=bank.class_count)
                    for a in range(len(bank.alias_names)):
                        selected = indices[a, c, present[a, c]]
                        labels = truth[selected]
                        valid = (labels >= 0) & (labels < bank.class_count)
                        totals["instances"][c] += np.bincount(labels[valid], minlength=bank.class_count)
                        totals["weighted"][c] += np.bincount(labels[valid], weights=quality[selected[valid], a], minlength=bank.class_count)
    result = {"status": "complete", "implementation": IMPLEMENTATION, "config": asdict(CONFIG), "action_config": asdict(ACTION),
        "dataset": args.dataset, "sample_keys": fixed, "processed_images": len(fixed), "audit_only": True,
        "target_masks_loaded_only_after_reference_construction": True, "used_to_change_config": False,
        "label_definition": "patch-center labels of candidate pools; not query-specific retrieved purity or independent-object truth",
        "elapsed_seconds": time.perf_counter() - started, "protocols": {}}
    for key, bank in banks.items():
        result["protocols"][key] = {}
        for name, totals in matrices[key].items():
            entries = []
            for c, classname in enumerate(bank.class_names):
                entries.append({"name": classname, **{field + "_count": float(matrix[c].sum()) for field, matrix in totals.items()},
                    **{field + "_purity_percent": float(100 * matrix[c, c] / matrix[c].sum()) if matrix[c].sum() else None
                       for field, matrix in totals.items()}})
            result["protocols"][key][name] = {"per_class": entries, "matrices": {field: value.tolist() for field, value in totals.items()}}
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"dataset": args.dataset, "protocols": {key: {name: group["per_class"] for name, group in groups.items()}
                   for key, groups in result["protocols"].items()}}), flush=True)


def launch(root):
    directory = root / "reference_audits"
    if directory.exists() or any(not idle(gpu) for gpu in (2, 3)):
        raise RuntimeError("Existing audit root or occupied GPUs2-3.")
    directory.mkdir()
    jobs = []
    for gpu, dataset in ((2, "potsdam"), (3, "oem")):
        _, data, vocabulary, _, _ = next(row for row in SETTINGS if row[0] == dataset)
        session, output, log = "gcca02_audit_" + dataset, directory / dataset, directory / (dataset + ".log")
        if alive(session):
            raise RuntimeError("Existing audit session.")
        args = [PYTHON, "-u", "scripts/audit_calibrated_alias_references.py", "--dataset", dataset,
                "--dinov3-repo", str(TOOL / "dinov3_hub"), "--checkpoint-dir", str(BASE / "ckpt/DINO"),
                "--data-root", str(Path("/data/test/datasets") / data), "--vocabulary-config", str(TOOL / "configs" / vocabulary),
                "--upstream-root", str(BASE / "third_party/VIP_official_5bd25ee"), "--output-dir", str(output),
                "--source-diagnostic", str(root / (dataset + "_samples.json")), "--num-shards", "1", "--shard-index", "0"]
        env = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
               "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "TOKENIZERS_PARALLELISM=false"]
        shell = f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(env)} nice -n 10 {shlex.join(args)} > {shlex.quote(str(log))} 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
        jobs.append({"dataset": dataset, "gpu": gpu, "session": session, "output": str(output), "log": str(log)})
    (directory / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    print(json.dumps(jobs), flush=True)


if __name__ == "__main__":
    if "--launch-root" in sys.argv:
        launch(Path(sys.argv[sys.argv.index("--launch-root") + 1]))
    else:
        audit(parse_args())
