"""Label-filtered candidate-reference diagnostic; never a proposed model."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
import torch

from dinotool.calibrated_competitive_alias import IMPLEMENTATION, METHODS, PRIMARY, CalibratedCompetitiveAliases
from dinotool.prompts import load_class_specs
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from eval_bounded_alias_anchored import make_models
from eval_calibrated_competitive_alias import ACTION, CONFIG, predict_image
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import parse_args
from eval_geometry_vip_reliability import summary
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, SETTINGS, idle
from run_sat_geometry_transport_suite import alive


class LabelFilteredReader(CalibratedCompetitiveAliases):
    target = None

    def read(self, features, coordinates, local, broad, old_reference, reference, crops, count, valid, tau=1.):
        coords = reference.coordinates.long().cpu().numpy()
        labels = torch.as_tensor(self.target[coords[:, 0], coords[:, 1]], device=features.device)
        declared = torch.arange(self.bank.class_count, device=features.device)[None, :, None]
        present = reference.pool_valid & (labels[reference.pool_indices] == declared)
        reference = replace(reference, pool_valid=present)
        output, diagnostics = super().read(features, coordinates, local, broad, old_reference, reference, crops, count, valid, tau)
        diagnostics["oracle_retained_pool_instances"] = float(present.sum())
        return output, diagnostics


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists() or args.dataset != "potsdam":
        raise ValueError("New Potsdam-only oracle audit output required.")
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    fixed = json.loads(args.source_diagnostic.read_text())["signature"]["samples"]
    lookup = {sample.key: sample for sample in samples}
    expected = json.loads((args.source_diagnostic.parent / "potsdam/merged.json").read_text())
    geometry, banks, vip, queries, _ = make_models(args, specs)
    readers = {key: LabelFilteredReader(bank, CONFIG, ACTION) for key, bank in banks.items()}
    matrices = {key: {method: np.zeros((bank.class_count,) * 2, np.int64) for method in METHODS} for key, bank in banks.items()}
    per_image = {key: {method: [] for method in METHODS} for key in banks}
    transitions = {key: np.zeros((bank.class_count,) * 3, np.int64) for key, bank in banks.items()}
    diagnostics, ignored = {key: {} for key in banks}, dict.fromkeys(banks, 0)
    output.mkdir(parents=True)
    started = time.perf_counter()
    for sample_key in fixed:
        sample = lookup[sample_key]
        image = load_image(sample)
        targets = {key: load_mask(sample, key, tuple(image.shape[-2:])) for key in banks}
        for key in banks:
            readers[key].target = targets[key]
        predictions, current = predict_image(image, geometry, banks, vip, queries, readers, output)
        for key, bank in banks.items():
            target = targets[key]
            valid = (target >= 0) & (target < bank.class_count)
            ignored[key] += int((~valid).sum())
            for method in METHODS:
                encoded = target[valid].astype(np.int64) * bank.class_count + predictions[key][method][valid]
                cm = np.bincount(encoded, minlength=bank.class_count ** 2).reshape(bank.class_count, -1)
                matrices[key][method] += cm
                per_image[key][method].append(cm)
            transitions[key] += transition_counts(predictions[key]["Anchored_VIP"], predictions[key][PRIMARY], target, bank.class_count)
            for field, value in current[key].items():
                diagnostics[key][field] = diagnostics[key].get(field, 0.) + value
    for key, group in matrices.items():
        for method in ("Geometry", "BroadVIP", "Anchored_VIP", "StratifiedAlias_Coupled", "CalibratedReplay_Coupled", "TextPrior_Coupled"):
            if not np.array_equal(group[method], expected["metrics"][key][method]["confusion_matrix"]):
                raise ValueError("Unaffected control replay failed: " + method)
    result = {"status": "complete", "implementation": IMPLEMENTATION, "dataset": args.dataset,
        "config": asdict(CONFIG), "action_config": asdict(ACTION), "source_signature": expected["signature"],
        "sample_keys": fixed, "processed_images": len(fixed), "audit_only": True, "target_labels_used": True,
        "label_use": "filter already-selected group candidate references by true patch-center class; no new candidates or parameter fitting",
        "used_to_change_config": False, "is_proposed_method": False,
        "scope": "restricted reference-purification intervention, not an upper bound for arbitrary alias screening",
        "unchanged_controls_verified": True, "wall_seconds": time.perf_counter() - started,
        "metrics": {key: {method: summary(cm, banks[key].class_names, ignored[key]) for method, cm in group.items()} for key, group in matrices.items()},
        "transitions": {key: {"counts": counts.tolist(), **transition_summary(counts)} for key, counts in transitions.items()},
        "diagnostics": {key: {field: value / len(fixed) for field, value in group.items()} for key, group in diagnostics.items()}}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(output / "per_image_confusions.npz", sample_keys=np.asarray(fixed),
        **{key + "__" + method: np.stack(values) for key, group in per_image.items() for method, values in group.items()})
    print(json.dumps({"status": "complete", "audit_only": True, "metrics": {key: {method: value["mean_iou_percent"] for method, value in group.items()}
                     for key, group in result["metrics"].items()}}), flush=True)


def launch(root):
    dataset, data, vocabulary, _, _ = next(row for row in SETTINGS if row[0] == "potsdam")
    output, log = root / "reference_oracle_potsdam", root / "reference_oracle_potsdam_r3.log"
    session = "gcca02_oracle_potsdam_r3"
    if output.exists() or log.exists() or alive(session) or not idle(2):
        raise RuntimeError("Existing diagnostic output/log/session or occupied GPU2.")
    args = [PYTHON, "-u", "scripts/audit_calibrated_reference_oracle.py", "--dataset", dataset,
        "--dinov3-repo", str(TOOL / "dinov3_hub"), "--checkpoint-dir", str(BASE / "ckpt/DINO"),
        "--data-root", str(Path("/data/test/datasets") / data), "--vocabulary-config", str(TOOL / "configs" / vocabulary),
        "--upstream-root", str(BASE / "third_party/VIP_official_5bd25ee"), "--output-dir", str(output),
        "--source-diagnostic", str(root / "potsdam_samples.json"), "--num-shards", "1", "--shard-index", "0"]
    env = ["env", "CUDA_VISIBLE_DEVICES=2", f"PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}",
           "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "TOKENIZERS_PARALLELISM=false"]
    shell = f"cd {shlex.quote(str(TOOL))} && exec {shlex.join(env + args)} > {shlex.quote(str(log))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    print(session, "GPU2", flush=True)


if __name__ == "__main__":
    if "--launch-root" in sys.argv:
        launch(Path(sys.argv[sys.argv.index("--launch-root") + 1]))
    else:
        main(parse_args())
