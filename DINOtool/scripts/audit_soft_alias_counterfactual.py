"""Mask-free fresh-readout checks for cached local and broad alias deletion."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.prompts import load_class_specs
from dinotool.soft_competitive_alias import IMPLEMENTATION, SoftAliasConfig, leave_one_out_scores
from eval_bounded_alias_anchored import make_models
from eval_geometry_semantic_innovation import SETTINGS, parse_args
from eval_gear_ov import protocol
from eval_matched_head_fov import resize_rgb
from eval_matched_head_text import imagenet_geometry_logits
from eval_soft_competitive_alias import vip_leave_out_crop
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, SETTINGS as DATASET_SETTINGS, idle
from run_sat_geometry_transport_suite import alive


@torch.inference_mode()
def audit(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Existing audit output.")
    started = time.perf_counter()
    samples, specs, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    key = json.loads(args.source_diagnostic.read_text())["signature"]["samples"][0]
    sample = next(sample for sample in samples if sample.key == key)
    image = load_image(sample.image_path if args.dataset == "loveda" else sample)
    geometry, banks, vip, queries, _ = make_models(args, specs)
    prepared = geometry.prepare_image(_crop_at(image, 0, 0, 512).to(geometry.device))
    wide = resize_rgb(image, 448)[:, :336, :336]
    wide = F.pad(wide, (0, 336 - wide.shape[-1], 0, 336 - wide.shape[-2]))
    features = vip.crop_patch_features(wide)
    errors, checked = {}, {}
    for name, bank in banks.items():
        aliases = prepared.geometry_projected.float() @ F.normalize(bank.features.float(), dim=-1).T
        local = leave_one_out_scores(aliases, bank.parent_indices, bank.class_count)
        broad = vip_leave_out_crop(features, queries[name])
        entries = []
        for c in range(bank.class_count):
            members = (bank.parent_indices == c).nonzero().flatten().tolist()
            for a in (members[0], members[len(members) // 2], members[-1]):
                kept = torch.arange(len(bank.alias_names), device=geometry.device) != a
                fresh_local = alias_class_scores(aliases[..., kept], bank.parent_indices[kept], bank.class_count)[..., c]
                query = queries[name]
                reduced = replace(query, features=query.features[kept], parents=query.parents[kept],
                                  aliases=tuple(alias for i, alias in enumerate(query.aliases) if i != a))
                fresh_broad = imagenet_geometry_logits(features, reduced, SETTINGS)[c]
                entries.append({"alias": bank.alias_names[a], "class": bank.class_names[c],
                                "local_error": float((local[..., a] - fresh_local).abs().max()),
                                "wide_error": float((broad[a] - fresh_broad).abs().max())})
        checked[name] = entries
        errors[name + "_local"] = max(row["local_error"] for row in entries)
        errors[name + "_wide"] = max(row["wide_error"] for row in entries)
    frozen = all(not value.requires_grad for model in (geometry.backbone, vip.backbone) for value in model.model.parameters())
    passed = frozen and all(error <= 1e-4 for error in errors.values())
    result = {"status": "complete", "passed": passed, "implementation": IMPLEMENTATION, "dataset": args.dataset,
              "sample_key": key, "config": asdict(SoftAliasConfig()), "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "target_masks_loaded": False, "weights_frozen": frozen, "errors": errors, "checked_aliases": checked,
              "numerical_tolerance": 1e-4, "elapsed_seconds": time.perf_counter() - started,
              "note": "Three fixed positions per class; fresh score recalculation with unchanged cached visual/text features. Not performance or standalone latency."}
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError("Fresh-readout alias counterfactual mismatch.")


def launch_audits(root):
    settings = [(gpu, next(row for row in DATASET_SETTINGS if row[0] == dataset))
                for gpu, dataset in enumerate(("udd5", "vdd", "potsdam", "flair1"))]
    audit_root = root / "audits"
    if audit_root.exists() or any(not idle(gpu) for gpu, _ in settings):
        raise RuntimeError("Existing audits or occupied physical GPUs0-3.")
    audit_root.mkdir()
    jobs = []
    for gpu, (dataset, data, vocabulary, _, _) in settings:
        session = "gsca02_audit_" + dataset
        if alive(session):
            raise RuntimeError("Existing audit session.")
        output, log = audit_root / dataset, audit_root / (dataset + ".log")
        args = [PYTHON, "-u", "scripts/audit_soft_alias_counterfactual.py", "--dataset", dataset,
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
        launch_audits(Path(sys.argv[sys.argv.index("--launch-root") + 1]))
    else:
        audit(parse_args())
