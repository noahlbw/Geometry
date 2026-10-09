"""Mask-free decomposition of the frozen rejection source, not a new candidate."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from dinotool.calibrated_competitive_alias import profiled_logits
from dinotool.excess_alias_rejection import IMPLEMENTATION, ExcessAliasReader, canonical_support
from dinotool.gear_ov import _crop_at
from dinotool.prompts import load_class_specs
from dinotool.stratified_soft_alias import crop_stencil
from eval_bounded_alias_anchored import make_models
from eval_excess_alias_rejection import CONFIG, make_variants, prepare_wide
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import SETTINGS, parse_args
from eval_stratified_soft_alias import tile_coordinates


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Existing audit output.")
    samples, specs, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {sample.key: sample for sample in samples}
    keys = json.loads(args.source_diagnostic.read_text())["signature"]["samples"][:8]
    if len(keys) != 8 or len(set(keys)) != 8:
        raise ValueError("Eight unique fixed images required.")
    geometry, banks, vip, queries, _ = make_models(args, specs)
    variants, _ = make_variants(geometry, banks, vip, queries, specs)
    readers = {(scenario, key): ExcessAliasReader(bank, CONFIG)
               for scenario, (group, _) in variants.items() for key, bank in group.items()}
    totals = {scenario: {key: {} for key in banks} for scenario in variants}
    started = time.perf_counter()
    for index, sample_key in enumerate(keys, 1):
        sample = lookup[sample_key]
        image = load_image(sample.image_path if args.dataset == "loveda" else sample)
        image_size = tuple(image.shape[-2:])
        _, crops, auxiliary, count = prepare_wide(image, vip, variants)
        prepared = geometry.prepare_image(_crop_at(image, 0, 0, 512).to(geometry.device))
        coords = tile_coordinates(0, 0, geometry.device)
        valid = (coords[:, 0] < image_size[0]) & (coords[:, 1] < image_size[1])
        for scenario, (group, _) in variants.items():
            for key, bank in group.items():
                reader = readers[scenario, key]
                local = (prepared.geometry_projected.float() @ F.normalize(bank.features.float(), dim=-1).T)[0]
                support, supported, _ = canonical_support(local[:, reader.canonical] / CONFIG.local_temperature,
                    prepared.geometry_patch_conditional[0], coords, valid, CONFIG)
                wide, responsibilities = torch.zeros_like(local), torch.zeros_like(local)
                for crop, cosines in zip(crops[scenario, key], auxiliary[scenario, key]):
                    ids, coefficients = crop_stencil(crop, count, coords, image_size)
                    wide += (cosines[ids] * coefficients[..., None]).sum(1)
                    q = (SETTINGS.tau * profiled_logits(crop, reader.members)).softmax(-1)
                    sampled = (q[ids] * coefficients[..., None, None]).sum(1)
                    responsibilities[:, reader.members.flatten()] += sampled.flatten(-2)
                own, rival = support[:, bank.parent_indices, None], support[:, None]
                leakage = (rival-own).clamp_min(0) / (rival+own).clamp_min(CONFIG.epsilon)
                joint = (leakage * reader.conflict[None]).amax(-1) * supported[:, None]
                gain = ((wide-local) / CONFIG.local_temperature).clamp(0, 1)
                risk = joint * gain
                j, g, r = joint[valid], gain[valid], risk[valid]
                excess = (responsibilities[valid] - 1/reader.count).clamp_min(0)
                before = (excess * j).sum(-1)
                after = (excess * r).sum(-1)
                active = j > CONFIG.epsilon
                values = {"text_conflicted_alias_fraction": float((reader.conflict.amax(-1) > CONFIG.epsilon).float().mean()),
                    "text_geometry_positive_fraction": float(active.float().mean()),
                    "gain_positive_fraction": float((g > CONFIG.epsilon).float().mean()),
                    "joint_and_gain_positive_fraction": float((r > CONFIG.epsilon).float().mean()),
                    "mean_joint_before_gain": float(j.mean()), "mean_risk_after_gain": float(r.mean()),
                    "mean_gain_on_joint_positive": float(g[active].mean()) if bool(active.any()) else 0.,
                    "gain_mass_survival_ratio": float(r.sum() / j.sum().clamp_min(CONFIG.epsilon)),
                    "responsibility_excess_joint_mass": float(before.mean()),
                    "responsibility_excess_risk_mass": float(after.mean()),
                    "responsibility_gain_survival_ratio": float(after.sum() / before.sum().clamp_min(CONFIG.epsilon))}
                for field, value in values.items():
                    totals[scenario][key][field] = totals[scenario][key].get(field, 0.) + value
        print(json.dumps({"dataset": args.dataset, "audit_images": index, "total": 8}), flush=True)
    output.mkdir(parents=True)
    result = {"status": "complete", "implementation": IMPLEMENTATION, "config": asdict(CONFIG), "dataset": args.dataset,
              "sample_keys": keys, "target_masks_loaded": False, "audit_only": True, "used_to_change_config": False,
              "scope": "top-left original512 window per fixed image; not full-image prediction or semantic accuracy",
              "diagnostics": {scenario: {key: {field: value/8 for field, value in row.items()}
                  for key, row in group.items()} for scenario, group in totals.items()}, "wall_seconds": time.perf_counter()-started}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(parse_args())
