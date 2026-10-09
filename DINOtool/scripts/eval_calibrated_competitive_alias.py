"""Matched complete-image test of calibration and group-excluded alias utility."""
from contextlib import ExitStack
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.calibrated_competitive_alias import (IMPLEMENTATION, METHODS, PAIRS, PRIMARY, REPLAY, SHUFFLED,
    CalibratedAliasConfig, CalibratedCompetitiveAliases, action_and_group_scores, calibrated_wide_pairs, group_references)
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.geometry_semantic_innovation import anchored_innovation
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.model import checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from dinotool.stratified_soft_alias import StratifiedAliasConfig, build_image_references
from eval_bounded_alias_anchored import make_models
from eval_gear_ov import digest, protocol
from eval_geometry_semantic_innovation import SETTINGS, parse_args
from eval_geometry_vip_reliability import sample_broad, summary
from eval_stratified_soft_alias import prepare_wide, tile_coordinates
from eval_vip_official_eight import PINNED_COMMIT


CONFIG, ACTION = StratifiedAliasConfig(), CalibratedAliasConfig()
LOCAL_METHODS = tuple(method for method in METHODS if method != "BroadVIP")


@torch.inference_mode()
def references_for_image(image, geometry, banks, broad, without, crops, count, readers):
    height, width = image.shape[-2:]
    fields = {key: {name: [] for name in ("features", "coordinates", "aliases", "broad", "without")} for key in banks}
    text = {key: F.normalize(bank.features.float(), dim=-1) for key, bank in banks.items()}
    views, started = 0, time.perf_counter()
    for top in range(0, height, 512):
        for left in range(0, width, 512):
            prepared = geometry.prepare_image(_crop_at(image, top, left, 512).to(geometry.device))
            coords = tile_coordinates(top, left, geometry.device)
            valid = (coords[:, 0] < height) & (coords[:, 1] < width)
            for key, bank in banks.items():
                values = fields[key]
                values["features"].append(prepared.raw_patch_tokens[0, valid])
                values["coordinates"].append(coords[valid])
                values["aliases"].append((prepared.geometry_projected.float() @ text[key].T)[0, valid])
                values["broad"].append(sample_broad(broad[key], top, left, height, width).reshape(-1, bank.class_count)[valid])
                values["without"].append(sample_broad(without[key], top, left, height, width).reshape(-1, len(bank.alias_names))[valid])
            views += 1
    outputs = {}
    for key, bank in banks.items():
        values = {field: torch.cat(rows) for field, rows in fields[key].items()}
        reader = readers[key]
        old = build_image_references(values["features"], values["coordinates"], values["aliases"], values["broad"],
              values["without"], bank.parent_indices, bank.class_count, (height, width), CONFIG)
        marginal, group = action_and_group_scores(crops[key], count, values["coordinates"], (height, width),
              reader.members, reader.holdouts, ACTION.action_retention, SETTINGS.tau, CONFIG.query_chunk)
        new = group_references(values["features"], values["coordinates"], values["aliases"], group, marginal,
              reader.members, reader.holdouts, (height, width), CONFIG)
        outputs[key] = (old, new)
    return outputs, views, time.perf_counter() - started


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, readers, work):
    height, width = image.shape[-2:]
    broad, without, crops, count = prepare_wide(image, vip, queries)
    references, views, seconds = references_for_image(image, geometry, banks, broad, without, crops, count, readers)
    text = {key: F.normalize(bank.features.float(), dim=-1) for key, bank in banks.items()}
    totals = {key: {"tiles": 0, "reference_views": views, "reference_seconds": seconds} for key in banks}
    blend = hann_blend_window(512)
    with ExitStack() as stack:
        accumulators = {(key, method): stack.enter_context(ProbabilityAccumulator(bank.class_count, height, width, 256, work))
                        for key, bank in banks.items() for method in LOCAL_METHODS}
        for top in tile_starts(height, 512, 128):
            for left in tile_starts(width, 512, 128):
                prepared = geometry.prepare_image(_crop_at(image, top, left, 512).to(geometry.device))
                coords = tile_coordinates(top, left, geometry.device)
                valid = (coords[:, 0] < height) & (coords[:, 1] < width)
                ah, aw = min(512, height - top), min(512, width - left)
                for key, bank in banks.items():
                    aliases = prepared.geometry_projected.float() @ text[key].T
                    raw = alias_class_scores(aliases, bank.parent_indices, bank.class_count)
                    local = raw / .07
                    b = sample_broad(broad[key], top, left, height, width).reshape_as(local)
                    anchored, _ = anchored_innovation(local, b, prepared.geometry_patch_conditional, valid[None])
                    variants, diagnostics = readers[key].read(prepared.raw_patch_tokens[0], coords, local[0], b[0],
                          *references[key], crops[key], count, valid, SETTINGS.tau)
                    scores = {"Geometry": raw, "Anchored_VIP": anchored}
                    for method, observation in variants.items():
                        scores[method], _ = anchored_innovation(local, observation[None], prepared.geometry_patch_conditional, valid[None])
                    for method, value in scores.items():
                        dense = F.interpolate(value.transpose(1, 2).reshape(1, bank.class_count, 32, 32), (512, 512),
                                              mode="bilinear", align_corners=False)[0]
                        if method == "Geometry":
                            dense = dense / .07
                        accumulators[key, method].add(dense[:, :ah, :aw].softmax(0).cpu().numpy(), blend[:ah, :aw], left, top)
                    totals[key]["tiles"] += 1
                    for field, value in diagnostics.items():
                        totals[key][field] = totals[key].get(field, 0.) + value
        predictions = {key: {method: accumulators[key, method].finalize(None)[0] for method in LOCAL_METHODS} for key in banks}
        for key in banks:
            probabilities = F.interpolate(broad[key][None], (height, width), mode="bilinear", align_corners=False)[0].softmax(0)
            predictions[key]["BroadVIP"] = probabilities.argmax(0).cpu().numpy().astype(np.uint8)
    return predictions, {key: {field: value if field in ("tiles", "reference_views", "reference_seconds")
                else value / row["tiles"] for field, value in row.items()} for key, row in totals.items()}


@torch.inference_mode()
def smoke(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Existing smoke output.")
    samples, specs, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    geometry, banks, vip, queries, _ = make_models(args, specs)
    readers = {key: CalibratedCompetitiveAliases(bank, CONFIG, ACTION) for key, bank in banks.items()}
    states = {prefix + key: value.clone() for prefix, model in (("geo_", geometry.backbone), ("vip_", vip.backbone))
              for key, value in model.model.visual_model.head.state_dict().items()}
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == "loveda" else sample)
    height, width = image.shape[-2:]
    broad, without, crops, count = prepare_wide(image, vip, queries)
    references, _, _ = references_for_image(image, geometry, banks, broad, without, crops, count, readers)
    prepared = geometry.prepare_image(_crop_at(image, 0, 0, 512).to(geometry.device))
    coords = tile_coordinates(0, 0, geometry.device)
    valid = (coords[:, 0] < height) & (coords[:, 1] < width)
    errors, diagnostics = {}, {}
    for key, bank in banks.items():
        local = alias_class_scores(prepared.geometry_projected.float() @ F.normalize(bank.features.float(), dim=-1).T,
                                   bank.parent_indices, bank.class_count)[0] / .07
        b = sample_broad(broad[key], 0, 0, height, width).reshape_as(local)
        reader = readers[key]
        pairs = local.topk(2, -1).indices
        uniform = torch.full((*pairs.shape, reader.count), 1 / reader.count, device=geometry.device)
        replay = calibrated_wide_pairs(crops[key], count, coords, (height, width), pairs, uniform, reader.members, SETTINGS.tau)
        errors[key + "_uniform_stencil"] = float((replay - b.gather(-1, pairs)).abs().max())
        identity = CalibratedCompetitiveAliases(bank, replace(CONFIG, uniform_prior=1), ACTION)
        outputs, _ = identity.read(prepared.raw_patch_tokens[0], coords, local, b, *references[key], crops[key], count, valid, SETTINGS.tau)
        errors[key + "_uniform_identity"] = max(float((value - b).abs().max()) for value in outputs.values())
        _, diagnostics[key] = reader.read(prepared.raw_patch_tokens[0], coords, local, b, *references[key], crops[key], count, valid, SETTINGS.tau)
        if any(value > 1e-5 for field, value in diagnostics[key].items() if field.endswith("_error")):
            raise ValueError("Pair partition or matched shuffle failed.")
    unchanged = all(torch.equal(states[prefix + key], value) for prefix, model in (("geo_", geometry.backbone), ("vip_", vip.backbone))
                    for key, value in model.model.visual_model.head.state_dict().items())
    frozen = all(not value.requires_grad for model in (geometry.backbone, vip.backbone) for value in model.model.parameters())
    if not unchanged or not frozen or any(value > (1e-4 if field.endswith("_stencil") else 0.) for field, value in errors.items()):
        raise ValueError("Frozen calibrated smoke failed: " + json.dumps(errors))
    output.mkdir(parents=True)
    result = {"status": "complete", "implementation": IMPLEMENTATION, "config": asdict(CONFIG), "action_config": asdict(ACTION),
              "errors": errors, "diagnostics": diagnostics, "weights_frozen": frozen, "head_weights_unchanged": unchanged, "target_masks_loaded": False}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main(args):
    output = Path(args.output_dir)
    if output.exists() or args.num_shards != 1 or args.shard_index != 0:
        raise ValueError("New single-shard output required.")
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    fixed = json.loads(args.source_diagnostic.read_text())["signature"]["samples"]
    lookup = {sample.key: sample for sample in samples}
    selected = [lookup[key] for key in fixed]
    if not selected or len(fixed) != len(set(fixed)):
        raise ValueError("Empty or duplicate sequence.")
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    readers = {key: CalibratedCompetitiveAliases(bank, CONFIG, ACTION) for key, bank in banks.items()}
    signature = {"implementation": IMPLEMENTATION, "dataset": args.dataset, "methods": METHODS,
        "classes": {key: bank.class_names for key, bank in banks.items()},
        "gear": {"geometry": asdict(geometry.config), "primary": PRIMARY, "stratified_alias": asdict(CONFIG),
                 "calibrated_action": asdict(ACTION), "observation": asdict(SETTINGS), "upstream_commit": PINNED_COMMIT}, "competitive": None,
        "vocabulary": {"sha256": hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest(),
                       "aliases": {key: bank.alias_names for key, bank in banks.items()},
                       "counts": {key: [20] * bank.class_count for key, bank in banks.items()}},
        "checkpoints": checkpoint_manifest(checkpoints), "global_sample_count": len(fixed), "global_sample_keys_sha256": digest(fixed),
        "sample_keys": fixed, "sample_keys_sha256": digest(fixed), "num_shards": 1, "shard_index": 0,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "note": "Fixed20 aliases and original Geometry anchor. Calibrated broad mixtures with three-text-neighbor holdout and exact attenuation marginals. "
                "Single-image references, no target-label fitting, development screen."}
    output.mkdir(parents=True)
    (output / "selection.json").write_text(json.dumps({key: reader.report() for key, reader in readers.items()}, indent=2) + "\n")
    matrices = {key: {method: np.zeros((bank.class_count,) * 2, np.int64) for method in METHODS} for key, bank in banks.items()}
    transitions = {key: {method: np.zeros((bank.class_count,) * 3, np.int64) for method in METHODS if method != "Geometry"} for key, bank in banks.items()}
    pairs = {key: {name: np.zeros((bank.class_count,) * 3, np.int64) for name in PAIRS} for key, bank in banks.items()}
    per_image = {key: {method: [] for method in METHODS} for key in banks}
    ignored, diagnostics = dict.fromkeys(banks, 0), {key: {} for key in banks}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for number, sample in enumerate(selected, 1):
        image = load_image(sample.image_path if args.dataset == "loveda" else sample)
        predictions, current = predict_image(image, geometry, banks, vip, queries, readers, output)
        for key, bank in banks.items():
            target = load_mask(sample, key, tuple(image.shape[-2:]))
            valid = (target >= 0) & (target < bank.class_count)
            ignored[key] += int((~valid).sum())
            for method in METHODS:
                encoded = target[valid].astype(np.int64) * bank.class_count + predictions[key][method][valid]
                cm = np.bincount(encoded, minlength=bank.class_count ** 2).reshape(bank.class_count, -1)
                matrices[key][method] += cm
                per_image[key][method].append(cm)
                if method != "Geometry":
                    transitions[key][method] += transition_counts(predictions[key]["Geometry"], predictions[key][method], target, bank.class_count)
            for name, (before, after) in PAIRS.items():
                pairs[key][name] += transition_counts(predictions[key][before], predictions[key][after], target, bank.class_count)
            for field, value in current[key].items():
                diagnostics[key][field] = diagnostics[key].get(field, 0.) + value
        result = {"status": "complete" if number == len(selected) else "running", "processed_images": number, "total_images": len(selected),
            "signature": signature, "metrics": {key: {method: summary(cm, banks[key].class_names, ignored[key]) for method, cm in group.items()} for key, group in matrices.items()},
            "transitions": {key: {method: {"counts": value.tolist(), **transition_summary(value)} for method, value in group.items()} for key, group in transitions.items()},
            "factorial_transitions": {key: {name: {"methods": PAIRS[name], "counts": value.tolist(), **transition_summary(value)} for name, value in group.items()} for key, group in pairs.items()},
            "diagnostics": {key: {field: value / number for field, value in group.items()} for key, group in diagnostics.items()},
            "wall_seconds": time.perf_counter() - started, "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1048576}
        temporary = output / "results.tmp"
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(output / "results.json")
        print(json.dumps({"dataset": args.dataset, "processed": number, "total": len(selected),
              "miou": {key: {method: row["mean_iou_percent"] for method, row in group.items()} for key, group in result["metrics"].items()}}), flush=True)
    np.savez_compressed(output / "per_image_confusions.npz", sample_keys=np.asarray(fixed),
                        **{key + "__" + method: np.stack(values) for key, group in per_image.items() for method, values in group.items()})


if __name__ == "__main__":
    is_smoke = "--smoke" in sys.argv
    if is_smoke:
        sys.argv.remove("--smoke")
    (smoke if is_smoke else main)(parse_args())
