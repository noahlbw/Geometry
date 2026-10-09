"""Matched hard deletion and grounded soft alias allocation on complete images."""
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

from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.geometry_semantic_innovation import anchored_innovation
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.model import checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from dinotool.soft_competitive_alias import IMPLEMENTATION, METHODS, PAIRS, PRIMARY, SoftAliasConfig, SoftCompetitiveAliases
from eval_bounded_alias_anchored import make_models
from eval_gear_ov import digest, protocol
from eval_geometry_semantic_innovation import SETTINGS, parse_args
from eval_geometry_vip_reliability import sample_broad, summary
from eval_matched_head_fov import resize_rgb
from eval_matched_head_text import imagenet_geometry_logits
from eval_pair_conditional_alias import prepare_wide as old_wide
from eval_vip_official_eight import PINNED_COMMIT


CONFIG = SoftAliasConfig()
LOCAL_METHODS = tuple(method for method in METHODS if method not in ("BroadVIP", "MeanProb_VIP"))


@torch.inference_mode()
def vip_leave_out_crop(features, query):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        similarity = torch.einsum("bnd,mtd->bnmt", features, query.features.float()).mean(-1)[0]
        patch_mean = F.normalize(features.mean(1), dim=-1)
        text_mean = F.normalize(query.features.float().mean(1), dim=-1)
        salience = ((patch_mean @ text_mean.T)[0] / SETTINGS.tem).float()
        alias_logits = (similarity * SETTINGS.logit_scale).T.reshape(-1, 21, 21)
        out = torch.empty_like(alias_logits, dtype=torch.float32)
        for c in range(len(query.class_names)):
            ids = (query.parents == c).nonzero().flatten()
            count = len(ids)
            excluded = torch.eye(count, device=features.device, dtype=torch.bool)
            weights = salience[ids][None].expand(count, -1).masked_fill(excluded, -torch.inf).softmax(-1)
            scaled = alias_logits[ids][None] * ((count - 1) * weights)[..., None, None]
            scaled = scaled.masked_fill(excluded[..., None, None], -torch.inf)
            out[ids] = torch.logsumexp(SETTINGS.tau * scaled, 1) / SETTINGS.tau
        return F.interpolate(out[None], size=(336, 336), mode="bilinear", align_corners=False)[0].float()


@torch.inference_mode()
def prepare_wide(image, vip, queries):
    resized = resize_rgb(image, 448)
    h, w = resized.shape[-2:]
    maps = {key: torch.zeros(len(query.class_names), h, w, device=vip.device) for key, query in queries.items()}
    without = {key: torch.zeros(len(query.aliases), h, w, device=vip.device) for key, query in queries.items()}
    count = torch.zeros(h, w, device=vip.device)
    for top in tile_starts(h, 336, 224):
        for left in tile_starts(w, 336, 224):
            ah, aw = min(336, h - top), min(336, w - left)
            rgb = F.pad(resized[:, top:top + ah, left:left + aw], (0, 336 - aw, 0, 336 - ah))
            features = vip.crop_patch_features(rgb)
            for key, query in queries.items():
                maps[key][:, top:top + ah, left:left + aw] += imagenet_geometry_logits(features, query, SETTINGS)[:, :ah, :aw]
                without[key][:, top:top + ah, left:left + aw] += vip_leave_out_crop(features, query)[:, :ah, :aw]
            count[top:top + ah, left:left + aw] += 1
    if not bool((count > 0).all()):
        raise ValueError("Uncovered wide-view pixels.")
    return {key: value / count[None] for key, value in maps.items()}, {key: value / count[None] for key, value in without.items()}


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, readers, work):
    h, w = image.shape[-2:]
    before_empty, before_rows = vip.empty_rows, vip.observed_rows
    broad, leave_broad = prepare_wide(image, vip, queries)
    totals = {key: {"tiles": 0, "observer_empty_rows": vip.empty_rows - before_empty,
                    "observer_rows": vip.observed_rows - before_rows} for key in banks}
    texts = {key: F.normalize(bank.features.float(), dim=-1) for key, bank in banks.items()}
    blend = hann_blend_window(512)
    with ExitStack() as stack:
        accumulators = {(key, method): stack.enter_context(ProbabilityAccumulator(
            bank.class_count, h, w, 256, work)) for key, bank in banks.items() for method in LOCAL_METHODS}
        for top in tile_starts(h, 512, 128):
            for left in tile_starts(w, 512, 128):
                prepared = geometry.prepare_image(_crop_at(image, top, left, 512).to(geometry.device))
                ah, aw = min(512, h - top), min(512, w - left)
                yy = top + (torch.arange(32, device=geometry.device) + .5) * 16
                xx = left + (torch.arange(32, device=geometry.device) + .5) * 16
                valid = ((yy[:, None] < h) & (xx[None] < w)).reshape(1, -1)
                for key, bank in banks.items():
                    aliases = prepared.geometry_projected.float() @ texts[key].T
                    raw = alias_class_scores(aliases, bank.parent_indices, bank.class_count)
                    g = raw / .07
                    b = sample_broad(broad[key], top, left, h, w).reshape_as(g)
                    b_without = sample_broad(leave_broad[key], top, left, h, w).reshape_as(aliases)
                    anchored, _ = anchored_innovation(g, b, prepared.geometry_patch_conditional, valid)
                    hard, _ = readers[key].hard.read(aliases, valid)
                    hard_coupled, _ = anchored_innovation(hard["selected"] / .07, b, prepared.geometry_patch_conditional, valid)
                    variants, stats = readers[key].read(aliases, prepared.geometry_patch_conditional, b, b_without, valid)
                    scores = {"Geometry": raw, "MeanLogit_VIP": .5 * (g + b), "Anchored_VIP": anchored,
                              "PairAlias_Local": hard["selected"], "PairAlias_Coupled": hard_coupled}
                    for name, local, coupled in (("text", "SoftText_Local", "SoftText_Coupled"),
                            ("soft", "SoftCounterfactual_Local", PRIMARY),
                            ("shuffled", "ShuffledCounterfactual_Local", "ShuffledCounterfactual_Coupled")):
                        scores[local] = variants[name]["local"]
                        scores[coupled], diag = anchored_innovation(g, variants[name]["observation"], prepared.geometry_patch_conditional, valid)
                        if name == "soft":
                            stats.update({"solver_" + field: value for field, value in diag.items()})
                    for method, value in scores.items():
                        dense = F.interpolate(value.transpose(1, 2).reshape(1, bank.class_count, 32, 32),
                                              size=(512, 512), mode="bilinear", align_corners=False)[0]
                        if method in ("Geometry", "PairAlias_Local"):
                            dense = dense / .07
                        accumulators[key, method].add(dense[:, :ah, :aw].softmax(0).cpu().numpy(), blend[:ah, :aw], left, top)
                    totals[key]["tiles"] += 1
                    for field, value in stats.items():
                        totals[key][field] = totals[key].get(field, 0.) + value
        predictions = {key: {method: accumulators[key, method].finalize(None)[0] for method in LOCAL_METHODS} for key in banks}
        for key in banks:
            probabilities = F.interpolate(broad[key][None], size=(h, w), mode="bilinear", align_corners=False)[0].softmax(0).cpu().numpy()
            predictions[key]["BroadVIP"] = probabilities.argmax(0).astype(np.uint8)
            fused = np.empty((h, w), dtype=np.uint8)
            accumulator = accumulators[key, "Geometry"]
            for top in range(0, h, 128):
                end = min(top + 128, h)
                local = accumulator.probabilities[:, top:end] / accumulator.normalizer[None, top:end].clip(1e-8)
                fused[top:end] = (local + probabilities[:, top:end]).argmax(0).astype(np.uint8)
            predictions[key]["MeanProb_VIP"] = fused
    return predictions, {key: {field: value if field in ("tiles", "observer_empty_rows", "observer_rows")
                                else value / row["tiles"] for field, value in row.items()} for key, row in totals.items()}


@torch.inference_mode()
def smoke(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Existing smoke output.")
    samples, specs, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    geometry, banks, vip, queries, _ = make_models(args, specs)
    states = {prefix + key: value.detach().clone() for prefix, model in (("geo_", geometry.backbone), ("vip_", vip.backbone))
              for key, value in model.model.visual_model.head.state_dict().items()}
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == "loveda" else sample)
    prepared = geometry.prepare_image(_crop_at(image, 0, 0, 512).to(geometry.device))
    broad, leave = prepare_wide(image, vip, queries)
    replay = old_wide(image, vip, queries)
    errors, diagnostics = {}, {}
    for key, bank in banks.items():
        aliases = prepared.geometry_projected.float() @ F.normalize(bank.features.float(), dim=-1).T
        valid = torch.ones(aliases.shape[:2], device=aliases.device, dtype=torch.bool)
        b = sample_broad(broad[key], 0, 0, *image.shape[-2:]).reshape(*aliases.shape[:2], bank.class_count)
        b_without = sample_broad(leave[key], 0, 0, *image.shape[-2:]).reshape_as(aliases)
        reader = SoftCompetitiveAliases(bank, CONFIG)
        variants, diagnostics[key] = reader.read(aliases, prepared.geometry_patch_conditional, b, b_without, valid)
        if not all(bool(torch.isfinite(row[field]).all()) for row in variants.values() for field in ("local", "observation", "weights")):
            raise ValueError("Nonfinite soft allocation.")
        no_adjustment = SoftCompetitiveAliases(bank, replace(CONFIG, uniform_prior=1))
        uniform, _ = no_adjustment.read(aliases, prepared.geometry_patch_conditional, b, b_without, valid)
        original = alias_class_scores(aliases, bank.parent_indices, bank.class_count) / .07
        errors[key + "_uniform_identity"] = max(float((row["local"] - original).abs().max()) for row in uniform.values())
        errors[key + "_uniform_observer_identity"] = max(float((row["observation"] - b).abs().max()) for row in uniform.values())
        errors[key + "_wide_original_replay"] = float((broad[key] - replay[key]).abs().max())
        if diagnostics[key]["shuffled_weight_spectrum_error"] != 0 or diagnostics[key]["pair_partition_error"] > 1e-5:
            raise ValueError("Weight or pair partition check failed.")
    unchanged = all(torch.equal(states[prefix + key], value) for prefix, model in (("geo_", geometry.backbone), ("vip_", vip.backbone))
                    for key, value in model.model.visual_model.head.state_dict().items())
    frozen = all(not value.requires_grad for model in (geometry.backbone, vip.backbone) for value in model.model.parameters())
    if any(errors.values()) or not unchanged or not frozen:
        raise ValueError("Frozen identity smoke failed.")
    output.mkdir(parents=True)
    result = {"status": "complete", "implementation": IMPLEMENTATION, "config": asdict(CONFIG), "errors": errors,
              "head_weights_unchanged": unchanged, "weights_frozen": frozen, "target_masks_loaded": False, "diagnostics": diagnostics}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main(args):
    output = Path(args.output_dir)
    if output.exists() or args.num_shards != 1 or args.shard_index != 0:
        raise ValueError("New single-shard screen output required.")
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    fixed = json.loads(args.source_diagnostic.read_text())["signature"]["samples"]
    lookup = {sample.key: sample for sample in samples}
    selected = [lookup[key] for key in fixed]
    if not selected or len(fixed) != len(set(fixed)):
        raise ValueError("Empty or duplicate sample sequence.")
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    readers = {key: SoftCompetitiveAliases(bank, CONFIG) for key, bank in banks.items()}
    signature = {"implementation": IMPLEMENTATION, "dataset": args.dataset, "methods": METHODS,
        "classes": {key: bank.class_names for key, bank in banks.items()},
        "gear": {"geometry": asdict(geometry.config), "primary": PRIMARY, "soft_alias": asdict(CONFIG),
                 "observation": asdict(SETTINGS), "upstream_commit": PINNED_COMMIT}, "competitive": None,
        "vocabulary": {"sha256": hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest(),
                       "aliases": {key: bank.alias_names for key, bank in banks.items()},
                       "counts": {key: [20] * bank.class_count for key, bank in banks.items()}},
        "checkpoints": checkpoint_manifest(checkpoints), "global_sample_count": len(fixed),
        "global_sample_keys_sha256": digest(fixed), "sample_keys": fixed, "sample_keys_sha256": digest(fixed),
        "num_shards": 1, "shard_index": 0,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "note": "Frozen region/rival-conditioned soft allocation. Tested alias absent from local and wide reference readouts. "
                "Original Geometry anchor, bounded pair correction on observation. Single-image image-only inference; "
                "correlated aliases may still self-confirm. Labels only evaluation. Development screen, not independent validation."}
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
            for name, (old, new) in PAIRS.items():
                pairs[key][name] += transition_counts(predictions[key][old], predictions[key][new], target, bank.class_count)
            for field, value in current[key].items():
                diagnostics[key][field] = diagnostics[key].get(field, 0.) + value
        result = {"status": "complete" if number == len(selected) else "running", "processed_images": number,
            "total_images": len(selected), "signature": signature,
            "metrics": {key: {method: summary(cm, banks[key].class_names, ignored[key]) for method, cm in group.items()} for key, group in matrices.items()},
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
