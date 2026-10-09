"""Matched alias aggregation by spatial reconstruction factorial evaluation."""
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.bounded_alias_readout import CONFIG, FACTORIAL_PAIRS, IMPLEMENTATION, METHODS, PRIMARY, bounded_classes, vip_profile_scores
from dinotool.finite_vip_observer import FiniteVIPObserver
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.geometry_semantic_innovation import anchored_innovation
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from eval_gear_ov import digest, protocol
from eval_geometry_semantic_innovation import SETTINGS, parse_args
from eval_geometry_vip_reliability import sample_broad, summary
from eval_matched_head_fov import resize_rgb
from eval_matched_head_text import imagenet_geometry_logits
from eval_stride_ov_loveda_e1 import make_checkpoints
from eval_vip_official_eight import PINNED_COMMIT


LOCAL_METHODS = tuple(method for method in METHODS if method not in
                      ("BroadVIP", "Bounded_BroadVIP", "MeanProb_VIP"))


def make_models(args, specs):
    commit = subprocess.check_output(["git", "-C", args.upstream_root, "rev-parse", "HEAD"], text=True).strip()
    if commit != PINNED_COMMIT:
        raise ValueError("Pinned VIP source changed.")
    checkpoints = make_checkpoints(args)
    geometry = TCPRSegmenter(DINOTextSegmenter(checkpoints, device=args.device),
                            TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    banks = {key: geometry.encode_text(classes) for key, classes in specs.items()}
    vip = FiniteVIPObserver(DINOTextSegmenter(checkpoints, device=args.device), Path(args.upstream_root))
    queries = {key: vip.encode_queries(bank.class_names, tuple(spec.synonyms for spec in specs[key]))
               for key, bank in banks.items()}
    for key, bank in banks.items():
        if (bank.alias_names != queries[key].aliases or not torch.equal(bank.parent_indices, queries[key].parents)
                or any(int((bank.parent_indices == c).sum()) != 20 for c in range(bank.class_count))):
            raise ValueError("Exact matched all20 alias groups required.")
    return geometry, banks, vip, queries, checkpoints


@torch.inference_mode()
def prepare_wide(image, vip, queries):
    resized = resize_rgb(image, 448)
    h, w = resized.shape[-2:]
    maps = {key: {arm: torch.zeros(len(query.class_names), h, w, device=vip.device)
                  for arm in ("original", "bounded")} for key, query in queries.items()}
    count = torch.zeros(h, w, device=vip.device)
    statistics = {key: {} for key in queries}
    crops = 0
    for top in tile_starts(h, 336, 224):
        for left in tile_starts(w, 336, 224):
            ah, aw = min(336, h - top), min(336, w - left)
            rgb = F.pad(resized[:, top:top + ah, left:left + aw], (0, 336 - aw, 0, 336 - ah))
            features = vip.crop_patch_features(rgb)
            for key, query in queries.items():
                original, bounded, diagnostics = vip_profile_scores(features, query, SETTINGS,
                                        CONFIG["responsibility_multiplier"])
                maps[key]["original"][:, top:top + ah, left:left + aw] += original[:, :ah, :aw]
                maps[key]["bounded"][:, top:top + ah, left:left + aw] += bounded[:, :ah, :aw]
                for field, value in diagnostics.items():
                    statistics[key][field] = statistics[key].get(field, 0.0) + value
            count[top:top + ah, left:left + aw] += 1
            crops += 1
    if not bool((count > 0).all()):
        raise ValueError("Uncovered wide-view pixels.")
    return ({key: {arm: value / count[None] for arm, value in group.items()}
             for key, group in maps.items()},
            {key: {field: value / crops for field, value in row.items()} for key, row in statistics.items()})


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work):
    h, w = image.shape[-2:]
    before_empty, before_rows = vip.empty_rows, vip.observed_rows
    broad, broad_stats = prepare_wide(image, vip, queries)
    totals = {key: {"tiles": 0, "observer_empty_rows": vip.empty_rows - before_empty,
                    "observer_rows": vip.observed_rows - before_rows,
                    **{"broad_" + field: value for field, value in broad_stats[key].items()}}
              for key in banks}
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
                    bounded, stats = bounded_classes(aliases, bank.parent_indices, bank.class_count,
                                   CONFIG["local_temperature"], CONFIG["responsibility_multiplier"])
                    g, gc = raw / .07, bounded / .07
                    b = sample_broad(broad[key]["original"], top, left, h, w).reshape_as(g)
                    bc = sample_broad(broad[key]["bounded"], top, left, h, w).reshape_as(g)
                    anchored, _ = anchored_innovation(g, b, prepared.geometry_patch_conditional, valid)
                    candidate, diag = anchored_innovation(gc, bc, prepared.geometry_patch_conditional, valid)
                    scores = {"Geometry": raw, "Bounded_Local": bounded, "MeanLogit_VIP": .5 * (g + b),
                              "Bounded_MeanLogit": .5 * (gc + bc), "Anchored_VIP": anchored, PRIMARY: candidate}
                    for method, value in scores.items():
                        dense = F.interpolate(value.transpose(1, 2).reshape(1, bank.class_count, 32, 32),
                                              size=(512, 512), mode="bilinear", align_corners=False)[0]
                        if method in ("Geometry", "Bounded_Local"):
                            dense = dense / .07
                        accumulators[key, method].add(dense[:, :ah, :aw].softmax(0).cpu().numpy(),
                                                     blend[:ah, :aw], left, top)
                    totals[key]["tiles"] += 1
                    for field, value in {**{"local_" + name: number for name, number in stats.items()},
                                         **{"solver_" + name: number for name, number in diag.items()}}.items():
                        totals[key][field] = totals[key].get(field, 0.0) + value
        predictions = {key: {method: accumulators[key, method].finalize(None)[0] for method in LOCAL_METHODS}
                       for key in banks}
        for key in banks:
            for arm, name in (("original", "BroadVIP"), ("bounded", "Bounded_BroadVIP")):
                probabilities = F.interpolate(broad[key][arm][None], size=(h, w), mode="bilinear",
                                              align_corners=False)[0].softmax(0).cpu().numpy()
                predictions[key][name] = probabilities.argmax(0).astype(np.uint8)
                if arm == "original":
                    fused = np.empty((h, w), dtype=np.uint8)
                    accumulator = accumulators[key, "Geometry"]
                    for top in range(0, h, 128):
                        end = min(top + 128, h)
                        local = accumulator.probabilities[:, top:end] / accumulator.normalizer[None, top:end].clip(1e-8)
                        fused[top:end] = (local + probabilities[:, top:end]).argmax(0).astype(np.uint8)
                    predictions[key]["MeanProb_VIP"] = fused
    diagnostics = {key: {field: value / row["tiles"] if field.startswith(("local_", "solver_")) else value
                         for field, value in row.items()} for key, row in totals.items()}
    return predictions, diagnostics


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
    rgb = resize_rgb(image, 448)[:, :336, :336]
    rgb = F.pad(rgb, (0, 336 - rgb.shape[-1], 0, 336 - rgb.shape[-2]))
    features = vip.crop_patch_features(rgb)
    errors, diagnostics = {}, {}
    for key, bank in banks.items():
        aliases = prepared.geometry_projected.float() @ F.normalize(bank.features.float(), dim=-1).T
        original = alias_class_scores(aliases, bank.parent_indices, bank.class_count)
        unlimited, _ = bounded_classes(aliases, bank.parent_indices, bank.class_count, .07, 20)
        bounded, local_diag = bounded_classes(aliases, bank.parent_indices, bank.class_count)
        saved = imagenet_geometry_logits(features, queries[key], SETTINGS)
        replay, changed, broad_diag = vip_profile_scores(features, queries[key], SETTINGS)
        _, unlimited_broad, _ = vip_profile_scores(features, queries[key], SETTINGS, 20)
        errors[key + "_local_identity"] = float((original - unlimited).abs().max())
        errors[key + "_vip_original_replay"] = float((saved - replay).abs().max())
        errors[key + "_vip_unrestricted_identity"] = float((saved - unlimited_broad).abs().max())
        errors[key + "_local_inactive_identity"] = local_diag["inactive_max_absolute_score_change"]
        errors[key + "_vip_inactive_identity"] = broad_diag["inactive_max_absolute_score_change"]
        if not bool(torch.isfinite(bounded).all() and torch.isfinite(changed).all()):
            raise ValueError("Nonfinite bounded scores.")
        diagnostics[key] = {"local": local_diag, "broad": broad_diag}
    unchanged = all(torch.equal(states[prefix + key], value) for prefix, model in (("geo_", geometry.backbone), ("vip_", vip.backbone))
                    for key, value in model.model.visual_model.head.state_dict().items())
    frozen = all(not value.requires_grad for model in (geometry.backbone, vip.backbone) for value in model.model.parameters())
    if any(errors.values()) or not unchanged or not frozen:
        raise ValueError(str({"errors": errors, "unchanged": unchanged, "frozen": frozen}))
    output.mkdir(parents=True)
    result = {"status": "complete", "implementation": IMPLEMENTATION, "config": CONFIG,
              "errors": errors, "head_weights_unchanged": unchanged, "weights_frozen": frozen,
              "target_masks_loaded": False, "diagnostics": diagnostics}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main(args):
    output = Path(args.output_dir)
    if output.exists() or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Existing output or invalid shard.")
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    if args.source_diagnostic:
        fixed = json.loads(args.source_diagnostic.read_text())["signature"]["samples"]
        lookup = {sample.key: sample for sample in samples}
        samples = [lookup[key] for key in fixed]
    keys = [sample.key for sample in samples]
    selected = samples[args.shard_index::args.num_shards]
    if not selected or len(keys) != len(set(keys)):
        raise ValueError("Empty or duplicate sample sequence.")
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    signature = {"implementation": IMPLEMENTATION, "dataset": args.dataset, "methods": METHODS,
        "classes": {key: bank.class_names for key, bank in banks.items()},
        "gear": {"geometry": asdict(geometry.config), "primary": PRIMARY, "bounded_alias": CONFIG,
                 "observation": asdict(SETTINGS), "upstream_commit": PINNED_COMMIT}, "competitive": None,
        "vocabulary": {"sha256": hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest(),
                       "aliases": {key: bank.alias_names for key, bank in banks.items()},
                       "counts": {key: [20] * bank.class_count for key, bank in banks.items()}},
        "checkpoints": checkpoint_manifest(checkpoints), "global_sample_count": len(keys),
        "global_sample_keys_sha256": digest(keys), "sample_keys": [sample.key for sample in selected],
        "sample_keys_sha256": digest([sample.key for sample in selected]),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "note": "Frozen rho=4 all20 factorial; original guarded VIP observation is borrowed. "
                "The cap limits effective-logit aggregation derivatives, not semantic correctness or the full VIP salience chain. "
                "No target-label tuning. Complete-image exploratory development screen; corrected IRRG Vaihingen."}
    output.mkdir(parents=True)
    matrices = {key: {method: np.zeros((bank.class_count,) * 2, np.int64) for method in METHODS} for key, bank in banks.items()}
    transitions = {key: {method: np.zeros((bank.class_count,) * 3, np.int64) for method in METHODS if method != "Geometry"}
                   for key, bank in banks.items()}
    pairs = {key: {name: np.zeros((bank.class_count,) * 3, np.int64) for name in FACTORIAL_PAIRS} for key, bank in banks.items()}
    per_image = {key: {method: [] for method in METHODS} for key in banks}
    ignored, diagnostics = dict.fromkeys(banks, 0), {key: {} for key in banks}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for number, sample in enumerate(selected, 1):
        image = load_image(sample.image_path if args.dataset == "loveda" else sample)
        predictions, current = predict_image(image, geometry, banks, vip, queries, output)
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
            for name, (old, new) in FACTORIAL_PAIRS.items():
                pairs[key][name] += transition_counts(predictions[key][old], predictions[key][new], target, bank.class_count)
            for field, value in current[key].items():
                diagnostics[key][field] = diagnostics[key].get(field, 0.0) + value
        result = {"status": "complete" if number == len(selected) else "running", "processed_images": number,
            "total_images": len(selected), "signature": signature,
            "metrics": {key: {method: summary(cm, banks[key].class_names, ignored[key]) for method, cm in group.items()}
                        for key, group in matrices.items()},
            "transitions": {key: {method: {"counts": value.tolist(), **transition_summary(value)} for method, value in group.items()}
                            for key, group in transitions.items()},
            "factorial_transitions": {key: {name: {"methods": FACTORIAL_PAIRS[name], "counts": value.tolist(), **transition_summary(value)}
                                           for name, value in group.items()} for key, group in pairs.items()},
            "diagnostics": {key: {field: value / number for field, value in group.items()} for key, group in diagnostics.items()},
            "wall_seconds": time.perf_counter() - started, "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1048576}
        temp = output / "results.tmp"
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(output / "results.json")
        print(json.dumps({"dataset": args.dataset, "processed": number, "total": len(selected),
              "miou": {key: {method: row["mean_iou_percent"] for method, row in group.items()}
                       for key, group in result["metrics"].items()}}), flush=True)
    np.savez_compressed(output / "per_image_confusions.npz", sample_keys=np.asarray(signature["sample_keys"]),
                        **{key + "__" + method: np.stack(values) for key, group in per_image.items() for method, values in group.items()})


if __name__ == "__main__":
    is_smoke = "--smoke" in sys.argv
    if is_smoke:
        sys.argv.remove("--smoke")
    args = parse_args()
    (smoke if is_smoke else main)(args)
