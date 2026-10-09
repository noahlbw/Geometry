"""Test rival-specific local aliases with unchanged Geometry and VIP observations."""
from contextlib import ExitStack
from dataclasses import asdict
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
from dinotool.pair_conditional_alias import IMPLEMENTATION, METHODS, PAIRS, PRIMARY, PairAliasConfig, PairConditionalAliases
from dinotool.prompts import load_class_specs
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from eval_bounded_alias_anchored import make_models
from eval_gear_ov import digest, protocol
from eval_geometry_semantic_innovation import SETTINGS, parse_args
from eval_geometry_vip_reliability import sample_broad, summary
from eval_matched_head_fov import resize_rgb
from eval_matched_head_text import imagenet_geometry_logits
from eval_vip_official_eight import PINNED_COMMIT


CONFIG = PairAliasConfig()
LOCAL_METHODS = tuple(method for method in METHODS if method not in ("BroadVIP", "MeanProb_VIP"))


@torch.inference_mode()
def prepare_wide(image, vip, queries):
    resized = resize_rgb(image, 448)
    h, w = resized.shape[-2:]
    maps = {key: torch.zeros(len(query.class_names), h, w, device=vip.device)
            for key, query in queries.items()}
    count = torch.zeros(h, w, device=vip.device)
    for top in tile_starts(h, 336, 224):
        for left in tile_starts(w, 336, 224):
            ah, aw = min(336, h - top), min(336, w - left)
            rgb = F.pad(resized[:, top:top + ah, left:left + aw], (0, 336 - aw, 0, 336 - ah))
            features = vip.crop_patch_features(rgb)
            for key, query in queries.items():
                logits = imagenet_geometry_logits(features, query, SETTINGS)
                maps[key][:, top:top + ah, left:left + aw] += logits[:, :ah, :aw]
            count[top:top + ah, left:left + aw] += 1
    if not bool((count > 0).all()):
        raise ValueError("Uncovered wide-view pixels.")
    return {key: value / count[None] for key, value in maps.items()}


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, readers, work):
    h, w = image.shape[-2:]
    before_empty, before_rows = vip.empty_rows, vip.observed_rows
    broad = prepare_wide(image, vip, queries)
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
                    original, _ = anchored_innovation(g, b, prepared.geometry_patch_conditional, valid)
                    variants, stats = readers[key].read(aliases, valid)
                    scores = {"Geometry": raw, "MeanLogit_VIP": .5 * (g + b), "Anchored_VIP": original}
                    for variant, local_name, coupled_name in (("selected", "PairAlias_Local", PRIMARY),
                            ("random", "RandomPair_Local", "RandomPair_Coupled"),
                            ("fixed_count", "FixedCount_Local", "FixedCount_Coupled")):
                        scores[local_name] = variants[variant]
                        scores[coupled_name], diag = anchored_innovation(variants[variant] / .07, b,
                                                                       prepared.geometry_patch_conditional, valid)
                        if variant == "selected":
                            stats.update({"solver_" + field: value for field, value in diag.items()})
                    for method, value in scores.items():
                        dense = F.interpolate(value.transpose(1, 2).reshape(1, bank.class_count, 32, 32),
                                              size=(512, 512), mode="bilinear", align_corners=False)[0]
                        if method in ("Geometry", "PairAlias_Local", "RandomPair_Local", "FixedCount_Local"):
                            dense = dense / .07
                        accumulators[key, method].add(dense[:, :ah, :aw].softmax(0).cpu().numpy(),
                                                     blend[:ah, :aw], left, top)
                    totals[key]["tiles"] += 1
                    for field, value in stats.items():
                        totals[key][field] = totals[key].get(field, 0.0) + value
        predictions = {key: {method: accumulators[key, method].finalize(None)[0] for method in LOCAL_METHODS}
                       for key in banks}
        for key in banks:
            probabilities = F.interpolate(broad[key][None], size=(h, w), mode="bilinear",
                                          align_corners=False)[0].softmax(0).cpu().numpy()
            predictions[key]["BroadVIP"] = probabilities.argmax(0).astype(np.uint8)
            fused = np.empty((h, w), dtype=np.uint8)
            accumulator = accumulators[key, "Geometry"]
            for top in range(0, h, 128):
                end = min(top + 128, h)
                local = accumulator.probabilities[:, top:end] / accumulator.normalizer[None, top:end].clip(1e-8)
                fused[top:end] = (local + probabilities[:, top:end]).argmax(0).astype(np.uint8)
            predictions[key]["MeanProb_VIP"] = fused
    diagnostics = {key: {field: value if field in ("tiles", "observer_empty_rows", "observer_rows") else value / row["tiles"]
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
    errors, diagnostics, selections = {}, {}, {}
    for key, bank in banks.items():
        reader = PairConditionalAliases(bank, CONFIG)
        aliases = prepared.geometry_projected.float() @ F.normalize(bank.features.float(), dim=-1).T
        variants, diagnostics[key] = reader.read(aliases)
        if not all(bool(torch.isfinite(value).all()) for value in variants.values()):
            raise ValueError("Nonfinite selected readout.")
        selections[key] = reader.report()
        reader.selected = [torch.ones_like(mask) for mask in reader.selected]
        reader.random = [torch.ones_like(mask) for mask in reader.random]
        unlimited, _ = reader.read(aliases)
        original = alias_class_scores(aliases, bank.parent_indices, bank.class_count)
        errors[key + "_unrestricted_identity"] = max(float((value - original).abs().max()) for value in unlimited.values())
    broad = prepare_wide(image, vip, queries)
    if not all(bool(torch.isfinite(value).all()) for value in broad.values()):
        raise ValueError("Nonfinite original wide observer.")
    unchanged = all(torch.equal(states[prefix + key], value) for prefix, model in (("geo_", geometry.backbone), ("vip_", vip.backbone))
                    for key, value in model.model.visual_model.head.state_dict().items())
    frozen = all(not value.requires_grad for model in (geometry.backbone, vip.backbone) for value in model.model.parameters())
    if any(errors.values()) or not unchanged or not frozen:
        raise ValueError("Exact fallback/frozen-weight smoke failed.")
    output.mkdir(parents=True)
    result = {"status": "complete", "implementation": IMPLEMENTATION, "config": asdict(CONFIG),
              "errors": errors, "head_weights_unchanged": unchanged, "weights_frozen": frozen,
              "target_masks_loaded": False, "diagnostics": diagnostics, "selection": selections}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "selection"}), flush=True)


def main(args):
    output = Path(args.output_dir)
    if output.exists() or args.num_shards != 1 or args.shard_index != 0:
        raise ValueError("This screen requires a new single-shard output.")
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    fixed = json.loads(args.source_diagnostic.read_text())["signature"]["samples"]
    lookup = {sample.key: sample for sample in samples}
    selected = [lookup[key] for key in fixed]
    if not selected or len(fixed) != len(set(fixed)):
        raise ValueError("Empty or duplicate sample sequence.")
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    readers = {key: PairConditionalAliases(bank, CONFIG) for key, bank in banks.items()}
    selections = {key: reader.report() for key, reader in readers.items()}
    signature = {"implementation": IMPLEMENTATION, "dataset": args.dataset, "methods": METHODS,
        "classes": {key: bank.class_names for key, bank in banks.items()},
        "gear": {"geometry": asdict(geometry.config), "primary": PRIMARY, "pair_alias": asdict(CONFIG),
                 "observation": asdict(SETTINGS), "upstream_commit": PINNED_COMMIT}, "competitive": None,
        "vocabulary": {"sha256": hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest(),
                       "aliases": {key: bank.alias_names for key, bank in banks.items()},
                       "counts": {key: [20] * bank.class_count for key, bank in banks.items()}},
        "checkpoints": checkpoint_manifest(checkpoints), "global_sample_count": len(fixed),
        "global_sample_keys_sha256": digest(fixed), "sample_keys": fixed, "sample_keys_sha256": digest(fixed),
        "num_shards": 1, "shard_index": 0,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "note": "Pair-specific text subsets; original Geometry top2; all20 remain available; "
                "same-count random and fixed original denominator controls. Broad VIP unchanged/borrowed. "
                "No masks in selection. Development screen, not independent validation or semantic truth guarantee."}
    output.mkdir(parents=True)
    (output / "selection.json").write_text(json.dumps(selections, indent=2) + "\n")
    matrices = {key: {method: np.zeros((bank.class_count,) * 2, np.int64) for method in METHODS} for key, bank in banks.items()}
    transitions = {key: {method: np.zeros((bank.class_count,) * 3, np.int64) for method in METHODS if method != "Geometry"}
                   for key, bank in banks.items()}
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
                diagnostics[key][field] = diagnostics[key].get(field, 0.0) + value
        result = {"status": "complete" if number == len(selected) else "running", "processed_images": number,
            "total_images": len(selected), "signature": signature,
            "metrics": {key: {method: summary(cm, banks[key].class_names, ignored[key]) for method, cm in group.items()}
                        for key, group in matrices.items()},
            "transitions": {key: {method: {"counts": value.tolist(), **transition_summary(value)} for method, value in group.items()}
                            for key, group in transitions.items()},
            "factorial_transitions": {key: {name: {"methods": PAIRS[name], "counts": value.tolist(), **transition_summary(value)}
                                           for name, value in group.items()} for key, group in pairs.items()},
            "diagnostics": {key: {field: value / number for field, value in group.items()} for key, group in diagnostics.items()},
            "wall_seconds": time.perf_counter() - started, "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1048576}
        temp = output / "results.tmp"
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(output / "results.json")
        print(json.dumps({"dataset": args.dataset, "processed": number, "total": len(selected),
                          "miou": {key: {method: row["mean_iou_percent"] for method, row in group.items()}
                                   for key, group in result["metrics"].items()}}), flush=True)
    np.savez_compressed(output / "per_image_confusions.npz", sample_keys=np.asarray(fixed),
                        **{key + "__" + method: np.stack(values) for key, group in per_image.items() for method, values in group.items()})


if __name__ == "__main__":
    is_smoke = "--smoke" in sys.argv
    if is_smoke:
        sys.argv.remove("--smoke")
    (smoke if is_smoke else main)(parse_args())
