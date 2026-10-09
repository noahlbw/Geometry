"""Complete-image contextual rejection and matched-count vocabulary stress."""
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

from dinotool.excess_alias_rejection import (IMPLEMENTATION, METHODS, PRIMARY,
    ExcessAliasReader, ExcessRejectConfig, excess_delta, vocabulary_variants)
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.geometry_semantic_innovation import anchored_innovation
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.model import checkpoint_manifest
from dinotool.prompts import load_class_specs, serialize_class_specs
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from dinotool.stratified_soft_alias import WideCrop
from eval_bounded_alias_anchored import make_models
from eval_gear_ov import digest, protocol
from eval_geometry_semantic_innovation import SETTINGS, parse_args
from eval_geometry_vip_reliability import sample_broad, summary
from eval_matched_head_fov import resize_rgb
from eval_matched_head_text import imagenet_geometry_logits
from eval_stratified_soft_alias import tile_coordinates
from eval_vip_official_eight import PINNED_COMMIT


CONFIG = ExcessRejectConfig()
LOCAL_METHODS = tuple(method for method in METHODS if method != "BroadVIP")
SCENARIOS = ("clean", "wrong_parent", "paraphrase")


def make_variants(geometry, banks, vip, queries, specs):
    variants = {"clean": (banks, queries)}
    manifest = {scenario: {} for scenario in SCENARIOS}
    for scenario in SCENARIOS[1:]:
        new_banks, new_queries = {}, {}
        for key, classes in specs.items():
            choices, replacements = vocabulary_variants(classes)
            new_banks[key] = geometry.encode_text(choices[scenario])
            new_queries[key] = vip.encode_queries(new_banks[key].class_names,
                                                  tuple(spec.synonyms for spec in choices[scenario]))
            if new_banks[key].alias_names != new_queries[key].aliases:
                raise ValueError("Mismatched contextual aliases.")
            manifest[scenario][key] = {"classes": serialize_class_specs(choices[scenario]),
                                      "replacements": replacements[scenario]}
        variants[scenario] = new_banks, new_queries
    return variants, manifest


@torch.inference_mode()
def prepare_wide(image, vip, variants):
    resized = resize_rgb(image, 448)
    height, width = resized.shape[-2:]
    maps, crops, auxiliary = {}, {}, {}
    for scenario, (banks, queries) in variants.items():
        for key, query in queries.items():
            maps[scenario, key] = torch.zeros(len(query.class_names), height, width, device=vip.device)
            crops[scenario, key], auxiliary[scenario, key] = [], []
    count = torch.zeros(height, width, device=vip.device)
    for top in tile_starts(height, 336, 224):
        for left in tile_starts(width, 336, 224):
            ah, aw = min(336, height - top), min(336, width - left)
            rgb = F.pad(resized[:, top:top+ah, left:left+aw], (0, 336-aw, 0, 336-ah))
            features = vip.crop_patch_features(rgb)
            for scenario, (banks, queries) in variants.items():
                for key, query in queries.items():
                    maps[scenario, key][:, top:top+ah, left:left+aw] += imagenet_geometry_logits(features, query, SETTINGS)[:, :ah, :aw]
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        similarity = torch.einsum("bnd,mtd->bnmt", features, query.features.float()).mean(-1)[0]
                        patch_mean = F.normalize(features.mean(1), dim=-1)
                        text_mean = F.normalize(query.features.float().mean(1), dim=-1)
                        salience = ((patch_mean @ text_mean.T)[0] / SETTINGS.tem).float()
                        logits = (similarity * SETTINGS.logit_scale).float()
                    crops[scenario, key].append(WideCrop(logits, salience, top, left, ah, aw))
                    # Reliability compares physical views under the SAME RS text bank.
                    auxiliary[scenario, key].append((features.float() @ F.normalize(banks[key].features.float(), dim=-1).T)[0])
            count[top:top+ah, left:left+aw] += 1
    if not bool((count > 0).all()):
        raise ValueError("Incomplete wide observation coverage.")
    return {key: value / count[None] for key, value in maps.items()}, crops, auxiliary, count


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, variants, readers, work, methods=METHODS):
    height, width = image.shape[-2:]
    local_methods = tuple(method for method in methods if method != "BroadVIP")
    broad, crops, auxiliary, count = prepare_wide(image, vip, variants)
    text = {(scenario, key): F.normalize(bank.features.float(), dim=-1)
            for scenario, (group, _) in variants.items() for key, bank in group.items()}
    totals = {scenario: {key: {"tiles": 0} for key in banks} for scenario in variants}
    blend = hann_blend_window(512)
    with ExitStack() as stack:
        accumulators = {(scenario, key, method): stack.enter_context(ProbabilityAccumulator(bank.class_count, height, width, 256, work))
            for scenario in variants for key, bank in banks.items() for method in local_methods}
        for top in tile_starts(height, 512, 128):
            for left in tile_starts(width, 512, 128):
                prepared = geometry.prepare_image(_crop_at(image, top, left, 512).to(geometry.device))
                coordinates = tile_coordinates(top, left, geometry.device)
                valid = (coordinates[:, 0] < height) & (coordinates[:, 1] < width)
                ah, aw = min(512, height-top), min(512, width-left)
                for key, bank in banks.items():
                    raw_aliases = prepared.geometry_projected.float() @ text["clean", key].T
                    raw = alias_class_scores(raw_aliases, bank.parent_indices, bank.class_count)
                    local = raw / CONFIG.local_temperature
                    for scenario, (group, _) in variants.items():
                        b = sample_broad(broad[scenario, key], top, left, height, width).reshape_as(local)
                        anchored, _ = anchored_innovation(local, b, prepared.geometry_patch_conditional, valid[None])
                        aliases = prepared.geometry_projected.float() @ text[scenario, key].T
                        corrected, diagnostics = readers[scenario, key].read(aliases[0], prepared.geometry_patch_conditional[0],
                            coordinates, valid, b[0], crops[scenario, key], auxiliary[scenario, key], count, (height, width), SETTINGS.tau)
                        scores = {"Geometry": raw, "Anchored_VIP": anchored}
                        for method, observation in corrected.items():
                            scores[method], _ = anchored_innovation(local, observation[None], prepared.geometry_patch_conditional, valid[None])
                        for method, value in scores.items():
                            dense = F.interpolate(value.transpose(1, 2).reshape(1, bank.class_count, 32, 32), (512, 512),
                                                  mode="bilinear", align_corners=False)[0]
                            if method == "Geometry":
                                dense = dense / CONFIG.local_temperature
                            accumulators[scenario, key, method].add(dense[:, :ah, :aw].softmax(0).cpu().numpy(), blend[:ah, :aw], left, top)
                        totals[scenario][key]["tiles"] += 1
                        for field, value in diagnostics.items():
                            totals[scenario][key][field] = totals[scenario][key].get(field, 0.) + value
        predictions = {scenario: {} for scenario in variants}
        for scenario in variants:
            for key in banks:
                predictions[scenario][key] = {method: accumulators[scenario, key, method].finalize(None)[0] for method in local_methods}
                probabilities = F.interpolate(broad[scenario, key][None], (height, width), mode="bilinear", align_corners=False)[0].softmax(0)
                predictions[scenario][key]["BroadVIP"] = probabilities.argmax(0).cpu().numpy().astype(np.uint8)
    return predictions, {scenario: {key: {field: value if field == "tiles" else value / row["tiles"]
        for field, value in row.items()} for key, row in group.items()} for scenario, group in totals.items()}


@torch.inference_mode()
def smoke(args, reader_type=ExcessAliasReader, implementation=IMPLEMENTATION, primary=PRIMARY):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Existing smoke output.")
    samples, specs, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    geometry, banks, vip, queries, _ = make_models(args, specs)
    variants, manifest = make_variants(geometry, banks, vip, queries, specs)
    states = {prefix + key: value.clone() for prefix, model in (("geo_", geometry.backbone), ("vip_", vip.backbone))
              for key, value in model.model.visual_model.head.state_dict().items()}
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == "loveda" else sample)
    height, width = image.shape[-2:]
    broad, crops, auxiliary, count = prepare_wide(image, vip, variants)
    prepared = geometry.prepare_image(_crop_at(image, 0, 0, 512).to(geometry.device))
    coordinates = tile_coordinates(0, 0, geometry.device)
    valid = (coordinates[:, 0] < height) & (coordinates[:, 1] < width)
    diagnostics, errors = {}, {}
    for scenario, (group, _) in variants.items():
        for key, bank in group.items():
            reader = reader_type(bank, CONFIG)
            aliases = (prepared.geometry_projected.float() @ F.normalize(bank.features.float(), dim=-1).T)[0]
            b = sample_broad(broad[scenario, key], 0, 0, height, width).reshape(-1, bank.class_count)
            outputs, diagnostics[scenario + "__" + key] = reader.read(aliases, prepared.geometry_patch_conditional[0], coordinates,
                valid, b, crops[scenario, key], auxiliary[scenario, key], count, (height, width), SETTINGS.tau)
            evidence = crops[scenario, key][0].alias_logits[:, reader.members]
            errors[scenario + "__" + key + "_uniform"] = float(excess_delta(evidence, torch.ones_like(evidence)).abs().max())
            errors[scenario + "__" + key + "_monotone"] = float((outputs[primary] - b).clamp_min(0).max())
            if not all(bool(torch.isfinite(value).all()) for value in outputs.values()):
                raise ValueError("Nonfinite candidate.")
    frozen = all(not value.requires_grad for model in (geometry.backbone, vip.backbone) for value in model.model.parameters())
    unchanged = all(torch.equal(states[prefix + key], value) for prefix, model in (("geo_", geometry.backbone), ("vip_", vip.backbone))
                    for key, value in model.model.visual_model.head.state_dict().items())
    if any(errors.values()) or not frozen or not unchanged:
        raise ValueError("Mask-free smoke failed.")
    output.mkdir(parents=True)
    result = {"status": "complete", "implementation": implementation, "config": asdict(CONFIG),
              "errors": errors, "diagnostics": diagnostics, "vocabulary_stress": manifest,
              "target_masks_loaded": False, "weights_frozen": frozen, "head_weights_unchanged": unchanged}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main(args, reader_type=ExcessAliasReader, implementation=IMPLEMENTATION, methods=METHODS):
    output = Path(args.output_dir)
    if output.exists() or args.num_shards != 1 or args.shard_index != 0:
        raise ValueError("New single-shard output required.")
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    keys = json.loads(args.source_diagnostic.read_text())["signature"]["samples"]
    lookup = {sample.key: sample for sample in samples}
    selected = [lookup[key] for key in keys]
    if not selected or len(keys) != len(set(keys)):
        raise ValueError("Empty or duplicate sample sequence.")
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    variants, manifest = make_variants(geometry, banks, vip, queries, specs)
    readers = {(scenario, key): reader_type(bank, CONFIG)
               for scenario, (group, _) in variants.items() for key, bank in group.items()}
    output.mkdir(parents=True)
    (output / "selection.json").write_text(json.dumps({scenario + "__" + key: reader.report()
        for (scenario, key), reader in readers.items()}, indent=2) + "\n")
    signature = {"implementation": implementation, "dataset": args.dataset, "methods": methods,
        "classes": {key: bank.class_names for key, bank in banks.items()},
        "gear": {"geometry": asdict(geometry.config), "observation": asdict(SETTINGS), "upstream_commit": PINNED_COMMIT,
                 "excess_reject": asdict(CONFIG)}, "vocabulary": {
            "sha256": hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest(),
            "aliases": {key: bank.alias_names for key, bank in banks.items()},
            "counts": {key: [20] * bank.class_count for key, bank in banks.items()}},
        "checkpoints": checkpoint_manifest(checkpoints), "sample_keys": keys, "sample_keys_sha256": digest(keys),
        "global_sample_count": len(keys), "global_sample_keys_sha256": digest(keys), "num_shards": 1, "shard_index": 0}
    matrices = {scenario: {key: {method: np.zeros((bank.class_count,) * 2, np.int64) for method in methods}
                          for key, bank in banks.items()} for scenario in SCENARIOS}
    pairs = {scenario: {key: {method: np.zeros((bank.class_count,) * 3, np.int64) for method in methods if method != "Anchored_VIP"}
                       for key, bank in banks.items()} for scenario in SCENARIOS}
    per_image = {scenario: {key: {method: [] for method in methods} for key in banks} for scenario in SCENARIOS}
    ignored = {scenario: dict.fromkeys(banks, 0) for scenario in SCENARIOS}
    totals = {scenario: {key: {} for key in banks} for scenario in SCENARIOS}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for number, sample in enumerate(selected, 1):
        image = load_image(sample.image_path if args.dataset == "loveda" else sample)
        active = variants if number <= 8 else {"clean": variants["clean"]}
        predictions, diagnostics = predict_image(image, geometry, banks, vip, active, readers, output, methods)
        # Masks are opened only after all candidate predictions for this image.
        for key, bank in banks.items():
            target = load_mask(sample, key, tuple(image.shape[-2:]))
            valid = (target >= 0) & (target < bank.class_count)
            for scenario in active:
                ignored[scenario][key] += int((~valid).sum())
                for method in methods:
                    pred = predictions[scenario][key][method]
                    encoded = target[valid].astype(np.int64) * bank.class_count + pred[valid]
                    cm = np.bincount(encoded, minlength=bank.class_count ** 2).reshape(bank.class_count, -1)
                    matrices[scenario][key][method] += cm
                    per_image[scenario][key][method].append(cm)
                    if method != "Anchored_VIP":
                        pairs[scenario][key][method] += transition_counts(predictions[scenario][key]["Anchored_VIP"], pred, target, bank.class_count)
                for field, value in diagnostics[scenario][key].items():
                    totals[scenario][key][field] = totals[scenario][key].get(field, 0.) + value
        metrics = {scenario: {key: {method: summary(cm, banks[key].class_names, ignored[scenario][key])
            for method, cm in group.items()} for key, group in protocols.items()} for scenario, protocols in matrices.items()}
        result = {"status": "complete" if number == len(selected) else "running", "processed_images": number,
            "total_images": len(keys), "signature": signature, "metrics": metrics["clean"],
            "stress_metrics": {scenario: metrics[scenario] for scenario in SCENARIOS[1:]},
            "stress_sample_keys": keys[:min(number, 8)], "vocabulary_stress": manifest,
            "transitions": {scenario: {key: {method: {"counts": value.tolist(), **transition_summary(value)}
                for method, value in group.items()} for key, group in protocols.items()} for scenario, protocols in pairs.items()},
            "diagnostics": {scenario: {key: {field: value / (number if scenario == "clean" else min(number, 8))
                for field, value in row.items()} for key, row in group.items()} for scenario, group in totals.items()},
            "wall_seconds": time.perf_counter() - started, "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1048576,
            "stress_scope": "Only broad vocabulary changes; local original20 anchor fixed. First8 complete images per dataset; not fresh LLM outputs."}
        temporary = output / "results.tmp"
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(output / "results.json")
        print(json.dumps({"dataset": args.dataset, "processed": number, "total": len(keys),
            "clean": {key: {method: row["mean_iou_percent"] for method, row in group.items()} for key, group in metrics["clean"].items()}}), flush=True)
    np.savez_compressed(output / "per_image_confusions.npz", sample_keys=np.asarray(keys),
        **{key + "__" + method: np.stack(values) for key, group in per_image["clean"].items() for method, values in group.items()})
    np.savez_compressed(output / "stress_per_image_confusions.npz", sample_keys=np.asarray(keys[:8]),
        **{scenario + "__" + key + "__" + method: np.stack(values) for scenario in SCENARIOS
           for key, group in per_image[scenario].items() for method, values in group.items()
           if scenario != "clean" or len(keys) == 8},
        **({"clean__" + key + "__" + method: np.stack(values[:8])
            for key, group in per_image["clean"].items() for method, values in group.items()} if len(keys) != 8 else {}))


if __name__ == "__main__":
    is_smoke = "--smoke" in sys.argv
    if is_smoke:
        sys.argv.remove("--smoke")
    (smoke if is_smoke else main)(parse_args())
