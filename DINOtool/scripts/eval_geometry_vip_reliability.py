#!/usr/bin/env python3
"""Full Geometry/VIP complementarity and image-only branch-selection audit."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.branch_reliability import BranchReliabilitySelector
from dinotool.gear_ov import GearOVSegmenter
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.semantic_correction_audit import transition_counts
from dinotool.vip_official_adapter import VIPOfficialAdapter, upstream_settings
from eval_matched_head_fov import CROP, STRIDE, digest, parse_args, resize_rgb
from eval_matched_head_text import imagenet_geometry_logits
from eval_vip_official_eight import Confusion, PINNED_COMMIT, checkpoint_config, protocol


IMPLEMENTATION = "geometry-vip-reliability-audit-v1-20260930"
ARMS = ("Geometry", "Multiscale", "VIP20", "MeanProb50", "MaxConfidence",
        "LeaveFamilyOut", "GroundedLeaveFamilyOut")
SELECTORS = ("MaxConfidence", "LeaveFamilyOut", "GroundedLeaveFamilyOut")


def summary(matrix: np.ndarray, names: tuple[str, ...], ignored: int) -> dict:
    confusion = Confusion(names, len(names))
    confusion.matrix = matrix
    confusion.ignored = ignored
    result = confusion.summary()
    target, predicted = matrix.sum(1), matrix.sum(0)
    for index, entry in enumerate(result["per_class"]):
        tp = int(matrix[index, index])
        entry.update(target_pixels=int(target[index]), predicted_pixels=int(predicted[index]),
                     precision_percent=round(100 * tp / max(int(predicted[index]), 1), 4),
                     recall_percent=round(100 * tp / max(int(target[index]), 1), 4),
                     predicted_area_percent=round(100 * int(predicted[index]) / max(int(target.sum()), 1), 4))
    return result


@torch.inference_mode()
def broad_vip(image, vip, queries, settings):
    resized = resize_rgb(image, 448)
    h, w = resized.shape[-2:]
    aliases = len(queries.aliases)
    classes = len(queries.class_names)
    totals = torch.zeros((classes, h, w), device=vip.device)
    alias_totals = torch.zeros((aliases, h, w), device=vip.device)
    count = torch.zeros((h, w), device=vip.device)
    crops = 0
    for top in tile_starts(h, CROP, CROP - STRIDE):
        for left in tile_starts(w, CROP, CROP - STRIDE):
            ah, aw = min(CROP, h - top), min(CROP, w - left)
            rgb = resized[:, top:top + ah, left:left + aw]
            if ah < CROP or aw < CROP:
                rgb = F.pad(rgb, (0, CROP - aw, 0, CROP - ah))
            features = vip.crop_patch_features(rgb)
            logits = imagenet_geometry_logits(features, queries, settings)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                scores = torch.einsum("bnd,mtd->bnmt", features,
                                      queries.features.float()).mean(-1)[0]
            alias_map = F.interpolate(scores.T.float().reshape(1, aliases, 21, 21),
                                      size=(CROP, CROP), mode="bilinear", align_corners=False)[0]
            totals[:, top:top + ah, left:left + aw] += logits[:, :ah, :aw]
            alias_totals[:, top:top + ah, left:left + aw] += alias_map[:, :ah, :aw]
            count[top:top + ah, left:left + aw] += 1
            crops += 1
    if not bool((count > 0).all()):
        raise ValueError("VIP broad view has uncovered pixels.")
    return totals / count[None], alias_totals / count[None], crops


def sample_broad(values, top, left, image_h, image_w):
    y = (top + (torch.arange(32, device=values.device).float() + 0.5) * 16) / image_h * 2 - 1
    x = (left + (torch.arange(32, device=values.device).float() + 0.5) * 16) / image_w * 2 - 1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xx, yy), -1)[None]
    return F.grid_sample(values[None], grid, mode="bilinear", padding_mode="border",
                         align_corners=False)[0].permute(1, 2, 0)


@torch.inference_mode()
def predict_image(image, geometry, text, selector, vip, queries, settings, work):
    h, w = image.shape[-2:]
    classes = text.class_count
    wide_logits, wide_alias, broad_crops = broad_vip(image, vip, queries, settings)
    vip_prob = F.interpolate(wide_logits[None], size=(h, w), mode="bilinear",
                             align_corners=False)[0].softmax(0).cpu().numpy()
    blend = hann_blend_window(512)
    gates = {key: np.zeros((h, w), dtype=np.float32)
             for key in ("LeaveFamilyOut", "GroundedLeaveFamilyOut")}
    normalizer = np.zeros((h, w), dtype=np.float32)
    diagnostics = {"tiles": 0, "broad_crops": broad_crops}
    with ExitStack() as stack:
        accumulators = {key: stack.enter_context(ProbabilityAccumulator(classes, h, w, 2048, work))
                        for key in ("Geometry", "Multiscale")}
        for top in tile_starts(h, 512, 128):
            for left in tile_starts(w, 512, 128):
                views = geometry.prepare_views(image, top, left)
                readout = geometry.read_views(views, text, reconstruct=False)
                ah, aw = views.actual_height, views.actual_width
                weights = blend[:ah, :aw]
                for key, logits in (("Geometry", readout.local_logits),
                                    ("Multiscale", readout.multiscale_logits)):
                    probability = (logits.float() / 0.07).softmax(1)[0, :, :ah, :aw]
                    accumulators[key].add(probability.cpu().numpy(), weights, left, top)
                local_alias = views.local_aligned.float() @ text.features.float().T
                vip_alias = sample_broad(wide_alias, top, left, h, w)
                vl = sample_broad(wide_logits, top, left, h, w).argmax(-1)
                gl = F.interpolate(readout.multiscale_logits.float(), size=(32, 32),
                                   mode="bilinear", align_corners=False)[0].argmax(0)
                raw = F.avg_pool2d(views.detail_raw.permute(2, 0, 1)[None], 2)[0].permute(1, 2, 0)
                valid = ((torch.arange(32, device=geometry.device)[:, None] * 16 < ah)
                         & (torch.arange(32, device=geometry.device)[None] * 16 < aw))
                decisions, current = selector.select(local_alias, vip_alias, raw, gl, vl, valid)
                for key, decision in decisions.items():
                    dense = F.interpolate(decision.float()[None, None], size=(512, 512),
                                          mode="bilinear", align_corners=False)[0, 0, :ah, :aw]
                    gates[key][top:top + ah, left:left + aw] += dense.cpu().numpy() * weights
                normalizer[top:top + ah, left:left + aw] += weights
                diagnostics["tiles"] += 1
                for key, value in current.items():
                    diagnostics[key] = diagnostics.get(key, 0) + value
                del views, readout
        if np.any(normalizer <= 0):
            raise ValueError("Geometry windows left image pixels uncovered.")
        predictions = {key: accumulators[key].finalize(None)[0]
                       for key in ("Geometry", "Multiscale")}
        gv = predictions["Multiscale"]
        vv = vip_prob.argmax(0).astype(np.uint8)
        predictions["VIP20"] = vv
        # Chunk final branch operations to avoid creating several full-size
        # class-probability copies on panorama images.
        choices = {key: np.zeros((h, w), dtype=bool) for key in SELECTORS}
        predictions["MeanProb50"] = np.empty((h, w), dtype=np.uint8)
        for top in range(0, h, 128):
            end = min(top + 128, h)
            accumulator = accumulators["Multiscale"]
            gp = accumulator.probabilities[:, top:end] / accumulator.normalizer[None, top:end].clip(1e-8)
            vp = vip_prob[:, top:end]
            predictions["MeanProb50"][top:end] = (gp + vp).argmax(0).astype(np.uint8)
            choices["MaxConfidence"][top:end] = vp.max(0) > gp.max(0)
            for key in gates:
                choices[key][top:end] = (gates[key][top:end] / normalizer[top:end]) >= 0.5
        for key, choice in choices.items():
            predictions[key] = np.where(choice, vv, gv).astype(np.uint8)
    return predictions, choices, diagnostics


def correctness_audit(predictions, choices, target, classes):
    valid = (target >= 0) & (target < classes)
    truth = target[valid]
    g = predictions["Multiscale"][valid]
    v = predictions["VIP20"][valid]
    gc, vc = g == truth, v == truth
    patterns = gc.astype(np.int64) + 2 * vc.astype(np.int64)
    counts = np.bincount(truth * 4 + patterns, minlength=classes * 4).reshape(classes, 4)
    result = {"correctness_by_true_class": counts.tolist(), "routing": {}, "changes": {},
              "pair_transition_counts": transition_counts(g, v, truth, classes).tolist()}
    for key, choice in choices.items():
        picked = choice[valid]
        beneficial = ~gc & vc
        harmful = gc & ~vc
        result["routing"][key] = {
            "selected_vip_on_disagreement": int((picked & (g != v)).sum()),
            "available_beneficial": int(beneficial.sum()),
            "retained_beneficial": int((picked & beneficial).sum()),
            "available_harmful": int(harmful.sum()),
            "rejected_harmful": int((~picked & harmful).sum()),
            "correct_xor_routes": int(((picked & beneficial) | (~picked & harmful)).sum()),
            "xor_total": int((beneficial | harmful).sum()),
        }
    for baseline in ("Multiscale", "VIP20"):
        base = predictions[baseline][valid]
        bc = base == truth
        result["changes"][baseline] = {}
        for key, prediction in predictions.items():
            other = prediction[valid]
            oc = other == truth
            fixed, broken = ~bc & oc, bc & ~oc
            result["changes"][baseline][key] = {
                "changed": int((base != other).sum()), "beneficial": int(fixed.sum()),
                "harmful": int(broken.sum()),
                "wrong_to_wrong": int(((base != other) & ~bc & ~oc).sum()),
                "beneficial_by_true_class": np.bincount(truth[fixed], minlength=classes).tolist(),
                "harmful_by_true_class": np.bincount(truth[broken], minlength=classes).tolist(),
            }
    return result


def add_counts(left, right):
    if isinstance(right, dict):
        old = left or {}
        return {key: add_counts(old.get(key), value) for key, value in right.items()}
    if isinstance(right, list):
        old = left if left is not None else [None] * len(right)
        if len(old) != len(right):
            raise ValueError("Count arrays have different lengths.")
        return [add_counts(a, b) for a, b in zip(old, right)]
    return (left or 0) + right


def main(args):
    if (not 0 < args.memory_fraction <= 1 or args.max_images < 0 or args.progress_every < 1
            or not 0 <= args.shard_index < args.num_shards):
        raise ValueError("Invalid evaluation parameters.")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing existing output: {output}")
    upstream = Path(args.upstream_root).resolve()
    commit = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if commit != PINNED_COMMIT:
        raise ValueError("Unexpected VIP revision.")
    all_samples, scored, load_image, load_mask = protocol(args)
    if args.max_images:
        random.Random(args.sample_seed).shuffle(all_samples)
        all_samples = all_samples[:args.max_images]
    keys = [item.key for item in all_samples]
    samples = all_samples[args.shard_index::args.num_shards]
    if not samples or len(keys) != len(set(keys)):
        raise ValueError("Empty shard or duplicate keys.")
    specs = load_class_specs(args.vocabulary_config)
    names = tuple(spec.name for spec in specs)
    if names != tuple(scored[args.dataset]) or any(len(spec.synonyms) != 20 for spec in specs):
        raise ValueError("This audit fixes the matched 20-alias taxonomy.")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    checkpoints = checkpoint_config(args)
    backbone = DINOTextSegmenter(checkpoints, device=args.device, amp=True)
    geometry = GearOVSegmenter(backbone)
    bank = geometry.encode_text(specs)
    text = geometry.text_basis(bank)
    selector = BranchReliabilitySelector(bank)
    # The official adapter converts its runtime to fp16. Keep a separate frozen
    # copy so this cannot alter the historical fp32-weight/bf16-AMP Geometry
    # control. Both copies load exactly the same checkpoint files.
    vip_backbone = DINOTextSegmenter(checkpoints, device=args.device, amp=True)
    vip = VIPOfficialAdapter(vip_backbone, upstream)
    queries = vip.encode_queries(names, tuple(spec.synonyms for spec in specs))
    if bank.alias_names != queries.aliases or not torch.equal(bank.parent_indices, queries.parents):
        raise ValueError("Geometry and VIP alias groups differ.")
    settings = upstream_settings(args.dataset)
    shard_keys = [item.key for item in samples]
    signature = {
        "implementation": IMPLEMENTATION, "dataset": args.dataset, "arms": ARMS,
        "class_names": names, "alias_names": bank.alias_names, "alias_counts": [20] * len(names),
        "geometry": {"config": geometry.config_dict(), "overlap": 128,
                     "output_temperature": 0.07, "blend": "Hann probabilities; existing Multiscale"},
        "vip": {**asdict(settings), "background": False, "prob_thd": None,
                "templates": "pinned ImageNet per-template mean similarities"},
        "selector": selector.config_dict(), "routing_patch_size": 16,
        "routing_blend_threshold": 0.5, "upstream_commit": commit,
        "vocabulary_sha256": hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest(),
        "checkpoint_manifest": checkpoint_manifest(checkpoints),
        "runtime_copies": "Separate frozen copies preserve Geometry fp32 weights/bf16 AMP and official VIP fp16 weights",
        "global_sample_count": len(keys), "global_sample_keys_sha256": digest(keys),
        "sample_keys": shard_keys, "sample_keys_sha256": digest(shard_keys),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "sample_seed": args.sample_seed, "max_images": args.max_images,
        "note": "Native strong branch profiles differ in view/text/scorer; same vocabulary, masks and set. "
                "Not a head-only matched factorial. Masks loaded only after all predictions and decisions. "
                "Oracle correctness patterns are audit-only; no target-label tuning.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "signature.json").write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    matrices = {key: Confusion(names, len(names)) for key in ARMS}
    audit, diagnostics = {}, {}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(backbone.device)
    for number, sample in enumerate(samples, 1):
        image = load_image(sample)
        prediction, choices, current_diagnostics = predict_image(
            image, geometry, text, selector, vip, queries, settings, output)
        target = load_mask(sample, args.dataset, tuple(image.shape[-2:]))
        for key in ARMS:
            matrices[key].update(prediction[key], target)
        audit = add_counts(audit, correctness_audit(prediction, choices, target, len(names)))
        diagnostics = add_counts(diagnostics, current_diagnostics)
        if number % args.progress_every == 0 or number == len(samples):
            payload = {
                "status": "complete" if number == len(samples) else "running",
                "processed_images": number, "total_images": len(samples),
                "metrics": {key: summary(value.matrix, names, value.ignored) for key, value in matrices.items()},
                "audit": audit, "diagnostics": diagnostics,
                "wall_seconds": time.perf_counter() - started,
                "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(backbone.device) / 1048576,
                "signature": signature,
            }
            (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"dataset": args.dataset, "shard": args.shard_index,
                              "processed": number, "total": len(samples),
                              "miou": {key: payload["metrics"][key]["mean_iou_percent"] for key in ARMS}}), flush=True)
    return payload


if __name__ == "__main__":
    main(parse_args())
