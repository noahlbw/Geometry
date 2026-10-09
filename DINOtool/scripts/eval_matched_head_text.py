#!/usr/bin/env python3
"""Matched visual head by text-template representation with VIP scoring."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.inference import tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from dinotool.vip_official_adapter import VIPOfficialAdapter, VIPQueries, VIPSettings, upstream_settings
from eval_matched_head_fov import CROP, STRIDE, count_changes, digest, parse_args, resize_rgb
from eval_matched_head_score import vip_class_logits
from eval_vip_official_eight import Confusion, PINNED_COMMIT, checkpoint_config, protocol


ARMS = ("Geometry_RS", "VIP_RS", "Geometry_ImageNet", "VIP_ImageNet")
LONG_EDGE = 448


def imagenet_geometry_logits(features: torch.Tensor, queries: VIPQueries,
                             settings: VIPSettings) -> torch.Tensor:
    if features.ndim != 3 or features.shape[1] != 21 * 21:
        raise ValueError("Geometry must return the same 21x21 patch grid as VIP.")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        similarities = torch.einsum("bnd,mtd->bnmt", features, queries.features.float()).mean(-1)[0]
        patch_mean = F.normalize(features.mean(1), dim=-1)
        text_mean = F.normalize(queries.features.float().mean(1), dim=-1)
        salience = ((patch_mean @ text_mean.T)[0] / settings.tem).float()
        alias_logits = (similarities * settings.logit_scale).T.reshape(-1, 21, 21)
        class_logits = []
        for index in range(len(queries.class_names)):
            members = queries.parents == index
            weights = salience[members].softmax(0)
            scaled = alias_logits[members] * (weights / weights.mean())[:, None, None]
            class_logits.append(torch.logsumexp(settings.tau * scaled, dim=0) / settings.tau)
        logits = torch.stack(class_logits)[None]
        return F.interpolate(logits, size=(CROP, CROP), mode="bilinear",
                             align_corners=False)[0].float()


@torch.inference_mode()
def predict_image(image: torch.Tensor, geometry: TCPRSegmenter,
                  vip: VIPOfficialAdapter, rs_text: torch.Tensor,
                  rs_parents: torch.Tensor, queries: VIPQueries,
                  settings: VIPSettings) -> tuple[dict[str, np.ndarray], int]:
    original_size = tuple(image.shape[-2:])
    resized = resize_rgb(image, LONG_EDGE)
    height, width = resized.shape[-2:]
    classes = len(queries.class_names)
    totals = {arm: torch.zeros((classes, height, width), device=geometry.device) for arm in ARMS}
    count = torch.zeros((height, width), device=geometry.device)
    crops = 0
    for top in tile_starts(height, CROP, CROP - STRIDE):
        for left in tile_starts(width, CROP, CROP - STRIDE):
            actual_height = min(CROP, height - top)
            actual_width = min(CROP, width - left)
            rgb = resized[:, top:top + actual_height, left:left + actual_width]
            if actual_height < CROP or actual_width < CROP:
                rgb = F.pad(rgb, (0, CROP - actual_width, 0, CROP - actual_height))
            prepared = geometry.prepare_image(rgb[None])
            geo_features = prepared.geometry_projected
            vip_features = vip.crop_patch_features(rgb)
            crop_logits = {
                "Geometry_RS": vip_class_logits(geo_features, rs_text, rs_parents,
                                                classes, settings.tem, settings.tau),
                "VIP_RS": vip_class_logits(vip_features, rs_text, rs_parents,
                                           classes, settings.tem, settings.tau),
                "Geometry_ImageNet": imagenet_geometry_logits(geo_features, queries, settings),
                "VIP_ImageNet": vip.crop_logits(rgb, queries, settings),
            }
            for arm, logits in crop_logits.items():
                totals[arm][:, top:top + actual_height, left:left + actual_width] += (
                    logits[:, :actual_height, :actual_width]
                )
            count[top:top + actual_height, left:left + actual_width] += 1
            crops += 1
            del prepared, geo_features, vip_features
    if not bool((count > 0).all()):
        raise AssertionError("Sliding windows left image pixels uncovered.")
    predictions = {}
    for arm in ARMS:
        averaged = totals[arm] / count[None]
        original_logits = F.interpolate(averaged[None], size=original_size,
                                        mode="bilinear", align_corners=False)[0]
        predictions[arm] = original_logits.argmax(dim=0).to(torch.uint8).cpu().numpy()
    return predictions, crops


def main(args) -> dict:
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
        raise ValueError(f"VIP source revision changed: {commit}")
    all_samples, scored, load_image, load_mask = protocol(args)
    if args.max_images:
        random.Random(args.sample_seed).shuffle(all_samples)
        all_samples = all_samples[:args.max_images]
    global_keys = [sample.key for sample in all_samples]
    if len(set(global_keys)) != len(global_keys):
        raise ValueError("Duplicate global sample keys.")
    samples = all_samples[args.shard_index::args.num_shards]
    if not samples:
        raise ValueError("Empty shard.")
    specs = load_class_specs(args.vocabulary_config)
    names = tuple(spec.name for spec in specs)
    if names != tuple(scored[args.dataset]) or any(len(spec.synonyms) != 20 for spec in specs):
        raise ValueError("The fixed 20-alias bank differs from the scored taxonomy.")
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    checkpoints = checkpoint_config(args)
    backbone = DINOTextSegmenter(checkpoints, device=args.device, amp=True)
    geometry = TCPRSegmenter(backbone, TCPRConfig(maximum_aliases_per_class=20,
                                                  geometry_depth=2, prefix_policy="preserve"))
    rs_bank = geometry.encode_text(specs)
    vip = VIPOfficialAdapter(backbone, upstream)
    queries = vip.encode_queries(names, tuple(spec.synonyms for spec in specs))
    if (queries.aliases != rs_bank.alias_names or
            not torch.equal(queries.parents, rs_bank.parent_indices)):
        raise ValueError("Both text encoders must use identical aliases and parent classes.")
    rs_text = F.normalize(rs_bank.features.float(), dim=-1)
    settings = upstream_settings(args.dataset)
    output.mkdir(parents=True, exist_ok=True)
    shard_keys = [sample.key for sample in samples]
    signature = {
        "implementation": "matched-head-text-20260930", "dataset": args.dataset,
        "arms": ARMS, "class_names": names, "alias_names": rs_bank.alias_names,
        "alias_counts": [20] * len(names),
        "rs_templates": "six remote-sensing templates, mean normalized alias feature",
        "vip_templates": "pinned OpenAI ImageNet templates, per-template mean similarity",
        "vip_tau": settings.tau, "vip_salience_temperature": settings.tem,
        "vip_logit_scale": settings.logit_scale,
        "crop": CROP, "stride": STRIDE, "long_edges": (LONG_EDGE,),
        "overlap_rule": "raw-logit mean", "threshold": None,
        "upstream_commit": commit,
        "vocabulary_sha256": hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest(),
        "checkpoint_manifest": checkpoint_manifest(checkpoints),
        "global_sample_count": len(global_keys), "global_sample_keys_sha256": digest(global_keys),
        "sample_keys": shard_keys, "sample_keys_sha256": digest(shard_keys),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "sample_seed": args.sample_seed, "max_images": args.max_images,
    }
    matrices = {arm: Confusion(names, len(names)) for arm in ARMS}
    pairwise = {}
    crops = {str(LONG_EDGE): 0}
    started = time.perf_counter()
    if backbone.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backbone.device)
    for number, sample in enumerate(samples, 1):
        image = load_image(sample)
        prediction, crop_count = predict_image(image, geometry, vip, rs_text,
                                               rs_bank.parent_indices, queries, settings)
        target = load_mask(sample, args.dataset, tuple(image.shape[-2:]))
        for arm in ARMS:
            matrices[arm].update(prediction[arm], target)
        current = count_changes(prediction, target, len(names), ARMS)
        for key, values in current.items():
            saved = pairwise.setdefault(key, {field: ([0] * len(names) if isinstance(value, list) else 0)
                                               for field, value in values.items()})
            for field, value in values.items():
                if isinstance(value, list):
                    saved[field] = [left + right for left, right in zip(saved[field], value)]
                else:
                    saved[field] += value
        crops[str(LONG_EDGE)] += crop_count
        if number % args.progress_every == 0 or number == len(samples):
            payload = {
                "status": "complete" if number == len(samples) else "running",
                "processed_images": number, "total_images": len(samples),
                "metrics": {arm: matrix.summary() for arm, matrix in matrices.items()},
                "pairwise": pairwise, "crops": crops,
                "wall_seconds": time.perf_counter() - started,
                "peak_cuda_memory_mb": (torch.cuda.max_memory_allocated(backbone.device) / 1048576
                                        if backbone.device.type == "cuda" else 0.0),
                "signature": signature,
            }
            (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n",
                                                 encoding="utf-8")
            print(json.dumps({"dataset": args.dataset, "shard": args.shard_index,
                              "processed": number, "total": len(samples),
                              "miou": {arm: payload["metrics"][arm]["mean_iou_percent"]
                                       for arm in ARMS}}), flush=True)
    return payload


if __name__ == "__main__":
    main(parse_args())
