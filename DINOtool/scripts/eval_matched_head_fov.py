#!/usr/bin/env python3
"""Matched Geometry/VIP head by image field-of-view diagnosis."""
from __future__ import annotations

import argparse
from itertools import combinations
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dinotool.gear_ov import class_scores
from dinotool.inference import tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from dinotool.vip_official_adapter import VIPOfficialAdapter
from eval_vip_official_eight import Confusion, PINNED_COMMIT, checkpoint_config, protocol


HEADS = ("Geometry", "VIP")
LONG_EDGES = (448, 1024)
ARMS = tuple(f"{head}_{edge}" for edge in LONG_EDGES for head in HEADS)
CROP = 336
STRIDE = 112
ALIAS_TEMPERATURE = 0.07
OUTPUT_TEMPERATURE = 0.07


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("vdd", "potsdam"), required=True)
    for name in ("dinov3-repo", "checkpoint-dir", "data-root", "vocabulary-config",
                 "upstream-root", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.6)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260930)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=2)
    return parser.parse_args()


def digest(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def resize_rgb(image: torch.Tensor, long_edge: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    ratio = long_edge / max(height, width)
    shape = (int(width * ratio + 0.5), int(height * ratio + 0.5))
    array = (image.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
    resized = cv2.resize(array, shape, interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).float().div_(255)


def dense_logits(features: torch.Tensor, text: torch.Tensor, parents: torch.Tensor,
                 classes: int) -> torch.Tensor:
    if features.ndim != 3 or features.shape[1] != 21 * 21:
        raise ValueError("Both visual heads must return the same 21x21 crop grid.")
    scaled = features.float() @ text.T / ALIAS_TEMPERATURE
    values = ALIAS_TEMPERATURE * class_scores(scaled, parents, classes)
    return F.interpolate(values.transpose(1, 2).reshape(1, classes, 21, 21),
                         size=(CROP, CROP), mode="bilinear", align_corners=False)[0]


@torch.inference_mode()
def predict_image(image: torch.Tensor, geometry: TCPRSegmenter,
                  vip: VIPOfficialAdapter, text: torch.Tensor,
                  parents: torch.Tensor, classes: int) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    original_size = tuple(image.shape[-2:])
    predictions = {}
    crop_counts = {}
    for edge in LONG_EDGES:
        resized = resize_rgb(image, edge)
        height, width = resized.shape[-2:]
        totals = {head: torch.zeros((classes, height, width), device=geometry.device)
                  for head in HEADS}
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
                features = {"Geometry": prepared.geometry_projected,
                            "VIP": vip.crop_patch_features(rgb)}
                for head in HEADS:
                    logits = dense_logits(features[head], text, parents, classes)
                    totals[head][:, top:top + actual_height, left:left + actual_width] += (
                        logits[:, :actual_height, :actual_width]
                    )
                count[top:top + actual_height, left:left + actual_width] += 1
                crops += 1
                del prepared, features
        if not bool((count > 0).all()):
            raise AssertionError("Sliding windows left image pixels uncovered.")
        crop_counts[str(edge)] = crops
        for head in HEADS:
            averaged = totals[head] / count[None]
            original_logits = F.interpolate(averaged[None], size=original_size,
                                            mode="bilinear", align_corners=False)[0]
            prediction = (original_logits / OUTPUT_TEMPERATURE).argmax(dim=0)
            predictions[f"{head}_{edge}"] = prediction.to(torch.uint8).cpu().numpy()
    return predictions, crop_counts


def count_changes(predictions: dict[str, np.ndarray], target: np.ndarray,
                  classes: int, arms: tuple[str, ...] = ARMS) -> dict[str, dict]:
    valid = (target >= 0) & (target < classes)
    truth = target[valid]
    labels = {arm: prediction[valid] for arm, prediction in predictions.items()}
    result = {}
    for first, second in combinations(arms, 2):
        a, b = labels[first], labels[second]
        fixed = (a != truth) & (b == truth)
        broken = (a == truth) & (b != truth)
        result[f"{first}__{second}"] = {
            "changed": int((a != b).sum()),
            "first_wrong_second_correct": int(fixed.sum()),
            "first_correct_second_wrong": int(broken.sum()),
            "fixed_by_true_class": np.bincount(truth[fixed], minlength=classes).tolist(),
            "broken_by_true_class": np.bincount(truth[broken], minlength=classes).tolist(),
        }
    return result


def main(args: argparse.Namespace) -> dict:
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
    bank = geometry.encode_text(specs)
    if any(int((bank.parent_indices == index).sum()) != 20 for index in range(len(names))):
        raise ValueError("Text encoding changed the 20-alias bank.")
    vip = VIPOfficialAdapter(backbone, upstream)
    text = F.normalize(bank.features.float(), dim=-1)
    output.mkdir(parents=True, exist_ok=True)
    shard_keys = [sample.key for sample in samples]
    signature = {
        "implementation": "matched-head-fov-20260930", "dataset": args.dataset,
        "arms": ARMS, "class_names": names, "alias_names": bank.alias_names,
        "alias_counts": [20] * len(names), "text_templates": "six remote-sensing templates",
        "aggregation": "normalized log-mean-exp", "alias_temperature": ALIAS_TEMPERATURE,
        "output_temperature": OUTPUT_TEMPERATURE, "crop": CROP, "stride": STRIDE,
        "long_edges": LONG_EDGES, "overlap_rule": "raw-logit mean",
        "threshold": None, "upstream_commit": commit,
        "vocabulary_sha256": hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest(),
        "checkpoint_manifest": checkpoint_manifest(checkpoints),
        "global_sample_count": len(global_keys), "global_sample_keys_sha256": digest(global_keys),
        "sample_keys": shard_keys, "sample_keys_sha256": digest(shard_keys),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "sample_seed": args.sample_seed, "max_images": args.max_images,
    }
    matrices = {arm: Confusion(names, len(names)) for arm in ARMS}
    pairwise = {}
    crops = {str(edge): 0 for edge in LONG_EDGES}
    started = time.perf_counter()
    if backbone.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backbone.device)
    for number, sample in enumerate(samples, 1):
        image = load_image(sample)
        prediction, current_crops = predict_image(image, geometry, vip, text,
                                                  bank.parent_indices, len(names))
        target = load_mask(sample, args.dataset, tuple(image.shape[-2:]))
        for arm in ARMS:
            matrices[arm].update(prediction[arm], target)
        current = count_changes(prediction, target, len(names))
        for key, values in current.items():
            saved = pairwise.setdefault(key, {field: ([0] * len(names) if isinstance(value, list) else 0)
                                               for field, value in values.items()})
            for field, value in values.items():
                if isinstance(value, list):
                    saved[field] = [left + right for left, right in zip(saved[field], value)]
                else:
                    saved[field] += value
        for edge, count in current_crops.items():
            crops[edge] += count
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
