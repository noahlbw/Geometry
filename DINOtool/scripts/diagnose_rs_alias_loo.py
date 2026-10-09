#!/usr/bin/env python3
"""Exact Multiscale leave-one-alias-out predictions on a fixed pixel lattice."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.gear_ov import GearOVSegmenter, GearText, class_scores
from dinotool.inference import hann_blend_window, tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs
from eval_gear_ov import protocol
from eval_stride_ov_loveda_e1 import make_checkpoints


STRIDE = 16
OFFSET = 8
TILE = 512
OVERLAP = 128
OUTPUT_TEMPERATURE = 0.07


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("vdd", "potsdam"), required=True)
    for name in ("dinov3-repo", "checkpoint-dir", "data-root", "vocabulary-config", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.6)
    parser.add_argument("--vdd-ontology", default="official")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def digest(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def sampled_map(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    # align_corners=False maps these coordinates to the same pixels as 512x512 bilinear upsampling.
    return F.grid_sample(values.permute(2, 0, 1)[None], grid, mode="bilinear",
                         padding_mode="border", align_corners=False)[0]


def sample_grid(local_y: np.ndarray, local_x: np.ndarray, device: torch.device) -> torch.Tensor:
    ys = torch.as_tensor(local_y, device=device, dtype=torch.float32)
    xs = torch.as_tensor(local_x, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((2 * (xx + 0.5) / TILE - 1,
                        2 * (yy + 0.5) / TILE - 1), dim=-1)[None]


def leave_one_out_deltas(scores: torch.Tensor, members: torch.Tensor,
                         temperature: float, grid: torch.Tensor) -> torch.Tensor:
    """All member deletions at once; only the parent class changes relative to rivals."""
    subset = scores[..., members]
    count = subset.shape[-1]
    if count < 2:
        raise ValueError("Each class needs at least two aliases for leave-one-out.")
    exp_values = torch.exp(subset - subset.max(dim=-1, keepdim=True).values)
    total = exp_values.sum(dim=-1, keepdim=True)
    delta = temperature * (
        (total - exp_values).clamp_min(1e-20).log() - total.log()
        + math.log(count / (count - 1))
    )
    return sampled_map(delta, grid)


def sampled_probabilities(views, text: GearText, grid: torch.Tensor,
                          check_reference: bool, model: GearOVSegmenter
                          ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    temperature = model.config.alias_temperature
    scores = tuple(view.float() @ text.features.T / temperature for view in (
        views.local_aligned, views.detail_aligned, views.context_aligned
    ))
    base = sum(
        sampled_map(temperature * class_scores(score, text.parents, text.class_count), grid)
        for score in scores
    ) / 3
    updates = []
    for class_index in range(text.class_count):
        members = torch.nonzero(text.parents == class_index).flatten()
        delta = sum(leave_one_out_deltas(score, members, temperature, grid)
                    for score in scores) / 3
        candidate = base[None].expand(len(members), -1, -1, -1).clone()
        candidate[:, class_index] += delta
        updates.append((members, (candidate / OUTPUT_TEMPERATURE).softmax(dim=1)))

    if check_reference:
        reference = model.read_views(views, text, reconstruct=False).multiscale_logits
        reference = F.grid_sample(reference, grid, mode="bilinear",
                                  padding_mode="border", align_corners=False)[0]
        observed = (base / OUTPUT_TEMPERATURE).softmax(dim=0)
        expected = (reference / OUTPUT_TEMPERATURE).softmax(dim=0)
        if float((observed - expected).abs().max()) > 2e-5:
            raise AssertionError("Cached Multiscale probabilities differ from the evaluator.")

        first = updates[0][0][0]
        keep = torch.arange(text.features.shape[0], device=text.features.device) != first
        reduced = GearText(text.features[keep], text.parents[keep], text.basis[keep],
                           text.whitening[keep][:, keep], text.class_count)
        reduced_reference = model.read_views(views, reduced, reconstruct=False).multiscale_logits
        reduced_reference = F.grid_sample(reduced_reference, grid, mode="bilinear",
                                          padding_mode="border", align_corners=False)[0]
        reduced_expected = (reduced_reference / OUTPUT_TEMPERATURE).softmax(dim=0)
        if float((updates[0][1][0] - reduced_expected).abs().max()) > 2e-5:
            raise AssertionError("Cached alias deletion differs from a fresh readout.")
    return (base / OUTPUT_TEMPERATURE).softmax(dim=0), updates


def confusion(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray,
              classes: int) -> np.ndarray:
    encoded = target[valid] * classes + prediction[valid]
    return np.bincount(encoded, minlength=classes * classes).reshape(classes, classes)


def write_progress(path: Path, processed: int, total: int, points: int,
                   baseline: np.ndarray, aliases: list[dict], signature: dict,
                   started: float, model: GearOVSegmenter) -> None:
    payload = {
        "status": "complete" if processed == total else "running",
        "processed_images": processed, "total_images": total,
        "sampled_pixels": points,
        "baseline_confusion": baseline.tolist(),
        "aliases": [{**item, "confusion_matrix": item["confusion_matrix"].tolist(),
                     "rival_fp_removed": item["rival_fp_removed"].tolist(),
                     "rival_fp_added": item["rival_fp_added"].tolist()} for item in aliases],
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_mb": (torch.cuda.max_memory_allocated(model.device) / 1048576
                                if model.device.type == "cuda" else 0.0),
        "signature": signature,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_index < args.num_shards or not 0 < args.memory_fraction <= 1:
        raise ValueError("Invalid shard or CUDA memory fraction.")
    if args.vdd_ontology != "official" or args.progress_every < 1 or args.max_images < 0:
        raise ValueError("Only the fixed official label mapping and positive progress are supported.")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing existing output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    vocab = Path(args.vocabulary_config).resolve()
    specs = load_class_specs(vocab)
    all_samples, groups, load_image, load_mask = protocol(args, specs)
    if args.max_images:
        all_samples = all_samples[:args.max_images]
    keys = [sample.key for sample in all_samples]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate global sample keys.")
    samples = all_samples[args.shard_index::args.num_shards]
    if not samples:
        raise ValueError("Empty shard.")
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    checkpoints = make_checkpoints(args)
    model = GearOVSegmenter(DINOTextSegmenter(checkpoints, device=args.device, amp=True))
    bank = model.encode_text(groups[args.dataset])
    text = model.text_basis(bank)
    counts = [int((text.parents == index).sum()) for index in range(text.class_count)]
    if counts != [20] * text.class_count:
        raise ValueError("The counterfactual requires the unchanged 20-alias vocabulary.")
    aliases = [{
        "index": index, "class_index": int(text.parents[index]),
        "class": bank.class_names[int(text.parents[index])], "alias": bank.alias_names[index],
        "confusion_matrix": np.zeros((text.class_count, text.class_count), dtype=np.int64),
        "own_tp_lost": 0, "own_tp_gained": 0,
        "rival_fp_removed": np.zeros(text.class_count, dtype=np.int64),
        "rival_fp_added": np.zeros(text.class_count, dtype=np.int64),
        "beneficial_flips": 0, "harmful_flips": 0, "changed_predictions": 0,
    } for index in range(len(bank.alias_names))]
    shard_keys = [sample.key for sample in samples]
    signature = {
        "implementation": "multiscale-exact-lattice-alias-loo-20260930",
        "dataset": args.dataset, "readout": "GEAR-v2 Multiscale, no reconstruction",
        "tile_size": TILE, "overlap": OVERLAP, "sample_stride": STRIDE,
        "sample_offset": OFFSET, "alias_temperature": model.config.alias_temperature,
        "output_temperature": OUTPUT_TEMPERATURE,
        "class_names": bank.class_names, "alias_names": bank.alias_names,
        "alias_counts": counts,
        "vocabulary_sha256": hashlib.sha256(vocab.read_bytes()).hexdigest(),
        "checkpoint_manifest": checkpoint_manifest(checkpoints),
        "global_sample_count": len(keys), "global_sample_keys_sha256": digest(keys),
        "sample_keys": shard_keys, "sample_keys_sha256": digest(shard_keys),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
    }
    baseline = np.zeros((text.class_count, text.class_count), dtype=np.int64)
    blend = hann_blend_window(TILE)
    sampled_pixels = 0
    started = time.perf_counter()
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    with torch.inference_mode():
        for image_number, sample in enumerate(samples, 1):
            image = load_image(sample)
            height, width = image.shape[-2:]
            target = np.asarray(load_mask(sample, args.dataset, (height, width)))
            ys = np.arange(OFFSET, height, STRIDE)
            xs = np.arange(OFFSET, width, STRIDE)
            target_grid = target[np.ix_(ys, xs)]
            valid = (target_grid >= 0) & (target_grid < text.class_count)
            sampled_pixels += int(valid.sum())
            accum = np.zeros((len(aliases) + 1, text.class_count, len(ys), len(xs)),
                             dtype=np.float32)
            normalizer = np.zeros((len(ys), len(xs)), dtype=np.float32)
            for top in tile_starts(height, TILE, OVERLAP):
                for left in tile_starts(width, TILE, OVERLAP):
                    views = model.prepare_views(image, top, left)
                    y0, y1 = np.searchsorted(ys, (top, top + views.actual_height))
                    x0, x1 = np.searchsorted(xs, (left, left + views.actual_width))
                    local_y, local_x = ys[y0:y1] - top, xs[x0:x1] - left
                    grid = sample_grid(local_y, local_x, model.device)
                    base_probs, updates = sampled_probabilities(
                        views, text, grid, image_number == 1 and top == 0 and left == 0, model
                    )
                    weights = blend[np.ix_(local_y, local_x)]
                    region = np.s_[:, y0:y1, x0:x1]
                    accum[0][region] += base_probs.cpu().numpy() * weights[None]
                    for members, probabilities in updates:
                        indices = members.cpu().numpy() + 1
                        accum[indices, :, y0:y1, x0:x1] += (
                            probabilities.cpu().numpy() * weights[None, None]
                        )
                    normalizer[y0:y1, x0:x1] += weights
                    del views, base_probs, updates
            if not np.all(normalizer > 0):
                raise AssertionError("The fixed sample grid was not fully covered.")
            full = (accum[0] / normalizer[None]).argmax(axis=0)
            baseline += confusion(full, target_grid, valid, text.class_count)
            full_correct = full == target_grid
            for index, item in enumerate(aliases):
                removed = (accum[index + 1] / normalizer[None]).argmax(axis=0)
                item["confusion_matrix"] += confusion(removed, target_grid, valid,
                                                      text.class_count)
                parent = item["class_index"]
                own = valid & (target_grid == parent)
                rival = valid & (target_grid != parent)
                new_correct = removed == target_grid
                removed_fp = rival & (full == parent) & (removed != parent)
                added_fp = rival & (full != parent) & (removed == parent)
                item["own_tp_lost"] += int((own & (full == parent) & (removed != parent)).sum())
                item["own_tp_gained"] += int((own & (full != parent) & (removed == parent)).sum())
                item["rival_fp_removed"] += np.bincount(
                    target_grid[removed_fp], minlength=text.class_count
                )
                item["rival_fp_added"] += np.bincount(
                    target_grid[added_fp], minlength=text.class_count
                )
                item["beneficial_flips"] += int((valid & ~full_correct & new_correct).sum())
                item["harmful_flips"] += int((valid & full_correct & ~new_correct).sum())
                item["changed_predictions"] += int((valid & (full != removed)).sum())
            if image_number % args.progress_every == 0 or image_number == len(samples):
                write_progress(output / "results.json", image_number, len(samples),
                               sampled_pixels, baseline, aliases, signature, started, model)
                print(json.dumps({"dataset": args.dataset, "shard": args.shard_index,
                                  "processed": image_number, "total": len(samples),
                                  "sampled_pixels": sampled_pixels}), flush=True)


if __name__ == "__main__":
    main(parse_args())
