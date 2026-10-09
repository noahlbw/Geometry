#!/usr/bin/env python3
"""Label-free, variable-count UDD5 alias screen with visual support protection.

Exploratory rule frozen before evaluating its target masks. An alias is removed
only when its native-counterfactual gain is positive with image-level z >= 1
AND it owns <1% of patches on which its class is in Geometry's top two.
No per-class retention count is imposed, apart from a four-word safety floor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from dinotool.gear_ov import GearOVSegmenter, class_scores
from dinotool.inference import tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs
from eval_gear_ov import protocol
from eval_stride_ov_loveda_e1 import crop_tile, make_checkpoints
from select_gear_disagreement_aliases import without_alias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("udd5",), required=True)
    for name in ("dinov3-repo", "checkpoint-dir", "data-root",
                 "vocabulary-config", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--images", type=int, default=40)
    parser.add_argument("--tiles-per-image", type=int, default=2)
    parser.add_argument("--sample-seed", type=int, default=20260929)
    parser.add_argument("--min-aliases", type=int, default=4)
    parser.add_argument("--z-threshold", type=float, default=1.0)
    parser.add_argument("--support-threshold", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.6)
    return parser.parse_args()


@torch.inference_mode()
def main(args: argparse.Namespace) -> dict:
    if args.images < 1 or args.tiles_per_image < 1 or not 1 <= args.min_aliases <= 20:
        raise ValueError("Invalid sampling or retention floor")
    if not 0 <= args.support_threshold <= 1 or args.z_threshold < 0:
        raise ValueError("Invalid screening threshold")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite existing selection: {output}")
    output.mkdir(parents=True, exist_ok=True)
    vocab_path = Path(args.vocabulary_config).resolve()
    specs = load_class_specs(vocab_path)
    if any(len(spec.synonyms) != 20 for spec in specs):
        raise ValueError("Expected 20 candidate aliases per class")
    all_samples, groups, load_image, _ = protocol(args, specs)
    samples = random.Random(args.sample_seed).sample(
        all_samples, min(args.images, len(all_samples))
    )
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    checkpoints = make_checkpoints(args)
    model = GearOVSegmenter(DINOTextSegmenter(checkpoints, device=args.device, amp=True))
    bank = model.encode_text(groups["udd5"])
    text = F.normalize(bank.features.float(), dim=-1)
    parents = bank.parent_indices
    count = len(bank.alias_names)
    support = torch.zeros(count, dtype=torch.int64, device=model.device)
    eligible = torch.zeros(bank.class_count, dtype=torch.int64, device=model.device)
    image_scores: list[torch.Tensor] = []
    started = time.perf_counter()

    for image_index, sample in enumerate(samples):
        image = load_image(sample)
        height, width = image.shape[-2:]
        positions = [(top, left) for top in tile_starts(height, 512, 128)
                     for left in tile_starts(width, 512, 128)]
        chosen = random.Random(args.sample_seed + image_index).sample(
            positions, min(args.tiles_per_image, len(positions))
        )
        score_sum = torch.zeros(count, dtype=torch.float64, device=model.device)
        image_patches = 0
        for top, left in chosen:
            actual_h, actual_w = min(512, height - top), min(512, width - left)
            prepared = model.base.prepare_image(
                crop_tile(image, left, top, 512).to(model.device))
            geometry_alias = (
                F.normalize(prepared.geometry_projected.float(), dim=-1) @ text.T
            ).reshape(32 * 32, count)
            native_alias = (
                F.normalize(prepared.native_projected.float(), dim=-1) @ text.T
            ).reshape(32 * 32, count)
            valid_y = torch.arange(32, device=model.device) * 16 < actual_h
            valid_x = torch.arange(32, device=model.device) * 16 < actual_w
            valid = (valid_y[:, None] & valid_x[None, :]).flatten()
            geometry_alias = geometry_alias[valid]
            native_alias = native_alias[valid]
            geometry_logits = 0.07 * class_scores(
                geometry_alias / 0.07, parents, bank.class_count)
            native_logits = 0.07 * class_scores(
                native_alias / 0.07, parents, bank.class_count)
            full_logp = (geometry_logits / 0.07).log_softmax(-1)
            top_two = geometry_logits.topk(min(2, bank.class_count), dim=-1).indices
            for class_index in range(bank.class_count):
                members = (parents == class_index).nonzero().flatten()
                class_eligible = (top_two == class_index).any(-1)
                eligible[class_index] += class_eligible.sum()
                winner = geometry_alias[:, members].argmax(-1)
                support[members] += torch.bincount(
                    winner[class_eligible], minlength=len(members))
            for index in range(count):
                reduced_geometry = without_alias(
                    geometry_logits, geometry_alias, parents, index, 0.07)
                reduced_native = without_alias(
                    native_logits, native_alias, parents, index, 0.07)
                teacher = (reduced_native / 0.07).softmax(-1)
                change = full_logp - (reduced_geometry / 0.07).log_softmax(-1)
                score_sum[index] += (teacher * change).sum().double()
            image_patches += int(valid.sum())
        image_scores.append(score_sum / max(image_patches, 1))
        print(json.dumps({"images": image_index + 1,
                          "total_images": len(samples)}), flush=True)

    scores = torch.stack(image_scores).cpu()
    mean = scores.mean(dim=0)
    standard_error = scores.std(dim=0, unbiased=len(scores) > 1) / len(scores) ** 0.5
    z = mean / standard_error.clamp_min(1e-12)
    support_cpu, eligible_cpu = support.cpu(), eligible.cpu()
    keep = [True] * count
    for class_index in range(bank.class_count):
        members = (parents == class_index).nonzero().flatten().tolist()
        for index in members:
            ratio = support_cpu[index].item() / max(eligible_cpu[class_index].item(), 1)
            if mean[index].item() > 0 and z[index].item() >= args.z_threshold \
                    and ratio < args.support_threshold:
                keep[index] = False
        if sum(keep[index] for index in members) < args.min_aliases:
            for index in sorted(members, key=lambda item: (mean[item].item(), item)):
                keep[index] = True
                if sum(keep[member] for member in members) >= args.min_aliases:
                    break

    selected = []
    rows = []
    for class_index, spec in enumerate(groups["udd5"]):
        members = (parents == class_index).nonzero().flatten().tolist()
        selected.append({"name": spec.name,
                         "synonyms": [bank.alias_names[index] for index in members
                                      if keep[index]]})
        for index in members:
            ratio = support_cpu[index].item() / max(eligible_cpu[class_index].item(), 1)
            rows.append({"class": spec.name, "alias": bank.alias_names[index],
                         "native_counterfactual_gain": mean[index].item(),
                         "image_level_z": z[index].item(),
                         "competitive_support": ratio,
                         "competitive_patches": support_cpu[index].item(),
                         "class_top2_patches": eligible_cpu[class_index].item(),
                         "selected": keep[index]})
    report = {
        "status": "complete", "dataset": "udd5",
        "method": "positive_counterfactual_and_low_competitive_support",
        "note": "Image-only UDD5 selection; labels are never read.",
        "sample_keys": [sample.key for sample in samples],
        "source_sha256": hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
        "checkpoints": checkpoint_manifest(checkpoints),
        "images": len(samples), "tiles_per_image": args.tiles_per_image,
        "z_threshold": args.z_threshold,
        "support_threshold": args.support_threshold,
        "min_aliases": args.min_aliases,
        "selected_counts": [len(item["synonyms"]) for item in selected],
        "aliases": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "selected_vocab.json").write_text(
        json.dumps({"classes": selected}, indent=2) + "\n")
    return report


if __name__ == "__main__":
    main(parse_args())
