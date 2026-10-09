#!/usr/bin/env python3
"""Measure each alias's signed local-readout effect; labels are audit-only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.gear_ov import GearOVSegmenter, class_scores
from dinotool.inference import tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs
from eval_gear_ov import protocol
from eval_stride_ov_loveda_e1 import crop_tile, make_checkpoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("loveda", "oem", "flair1", "udd5",
                                              "landcoverai"), required=True)
    for name in ("dinov3-repo", "checkpoint-dir", "data-root", "vocabulary-config",
                 "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--selection-file")
    parser.add_argument("--images", type=int, default=64)
    parser.add_argument("--tiles-per-image", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=20260929)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.6)
    return parser.parse_args()


def leave_one_out_class(full: torch.Tensor, aliases: torch.Tensor,
                        parent: torch.Tensor, index: int, temperature: float) -> torch.Tensor:
    """Exact normalized log-mean-exp after removing one non-sole alias."""
    members = (parent == int(parent[index])).nonzero().flatten()
    if len(members) < 2:
        raise ValueError("A class has fewer than two aliases.")
    remain = members[members != index]
    updated = full.clone()
    updated[:, parent[index]] = temperature * (
        torch.logsumexp(aliases[:, remain] / temperature, dim=-1)
        - torch.tensor(float(len(remain)), device=aliases.device).log()
    )
    return updated


def diagnose(args: argparse.Namespace) -> dict:
    if args.images < 1 or args.tiles_per_image < 1:
        raise ValueError("Positive image and tile budgets are required.")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    vocab_path = Path(args.vocabulary_config).resolve()
    specs = load_class_specs(vocab_path)
    all_samples, groups, load_image, load_mask = protocol(args, specs)
    key = "D" if args.dataset == "loveda" else args.dataset
    rng = random.Random(args.sample_seed)
    samples = rng.sample(all_samples, min(args.images, len(all_samples)))
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    checkpoints = make_checkpoints(args)
    model = GearOVSegmenter(DINOTextSegmenter(checkpoints, device=args.device, amp=True))
    bank = model.encode_text(groups[key])
    parents = bank.parent_indices
    text = F.normalize(bank.features.float(), dim=-1)
    count = len(bank.alias_names)
    selected = None
    if args.selection_file:
        payload = json.loads(Path(args.selection_file).read_text(encoding="utf-8"))
        if payload["signature"]["vocabulary_sha256"] != hashlib.sha256(vocab_path.read_bytes()).hexdigest():
            raise ValueError("Old selection uses a different candidate bank.")
        selected = payload["report"]["selected_mask"]
    totals = [{"pixels": 0, "gt_parent_pixels": 0, "mean_logprob_gain_sum": 0.0,
               "parent_logprob_gain_sum": 0.0, "other_logprob_gain_sum": 0.0,
               "beneficial_flips": 0, "harmful_flips": 0, "unique_top_patches": 0,
               "active_images": 0, "mean_class_delta_sum": 0.0,
               "native_counterfactual_gain_sum": 0.0} for _ in range(count)]
    started = time.perf_counter()
    tiles_seen = 0
    for image_index, sample in enumerate(samples):
        image = load_image(sample.image_path) if args.dataset == "loveda" else load_image(sample)
        target = np.asarray(load_mask(sample, key, tuple(image.shape[-2:])), dtype=np.int64)
        height, width = image.shape[-2:]
        positions = [(top, left) for top in tile_starts(height, 512, 128)
                     for left in tile_starts(width, 512, 128)]
        chosen = random.Random(args.sample_seed + image_index).sample(
            positions, min(args.tiles_per_image, len(positions))
        )
        image_active = torch.zeros(count, dtype=torch.bool, device=model.device)
        for top, left in chosen:
            actual_h, actual_w = min(512, height - top), min(512, width - left)
            tile = crop_tile(image, left, top, 512).to(model.device)
            prepared = model.base.prepare_image(tile)
            cosine = F.normalize(prepared.geometry_projected.float(), dim=-1) @ text.T
            native_cosine = F.normalize(prepared.native_projected.float(), dim=-1) @ text.T
            alias = cosine.reshape(32 * 32, count)
            native_alias = native_cosine.reshape(32 * 32, count)
            logits = 0.07 * class_scores(alias / 0.07, parents, bank.class_count)
            native_logits = 0.07 * class_scores(native_alias / 0.07, parents,
                                               bank.class_count)
            yy = top + np.minimum(np.arange(32) * 16 + 8, actual_h - 1)
            xx = left + np.minimum(np.arange(32) * 16 + 8, actual_w - 1)
            truth = torch.from_numpy(target[np.ix_(yy, xx)].copy().reshape(-1)).to(model.device)
            valid_y = torch.arange(32, device=model.device) * 16 < actual_h
            valid_x = torch.arange(32, device=model.device) * 16 < actual_w
            valid = (valid_y[:, None] & valid_x[None, :]).flatten()
            valid &= (truth >= 0) & (truth < bank.class_count)
            if not bool(valid.any()):
                continue
            alias, logits, truth = alias[valid], logits[valid], truth[valid]
            native_alias, native_logits = native_alias[valid], native_logits[valid]
            full_logp = (logits / 0.07).log_softmax(-1).gather(1, truth[:, None]).squeeze(1)
            full_logp_all = (logits / 0.07).log_softmax(-1)
            full_prediction = logits.argmax(-1)
            for class_index in range(bank.class_count):
                members = (parents == class_index).nonzero().flatten()
                winners = alias[:, members].argmax(-1)
                for within, alias_index in enumerate(members.tolist()):
                    member_top = winners == within
                    totals[alias_index]["unique_top_patches"] += int(member_top.sum())
                    image_active[alias_index] |= bool(member_top.any())
            for alias_index in range(count):
                removed = leave_one_out_class(logits, alias, parents, alias_index, 0.07)
                removed_logp_all = (removed / 0.07).log_softmax(-1)
                removed_logp = removed_logp_all.gather(1, truth[:, None]).squeeze(1)
                native_without = leave_one_out_class(
                    native_logits, native_alias, parents, alias_index, 0.07
                )
                native_teacher = (native_without / 0.07).softmax(-1)
                proxy_gain = (native_teacher * (full_logp_all - removed_logp_all)).sum(-1)
                gain = full_logp - removed_logp
                own = truth == parents[alias_index]
                new_prediction = removed.argmax(-1)
                item = totals[alias_index]
                item["pixels"] += len(truth)
                item["gt_parent_pixels"] += int(own.sum())
                item["mean_logprob_gain_sum"] += float(gain.sum())
                item["mean_class_delta_sum"] += float(
                    (logits[:, parents[alias_index]] - removed[:, parents[alias_index]]).sum()
                )
                item["native_counterfactual_gain_sum"] += float(proxy_gain.sum())
                item["parent_logprob_gain_sum"] += float(gain[own].sum())
                item["other_logprob_gain_sum"] += float(gain[~own].sum())
                item["beneficial_flips"] += int(((full_prediction == truth) &
                                                  (new_prediction != truth)).sum())
                item["harmful_flips"] += int(((full_prediction != truth) &
                                               (new_prediction == truth)).sum())
            tiles_seen += 1
        for alias_index, active in enumerate(image_active.tolist()):
            totals[alias_index]["active_images"] += int(active)
        print(json.dumps({"dataset": args.dataset, "images": image_index + 1,
                          "total_images": len(samples), "tiles": tiles_seen}), flush=True)
    entries = []
    for index, item in enumerate(totals):
        pixels = max(item["pixels"], 1)
        entries.append({"class": bank.class_names[int(parents[index])],
                        "alias": bank.alias_names[index],
                        "old_selected": None if selected is None else selected[index],
                        **item,
                        "mean_logprob_gain": item["mean_logprob_gain_sum"] / pixels,
                        "mean_class_delta": item["mean_class_delta_sum"] / pixels,
                        "native_counterfactual_gain": (
                            item["native_counterfactual_gain_sum"] / pixels),
                        "parent_logprob_gain": (item["parent_logprob_gain_sum"] /
                                                max(item["gt_parent_pixels"], 1)),
                        "other_logprob_gain": (item["other_logprob_gain_sum"] /
                                               max(pixels - item["gt_parent_pixels"], 1)),
                        "flip_balance": item["beneficial_flips"] - item["harmful_flips"],
                        "unique_top_fraction": item["unique_top_patches"] / pixels})
    result = {"status": "complete", "dataset": args.dataset,
              "note": "GT is diagnostic only, never used to select aliases or tune inference.",
              "images": len(samples), "tiles": tiles_seen,
              "sample_keys": [sample.key for sample in samples],
              "candidate_sha256": hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
              "checkpoint_manifest": checkpoint_manifest(checkpoints),
              "elapsed_seconds": time.perf_counter() - started,
              "aliases": entries}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    diagnose(parse_args())
