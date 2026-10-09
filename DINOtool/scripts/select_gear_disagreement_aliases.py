#!/usr/bin/env python3
"""Unlabeled counterfactual alias selection from native/Geometry disagreement."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("loveda", "oem", "flair1", "udd5",
                                              "landcoverai"), required=True)
    for name in ("dinov3-repo", "checkpoint-dir", "data-root", "vocabulary-config",
                 "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--images", type=int, default=64)
    parser.add_argument("--tiles-per-image", type=int, default=2)
    parser.add_argument("--sample-seed", type=int, default=20260929)
    parser.add_argument("--keep-per-class", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.6)
    return parser.parse_args()


def without_alias(logits, alias, parents, index, temperature):
    members = (parents == int(parents[index])).nonzero().flatten()
    remain = members[members != index]
    updated = logits.clone()
    updated[:, parents[index]] = temperature * (
        torch.logsumexp(alias[:, remain] / temperature, -1)
        - torch.tensor(float(len(remain)), device=alias.device).log()
    )
    return updated


def main(args: argparse.Namespace):
    if args.images < 1 or args.tiles_per_image < 1 or not 1 <= args.keep_per_class <= 20:
        raise ValueError("Invalid sampling or retention budget.")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite selection output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    vocab_path = Path(args.vocabulary_config).resolve()
    specs = load_class_specs(vocab_path)
    if any(len(spec.synonyms) != 20 for spec in specs):
        raise ValueError("Expected exactly 20 candidate aliases per class.")
    all_samples, groups, load_image, _ = protocol(args, specs)
    key = "D" if args.dataset == "loveda" else args.dataset
    samples = random.Random(args.sample_seed).sample(all_samples,
                                                     min(args.images, len(all_samples)))
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    checkpoints = make_checkpoints(args)
    model = GearOVSegmenter(DINOTextSegmenter(checkpoints, device=args.device, amp=True))
    bank = model.encode_text(groups[key])
    text = F.normalize(bank.features.float(), dim=-1)
    parents = bank.parent_indices
    sums = torch.zeros(len(bank.alias_names), device=model.device, dtype=torch.float64)
    patches = 0
    started = time.perf_counter()
    for image_index, sample in enumerate(samples):
        image = load_image(sample.image_path) if args.dataset == "loveda" else load_image(sample)
        height, width = image.shape[-2:]
        positions = [(top, left) for top in tile_starts(height, 512, 128)
                     for left in tile_starts(width, 512, 128)]
        chosen = random.Random(args.sample_seed + image_index).sample(
            positions, min(args.tiles_per_image, len(positions))
        )
        for top, left in chosen:
            actual_h, actual_w = min(512, height - top), min(512, width - left)
            prepared = model.base.prepare_image(crop_tile(image, left, top, 512).to(model.device))
            geometry_alias = (F.normalize(prepared.geometry_projected.float(), dim=-1)
                              @ text.T).reshape(32 * 32, -1)
            native_alias = (F.normalize(prepared.native_projected.float(), dim=-1)
                            @ text.T).reshape(32 * 32, -1)
            geometry_logits = 0.07 * class_scores(geometry_alias / 0.07,
                                                  parents, bank.class_count)
            native_logits = 0.07 * class_scores(native_alias / 0.07,
                                                parents, bank.class_count)
            valid_y = torch.arange(32, device=model.device) * 16 < actual_h
            valid_x = torch.arange(32, device=model.device) * 16 < actual_w
            valid = (valid_y[:, None] & valid_x[None, :]).flatten()
            geometry_alias, native_alias = geometry_alias[valid], native_alias[valid]
            geometry_logits, native_logits = geometry_logits[valid], native_logits[valid]
            full_logp = (geometry_logits / 0.07).log_softmax(-1)
            for index in range(len(bank.alias_names)):
                reduced_geometry = without_alias(geometry_logits, geometry_alias,
                                                 parents, index, 0.07)
                reduced_native = without_alias(native_logits, native_alias,
                                               parents, index, 0.07)
                teacher = (reduced_native / 0.07).softmax(-1)
                change = full_logp - (reduced_geometry / 0.07).log_softmax(-1)
                sums[index] += (teacher * change).sum().double()
            patches += int(valid.sum())
        print(json.dumps({"dataset": args.dataset, "images": image_index + 1,
                          "total_images": len(samples)}), flush=True)
    scores = (sums / max(patches, 1)).cpu().tolist()
    selected = []
    selected_indices = []
    for class_index, spec in enumerate(groups[key]):
        members = (parents == class_index).nonzero().flatten().tolist()
        chosen = sorted(members, key=lambda index: (scores[index], index))[:args.keep_per_class]
        selected_indices.extend(chosen)
        selected.append({"name": spec.name, "synonyms": [bank.alias_names[index]
                                                           for index in members if index in chosen]})
    result = {
        "status": "complete", "method": "native_counterfactual_disagreement_low15",
        "dataset": args.dataset,
        "note": "Unlabeled image-only selection. Low native-teacher counterfactual gain is retained; labels never loaded.",
        "sample_keys": [sample.key for sample in samples],
        "source_sha256": hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
        "checkpoints": checkpoint_manifest(checkpoints),
        "keep_per_class": args.keep_per_class,
        "scores": [{"class": bank.class_names[int(parents[index])],
                    "alias": bank.alias_names[index], "native_counterfactual_gain": score,
                    "selected": index in selected_indices} for index, score in enumerate(scores)],
        "patches": patches,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "selected_vocab.json").write_text(json.dumps({"classes": selected}, indent=2) + "\n")
    return result


if __name__ == "__main__":
    main(parse_args())
