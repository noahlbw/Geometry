#!/usr/bin/env python3
"""Select aliases from unlabeled images for a matched GEAR-OV vocabulary test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from dinotool.gear_ov import GearOVSegmenter
from dinotool.inference import tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs, serialize_class_specs
from dinotool.vip_alias_distillation import (
    VIPAliasAccumulator, VIPDistillationConfig, multilayer_attention_affinity,
)
from eval_gear_ov import protocol
from eval_stride_ov_loveda_e1 import crop_tile, make_checkpoints, patch_valid_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("loveda", "udd5", "oem", "vaihingen",
                                              "landcoverai", "flair1"), required=True)
    for name in ("dinov3-repo", "checkpoint-dir", "data-root", "vocabulary-config",
                 "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selection-images", type=int, default=64)
    parser.add_argument("--tiles-per-image", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=20260929)
    parser.add_argument("--memory-fraction", type=float, default=0.6)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def selected_vocabularies(specs, bank, report, seed: int):
    selected = torch.tensor(report["selected_mask"], dtype=torch.bool)
    rng = random.Random(seed)
    selected_specs, random_specs = [], []
    for index, spec in enumerate(specs):
        members = (bank.parent_indices.cpu() == index).nonzero().flatten().tolist()
        kept = [bank.alias_names[i] for i in members if selected[i]]
        if not kept or kept[0] != spec.name:
            raise ValueError(f"Selection lost the canonical alias for {spec.name}.")
        alternatives = [alias for alias in spec.synonyms if alias != spec.name]
        random_kept = {spec.name, *rng.sample(alternatives, len(kept) - 1)}
        selected_specs.append({"name": spec.name, "synonyms": kept})
        random_specs.append({"name": spec.name, "synonyms": [
            alias for alias in spec.synonyms if alias in random_kept
        ]})
    return selected_specs, random_specs


def main(args: argparse.Namespace) -> dict:
    if args.selection_images < 1 or args.tiles_per_image < 1:
        raise ValueError("Selection sample and tile budgets must be positive.")
    if not 0 < args.memory_fraction <= 1:
        raise ValueError("Invalid CUDA memory fraction.")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite selection output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    vocabulary_path = Path(args.vocabulary_config).resolve()
    specs = load_class_specs(vocabulary_path)
    if any(len(spec.synonyms) != 20 for spec in specs):
        raise ValueError("Selection expects exactly 20 candidate aliases per class.")
    all_samples, _, load_image, _ = protocol(args, specs)
    rng = random.Random(args.sample_seed)
    samples = rng.sample(all_samples, min(args.selection_images, len(all_samples)))
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    checkpoints = make_checkpoints(args)
    model = GearOVSegmenter(DINOTextSegmenter(
        checkpoints, device=args.device, amp=not args.no_amp
    ))
    bank = model.encode_text(specs)
    config = VIPDistillationConfig()
    stats = VIPAliasAccumulator(bank.features.shape[0], device=model.device)
    text = F.normalize(bank.features.float(), dim=-1)
    started = time.perf_counter()
    for image_index, sample in enumerate(samples):
        image = (load_image(sample.image_path) if args.dataset == "loveda"
                 else load_image(sample))
        height, width = image.shape[-2:]
        positions = [(top, left) for top in tile_starts(height, 512, 128)
                     for left in tile_starts(width, 512, 128)]
        tile_rng = random.Random(args.sample_seed + image_index)
        chosen = tile_rng.sample(positions, min(args.tiles_per_image, len(positions)))
        for top, left in chosen:
            actual_h, actual_w = min(512, height - top), min(512, width - left)
            tile = crop_tile(image, left, top, 512).to(model.device)
            prepared = model.base.prepare_image(tile)
            attention = multilayer_attention_affinity(model.base.backbone, tile)
            valid = patch_valid_mask(actual_h, actual_w, 512, model.base.patch_size,
                                     model.device).reshape(1, -1)
            aliases = F.normalize(prepared.geometry_projected.float(), dim=-1) @ text.T
            stats.update_tile(aliases, attention, bank.parent_indices,
                              bank.class_count, valid, config)
        stats.finalize_image()
        print(json.dumps({"dataset": args.dataset, "selected_images": image_index + 1,
                          "total_images": len(samples)}), flush=True)
    report = stats.report(bank)
    selected, random_control = selected_vocabularies(specs, bank, report, args.sample_seed)
    signature = {
        "implementation": "gear-alias-selection-v1-20260929",
        "dataset": args.dataset,
        "source": "unlabeled evaluation images; transductive vocabulary selection",
        "sample_keys": [sample.key for sample in samples],
        "selection_images": len(samples),
        "tiles_per_image": args.tiles_per_image,
        "sample_seed": args.sample_seed,
        "vocabulary_sha256": hashlib.sha256(vocabulary_path.read_bytes()).hexdigest(),
        "vocabulary": serialize_class_specs(specs),
        "checkpoint_manifest": checkpoint_manifest(checkpoints),
        "distillation": vars(config),
    }
    result = {"status": "complete", "signature": signature, "report": report,
              "selected_counts_per_class": report["selected_counts_per_class"],
              "wall_seconds": time.perf_counter() - started,
              "peak_cuda_memory_mb": (torch.cuda.max_memory_allocated(model.device) / 1048576
                                      if model.device.type == "cuda" else 0)}
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "selected_vocab.json").write_text(json.dumps({"classes": selected}, indent=2) + "\n")
    (output / "random_vocab.json").write_text(json.dumps({"classes": random_control}, indent=2) + "\n")
    return result


if __name__ == "__main__":
    main(parse_args())
