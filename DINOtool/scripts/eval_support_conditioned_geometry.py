#!/usr/bin/env python3
"""Complete support-conditioned model with same-run Geometry/Multiscale controls."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs, serialize_class_specs
from dinotool.support_conditioned_geometry import (
    DIAGNOSTICS, IMPLEMENTATION, METHODS, SupportConditionedGeometry,
    SupportConditionedReadout,
)
from eval_gear_ov import digest, protocol
from eval_grounded_context import write_json
from eval_stride_ov_loveda_e1 import Confusion, make_checkpoints


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("loveda", "udd5", "oem", "vdd", "potsdam",
                                             "vaihingen", "landcoverai", "flair1"), required=True)
    for name in ("dinov3-repo", "checkpoint-dir", "data-root", "vocabulary-config", "output-dir",
                 "reference-manifest"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=.60)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--output-temperature", type=float, default=.07)
    parser.add_argument("--vdd-ontology", default="official", choices=("official",))
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def changed_pixel_counts(base, final, target, classes):
    valid = (target >= 0) & (target < classes)
    changed = (base != final) & valid
    beneficial = changed & (base != target) & (final == target)
    harmful = changed & (base == target) & (final != target)
    return {"valid": int(valid.sum()), "changed": int(changed.sum()),
            "beneficial": int(beneficial.sum()), "harmful": int(harmful.sum()),
            "wrong_to_wrong": int((changed & (base != target) & (final != target)).sum()),
            "beneficial_by_true_class": np.bincount(target[beneficial], minlength=classes).tolist(),
            "harmful_by_true_class": np.bincount(target[harmful], minlength=classes).tolist()}


@torch.inference_mode()
def predict_image(model, image, banks, texts, readers, args, work):
    height, width = image.shape[-2:]
    blend = hann_blend_window(512)
    totals = {key: {"tiles": 0, **dict.fromkeys(DIAGNOSTICS, 0.0)} for key in banks}
    # LoveDA shares support observations selected with its larger D bank.
    priority_key = max(banks, key=lambda key: banks[key].class_count)
    with ExitStack() as stack:
        buffers = {(key, method): stack.enter_context(ProbabilityAccumulator(
            bank.class_count, height, width, 2048, work))
            for key, bank in banks.items() for method in METHODS}
        for top in tile_starts(height, 512, 128):
            for left in tile_starts(width, 512, 128):
                views = model.base.prepare_views(image, top, left)
                base = {key: model.base.read_views(views, texts[key], reconstruct=False)
                        for key in banks}
                observations, common = model.observe(image, top, left, views,
                                                       base[priority_key].multiscale_logits)
                for key in banks:
                    final, current = readers[key].solve(views, base[key].multiscale_logits, observations)
                    logits = {"Geometry": base[key].local_logits,
                              "Multiscale": base[key].multiscale_logits,
                              "SupportConditioned": final}
                    for method, scores in logits.items():
                        if not bool(torch.isfinite(scores).all()):
                            raise ValueError("Non-finite model output.")
                        probability = (scores.float() / args.output_temperature).softmax(1)
                        buffers[key, method].add(
                            probability[0, :, :views.actual_height, :views.actual_width].cpu().numpy(),
                            blend[:views.actual_height, :views.actual_width], left, top)
                    totals[key]["tiles"] += 1
                    for name in DIAGNOSTICS:
                        totals[key][name] += (common | current)[name]
                del views, base, observations
        predictions = {key: {method: buffers[key, method].finalize(None)[0] for method in METHODS}
                       for key in banks}
    return predictions, totals


def main(args):
    if (args.tile_size, args.overlap, args.output_temperature, args.sample_seed) != (512, 128, .07, 20260923):
        raise ValueError("The complete candidate uses the fixed historical protocol.")
    if (not 0 < args.memory_fraction <= 1 or args.num_shards < 1
            or not 0 <= args.shard_index < args.num_shards
            or args.max_images < 0 or args.progress_every < 1):
        raise ValueError("Invalid evaluation settings.")
    vocabulary_path = Path(args.vocabulary_config).resolve()
    vocabulary = load_class_specs(vocabulary_path)
    samples, specs, load_image, load_mask = protocol(args, vocabulary)
    reference = json.loads(Path(args.reference_manifest).read_text())[args.dataset]
    vocab_sha = hashlib.sha256(vocabulary_path.read_bytes()).hexdigest()
    full_keys = [sample.key for sample in samples]
    if (len(full_keys) != reference["global_sample_count"]
            or digest(full_keys) != reference["global_sample_keys_sha256"]
            or vocab_sha != reference["vocabulary_sha256"]):
        raise ValueError("Dataset or vocabulary differs from the historical full control.")
    if args.max_images:
        if args.dataset != "loveda":
            random.Random(args.sample_seed).shuffle(samples)
        samples = samples[:args.max_images]
    keys = [sample.key for sample in samples]
    shard = samples[args.shard_index::args.num_shards]
    if not shard:
        raise ValueError("Empty shard.")
    output = Path(args.output_dir).resolve()
    if (output / "results.json").exists():
        raise ValueError("Refusing to overwrite an existing result.")
    output.mkdir(parents=True, exist_ok=True)
    work = output / ".accumulators"
    work.mkdir(exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device.index or 0)
    checkpoints = make_checkpoints(args)
    backbone = DINOTextSegmenter(checkpoints, device=args.device, amp=not args.no_amp)
    model = SupportConditionedGeometry(backbone)
    banks = {key: model.base.encode_text(group) for key, group in specs.items()}
    texts = {key: model.base.text_basis(bank) for key, bank in banks.items()}
    readers = {key: SupportConditionedReadout(bank, backbone, model.config) for key, bank in banks.items()}
    names = {key: list(bank.class_names) for key, bank in banks.items()}
    counts = {key: [int((bank.parent_indices == c).sum()) for c in range(bank.class_count)]
              for key, bank in banks.items()}
    if any(count != 20 for values in counts.values() for count in values):
        raise ValueError("The complete candidate requires the exact historical 20 slots per class.")
    signature = {
        "implementation": IMPLEMENTATION, "dataset": args.dataset, "methods": list(METHODS),
        "classes": names, "sample_keys": [sample.key for sample in shard],
        "sample_count": len(shard), "sample_keys_sha256": digest([sample.key for sample in shard]),
        "global_sample_count": len(keys), "global_sample_keys_sha256": digest(keys),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "config": {key: value for key, value in vars(args).items() if key != "device"},
        "gear": model.config_dict(), "competitive": None,
        "vocabulary": {"sha256": vocab_sha, "alias_counts": counts,
                       "classes": {key: serialize_class_specs(group) for key, group in specs.items()}},
        "checkpoints": checkpoint_manifest(checkpoints),
        "note": "Same Geometry/Multiscale controls; actual raw-feature support crops; full native "
                "global/text verifies held-out-family ownership; same Geometry dense proposal; "
                "signed replacements retain all20 denominators. Masks enter after predictions only.",
    }
    write_json(output / "signature.json", signature)
    matrices = {key: {method: Confusion(tuple(group)) for method in METHODS} for key, group in names.items()}
    totals = {key: {"tiles": 0, **dict.fromkeys(DIAGNOSTICS, 0.0)} for key in banks}
    changes = {key: {name: 0 for name in ("valid", "changed", "beneficial", "harmful", "wrong_to_wrong")}
               | {name: [0] * bank.class_count for name in ("beneficial_by_true_class", "harmful_by_true_class")}
               for key, bank in banks.items()}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for number, sample in enumerate(shard, 1):
        image = load_image(sample.image_path) if args.dataset == "loveda" else load_image(sample)
        predictions, current = predict_image(model, image, banks, texts, readers, args, work)
        for key in banks:
            target = load_mask(sample, key, tuple(image.shape[-2:]))
            for method in METHODS:
                matrices[key][method].update(predictions[key][method], target)
            for name in totals[key]:
                totals[key][name] += current[key][name]
            change = changed_pixel_counts(predictions[key]["Multiscale"],
                                          predictions[key]["SupportConditioned"], target, banks[key].class_count)
            for name, value in change.items():
                changes[key][name] = ([a + b for a, b in zip(changes[key][name], value)]
                                      if isinstance(value, list) else changes[key][name] + value)
        if number % args.progress_every == 0 or number == len(shard):
            summary = {}
            for key in banks:
                summary[key] = {}
                for method in METHODS:
                    metric = matrices[key][method].summary()
                    if key in ("udd5", "landcoverai"):
                        values = [item["iou_percent"] for item in metric["per_class"]
                                  if item["name"] not in ("other", "background") and item["iou_percent"] is not None]
                        metric["non_residual_mean_iou_percent"] = round(sum(values) / len(values), 4)
                    summary[key][method] = metric
            result = {"status": "complete" if number == len(shard) else "running",
                      "processed_images": number, "total_images": len(shard), "metrics": summary,
                      "diagnostics": {key: {name: value if name == "tiles" else value / max(totals[key]["tiles"], 1)
                                            for name, value in totals[key].items()} for key in banks},
                      "changes": changes, "wall_seconds": time.perf_counter() - started,
                      "peak_cuda_memory_mb": (torch.cuda.max_memory_allocated(device) / 1048576
                                              if device.type == "cuda" else 0), "signature": signature}
            write_json(output / "results.json", result)
            print(json.dumps({"dataset": args.dataset, "processed": number, "total": len(shard),
                              "miou": {key: {method: group[method]["mean_iou_percent"] for method in METHODS}
                                       for key, group in summary.items()},
                              "diagnostics": result["diagnostics"]}), flush=True)
    return result


if __name__ == "__main__":
    main(parse_args())
