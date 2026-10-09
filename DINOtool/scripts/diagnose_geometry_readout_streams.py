#!/usr/bin/env python3
"""Fixed eight-image observation of unchanged Geometry readout streams."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.geometry_readout_trace import (
    alias_class_scores, class_attribution, exact_alias_attribution,
    probe_scores, trace_geometry_head,
)
from dinotool.gear_ov import _crop_at
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from eval_gear_ov import digest, protocol
from eval_grounded_context import write_json
from eval_stride_ov_loveda_e1 import Confusion, make_checkpoints


SEED = 20261001
IMPLEMENTATION = "geometry-readout-observation-20261001"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("vdd", "potsdam"))
    for name in ("dinov3-repo", "checkpoint-dir", "data-root", "vocabulary-config",
                 "reference-manifest", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-seed", type=int, default=SEED, choices=(SEED,))
    parser.add_argument("--vdd-ontology", default="official", choices=("official",))
    return parser.parse_args()


def add_patch_attribution(totals, classes, contributions, truth, valid):
    prediction = classes.argmax(-1)
    for actual in range(classes.shape[-1]):
        for predicted in range(classes.shape[-1]):
            selected = valid & (truth == actual) & (prediction == predicted)
            count = int(selected.sum())
            if not count:
                continue
            key = f"{actual}->{predicted}"
            entry = totals.setdefault(key, {"patch_centers": 0, "margin_sum": 0., "components": {}})
            entry["patch_centers"] += count
            comparison = classes.clone()
            comparison[..., predicted] = -torch.inf
            competitor = (comparison.argmax(-1) if actual == predicted
                          else torch.full_like(prediction, actual))
            margin = classes[..., predicted] - classes.gather(-1, competitor[..., None])[..., 0]
            entry["margin_sum"] += float(margin[selected].sum())
            for name, component in contributions.items():
                value = component[..., predicted] - component.gather(-1, competitor[..., None])[..., 0]
                entry["components"][name] = entry["components"].get(name, 0.) + float(value[selected].sum())


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Refusing an existing diagnostic output directory.")
    vocabulary_path = Path(args.vocabulary_config)
    vocabulary = load_class_specs(vocabulary_path)
    samples, specs, load_rgb, load_mask = protocol(args, vocabulary)
    reference = json.loads(Path(args.reference_manifest).read_text())[args.dataset]
    import hashlib
    if (len(samples) != reference["global_sample_count"]
            or digest([sample.key for sample in samples]) != reference["global_sample_keys_sha256"]
            or hashlib.sha256(vocabulary_path.read_bytes()).hexdigest() != reference["vocabulary_sha256"]):
        raise ValueError("Historical sample set or vocabulary changed.")
    random.Random(SEED).shuffle(samples)
    samples = samples[:8]
    if len(samples) != 8 or len({sample.key for sample in samples}) != 8:
        raise ValueError("Requires exactly eight distinct fixed samples.")
    output.mkdir(parents=True)
    work = output / "temporary"
    work.mkdir()
    snapshots = output / "features"
    snapshots.mkdir()
    config = TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2,
                        prefix_policy="preserve", relation_policy="dense")
    checkpoints = make_checkpoints(args)
    backbone = DINOTextSegmenter(checkpoints, device=args.device)
    reader = TCPRSegmenter(backbone, config)
    bank = reader.encode_text(specs[args.dataset])
    text = F.normalize(bank.features.float(), dim=-1)
    if any(int((bank.parent_indices == c).sum()) != 20 for c in range(bank.class_count)):
        raise ValueError("The unchanged Geometry vocabulary requires exactly 20 aliases.")
    signature = {"implementation": IMPLEMENTATION, "dataset": args.dataset,
                 "samples": [sample.key for sample in samples], "seed": SEED,
                 "geometry": asdict(config), "tile_size": 512, "overlap": 128,
                 "temperature": .07, "vocabulary_sha256": reference["vocabulary_sha256"],
                 "global_sample_keys_sha256": reference["global_sample_keys_sha256"],
                 "checkpoints": checkpoint_manifest(checkpoints),
                 "class_names": list(bank.class_names), "alias_names": list(bank.alias_names),
                 "stage_probe_note": "Intermediate states are read through the frozen final LN/projection. "
                                      "They are diagnostics, not deployed model variants.",
                 "attribution_note": "Conditional additive margins use observed LN/L2 denominators and "
                                      "final alias responsibilities; not causal feature-removal effects."}
    write_json(output / "signature.json", signature)
    torch.save({"features": bank.features.cpu(), "parents": bank.parent_indices.cpu()}, output / "text_bank.pt")
    matrices, patch_totals, image_records = {}, {}, []
    maximum_replay_error = maximum_attribution_error = 0.
    windows = unchanged_windows = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    blend = hann_blend_window(512)
    for image_index, sample in enumerate(samples):
        image = load_rgb(sample)
        h, w = image.shape[-2:]
        coordinates = [(top, left) for top in tile_starts(h, 512, 128)
                       for left in tile_starts(w, 512, 128)]
        saved_coordinates = set(random.Random(SEED + image_index).sample(coordinates, min(2, len(coordinates))))
        record = {"sample_key": sample.key, "height": h, "width": w,
                  "windows": len(coordinates), "saved_windows": sorted(saved_coordinates)}
        buffers = {}
        target = None
        try:
            for window_index, (top, left) in enumerate(coordinates):
                rgb = _crop_at(image, top, left, 512).to(reader.device)
                prepared = reader.prepare_image(rgb)
                head = backbone.model.visual_model.head
                with backbone._autocast():
                    trace = trace_geometry_head(head, prepared, config)
                    stages, aliases = probe_scores(head, trace, bank, prepared.prefix_tokens)
                replay = F.normalize(trace["projected"][:, prepared.prefix_tokens:].float(), dim=-1)
                replay_error = float((replay - prepared.geometry_projected).abs().max())
                maximum_replay_error = max(maximum_replay_error, replay_error)
                original_alias = prepared.geometry_projected.float() @ text.T
                original = alias_class_scores(original_alias, bank.parent_indices, bank.class_count)
                if not torch.equal(original.argmax(-1), stages["final"].argmax(-1)) or replay_error > 1e-6:
                    raise AssertionError("Observed replay changed the original Geometry output.")
                # Disable autocast for linear attribution so the explicit rounding
                # components account for the original mixed-precision path.
                components = exact_alias_attribution(head, trace, bank, prepared.prefix_tokens)
                contributions = class_attribution(original_alias, components, bank.parent_indices, bank.class_count)
                attribution_error = float((sum(contributions.values()) - original).abs().max())
                maximum_attribution_error = max(maximum_attribution_error, attribution_error)
                if attribution_error > 2e-5:
                    raise AssertionError("Conditional attribution failed to reconstruct class scores.")
                stages["Geometry_original"] = original
                ah, aw = min(512, h - top), min(512, w - left)
                for name, values in stages.items():
                    dense = F.interpolate(values.permute(0, 2, 1).reshape(1, bank.class_count, 32, 32),
                                          size=(512, 512), mode="bilinear", align_corners=False)
                    if name not in buffers:
                        buffers[name] = ProbabilityAccumulator(bank.class_count, h, w, 256, work)
                    buffers[name].add((dense[0, :, :ah, :aw] / .07).softmax(0).cpu().numpy(),
                                      blend[:ah, :aw], left, top)
                # Labels are loaded only after the fixed observation and score path.
                if target is None:
                    target = load_mask(sample, args.dataset, (h, w))
                cy = top + torch.arange(32) * 16 + 8
                cx = left + torch.arange(32) * 16 + 8
                valid = ((cy[:, None] < h) & (cx[None, :] < w)).reshape(1, -1).to(reader.device)
                centers = target[np.minimum(cy.numpy(), h - 1)[:, None], np.minimum(cx.numpy(), w - 1)[None, :]]
                truth = torch.from_numpy(centers.copy()).reshape(1, -1).to(reader.device)
                valid = valid & (truth >= 0) & (truth < bank.class_count)
                add_patch_attribution(patch_totals, original, contributions, truth, valid)
                if (top, left) in saved_coordinates:
                    cache = {"sample_key": sample.key, "top": top, "left": left,
                             "prefix_tokens": prepared.prefix_tokens,
                             "features": {key: value.detach().cpu() for key, value in trace["features"].items()},
                             "geometry_relation": prepared.geometry_patch_conditional.cpu(),
                             "original_alias_scores": original_alias.cpu(),
                             "stage_scores": {key: value.cpu() for key, value in stages.items()},
                             "alias_contributions": {key: value.cpu() for key, value in components.items()},
                             "target_patch_centers": truth.cpu(), "valid_patch_centers": valid.cpu()}
                    torch.save(cache, snapshots / f"image{image_index:02d}_window{window_index:04d}.pt")
                windows += 1
                unchanged_windows += 1
                del rgb, prepared, trace, stages, components, contributions
            for name, buffer in buffers.items():
                prediction = buffer.finalize(None)[0]
                if name not in matrices:
                    matrices[name] = Confusion(tuple(bank.class_names))
                matrices[name].update(prediction, target)
        finally:
            for buffer in buffers.values():
                buffer.close()
        image_records.append(record)
        result = {"status": "complete" if len(image_records) == 8 else "running",
                  "processed_images": len(image_records), "total_images": 8,
                  "signature": signature, "images": image_records,
                  "windows": windows, "unchanged_windows": unchanged_windows,
                  "maximum_replay_feature_error": maximum_replay_error,
                  "maximum_class_attribution_error": maximum_attribution_error,
                  "stage_probe_metrics": {name: value.summary() for name, value in matrices.items()},
                  "patch_center_attribution": patch_totals,
                  "wall_seconds": time.perf_counter() - started,
                  "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1048576}
        write_json(output / "results.json", result)
        print(json.dumps({"dataset": args.dataset, "processed": len(image_records),
                          "windows": windows, "replay_error": maximum_replay_error}), flush=True)
    if matrices["final"].matrix.tolist() != matrices["Geometry_original"].matrix.tolist():
        raise AssertionError("Observed final predictions differ from original Geometry.")
    return result


if __name__ == "__main__":
    main(parse_args())
