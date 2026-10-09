"""Full-image fixed Geometry/internal cross-view Value innovation evaluation."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.cross_view_value_innovation import (
    IMPLEMENTATION, PRIMARY, METHODS, ValueInnovationConfig, cross_view_prior, read_value_innovation)
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.matched_readout_controls import run_head
from dinotool.model import DINOTextSegmenter
from dinotool.prompts import load_class_specs
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
import eval_frozen_semantic_path as evaluator
from eval_gear_ov import protocol
from eval_stride_ov_loveda_e1 import make_checkpoints


CONFIG = ValueInnovationConfig()


def head_fingerprint(head):
    digest = hashlib.sha256()
    for name, value in head.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def coordinates(geometry, h, w, top, left):
    cy, cx = torch.meshgrid(torch.arange(32, device=geometry.device),
                            torch.arange(32, device=geometry.device), indexing="ij")
    fine_xy = torch.stack(((cy.flatten()+.5)*16, (cx.flatten()+.5)*16), -1)/512
    context_xy = torch.stack(((cy.flatten()+.5)*32-256, (cx.flatten()+.5)*32-256), -1)/512
    fine_valid = (((cy+.5)*16 < min(512, h-top)) & ((cx+.5)*16 < min(512, w-left))).reshape(1, -1)
    yy, xx = top-256+(cy+.5)*32, left-256+(cx+.5)*32
    context_valid = ((yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)).reshape(1, -1)
    return fine_xy, context_xy, fine_valid, context_valid


@torch.inference_mode()
def tile_features(image, geometry, prepared, top, left, *, controls=True):
    h, w = image.shape[-2:]
    context_rgb = F.interpolate(_crop_at(image, top-256, left-256, 1024).to(geometry.device),
                                (512, 512), mode="bilinear", align_corners=False)
    context = geometry.prepare_image(context_rgb)
    fine_xy, context_xy, valid, context_valid = coordinates(geometry, h, w, top, left)
    prior = cross_view_prior(prepared.raw_patch_tokens, context.raw_patch_tokens, fine_xy, context_xy,
        context_valid, temperature=geometry.config.geometry_temperature, spatial_sigma=geometry.config.spatial_sigma)
    head = geometry.backbone.model.visual_model.head
    with geometry.backbone._autocast():
        features, diagnostics = read_value_innovation(head, prepared, context, valid, context_valid, prior)
        output = {"Geometry": prepared.geometry_projected, PRIMARY: features}
        if controls:
            active = context_valid[0].nonzero().flatten()
            permutation = torch.arange(context_valid.shape[1], device=geometry.device)
            permutation[active] = active.roll(len(active)//2)
            output["Shuffled_ValueInnovation"], _ = read_value_innovation(head, prepared, context, valid,
                context_valid, prior, context_permutation=permutation)
            raw = prepared.backbone_tokens[:, prepared.prefix_tokens:]
            for method in ("SCLIP_Two", "VIPProxy_Two"):
                output[method] = run_head(head, prepared.backbone_tokens, raw,
                    prepared.geometry_patch_conditional, prepared.prefix_tokens, method, prepared.block_index)[0]
    if any(not torch.isfinite(value).all() for value in output.values()):
        raise RuntimeError("Nonfinite complete readout.")
    return output, context, diagnostics


@torch.inference_mode()
def predict_image(image, geometry, banks, work, methods, reader, *, reader_uses_rgb=False):
    h, w = image.shape[-2:]
    blend = hann_blend_window(512)
    texts = {key: F.normalize(bank.features.float(), dim=-1) for key, bank in banks.items()}
    totals = {key: {"tiles": 0} for key in banks}
    with ExitStack() as stack:
        accumulators = {(key, method): stack.enter_context(ProbabilityAccumulator(bank.class_count, h, w, 256, work))
                        for key, bank in banks.items() for method in methods}
        for top in tile_starts(h, 512, 128):
            for left in tile_starts(w, 512, 128):
                prepared = geometry.prepare_image(_crop_at(image, top, left, 512).to(geometry.device))
                features, context, diagnostics = tile_features(image, geometry, prepared, top, left)
                ah, aw = min(512, h-top), min(512, w-left)
                for key, bank in banks.items():
                    scores = {method: alias_class_scores(feature.float() @ texts[key].T,
                        bank.parent_indices, bank.class_count) for method, feature in features.items()}
                    coarse = alias_class_scores(context.geometry_projected.float() @ texts[key].T,
                        bank.parent_indices, bank.class_count).transpose(1, 2).reshape(1, bank.class_count, 32, 32)
                    scores["ContextGeometry"] = F.interpolate(coarse[:, :, 8:24, 8:24], (32, 32),
                        mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)
                    scores["MeanLogit_ContextGeometry"] = .5*(scores["Geometry"]+scores["ContextGeometry"])
                    for method in methods:
                        dense = F.interpolate(scores[method].transpose(1, 2).reshape(1, bank.class_count, 32, 32),
                            (512, 512), mode="bilinear", align_corners=False)[0]/.07
                        accumulators[key, method].add(dense[:, :ah, :aw].softmax(0).cpu().numpy(), blend[:ah, :aw], left, top)
                    totals[key]["tiles"] += 1
                    for field, value in diagnostics.items():
                        totals[key][field] = totals[key].get(field, 0.)+value
        predictions = {key: {method: accumulators[key, method].finalize(None)[0] for method in methods} for key in banks}
    return predictions, totals


@torch.inference_mode()
def smoke(args):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError("Existing smoke output.")
    samples, specs, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    geometry = TCPRSegmenter(DINOTextSegmenter(make_checkpoints(args), device=args.device),
        TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == "loveda" else sample)
    prepared = geometry.prepare_image(_crop_at(image, 0, 0, 512).to(geometry.device))
    _, _, valid, _ = coordinates(geometry, *image.shape[-2:], 0, 0)
    head = geometry.backbone.model.visual_model.head
    fingerprint = head_fingerprint(head)
    started = time.perf_counter()
    with geometry.backbone._autocast():
        duplicate, _ = read_value_innovation(geometry.backbone.model.visual_model.head,
            prepared, prepared, valid, valid, prepared.geometry_patch_conditional)
        absent, _ = read_value_innovation(geometry.backbone.model.visual_model.head,
            prepared, prepared, valid, torch.zeros_like(valid), prepared.geometry_patch_conditional)
    features, _, diagnostics = tile_features(image, geometry, prepared, 0, 0)
    errors = {"duplicate_view_max_error": float((duplicate-prepared.geometry_projected).abs().max()),
              "no_context_max_error": float((absent-prepared.geometry_projected).abs().max())}
    if any(errors.values()) or fingerprint != head_fingerprint(head):
        raise RuntimeError("Original Geometry/weight identity failed.")
    output.mkdir(parents=True)
    result = {"status": "complete", "implementation": IMPLEMENTATION, "target_masks_loaded": False,
        "config": CONFIG.signature(), "errors": errors, "diagnostics": diagnostics,
        "methods_finite": all(bool(torch.isfinite(value).all()) for value in features.values()),
        "weights_frozen": all(not p.requires_grad for p in geometry.backbone.model.parameters()),
        "head_weights_unchanged": True, "head_sha256": fingerprint, "wall_seconds": time.perf_counter()-started}
    (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    for field in ("dinov3-repo", "checkpoint-dir", "data-root", "vocabulary-config", "output-dir"):
        parser.add_argument("--"+field, required=True)
    parser.add_argument("--source-diagnostic", type=Path)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--vdd-ontology", default="official")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke:
        smoke(args)
    else:
        evaluator.predict_image = predict_image
        evaluator.main(args, methods=METHODS, implementation=IMPLEMENTATION, primary=PRIMARY,
            readout_settings=CONFIG.signature(), save_per_image=True,
            signature_note="Frozen same-encoder internal matched fine/context Value innovation; "
                "primary does not use VIP or external teachers. Both original head blocks and all20 aliases. "
                "Native QK is not a correctness oracle; no trained gate, label fitting or dataset routing. "
                "Full-image exploratory96 screen; nearest operators are adaptations, not official systems.")
