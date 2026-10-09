"""Matched full-image evaluation of Geometry-conditioned paired context reads."""
import json
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.geometry_semantic_innovation import anchored_innovation
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.model import DINOTextSegmenter
from dinotool.paired_context_readout import CONFIG, IMPLEMENTATION, METHODS, PRIMARY, PairedContextObserver, fixed_profile_logits, footprint_overlap, patch_boxes, transported_support
from dinotool.prompts import load_class_specs
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
import eval_frozen_semantic_path as evaluator
from eval_gear_ov import protocol
from eval_geometry_semantic_innovation import SETTINGS, parse_args
from eval_geometry_vip_reliability import sample_broad
from eval_matched_head_fov import resize_rgb
from eval_matched_head_text import imagenet_geometry_logits
from eval_stride_ov_loveda_e1 import make_checkpoints
from eval_vip_official_eight import PINNED_COMMIT


LOCAL_METHODS = tuple(method for method in METHODS if method not in ("BroadVIP", "MeanProb_VIP"))


def make_observer(args, banks):
    commit = subprocess.check_output(["git", "-C", args.upstream_root, "rev-parse", "HEAD"], text=True).strip()
    if commit != PINNED_COMMIT:
        raise ValueError("Pinned VIP source changed.")
    vip = PairedContextObserver(DINOTextSegmenter(make_checkpoints(args), device=args.device), Path(args.upstream_root))
    queries = {key: vip.encode_queries(bank.class_names, tuple(spec.synonyms for spec in classes))
               for key, (bank, classes) in banks.items()}
    for key, (bank, _) in banks.items():
        if bank.alias_names != queries[key].aliases or not torch.equal(bank.parent_indices, queries[key].parents):
            raise ValueError("Unmatched alias groups.")
    return vip, queries


@torch.inference_mode()
def prepare_wide(image, vip, queries):
    original_h, original_w = image.shape[-2:]
    resized = resize_rgb(image, 448)
    h, w = resized.shape[-2:]
    maps = {key: torch.zeros(len(query.class_names), h, w, device=vip.device) for key, query in queries.items()}
    count = torch.zeros(h, w, device=vip.device)
    crops = []
    for top in tile_starts(h, 336, 224):
        for left in tile_starts(w, 336, 224):
            ah, aw = min(336, h-top), min(336, w-left)
            rgb = F.pad(resized[:, top:top+ah, left:left+aw], (0, 336-aw, 0, 336-ah))
            cache = vip.prepare_crop(rgb)
            logits = {key: imagenet_geometry_logits(cache.full, query, SETTINGS) for key, query in queries.items()}
            boxes = patch_boxes(21, 21, top*original_h/h, left*original_w/w,
                                16*original_h/h, 16*original_w/w, vip.device)
            for key in queries:
                maps[key][:, top:top+ah, left:left+aw] += logits[key][:, :ah, :aw]
            count[top:top+ah, left:left+aw] += 1
            crops.append((top, left, ah, aw, boxes, cache, logits))
    if not bool((count > 0).all()):
        raise ValueError("Incomplete wide-view coverage.")
    return {key: value/count[None] for key, value in maps.items()}, count, crops


@torch.inference_mode()
def context_maps(image, prepared, valid, top, left, vip, queries, full_maps, count, crops):
    h, w = image.shape[-2:]
    fine_boxes = patch_boxes(32, 32, top, left, 16, 16, vip.device)
    differences = {key: torch.zeros_like(value) for key, value in full_maps.items()}
    stats = {"reference_crops": 0., "mapped_queries": 0., "reference_empty_rows": 0.,
             "reference_rows": 0., "reference_support_fraction_sum": 0.}
    for y, x, ah, aw, boxes, cache, full in crops:
        overlap = footprint_overlap(boxes, fine_boxes, h, w)
        support, mapped = transported_support(prepared.geometry_patch_conditional, valid, overlap)
        if not bool(mapped.any()):
            continue
        reference, empty = vip.read_tokens(cache.tokens, cache.raw, support)
        stats["reference_crops"] += 1
        stats["mapped_queries"] += int(mapped.sum())
        stats["reference_empty_rows"] += empty
        stats["reference_rows"] += 2*len(support)
        stats["reference_support_fraction_sum"] += float(support[mapped].float().mean())
        for key, query in queries.items():
            ref_logits = fixed_profile_logits(reference, cache.full, query, SETTINGS)
            differences[key][:, y:y+ah, x:x+aw] += full[key][:, :ah, :aw]-ref_logits[:, :ah, :aw]
    return {key: value/count[None] for key, value in differences.items()}, stats


@torch.inference_mode()
def predict_image(image, geometry, banks, work, methods, reader, *, reader_uses_rgb=False):
    vip, queries = reader
    h, w = image.shape[-2:]
    full_maps, count, crops = prepare_wide(image, vip, queries)
    texts = {key: F.normalize(bank.features.float(), dim=-1) for key, bank in banks.items()}
    blend = hann_blend_window(512)
    totals = {key: {"tiles": 0, "wide_crops": len(crops)} for key in banks}
    with ExitStack() as stack:
        accumulators = {(key, method): stack.enter_context(ProbabilityAccumulator(bank.class_count, h, w, 256, work))
                        for key, bank in banks.items() for method in LOCAL_METHODS}
        for top in tile_starts(h, 512, 128):
            for left in tile_starts(w, 512, 128):
                prepared = geometry.prepare_image(_crop_at(image, top, left, 512).to(geometry.device))
                ah, aw = min(512, h-top), min(512, w-left)
                yy = top+(torch.arange(32, device=geometry.device)+.5)*16
                xx = left+(torch.arange(32, device=geometry.device)+.5)*16
                valid = ((yy[:, None] < h) & (xx[None] < w)).reshape(1, -1)
                delta_maps, stats = context_maps(image, prepared, valid, top, left, vip, queries, full_maps, count, crops)
                for key, bank in banks.items():
                    raw = alias_class_scores(prepared.geometry_projected.float() @ texts[key].T, bank.parent_indices, bank.class_count)
                    g = raw/.07
                    b = sample_broad(full_maps[key], top, left, h, w).reshape(1, 1024, bank.class_count)
                    delta = sample_broad(delta_maps[key], top, left, h, w).reshape_as(g)
                    anchored, _ = anchored_innovation(g, b, prepared.geometry_patch_conditional, valid)
                    coupled, diag = anchored_innovation(g, g+delta, prepared.geometry_patch_conditional, valid)
                    shuffled_relation = prepared.geometry_patch_conditional.roll((512, 512), (1, 2))
                    shuffled, _ = anchored_innovation(g, g+delta, shuffled_relation, valid)
                    scores = {"Geometry": raw, "MeanLogit_VIP": .5*(g+b), "Anchored_VIP": anchored,
                              "MeanLogit_Context": g+.5*delta, "ShuffledContextWriteback": shuffled, PRIMARY: coupled}
                    for method, value in scores.items():
                        dense = F.interpolate(value.transpose(1, 2).reshape(1, bank.class_count, 32, 32),
                                              size=(512, 512), mode="bilinear", align_corners=False)[0]
                        if method == "Geometry":
                            dense = dense/.07
                        accumulators[key, method].add(dense[:, :ah, :aw].softmax(0).cpu().numpy(), blend[:ah, :aw], left, top)
                    totals[key]["tiles"] += 1
                    for field, value in {**stats, **diag, "mean_absolute_context_delta": float(delta.abs().mean())}.items():
                        totals[key][field] = totals[key].get(field, 0.)+value
        predictions = {key: {method: accumulators[key, method].finalize(None)[0] for method in LOCAL_METHODS} for key in banks}
        for key in banks:
            broad = F.interpolate(full_maps[key][None], size=(h, w), mode="bilinear", align_corners=False)[0].softmax(0).cpu().numpy()
            predictions[key]["BroadVIP"] = broad.argmax(0).astype(np.uint8)
            fused = np.empty((h, w), dtype=np.uint8)
            accumulator = accumulators[key, "Geometry"]
            for top in range(0, h, 128):
                end = min(top+128, h)
                local = accumulator.probabilities[:, top:end]/accumulator.normalizer[None, top:end].clip(1e-8)
                fused[top:end] = (local+broad[:, top:end]).argmax(0).astype(np.uint8)
            predictions[key]["MeanProb_VIP"] = fused
    return predictions, totals


def smoke(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Existing smoke output.")
    vocabulary = load_class_specs(args.vocabulary_config)
    samples, specs, load_image, _ = protocol(args, vocabulary)
    geometry = TCPRSegmenter(DINOTextSegmenter(make_checkpoints(args), device=args.device), TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2))
    banks = {key: geometry.encode_text(classes) for key, classes in specs.items()}
    vip, queries = make_observer(args, {key: (banks[key], classes) for key, classes in specs.items()})
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == "loveda" else sample)
    rgb = resize_rgb(image, 448)[:, :336, :336]
    rgb = F.pad(rgb, (0, 336-rgb.shape[-1], 0, 336-rgb.shape[-2]))
    started = time.perf_counter()
    cache = vip.prepare_crop(rgb)
    inherited = vip.crop_patch_features(rgb)
    same, _ = vip.read_tokens(cache.tokens, cache.raw, torch.ones(441, 441, device=vip.device, dtype=torch.bool))
    prepared = geometry.prepare_image(_crop_at(image, 0, 0, 512).to(geometry.device))
    valid = torch.ones(1, 1024, device=vip.device, dtype=torch.bool)
    overlap = footprint_overlap(patch_boxes(21, 21, 0, 0, image.shape[-2]/28, image.shape[-1]/28, vip.device),
                                patch_boxes(32, 32, 0, 0, 16, 16, vip.device), *image.shape[-2:])
    support, _ = transported_support(prepared.geometry_patch_conditional, valid, overlap)
    ref, _ = vip.read_tokens(cache.tokens, cache.raw, support)
    checks = {"full_feature_replay_max_error": float((inherited-cache.full).abs().max()),
              "full_support_identity_max_error": float((same-cache.full).abs().max()),
              "finite_reference": bool(torch.isfinite(ref).all()),
              "weights_frozen": all(not parameter.requires_grad for parameter in vip.backbone.model.parameters())}
    checks["full_profile_logits_replay_max_error"] = max(float((imagenet_geometry_logits(cache.full, query, SETTINGS)-
                 fixed_profile_logits(cache.full, cache.full, query, SETTINGS)).abs().max()) for query in queries.values())
    if any(checks[field] != 0. for field in ("full_feature_replay_max_error", "full_support_identity_max_error", "full_profile_logits_replay_max_error")) or not checks["finite_reference"] or not checks["weights_frozen"]:
        raise ValueError(str(checks))
    torch.cuda.synchronize()
    output.mkdir(parents=True)
    result = {"status": "complete", "implementation": IMPLEMENTATION, "checks": checks,
              "wall_seconds": time.perf_counter()-started, "peak_cuda_memory_mb": torch.cuda.max_memory_allocated()/1048576}
    (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


def main(args):
    classes = load_class_specs(args.vocabulary_config)
    _, specs, _, _ = protocol(args, classes)
    def factory(geometry, banks):
        return make_observer(args, {key: (bank, specs[key]) for key, bank in banks.items()})
    evaluator.predict_image = predict_image
    evaluator.main(args, methods=METHODS, implementation=IMPLEMENTATION, primary=PRIMARY,
        reader_factory=factory, save_per_image=True,
        readout_settings={"paired_context": CONFIG, "observation": SETTINGS.__dict__, "upstream_commit": PINNED_COMMIT},
        signature_note="Frozen all20 paired contextual readout. Same wide backbone tokens and full-read text profile; "
            "only donor support differs between paired reads. Original Geometry and guarded wide VIP are attributed controls. "
            "Reference retains backbone context and two-block indirect influences; not semantic correctness certification. "
            "All eight domains are exploratory development; labels are loaded after complete-image predictions.")


if __name__ == "__main__":
    is_smoke = "--smoke" in sys.argv
    if is_smoke:
        sys.argv.remove("--smoke")
    args = parse_args()
    (smoke if is_smoke else main)(args)
