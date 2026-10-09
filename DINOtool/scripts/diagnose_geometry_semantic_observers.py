#!/usr/bin/env python3
"""Matched dense semantic observations on the fixed Geometry diagnostic samples."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from dinotool.semantic_observer_trace import matched_vip_features, nonself_supported
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from dinotool.vip_official_adapter import VIPOfficialAdapter, upstream_settings
from eval_gear_ov import digest, protocol
from eval_geometry_vip_reliability import sample_broad, summary
from eval_matched_head_fov import resize_rgb
from eval_matched_head_text import imagenet_geometry_logits
from eval_stride_ov_loveda_e1 import make_checkpoints
from eval_vip_official_eight import PINNED_COMMIT


IMPLEMENTATION = "geometry-semantic-observer-diagnostic-20261001"
LOCAL = ("Geometry", "Native_Local_RS", "VIP_Matched_Local_RS")
WIDE = ("VIP_Broad_RS", "VIP_Broad_ImageNet_LME", "VIP_Broad_Profile")
METHODS = LOCAL + WIDE + ("MeanProb50",)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("vdd", "potsdam"))
    for name in ("dinov3-repo", "checkpoint-dir", "data-root", "vocabulary-config",
                 "upstream-root", "source-diagnostic-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--num-shards", type=int, default=4, choices=(4,))
    parser.add_argument("--shard-index", type=int, required=True, choices=range(4))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-seed", type=int, default=20261001, choices=(20261001,))
    parser.add_argument("--vdd-ontology", default="official", choices=("official",))
    return parser.parse_args()


@torch.inference_mode()
def wide_observations(image, vip, queries, bank, settings):
    resized = resize_rgb(image, 448)
    h, w = resized.shape[-2:]
    maps = {method: torch.zeros((bank.class_count, h, w), device=vip.device) for method in WIDE}
    counts = torch.zeros((h, w), device=vip.device)
    text = F.normalize(bank.features.float(), dim=-1)
    imagenet_text = queries.features.float().mean(1)
    for top in tile_starts(h, 336, 224):
        for left in tile_starts(w, 336, 224):
            ah, aw = min(336, h-top), min(336, w-left)
            rgb = resized[:, top:top+ah, left:left+aw]
            rgb = F.pad(rgb, (0, 336-aw, 0, 336-ah))
            features = vip.crop_patch_features(rgb)
            scores = {
                "VIP_Broad_RS": alias_class_scores(features.float() @ text.T,
                                                     bank.parent_indices, bank.class_count),
                "VIP_Broad_ImageNet_LME": alias_class_scores(features.float() @ imagenet_text.T,
                                                               bank.parent_indices, bank.class_count),
            }
            dense = {method: F.interpolate(value.transpose(1, 2).reshape(1, bank.class_count, 21, 21),
                                            size=(336, 336), mode="bilinear", align_corners=False)[0]
                     for method, value in scores.items()}
            dense["VIP_Broad_Profile"] = imagenet_geometry_logits(features, queries, settings)
            for method, value in dense.items():
                maps[method][:, top:top+ah, left:left+aw] += value[:, :ah, :aw]
            counts[top:top+ah, left:left+aw] += 1
    if not bool((counts > 0).all()):
        raise ValueError("Uncovered wide-view pixels.")
    return {method: value / counts[None] for method, value in maps.items()}


@torch.inference_mode()
def observe_image(image, sample, image_index, record, geometry, bank, vip, queries, settings,
                  output, load_target, dataset):
    h, w = image.shape[-2:]
    classes = bank.class_count
    text = F.normalize(bank.features.float(), dim=-1)
    wide = wide_observations(image, vip, queries, bank, settings)
    saved_coordinates = {tuple(value) for value in record["saved_windows"]}
    target = None
    blend = hann_blend_window(512)
    window_count = 0
    with ExitStack() as stack:
        accumulators = {method: stack.enter_context(ProbabilityAccumulator(classes, h, w, 256, output))
                        for method in LOCAL}
        for top in tile_starts(h, 512, 128):
            for left in tile_starts(w, 512, 128):
                rgb = _crop_at(image, top, left, 512).to(geometry.device)
                prepared = geometry.prepare_image(rgb)
                with geometry.backbone._autocast():
                    matched = matched_vip_features(geometry.backbone.model.visual_model.head,
                                                  prepared, vip.vip_head.proxy_attn)
                features = {"Geometry": prepared.geometry_projected,
                            "Native_Local_RS": prepared.native_projected,
                            "VIP_Matched_Local_RS": matched}
                aliases = {method: value.float() @ text.T for method, value in features.items()}
                scores = {method: alias_class_scores(value, bank.parent_indices, classes)
                          for method, value in aliases.items()}
                ah, aw = min(512, h-top), min(512, w-left)
                for method in LOCAL:
                    dense = F.interpolate(scores[method].transpose(1, 2).reshape(1, classes, 32, 32),
                                          size=(512, 512), mode="bilinear", align_corners=False)[0]
                    accumulators[method].add((dense[:, :ah, :aw] / .07).softmax(0).cpu().numpy(),
                                             blend[:ah, :aw], left, top)
                for method, value in wide.items():
                    sampled = sample_broad(value, top, left, h, w).reshape(1, 1024, classes)
                    scores[method] = sampled / 40 if method == "VIP_Broad_Profile" else sampled
                # Masks are used only after all observer scores are computed.
                if target is None:
                    target = load_target(sample, dataset, (h, w))
                cy, cx = top + np.arange(32)*16 + 8, left + np.arange(32)*16 + 8
                truth = target[np.minimum(cy, h-1)[:, None], np.minimum(cx, w-1)[None, :]]
                valid = torch.from_numpy(((cy[:, None] < h) & (cx[None, :] < w)).reshape(1, -1)).to(geometry.device)
                supported = {method: nonself_supported(value, prepared.geometry_patch_conditional, valid)
                             for method, value in scores.items()}
                payload = {"sample_key": sample.key, "top": top, "left": left,
                           "scores": {method: value.cpu() for method, value in scores.items()},
                           "supported_scores": {method: value.cpu() for method, value in supported.items()},
                           "target_patch_centers": torch.from_numpy(truth.copy()).reshape(1, -1),
                           "valid_patch_centers": valid.cpu()}
                if (top, left) in saved_coordinates:
                    payload.update(alias_scores={method: value.cpu() for method, value in aliases.items()},
                                   geometry_relation=prepared.geometry_patch_conditional.cpu())
                torch.save(payload, output / "observations" / f"image{image_index:02d}_window{window_count:04d}.pt")
                window_count += 1
                del prepared, matched, features, aliases, scores, supported, payload
        if window_count != record["windows"]:
            raise ValueError("Original window sequence changed.")
        predictions = {method: accumulator.finalize(None)[0] for method, accumulator in accumulators.items()}
        vip_probabilities = None
        for method, values in wide.items():
            dense = F.interpolate(values[None], size=(h, w), mode="bilinear", align_corners=False)[0]
            probabilities = (dense / (1. if method == "VIP_Broad_Profile" else .07)).softmax(0)
            predictions[method] = probabilities.argmax(0).to(torch.uint8).cpu().numpy()
            if method == "VIP_Broad_Profile":
                vip_probabilities = probabilities.cpu().numpy()
            del dense, probabilities
        predictions["MeanProb50"] = np.empty((h, w), dtype=np.uint8)
        accumulator = accumulators["Geometry"]
        for top in range(0, h, 128):
            end = min(top+128, h)
            gp = accumulator.probabilities[:, top:end] / accumulator.normalizer[None, top:end].clip(1e-8)
            predictions["MeanProb50"][top:end] = (gp + vip_probabilities[:, top:end]).argmax(0).astype(np.uint8)
        for method, prediction in predictions.items():
            Image.fromarray(prediction).save(output / "predictions" / f"image{image_index:02d}_{method}.png")
    return predictions, target, window_count


def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Refusing an existing output directory.")
    source = json.loads((Path(args.source_diagnostic_dir) / "results.json").read_text())
    if source["status"] != "complete" or source["processed_images"] != 8:
        raise ValueError("Requires the verified original eight-image diagnostic.")
    vocabulary = load_class_specs(args.vocabulary_config)
    all_samples, specs, load_image, load_target = protocol(args, vocabulary)
    lookup = {sample.key: sample for sample in all_samples}
    keys = source["signature"]["samples"]
    if (len(keys) != 8 or len(set(keys)) != 8
            or digest([sample.key for sample in all_samples]) != source["signature"]["global_sample_keys_sha256"]
            or hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest() != source["signature"]["vocabulary_sha256"]):
        raise ValueError("Source sample sequence or vocabulary changed.")
    upstream = Path(args.upstream_root)
    commit = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    if commit != PINNED_COMMIT:
        raise ValueError("Pinned VIP source changed.")
    output.mkdir(parents=True)
    (output / "observations").mkdir()
    (output / "predictions").mkdir()
    checkpoints = make_checkpoints(args)
    backbone = DINOTextSegmenter(checkpoints, device=args.device)
    config = TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2)
    geometry = TCPRSegmenter(backbone, config)
    bank = geometry.encode_text(specs[args.dataset])
    vip_backbone = DINOTextSegmenter(checkpoints, device=args.device)
    vip = VIPOfficialAdapter(vip_backbone, upstream)
    queries = vip.encode_queries(bank.class_names, tuple(spec.synonyms for spec in vocabulary))
    if bank.alias_names != queries.aliases or not torch.equal(bank.parent_indices, queries.parents):
        raise ValueError("Observer alias groups differ.")
    settings = upstream_settings(args.dataset)
    indices = list(range(8))[args.shard_index::args.num_shards]
    signature = {"implementation": IMPLEMENTATION, "dataset": args.dataset,
                 "methods": METHODS, "class_names": bank.class_names, "alias_names": bank.alias_names,
                 "alias_counts": [20]*bank.class_count, "geometry": asdict(config),
                 "vocabulary_sha256": source["signature"]["vocabulary_sha256"],
                 "global_sample_keys": keys, "global_sample_keys_sha256": digest(keys),
                 "sample_keys": [keys[index] for index in indices], "source": str(args.source_diagnostic_dir),
                 "checkpoints": checkpoint_manifest(checkpoints), "upstream_commit": commit,
                 "shard_index": args.shard_index, "num_shards": args.num_shards,
                 "note": "Local observers share input, Geometry fp32-weight/bf16 AMP runtime, RS text and LME. "
                         "Matched VIP uses pinned proxy at 32x32, not its official profile. Broad VIP uses a "
                         "separate fp16-weight runtime, 448 long edge, 336 crops/112 stride, no background threshold. "
                         "Profile scores in patch caches divide by 40; full-profile softmax uses original units. "
                         "Labels only audit observations. This is exploratory development, not independent test."}
    matrices = {method: np.zeros((bank.class_count, bank.class_count), np.int64) for method in METHODS}
    transitions = {method: np.zeros((bank.class_count,)*3, np.int64) for method in METHODS if method != "Geometry"}
    records, windows, ignored = [], 0, 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for index in indices:
        sample = lookup[keys[index]]
        if source["images"][index]["sample_key"] != sample.key:
            raise ValueError("Source image metadata order changed.")
        image = load_image(sample)
        predictions, target, count = observe_image(image, sample, index, source["images"][index],
                                                  geometry, bank, vip, queries, settings, output,
                                                  load_target, args.dataset)
        valid = (target >= 0) & (target < bank.class_count)
        ignored += int((~valid).sum())
        for method, prediction in predictions.items():
            encoded = target[valid].astype(np.int64)*bank.class_count + prediction[valid]
            matrices[method] += np.bincount(encoded, minlength=bank.class_count**2).reshape(bank.class_count, -1)
            if method != "Geometry":
                transitions[method] += transition_counts(predictions["Geometry"], prediction, target, bank.class_count)
        windows += count
        records.append({"sample_key": sample.key, "height": image.shape[-2], "width": image.shape[-1],
                        "windows": count, "image_index": index})
        result = {"status": "complete" if len(records) == len(indices) else "running",
                  "processed_images": len(records), "total_images": len(indices), "signature": signature,
                  "metrics": {method: summary(matrix, bank.class_names, ignored) for method, matrix in matrices.items()},
                  "transitions": {method: {"counts": counts.tolist(), **transition_summary(counts)}
                                  for method, counts in transitions.items()},
                  "images": records, "windows": windows, "wall_seconds": time.perf_counter()-started,
                  "peak_cuda_memory_mb": torch.cuda.max_memory_allocated()/1048576}
        (output / "results.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({"dataset": args.dataset, "shard": args.shard_index,
                          "processed": len(records), "total": len(indices),
                          "miou": {method: value["mean_iou_percent"] for method, value in result["metrics"].items()}}), flush=True)
    return result


if __name__ == "__main__":
    main(parse_args())
