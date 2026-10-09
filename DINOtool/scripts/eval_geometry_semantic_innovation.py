#!/usr/bin/env python3
"""Fixed semantic-innovation candidate and matched-source fusion controls."""
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
import torch
import torch.nn.functional as F

from dinotool.gear_ov import _crop_at
from dinotool.finite_vip_observer import FiniteVIPObserver
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.geometry_semantic_innovation import IMPLEMENTATION, anchored_innovation, semantic_innovation
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from dinotool.vip_official_adapter import VIPOfficialAdapter, VIPSettings
from eval_gear_ov import digest, protocol
from eval_geometry_vip_reliability import sample_broad, summary
from eval_matched_head_fov import resize_rgb
from eval_matched_head_text import imagenet_geometry_logits
from eval_stride_ov_loveda_e1 import make_checkpoints
from eval_vip_official_eight import PINNED_COMMIT


METHODS = ("Geometry", "BroadGeometry", "BroadVIP", "MeanProb_Geo", "MeanLogit_Geo",
           "Innovation_Geo", "Anchored_Geo", "MeanProb_VIP", "MeanLogit_VIP", "Innovation_VIP", "Anchored_VIP")
LOCAL_METHODS = ("Geometry", "Innovation_Geo", "Innovation_VIP", "MeanLogit_Geo", "MeanLogit_VIP", "Anchored_Geo", "Anchored_VIP")
SETTINGS = VIPSettings(tau=1., tem=1., background=False, prob_thd=0.)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("loveda", "udd5", "oem", "vdd",
                        "potsdam", "vaihingen", "landcoverai", "flair1"))
    for name in ("dinov3-repo", "checkpoint-dir", "data-root", "vocabulary-config", "upstream-root", "output-dir"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--source-diagnostic", type=Path)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--vdd-ontology", default="official", choices=("official",))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.inference_mode()
def wide_observations(image, geometry, vip, queries_by_protocol):
    resized = resize_rgb(image, 448)
    h, w = resized.shape[-2:]
    maps = {key: {source: torch.zeros(len(queries.class_names), h, w, device=geometry.device)
                  for source in ("Geo", "VIP")} for key, queries in queries_by_protocol.items()}
    count = torch.zeros(h, w, device=geometry.device)
    for top in tile_starts(h, 336, 224):
        for left in tile_starts(w, 336, 224):
            ah, aw = min(336, h-top), min(336, w-left)
            rgb = F.pad(resized[:, top:top+ah, left:left+aw], (0, 336-aw, 0, 336-ah))
            prepared = geometry.prepare_image(rgb[None].to(geometry.device))
            features = {"Geo": prepared.geometry_projected, "VIP": vip.crop_patch_features(rgb)}
            for key, queries in queries_by_protocol.items():
                for source, value in features.items():
                    logits = imagenet_geometry_logits(value, queries, SETTINGS)
                    maps[key][source][:, top:top+ah, left:left+aw] += logits[:, :ah, :aw]
            count[top:top+ah, left:left+aw] += 1
    if not bool((count > 0).all()):
        raise ValueError("Uncovered broad-view pixels.")
    return {key: {source: value/count[None] for source, value in sources.items()}
            for key, sources in maps.items()}


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work):
    h, w = image.shape[-2:]
    before_empty, before_rows = vip.empty_rows, vip.observed_rows
    broad = wide_observations(image, geometry, vip, queries)
    blend = hann_blend_window(512)
    texts = {key: F.normalize(bank.features.float(), dim=-1) for key, bank in banks.items()}
    diagnostics = {key: {"tiles": 0, "mean_absolute_innovation_Geo": 0.,
                        "mean_absolute_innovation_VIP": 0., "changed_patch_fraction_Geo": 0.,
                        "changed_patch_fraction_VIP": 0.} for key in banks}
    for values in diagnostics.values():
        values["observer_empty_rows"] = vip.empty_rows-before_empty
        values["observer_total_rows"] = vip.observed_rows-before_rows
        for source in ("Geo", "VIP"):
            values.update({"anchored_"+field+"_"+source: 0. for field in
                           ("mean_absolute_innovation", "changed_patch_fraction", "solver_relative_residual",
                            "solver_iterations", "energy_before", "energy_after")})
    with ExitStack() as stack:
        accumulators = {(key, method): stack.enter_context(ProbabilityAccumulator(
                        bank.class_count, h, w, 256, work)) for key, bank in banks.items()
                        for method in LOCAL_METHODS}
        for top in tile_starts(h, 512, 128):
            for left in tile_starts(w, 512, 128):
                prepared = geometry.prepare_image(_crop_at(image, top, left, 512).to(geometry.device))
                ah, aw = min(512, h-top), min(512, w-left)
                yy = top + (torch.arange(32, device=geometry.device) + .5)*16
                xx = left + (torch.arange(32, device=geometry.device) + .5)*16
                valid = ((yy[:, None] < h) & (xx[None] < w)).reshape(1, -1)
                for key, bank in banks.items():
                    raw_scores = alias_class_scores(prepared.geometry_projected.float() @ texts[key].T,
                                                    bank.parent_indices, bank.class_count)
                    scores = raw_scores / .07
                    outputs = {"Geometry": scores}
                    for source in ("Geo", "VIP"):
                        observation = sample_broad(broad[key][source], top, left, h, w).reshape(1, 1024, bank.class_count)
                        corrected, current = semantic_innovation(scores, observation,
                                                prepared.geometry_patch_conditional, valid)
                        outputs["Innovation_"+source] = corrected
                        outputs["MeanLogit_"+source] = .5*(scores+observation)
                        anchored, solver = anchored_innovation(scores, observation,
                                               prepared.geometry_patch_conditional, valid)
                        outputs["Anchored_"+source] = anchored
                        for field, value in solver.items():
                            diagnostics[key]["anchored_"+field+"_"+source] += value
                        for field in ("mean_absolute_innovation", "changed_patch_fraction"):
                            diagnostics[key][field+"_"+source] += current[field]
                    for method, logits in outputs.items():
                        # Keep the original control's interpolation-before-temperature order.
                        value = raw_scores if method == "Geometry" else logits
                        dense = F.interpolate(value.transpose(1, 2).reshape(1, bank.class_count, 32, 32),
                                               size=(512, 512), mode="bilinear", align_corners=False)[0]
                        if method == "Geometry":
                            dense = dense/.07
                        accumulators[key, method].add(dense[:, :ah, :aw].softmax(0).cpu().numpy(),
                                                       blend[:ah, :aw], left, top)
                    diagnostics[key]["tiles"] += 1
        predictions = {key: {method: accumulators[key, method].finalize(None)[0]
                            for method in LOCAL_METHODS}
                       for key in banks}
        for key, bank in banks.items():
            for source in ("Geo", "VIP"):
                dense = F.interpolate(broad[key][source][None], size=(h, w), mode="bilinear", align_corners=False)[0]
                probabilities = dense.softmax(0).cpu().numpy()
                predictions[key]["Broad"+("Geometry" if source == "Geo" else "VIP")] = probabilities.argmax(0).astype(np.uint8)
                fused = np.empty((h, w), dtype=np.uint8)
                accumulator = accumulators[key, "Geometry"]
                for top in range(0, h, 128):
                    end = min(top+128, h)
                    gp = accumulator.probabilities[:, top:end] / accumulator.normalizer[None, top:end].clip(1e-8)
                    fused[top:end] = (gp+probabilities[:, top:end]).argmax(0).astype(np.uint8)
                predictions[key]["MeanProb_"+source] = fused
    return predictions, diagnostics


def main(args):
    output = Path(args.output_dir)
    if output.exists() or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Existing output or invalid shard.")
    vocabulary = load_class_specs(args.vocabulary_config)
    samples, specs, load_image, load_mask = protocol(args, vocabulary)
    if args.source_diagnostic:
        source = json.loads(args.source_diagnostic.read_text())
        lookup = {sample.key: sample for sample in samples}
        samples = [lookup[key] for key in source["signature"]["samples"]]
    keys = [sample.key for sample in samples]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate global keys.")
    selected = samples[args.shard_index::args.num_shards]
    if not selected:
        raise ValueError("Empty shard.")
    upstream = Path(args.upstream_root)
    commit = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if commit != PINNED_COMMIT:
        raise ValueError("Pinned VIP changed.")
    checkpoints = make_checkpoints(args)
    backbone = DINOTextSegmenter(checkpoints, device=args.device)
    config = TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2)
    geometry = TCPRSegmenter(backbone, config)
    banks = {key: geometry.encode_text(classes) for key, classes in specs.items()}
    vip = FiniteVIPObserver(DINOTextSegmenter(checkpoints, device=args.device), upstream)
    queries = {key: vip.encode_queries(bank.class_names, tuple(spec.synonyms for spec in specs[key]))
               for key, bank in banks.items()}
    for key, bank in banks.items():
        if (bank.alias_names != queries[key].aliases or not torch.equal(bank.parent_indices, queries[key].parents)
                or any(int((bank.parent_indices == c).sum()) != 20 for c in range(bank.class_count))):
            raise ValueError("Requires matched exact 20-alias groups.")
    signature = {"implementation": IMPLEMENTATION, "dataset": args.dataset, "methods": METHODS,
        "classes": {key: bank.class_names for key, bank in banks.items()}, "gear": {"geometry": asdict(config),
        "observation": asdict(SETTINGS), "innovation": "g + A(b-g)",
        "anchored": "min_z .5||z-g||^2+.5||A(z-b)||^2; CG 32, relative residual 1e-6",
        "observer_empty_support_policy": "identity Value on empty rows; other rows unchanged from pinned VIP",
        "upstream_commit": commit},
        "competitive": None, "vocabulary": {"sha256": hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest(),
        "aliases": {key: bank.alias_names for key, bank in banks.items()},
        "counts": {key: [20]*bank.class_count for key, bank in banks.items()}}, "checkpoints": checkpoint_manifest(checkpoints),
        "global_sample_count": len(keys), "global_sample_keys_sha256": digest(keys),
        "sample_keys": [sample.key for sample in selected], "sample_keys_sha256": digest([sample.key for sample in selected]),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "note": "Frozen common rule; no label-dependent selector. Wide Geometry and VIP share ImageNet/profile scoring, "
                "448 long edge and 336/112 windows; no background threshold, fixed tau=tem=1 on all domains. "
                "VIP is an information-source control, not a new contribution. All datasets are exploratory development."}
    output.mkdir(parents=True)
    matrices = {key: {method: np.zeros((bank.class_count, bank.class_count), np.int64) for method in METHODS}
                for key, bank in banks.items()}
    transitions = {key: {method: np.zeros((bank.class_count,)*3, np.int64) for method in METHODS if method != "Geometry"}
                   for key, bank in banks.items()}
    ignored = dict.fromkeys(banks, 0)
    diagnostics = {key: {} for key in banks}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for number, sample in enumerate(selected, 1):
        image = load_image(sample.image_path if args.dataset == "loveda" else sample)
        predictions, current = predict_image(image, geometry, banks, vip, queries, output)
        for key, bank in banks.items():
            target = load_mask(sample, key, tuple(image.shape[-2:]))
            valid = (target >= 0) & (target < bank.class_count)
            ignored[key] += int((~valid).sum())
            for method in METHODS:
                encoded = target[valid].astype(np.int64)*bank.class_count+predictions[key][method][valid]
                matrices[key][method] += np.bincount(encoded, minlength=bank.class_count**2).reshape(bank.class_count, -1)
                if method != "Geometry":
                    transitions[key][method] += transition_counts(predictions[key]["Geometry"], predictions[key][method], target, bank.class_count)
            for field, value in current[key].items():
                diagnostics[key][field] = diagnostics[key].get(field, 0)+value
        result = {"status": "complete" if number == len(selected) else "running",
            "processed_images": number, "total_images": len(selected), "signature": signature,
            "metrics": {key: {method: summary(matrix, banks[key].class_names, ignored[key]) for method, matrix in methods.items()}
                        for key, methods in matrices.items()},
            "transitions": {key: {method: {"counts": counts.tolist(), **transition_summary(counts)}
                                    for method, counts in methods.items()} for key, methods in transitions.items()},
            "diagnostics": {key: {field: value if field == "tiles" else value/max(values["tiles"], 1)
                                  for field, value in values.items()} for key, values in diagnostics.items()},
            "wall_seconds": time.perf_counter()-started, "peak_cuda_memory_mb": torch.cuda.max_memory_allocated()/1048576}
        (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({"dataset": args.dataset, "processed": number, "total": len(selected),
              "miou": {key: {method: row["mean_iou_percent"] for method, row in values.items()}
                       for key, values in result["metrics"].items()}}), flush=True)


if __name__ == "__main__":
    main(parse_args())
