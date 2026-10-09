#!/usr/bin/env python3
"""Matched broad readout contrast on one shared frozen backbone."""
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
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.geometry_semantic_innovation import anchored_innovation, semantic_innovation
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.matched_semantic_innovation import IMPLEMENTATION, counterfactual_innovation, matched_proxy_features
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from dinotool.vip_official_adapter import VIPQueries, _load_upstream
from eval_geometry_semantic_innovation import SETTINGS, parse_args
from eval_geometry_vip_reliability import sample_broad, summary
from eval_gear_ov import digest, protocol
from eval_matched_head_fov import resize_rgb
from eval_matched_head_text import imagenet_geometry_logits
from eval_stride_ov_loveda_e1 import make_checkpoints
from eval_vip_official_eight import PINNED_COMMIT


METHODS = ("Geometry", "BroadGeometry", "BroadProxy", "MeanProb50", "MeanLogit50",
           "AbsoluteAnchor", "CounterfactualDirect", "CounterfactualGSI")
LOCAL_METHODS = ("Geometry", "MeanLogit50", "AbsoluteAnchor", "CounterfactualDirect", "CounterfactualGSI")


@torch.inference_mode()
def encode_queries(backbone, bank, template_source):
    templates = _load_upstream(template_source, "matched_observer_templates").get_text_template("openai_imagenet_template")
    prompts = [template(alias) for alias in bank.alias_names for template in templates]
    features = []
    for start in range(0, len(prompts), 64):
        tokens = backbone.tokenize(prompts[start:start+64]).to(backbone.device)
        with backbone._autocast():
            encoded = backbone.model.encode_text(tokens, normalize=False)
        features.append(F.normalize(encoded[:, encoded.shape[-1]//2:].float(), dim=-1))
    return VIPQueries(torch.cat(features).reshape(len(bank.alias_names), len(templates), -1),
                      bank.parent_indices, bank.class_names, bank.alias_names)


@torch.inference_mode()
def wide_observations(image, geometry, queries):
    resized = resize_rgb(image, 448)
    h, w = resized.shape[-2:]
    maps = {key: {source: torch.zeros(len(query.class_names), h, w, device=geometry.device)
                  for source in ("Geo", "Proxy")} for key, query in queries.items()}
    count = torch.zeros(h, w, device=geometry.device)
    diagnostics = {"observer_empty_rows": 0, "observer_total_rows": 0}
    for top in tile_starts(h, 336, 224):
        for left in tile_starts(w, 336, 224):
            ah, aw = min(336, h-top), min(336, w-left)
            rgb = F.pad(resized[:, top:top+ah, left:left+aw], (0, 336-aw, 0, 336-ah))
            prepared = geometry.prepare_image(rgb[None].to(geometry.device))
            with geometry.backbone._autocast():
                proxy, counts = matched_proxy_features(geometry.backbone.model.visual_model.head, prepared)
            diagnostics["observer_empty_rows"] += counts["empty_rows"]
            diagnostics["observer_total_rows"] += counts["observed_rows"]
            for key, query in queries.items():
                for source, feature in (("Geo", prepared.geometry_projected), ("Proxy", proxy)):
                    logits = imagenet_geometry_logits(feature, query, SETTINGS)
                    if not bool(torch.isfinite(logits).all()):
                        raise ValueError("Nonfinite matched broad logits.")
                    maps[key][source][:, top:top+ah, left:left+aw] += logits[:, :ah, :aw]
            count[top:top+ah, left:left+aw] += 1
    if not bool((count > 0).all()):
        raise ValueError("Uncovered broad pixels.")
    return {key: {source: value/count[None] for source, value in sources.items()}
            for key, sources in maps.items()}, diagnostics


@torch.inference_mode()
def predict_image(image, geometry, banks, queries, work):
    h, w = image.shape[-2:]
    broad, observer = wide_observations(image, geometry, queries)
    texts = {key: F.normalize(bank.features.float(), dim=-1) for key, bank in banks.items()}
    diagnostics = {key: {"tiles": 0, **observer, "mean_absolute_innovation": 0.,
                        "changed_patch_fraction": 0., "solver_relative_residual": 0.,
                        "solver_iterations": 0., "energy_before": 0., "energy_after": 0.} for key in banks}
    blend = hann_blend_window(512)
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
                for key, bank in banks.items():
                    raw = alias_class_scores(prepared.geometry_projected.float() @ texts[key].T, bank.parent_indices, bank.class_count)
                    local = raw/.07
                    control, treated = [sample_broad(broad[key][source], top, left, h, w).reshape(1, 1024, bank.class_count)
                                        for source in ("Geo", "Proxy")]
                    relation = prepared.geometry_patch_conditional
                    corrected, solver = counterfactual_innovation(local, treated, control, relation, valid)
                    direct, _ = semantic_innovation(local, local+(treated-control), relation, valid)
                    absolute, _ = anchored_innovation(local, treated, relation, valid)
                    outputs = {"Geometry": raw, "MeanLogit50": .5*(local+treated), "AbsoluteAnchor": absolute,
                               "CounterfactualDirect": direct, "CounterfactualGSI": corrected}
                    for method, value in outputs.items():
                        dense = F.interpolate(value.transpose(1, 2).reshape(1, bank.class_count, 32, 32),
                                               size=(512, 512), mode="bilinear", align_corners=False)[0]
                        if method == "Geometry":
                            dense = dense/.07
                        accumulators[key, method].add(dense[:, :ah, :aw].softmax(0).cpu().numpy(), blend[:ah, :aw], left, top)
                    diagnostics[key]["tiles"] += 1
                    for field, value in solver.items():
                        diagnostics[key][field] += value
        predictions = {key: {method: accumulators[key, method].finalize(None)[0] for method in LOCAL_METHODS} for key in banks}
        for key in banks:
            for source in ("Geo", "Proxy"):
                dense = F.interpolate(broad[key][source][None], size=(h, w), mode="bilinear", align_corners=False)[0]
                probability = dense.softmax(0).cpu().numpy()
                predictions[key]["Broad"+("Geometry" if source == "Geo" else "Proxy")] = probability.argmax(0).astype(np.uint8)
                if source == "Proxy":
                    fused = np.empty((h, w), np.uint8)
                    accumulator = accumulators[key, "Geometry"]
                    for top in range(0, h, 128):
                        end = min(top+128, h)
                        prior = accumulator.probabilities[:, top:end]/accumulator.normalizer[None, top:end].clip(1e-8)
                        fused[top:end] = (prior+probability[:, top:end]).argmax(0).astype(np.uint8)
                    predictions[key]["MeanProb50"] = fused
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
    selected = samples[args.shard_index::args.num_shards]
    if not selected or len(keys) != len(set(keys)):
        raise ValueError("Empty shard or duplicate samples.")
    upstream = Path(args.upstream_root)
    if subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() != PINNED_COMMIT:
        raise ValueError("Pinned source changed.")
    checkpoints = make_checkpoints(args)
    backbone = DINOTextSegmenter(checkpoints, device=args.device)
    config = TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2)
    geometry = TCPRSegmenter(backbone, config)
    banks = {key: geometry.encode_text(classes) for key, classes in specs.items()}
    if any(int((bank.parent_indices == c).sum()) != 20 for bank in banks.values() for c in range(bank.class_count)):
        raise ValueError("Requires exact 20-alias groups.")
    queries = {key: encode_queries(backbone, bank, upstream/"prompts/imagenet_template.py") for key, bank in banks.items()}
    signature = {"implementation": IMPLEMENTATION, "dataset": args.dataset, "methods": METHODS,
        "classes": {key: bank.class_names for key, bank in banks.items()}, "gear": {"geometry": asdict(config),
        "observation": asdict(SETTINGS), "counterfactual": "(I+A^T A)delta=A^T A(b_Proxy-b_Geo)",
        "runtime": "one fp32-weight backbone/head, shared exact broad tokens, bf16 AMP; shared fp32 text queries",
        "empty_support": "self Value fallback", "upstream_commit": PINNED_COMMIT}, "competitive": None,
        "vocabulary": {"sha256": hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest(),
        "aliases": {key: bank.alias_names for key, bank in banks.items()}, "counts": {key: [20]*bank.class_count for key, bank in banks.items()}},
        "checkpoints": checkpoint_manifest(checkpoints), "global_sample_count": len(keys), "global_sample_keys_sha256": digest(keys),
        "sample_keys": [sample.key for sample in selected], "sample_keys_sha256": digest([sample.key for sample in selected]),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "note": "Fixed matched readout contrast, no class/dataset routing or target-label parameter fitting. "
                "Proxy operator is borrowed, finite-guarded and uses Geometry precision; not official full VIP. "
                "All datasets are exploratory development. Labels load only after all predictions."}
    output.mkdir(parents=True)
    matrices = {key: {method: np.zeros((bank.class_count, bank.class_count), np.int64) for method in METHODS} for key, bank in banks.items()}
    transitions = {key: {method: np.zeros((bank.class_count,)*3, np.int64) for method in METHODS if method != "Geometry"} for key, bank in banks.items()}
    comparisons = {key: {base: np.zeros((bank.class_count,)*3, np.int64) for base in ("MeanProb50", "MeanLogit50", "AbsoluteAnchor")} for key, bank in banks.items()}
    ignored, diagnostics = dict.fromkeys(banks, 0), {key: {} for key in banks}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for number, sample in enumerate(selected, 1):
        image = load_image(sample.image_path if args.dataset == "loveda" else sample)
        predictions, current = predict_image(image, geometry, banks, queries, output)
        for key, bank in banks.items():
            target = load_mask(sample, key, tuple(image.shape[-2:]))
            valid = (target >= 0) & (target < bank.class_count)
            ignored[key] += int((~valid).sum())
            for method in METHODS:
                encoded = target[valid].astype(np.int64)*bank.class_count+predictions[key][method][valid]
                matrices[key][method] += np.bincount(encoded, minlength=bank.class_count**2).reshape(bank.class_count, -1)
                if method != "Geometry":
                    transitions[key][method] += transition_counts(predictions[key]["Geometry"], predictions[key][method], target, bank.class_count)
            for baseline in comparisons[key]:
                comparisons[key][baseline] += transition_counts(predictions[key][baseline], predictions[key]["CounterfactualGSI"], target, bank.class_count)
            for field, value in current[key].items():
                diagnostics[key][field] = diagnostics[key].get(field, 0)+value
        result = {"status": "complete" if number == len(selected) else "running", "processed_images": number,
            "total_images": len(selected), "signature": signature,
            "metrics": {key: {method: summary(matrix, banks[key].class_names, ignored[key]) for method, matrix in methods.items()} for key, methods in matrices.items()},
            "transitions": {key: {method: {"counts": counts.tolist(), **transition_summary(counts)} for method, counts in methods.items()} for key, methods in transitions.items()},
            "direct_comparisons": {key: {base: {"counts": counts.tolist(), **transition_summary(counts)} for base, counts in methods.items()} for key, methods in comparisons.items()},
            "diagnostics": {key: {field: value if field == "tiles" else value/max(values["tiles"], 1) for field, value in values.items()} for key, values in diagnostics.items()},
            "wall_seconds": time.perf_counter()-started, "peak_cuda_memory_mb": torch.cuda.max_memory_allocated()/1048576}
        (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({"dataset": args.dataset, "processed": number, "total": len(selected),
              "miou": {key: {method: row["mean_iou_percent"] for method, row in values.items()} for key, values in result["metrics"].items()}}), flush=True)


if __name__ == "__main__":
    main(parse_args())
