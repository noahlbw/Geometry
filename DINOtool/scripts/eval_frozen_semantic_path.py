#!/usr/bin/env python3
"""Fixed local Geometry semantic-path intervention, without VIP or routing."""
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.frozen_semantic_path import IMPLEMENTATION, METHODS, read_frozen_path
from dinotool.gear_ov import _crop_at
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from eval_geometry_semantic_innovation import parse_args
from eval_geometry_vip_reliability import summary
from eval_gear_ov import digest, protocol
from eval_stride_ov_loveda_e1 import make_checkpoints


@torch.inference_mode()
def frozen_scores(geometry, prepared, banks, texts, valid):
    with geometry.backbone._autocast():
        features, diagnostics = read_frozen_path(geometry.backbone.model.visual_model.head, prepared)
    error = float((features["Native"]-prepared.native_projected).abs().max())
    if error != 0.:
        raise ValueError(f"Native path replay differs: {error}")
    diagnostics["native_replay_max_error"] = error
    features["Geometry"] = prepared.geometry_projected
    scores = {key: {method: alias_class_scores(feature.float() @ texts[key].T, bank.parent_indices, bank.class_count)
                    for method, feature in features.items()} for key, bank in banks.items()}
    return scores, diagnostics


@torch.inference_mode()
def predict_image(image, geometry, banks, work, methods=METHODS, read_scores=frozen_scores,
                  *, reader_uses_rgb=False):
    h, w = image.shape[-2:]
    blend = hann_blend_window(512)
    texts = {key: F.normalize(bank.features.float(), dim=-1) for key, bank in banks.items()}
    totals = {key: {"tiles": 0} for key in banks}
    with ExitStack() as stack:
        accumulators = {(key, method): stack.enter_context(ProbabilityAccumulator(bank.class_count, h, w, 256, work))
                        for key, bank in banks.items() for method in methods}
        for top in tile_starts(h, 512, 128):
            for left in tile_starts(w, 512, 128):
                rgb = _crop_at(image, top, left, 512).to(geometry.device)
                prepared = geometry.prepare_image(rgb)
                ah, aw = min(512, h-top), min(512, w-left)
                yy = (torch.arange(32, device=geometry.device)+.5)*16
                xx = (torch.arange(32, device=geometry.device)+.5)*16
                valid = ((yy[:, None] < ah) & (xx[None] < aw)).reshape(1, -1)
                if reader_uses_rgb:
                    patch_scores, diagnostics = read_scores(geometry, prepared, banks, texts, valid, rgb)
                else:
                    patch_scores, diagnostics = read_scores(geometry, prepared, banks, texts, valid)
                for key, bank in banks.items():
                    for method in methods:
                        scores = patch_scores[key][method]
                        dense = F.interpolate(scores.transpose(1, 2).reshape(1, bank.class_count, 32, 32),
                                              size=(512, 512), mode="bilinear", align_corners=False)[0]/.07
                        accumulators[key, method].add(dense[:, :ah, :aw].softmax(0).cpu().numpy(), blend[:ah, :aw], left, top)
                    totals[key]["tiles"] += 1
                    for field, value in diagnostics.items():
                        totals[key][field] = totals[key].get(field, 0)+value
        predictions = {key: {method: accumulators[key, method].finalize(None)[0] for method in methods} for key in banks}
    return predictions, totals


def main(args, *, methods=METHODS, implementation=IMPLEMENTATION, primary="FrozenPathGeometry",
         reader_factory=None, readout_settings=None, reader_uses_rgb=False, save_per_image=False,
         signature_note=None):
    output = Path(args.output_dir)
    if output.exists() or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Existing output or invalid shard.")
    vocabulary = load_class_specs(args.vocabulary_config)
    samples, specs, load_image, load_mask = protocol(args, vocabulary)
    if args.source_diagnostic:
        fixed = json.loads(args.source_diagnostic.read_text())
        lookup = {sample.key: sample for sample in samples}
        samples = [lookup[key] for key in fixed["signature"]["samples"]]
    keys = [sample.key for sample in samples]
    selected = samples[args.shard_index::args.num_shards]
    if not selected or len(keys) != len(set(keys)):
        raise ValueError("Empty or duplicate sample sequence.")
    checkpoints = make_checkpoints(args)
    backbone = DINOTextSegmenter(checkpoints, device=args.device)
    config = TCPRConfig(maximum_aliases_per_class=20, geometry_depth=2)
    geometry = TCPRSegmenter(backbone, config)
    banks = {key: geometry.encode_text(classes) for key, classes in specs.items()}
    counts = {key: [int((bank.parent_indices == c).sum()) for c in range(bank.class_count)] for key, bank in banks.items()}
    if any(count != 20 for group in counts.values() for count in group):
        raise ValueError("Expected all 20 aliases per class.")
    reader = frozen_scores if reader_factory is None else reader_factory(geometry, banks)
    signature = {"implementation": implementation, "dataset": args.dataset, "methods": methods,
        "classes": {key: bank.class_names for key, bank in banks.items()},
        "gear": {"geometry": asdict(config), "primary": primary,
                 "transport": "native final state + sum_l(GeoAttention(native_l)-NativeAttention(native_l))",
                 "precision": "unchanged fp32 weights with bf16 AMP", "text": "unchanged six RS templates and normalized LME .07"},
        "competitive": None, "vocabulary": {"sha256": hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest(),
        "aliases": {key: bank.alias_names for key, bank in banks.items()}, "counts": counts},
        "checkpoints": checkpoint_manifest(checkpoints), "global_sample_count": len(keys), "global_sample_keys_sha256": digest(keys),
        "sample_keys": [sample.key for sample in selected], "sample_keys_sha256": digest([sample.key for sample in selected]),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "note": "Frozen local readout, no VIP operator, no alias selection, no fitted gate. Labels used after predictions. Exploratory development."}
    if readout_settings is not None:
        signature["gear"] = {"geometry": asdict(config), "primary": primary, **readout_settings}
    if signature_note is not None:
        signature["note"] = signature_note
    output.mkdir(parents=True)
    matrices = {key: {method: np.zeros((bank.class_count,)*2, np.int64) for method in methods} for key, bank in banks.items()}
    transitions = {key: {method: np.zeros((bank.class_count,)*3, np.int64) for method in methods if method != "Geometry"} for key, bank in banks.items()}
    ignored, diagnostics = dict.fromkeys(banks, 0), {key: {} for key in banks}
    image_matrices = {key: {method: [] for method in methods} for key in banks} if save_per_image else None
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for number, sample in enumerate(selected, 1):
        image = load_image(sample.image_path if args.dataset == "loveda" else sample)
        predictions, current = predict_image(image, geometry, banks, output, methods, reader,
                                             reader_uses_rgb=reader_uses_rgb)
        for key, bank in banks.items():
            target = load_mask(sample, key, tuple(image.shape[-2:]))
            valid = (target >= 0) & (target < bank.class_count)
            ignored[key] += int((~valid).sum())
            for method in methods:
                encoded = target[valid].astype(np.int64)*bank.class_count+predictions[key][method][valid]
                current_matrix = np.bincount(encoded, minlength=bank.class_count**2).reshape(bank.class_count, -1)
                matrices[key][method] += current_matrix
                if image_matrices is not None:
                    image_matrices[key][method].append(current_matrix)
                if method != "Geometry":
                    transitions[key][method] += transition_counts(predictions[key]["Geometry"], predictions[key][method], target, bank.class_count)
            for field, value in current[key].items():
                diagnostics[key][field] = diagnostics[key].get(field, 0)+value
        result = {"status": "complete" if number == len(selected) else "running", "processed_images": number,
            "total_images": len(selected), "signature": signature,
            "metrics": {key: {method: summary(cm, banks[key].class_names, ignored[key]) for method, cm in methods.items()} for key, methods in matrices.items()},
            "transitions": {key: {method: {"counts": tensor.tolist(), **transition_summary(tensor)} for method, tensor in methods.items()} for key, methods in transitions.items()},
            "diagnostics": {key: {field: value if field == "tiles" else value/max(values["tiles"], 1) for field, value in values.items()} for key, values in diagnostics.items()},
            "wall_seconds": time.perf_counter()-started, "peak_cuda_memory_mb": torch.cuda.max_memory_allocated()/1048576}
        (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({"dataset": args.dataset, "processed": number, "total": len(selected),
              "miou": {key: {method: row["mean_iou_percent"] for method, row in group.items()} for key, group in result["metrics"].items()}}), flush=True)
    if image_matrices is not None:
        np.savez_compressed(output/"per_image_confusions.npz", sample_keys=np.asarray(signature["sample_keys"]),
                            **{key+"__"+method: np.stack(values)
                               for key, group in image_matrices.items() for method, values in group.items()})


if __name__ == "__main__":
    main(parse_args())
