#!/usr/bin/env python3
"""Evaluate contextual phrase readout on fixed or full LoveDA P/D protocols."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from dinotool.contextual_phrase_readout import (
    ContextualPhraseConfig,
    ContextualPhraseSegmenter,
)
from dinotool.inference import ProbabilityAccumulator, hann_blend_window, tile_starts
from dinotool.loveda import discover_loveda_samples
from dinotool.model import DINOTextSegmenter, checkpoint_manifest
from dinotool.prompts import load_class_specs, serialize_class_specs
from dinotool.tcpr import TCPRConfig

from eval_competitive_evidence_loveda_e1 import (
    _change_summary,
    _empty_change,
    _load_prediction,
    _update_change,
    protocol_classes,
)
from eval_stride_ov_loveda_e1 import (
    Confusion,
    D_CLASSES,
    P_CLASSES,
    crop_tile,
    load_rgb,
    make_checkpoints,
    patch_valid_mask,
    save_prediction,
    target_ids,
)


METHODS = (
    "G_all20_uniform",
    "PhraseResidual",
    "OverviewFusion",
    "ContextualPhrase",
    "ClassScaleUnion",
    "AliasScaleUnion",
    "StructuredAliasUnion",
)
DEFAULT_VOCABULARY = Path(__file__).resolve().parents[1] / "configs" / "tcpr_loveda_vip_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov3-repo", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocabulary-config", default=str(DEFAULT_VOCABULARY))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--overview-size", type=int, default=512)
    parser.add_argument("--output-temperature", type=float, default=0.07)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


class Diagnostics:
    FIELDS = (
        "mean_direct_residual",
        "mean_structured_residual",
        "mean_geometry_overview_disagreement",
        "changed_phrase_from_geometry",
        "changed_overview_fusion_from_geometry",
        "changed_contextual_from_geometry",
        "changed_class_union_from_geometry",
        "changed_alias_union_from_geometry",
        "changed_structured_alias_union_from_geometry",
    )

    def __init__(self) -> None:
        self.tiles = 0
        self.values = {field: 0.0 for field in self.FIELDS}

    def add(self, value) -> None:
        self.tiles += 1
        summary = value.summary()
        for field in self.FIELDS:
            self.values[field] += summary[field]

    def summary(self) -> dict[str, float | int]:
        count = max(self.tiles, 1)
        return {"tiles": self.tiles, **{field: value / count for field, value in self.values.items()}}


@torch.inference_mode()
def predict_image(
    model: ContextualPhraseSegmenter,
    image_path: Path,
    text_banks: dict[str, object],
    args: argparse.Namespace,
    work_dir: Path,
    diagnostics: dict[str, Diagnostics],
) -> dict[str, dict[str, np.ndarray]]:
    image = load_rgb(image_path)
    height, width = image.shape[-2:]
    overview = model.prepare_overview(image.unsqueeze(0).to(model.device, non_blocking=True))
    overview_aliases = {
        protocol: model.overview_alias_map(overview, bank)
        for protocol, bank in text_banks.items()
    }
    starts_x = tile_starts(width, args.tile_size, args.overlap)
    starts_y = tile_starts(height, args.tile_size, args.overlap)
    blend = hann_blend_window(args.tile_size)
    with ExitStack() as stack:
        accumulators = {
            (protocol, method): stack.enter_context(
                ProbabilityAccumulator(
                    len(P_CLASSES if protocol == "P" else D_CLASSES),
                    height,
                    width,
                    2048,
                    work_dir,
                )
            )
            for protocol in text_banks
            for method in METHODS
        }
        for top in starts_y:
            for left in starts_x:
                actual_h = min(args.tile_size, height - top)
                actual_w = min(args.tile_size, width - left)
                rgb = crop_tile(image, left, top, args.tile_size).to(model.device, non_blocking=True)
                valid = patch_valid_mask(
                    actual_h, actual_w, args.tile_size, model.patch_size, model.device
                )
                prepared = model.prepare_image(rgb)
                blend_weights = blend[:actual_h, :actual_w]
                for protocol, bank in text_banks.items():
                    result = model.read_prepared(
                        prepared,
                        overview_aliases[protocol],
                        bank,
                        image_height=height,
                        image_width=width,
                        tile_top=top,
                        tile_left=left,
                        valid_mask=valid,
                    )
                    diagnostics[protocol].add(result.diagnostics)
                    logits_by_method = {
                        "G_all20_uniform": result.geometry_logits,
                        "PhraseResidual": result.phrase_residual_logits,
                        "OverviewFusion": result.overview_fusion_logits,
                        "ContextualPhrase": result.contextual_phrase_logits,
                        "ClassScaleUnion": result.class_scale_union_logits,
                        "AliasScaleUnion": result.alias_scale_union_logits,
                        "StructuredAliasUnion": result.structured_alias_union_logits,
                    }
                    for method, logits in logits_by_method.items():
                        dense = F.interpolate(
                            logits,
                            size=(args.tile_size, args.tile_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                        probability = torch.softmax(
                            dense.float() / args.output_temperature, dim=1
                        )
                        accumulators[(protocol, method)].add(
                            probability[0, :, :actual_h, :actual_w].cpu().numpy(),
                            blend_weights,
                            left,
                            top,
                        )
        return {
            protocol: {
                method: accumulators[(protocol, method)].finalize(None)[0]
                for method in METHODS
            }
            for protocol in text_banks
        }


def main(args: argparse.Namespace) -> dict[str, object]:
    if args.tile_size % 16 or args.overview_size % 16 or not 0 <= args.overlap < args.tile_size:
        raise ValueError("Tile and overview sizes must be patch aligned; overlap must be smaller.")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= shard-index < num-shards.")
    samples = discover_loveda_samples(args.data_root)
    random.Random(args.sample_seed).shuffle(samples)
    if args.max_images > 0:
        samples = samples[:args.max_images]
    global_keys = [sample.key for sample in samples]
    global_hash = hashlib.sha256("\n".join(global_keys).encode()).hexdigest()
    samples = samples[args.shard_index::args.num_shards]
    sample_hash = hashlib.sha256("\n".join(sample.key for sample in samples).encode()).hexdigest()

    vocabulary_path = Path(args.vocabulary_config).resolve()
    vocabulary = load_class_specs(vocabulary_path)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work_dir = output / ".accumulators"
    work_dir.mkdir(exist_ok=True)
    checkpoints = make_checkpoints(args)
    backbone = DINOTextSegmenter(checkpoints, device=args.device, amp=not args.no_amp)
    config = ContextualPhraseConfig(overview_size=args.overview_size)
    tcpr_config = TCPRConfig(maximum_aliases_per_class=20)
    model = ContextualPhraseSegmenter(backbone, config, tcpr_config)
    class_specs = {
        protocol: protocol_classes(protocol, vocabulary) for protocol in ("P", "D")
    }
    text_banks = {
        protocol: model.encode_text(specs) for protocol, specs in class_specs.items()
    }
    alias_counts = {
        protocol: [int((bank.parent_indices == cls).sum()) for cls in range(bank.class_count)]
        for protocol, bank in text_banks.items()
    }
    if any(count != 20 for counts in alias_counts.values() for count in counts):
        raise ValueError(f"Expected exactly 20 aliases per class: {alias_counts}")

    matrices = {
        (protocol, method): Confusion(P_CLASSES if protocol == "P" else D_CLASSES)
        for protocol in ("P", "D")
        for method in METHODS
    }
    changes = {
        (protocol, method): _empty_change()
        for protocol in ("P", "D")
        for method in METHODS
        if method != "G_all20_uniform"
    }
    diagnostics = {protocol: Diagnostics() for protocol in ("P", "D")}
    signature = {
        "method": "Contextual Phrase Readout",
        "date": "2026-09-27",
        "sample_count": len(samples),
        "sample_keys_sha256": sample_hash,
        "sample_keys": [sample.key for sample in samples],
        "global_sample_count": len(global_keys),
        "global_sample_keys_sha256": global_hash,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "methods": list(METHODS),
        "implementation": "contextual-phrase-overview-v2-scale-union",
        "config": {key: value for key, value in vars(args).items() if key != "device"},
        "contextual_phrase": asdict(config),
        "tcpr": asdict(tcpr_config),
        "vocabulary": {
            "path": str(vocabulary_path),
            "sha256": hashlib.sha256(vocabulary_path.read_bytes()).hexdigest(),
            "classes": {
                protocol: serialize_class_specs(specs) for protocol, specs in class_specs.items()
            },
            "alias_counts_per_class": alias_counts,
        },
        "checkpoints": checkpoint_manifest(checkpoints),
        "protocol_note": (
            "Frozen DINO.text and Geometry with 20 aliases per class. One 512-square whole-image "
            "overview is shared by all local tiles and arms. PhraseResidual keeps alias responses "
            "until local-conditioned context aggregation. OverviewFusion is the equal-observation "
            "ordinary multiscale control. ContextualPhrase additionally reads overview aliases "
            "through the local Geometry relation and adds only the conditioned-minus-uniform "
            "context residual to Geometry. ClassScaleUnion performs a normalized evidence union "
            "after class aggregation. AliasScaleUnion and StructuredAliasUnion perform that union "
            "per phrase before class aggregation, with the latter using the Geometry relation. "
            "Target masks are used only for evaluation."
        ),
    }
    signature_path = output / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Output directory is bound to a different experiment signature.")
    signature_path.write_text(json.dumps(signature, indent=2, ensure_ascii=True))

    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()
    result: dict[str, object] = {}
    for index, sample in enumerate(samples, start=1):
        with Image.open(sample.image_path) as image:
            shape = (image.height, image.width)
        predictions: dict[str, dict[str, np.ndarray]] = {"P": {}, "D": {}}
        complete = True
        for protocol in ("P", "D"):
            class_count = len(P_CLASSES if protocol == "P" else D_CLASSES)
            for method in METHODS:
                cached = _load_prediction(
                    output / "predictions" / protocol / method / sample.key,
                    shape,
                    class_count,
                )
                complete = complete and cached is not None
                if cached is not None:
                    predictions[protocol][method] = cached
        if not complete:
            predictions = predict_image(
                model, sample.image_path, text_banks, args, work_dir, diagnostics
            )
            for protocol in ("P", "D"):
                for method, prediction in predictions[protocol].items():
                    save_prediction(
                        output / "predictions" / protocol / method / sample.key, prediction
                    )

        for protocol in ("P", "D"):
            target = target_ids(sample.mask_path, protocol)
            base = predictions[protocol]["G_all20_uniform"]
            for method in METHODS:
                matrices[(protocol, method)].update(predictions[protocol][method], target)
                if method != "G_all20_uniform":
                    _update_change(changes[(protocol, method)], base, predictions[protocol][method], target)

        if index % max(args.progress_every, 1) == 0 or index == len(samples):
            metrics = {
                protocol: {
                    method: matrices[(protocol, method)].summary() for method in METHODS
                }
                for protocol in ("P", "D")
            }
            result = {
                "status": "complete" if index == len(samples) else "running",
                "processed_images": index,
                "total_images": len(samples),
                "metrics": metrics,
                "delta_miou_vs_geometry": {
                    protocol: {
                        method: round(
                            metrics[protocol][method]["mean_iou_percent"]
                            - metrics[protocol]["G_all20_uniform"]["mean_iou_percent"],
                            4,
                        )
                        for method in METHODS
                    }
                    for protocol in ("P", "D")
                },
                "changes_vs_geometry": {
                    protocol: {
                        method: _change_summary(changes[(protocol, method)])
                        for method in METHODS
                        if method != "G_all20_uniform"
                    }
                    for protocol in ("P", "D")
                },
                "diagnostics": {
                    protocol: diagnostics[protocol].summary() for protocol in ("P", "D")
                },
                "wall_seconds": round(time.perf_counter() - started, 3),
                "peak_cuda_memory_mb": (
                    round(torch.cuda.max_memory_allocated(model.device) / 1048576, 2)
                    if model.device.type == "cuda" else None
                ),
                "signature": signature,
            }
            (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=True))
            print(json.dumps({
                "processed": index,
                "P": {method: metrics["P"][method]["mean_iou_percent"] for method in METHODS},
                "D": {method: metrics["D"][method]["mean_iou_percent"] for method in METHODS},
            }), flush=True)
    return result


if __name__ == "__main__":
    main(parse_args())
