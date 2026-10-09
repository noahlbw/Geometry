#!/usr/bin/env python3
"""Run a locked CAFe-DINO/DINOv3 semantic-spatial depth diagnosis.

The script intentionally implements diagnostics rather than a new segmentation
model. Alignment is fitted only on OpenEarthMap. Target labels are used only by
the evaluator and by explicitly named oracle summaries.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import Tensor

# Evaluation is launched once per GPU. Avoid a large OpenCV thread pool in each
# process starving the GPU during metric bookkeeping.
cv2.setNumThreads(1)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cafedino_locked_loveda import (  # noqa: E402
    LOVE_DA_CLASS_NAMES,
    build_model,
    discover_samples,
    image_tensor,
    target_ids,
)
from dinotool.depth_diagnosis import (  # noqa: E402
    AlignmentStatistics,
    apply_alignment,
    binary_spatial_metrics,
    combine_edge_affinities,
    cosine_logits,
    grid_edge_affinity,
    gt_seed_labels,
    oracle_pair_matrix,
    seeded_grid_diffusion,
    seeded_grid_diffusion_batched,
    semantic_seed_labels,
)
from dinotool.oem import OEM_CLASSES, OEM_RAW_TO_CLASS_INDEX, discover_oem_samples  # noqa: E402


SEMANTIC_METHODS = ("raw", "procrustes", "ridge")
AFFINITY_KINDS = ("feature", "query", "key", "value", "qkv")


class Confusion:
    def __init__(self, class_names: Sequence[str]) -> None:
        self.class_names = tuple(class_names)
        self.matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        self.matrix += _confusion_matrix(prediction, target, len(self.class_names))

    def summary(self) -> dict[str, Any]:
        intersection = np.diag(self.matrix)
        target_count = self.matrix.sum(axis=1)
        predicted_count = self.matrix.sum(axis=0)
        union = target_count + predicted_count - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.full(intersection.shape, np.nan, dtype=np.float64),
            where=union > 0,
        )
        return {
            "mean_iou_percent": round(float(np.nanmean(iou)) * 100.0, 4),
            "per_class_iou_percent": {
                name: None if np.isnan(iou[index]) else round(float(iou[index]) * 100.0, 4)
                for index, name in enumerate(self.class_names)
            },
            "confusion_matrix": self.matrix.tolist(),
        }


def _confusion_matrix(
    prediction: np.ndarray,
    target: np.ndarray,
    class_count: int,
) -> np.ndarray:
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ.")
    valid = (target >= 0) & (target < class_count)
    encoded = target[valid].astype(np.int64) * class_count + prediction[valid].astype(np.int64)
    return np.bincount(encoded, minlength=class_count * class_count).reshape(
        (class_count, class_count)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit-alignments", help="Fit category-free layer-to-final transforms on OEM.")
    _add_model_arguments(fit)
    fit.add_argument("--source-data-root", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--layers", nargs="+", type=int, default=(4, 8, 12, 16, 20, 24))
    fit.add_argument("--crop-size", type=int, default=224)
    fit.add_argument("--max-train-images", type=int, default=None)
    fit.add_argument("--max-val-images", type=int, default=None)
    fit.add_argument("--max-patches-per-image", type=int, default=196)
    fit.add_argument("--ridge-lambdas", nargs="+", type=float, default=(1e-4, 1e-3, 1e-2, 1e-1, 1.0))
    fit.add_argument("--allow-missing-source-images", action="store_true")

    evaluate = subparsers.add_parser("evaluate-loveda", help="Run target LoveDA diagnostics.")
    _add_evaluation_arguments(evaluate)
    evaluate_oem = subparsers.add_parser("evaluate-oem", help="Run source OEM validation diagnostics.")
    _add_evaluation_arguments(evaluate_oem)
    evaluate_oem.add_argument(
        "--allow-missing-source-images",
        action="store_true",
        help="Explicitly allow the official OpenEarthMap_wo_xBD archive to omit xBD RGB images.",
    )
    return parser


def _add_evaluation_arguments(evaluate: argparse.ArgumentParser) -> None:
    _add_model_arguments(evaluate)
    evaluate.add_argument("--data-root", required=True)
    evaluate.add_argument("--alignment-checkpoint", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--layers", nargs="+", type=int, default=(4, 8, 12, 16, 20, 24))
    evaluate.add_argument("--evaluation-size", type=int, default=512)
    evaluate.add_argument("--window-size", type=int, default=224)
    evaluate.add_argument("--stride", type=int, default=112)
    evaluate.add_argument("--window-batch-size", type=int, default=16)
    evaluate.add_argument("--max-images", type=int, default=None)
    evaluate.add_argument("--start-index", type=int, default=0)
    evaluate.add_argument("--progress-every", type=int, default=5)
    evaluate.add_argument("--affinity-temperature", type=float, default=0.15)
    evaluate.add_argument("--diffusion-iterations", type=int, default=20)
    evaluate.add_argument("--diffusion-restart", type=float, default=0.15)
    evaluate.add_argument(
        "--semantic-seed-threshold",
        type=float,
        default=0.35,
        help="Source-calibrated top-1 softmax threshold for semantic seeds.",
    )
    evaluate.add_argument(
        "--semantic-seed-logit-scale",
        type=float,
        default=None,
        help="Scale used only for seed confidence; defaults to the checkpoint text logit scale.",
    )
    evaluate.add_argument("--semantic-seed-minimum", type=int, default=0)
    evaluate.add_argument("--gt-seed-fraction", type=float, default=0.10)
    evaluate.add_argument("--pair-semantic-method", choices=SEMANTIC_METHODS, default="ridge")
    evaluate.add_argument("--pair-affinity", choices=AFFINITY_KINDS, default="qkv")
    evaluate.add_argument(
        "--locked-selection",
        help="Optional source-validation results.json whose fixed same/dual layer choices are applied here.",
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bpe-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-mode", choices=("eval", "official-script"), default="eval")


def _model_args(args: argparse.Namespace, *, window_size: int) -> argparse.Namespace:
    return argparse.Namespace(
        official_root=args.official_root,
        checkpoint=args.checkpoint,
        bpe_path=args.bpe_path,
        device=args.device,
        model_mode=args.model_mode,
        window_size=window_size,
    )


def validate_layers(model: torch.nn.Module, layers: Sequence[int]) -> tuple[int, ...]:
    block_count = len(model.backbone.visual_model.backbone.blocks)
    selected = tuple(int(layer) for layer in layers)
    if len(set(selected)) != len(selected) or any(layer < 1 or layer > block_count for layer in selected):
        raise ValueError(f"Layers must be unique one-based values in [1,{block_count}].")
    if block_count not in selected:
        raise ValueError(f"The final layer L{block_count} must be included as the alignment target.")
    return selected


@torch.inference_mode()
def extract_layers(
    model: torch.nn.Module,
    image: Tensor,
    layers: Sequence[int],
    *,
    capture_qkv: bool,
) -> tuple[dict[int, Tensor], dict[int, dict[str, Tensor]]]:
    feature_batches, qkv_batches = extract_layers_batched(
        model,
        image,
        layers,
        capture_qkv=capture_qkv,
    )
    feature_layers = {layer: values[0] for layer, values in feature_batches.items()}
    structured = {
        layer: {kind: values[0] for kind, values in layer_values.items()}
        for layer, layer_values in qkv_batches.items()
    }
    return feature_layers, structured


@torch.inference_mode()
def extract_layers_batched(
    model: torch.nn.Module,
    image: Tensor,
    layers: Sequence[int],
    *,
    capture_qkv: bool,
) -> tuple[dict[int, Tensor], dict[int, dict[str, Tensor]]]:
    backbone = model.backbone.visual_model.backbone
    layer_indices = tuple(layer - 1 for layer in layers)
    qkv_outputs: dict[int, Tensor] = {}
    handles = []
    if capture_qkv:
        for one_based, index in zip(layers, layer_indices):
            def hook(_module, _inputs, output, *, layer=one_based):
                qkv_outputs[layer] = output.detach()

            handles.append(backbone.blocks[index].attn.qkv.register_forward_hook(hook))
    try:
        outputs = backbone.get_intermediate_layers(image, n=layer_indices, norm=True)
    finally:
        for handle in handles:
            handle.remove()
    feature_layers = {layer: output.float() for layer, output in zip(layers, outputs)}
    if not capture_qkv:
        return feature_layers, {}
    height = image.shape[-2] // backbone.patch_size
    width = image.shape[-1] // backbone.patch_size
    rope = backbone.rope_embed(H=height, W=width) if backbone.rope_embed is not None else None
    prefix = 1 + backbone.n_storage_tokens
    structured: dict[int, dict[str, Tensor]] = {}
    for one_based, index in zip(layers, layer_indices):
        raw = qkv_outputs.get(one_based)
        if raw is None:
            raise RuntimeError(f"DINO QKV hook did not fire for layer {one_based}.")
        attention = backbone.blocks[index].attn
        batch, tokens, _ = raw.shape
        qkv = raw.reshape(batch, tokens, 3, attention.num_heads, -1)
        query, key, value = [item.transpose(1, 2) for item in torch.unbind(qkv, dim=2)]
        if rope is not None:
            query, key = attention.apply_rope(query, key, rope)
        structured[one_based] = {
            "query": query[:, :, prefix:].permute(0, 2, 1, 3).float(),
            "key": key[:, :, prefix:].permute(0, 2, 1, 3).float(),
            "value": value[:, :, prefix:].permute(0, 2, 1, 3).float(),
        }
    return feature_layers, structured


@torch.inference_mode()
def extract_alignment_features(
    model: torch.nn.Module,
    image: Tensor,
    layers: Sequence[int],
) -> tuple[dict[int, Tensor], Tensor]:
    """Return raw layer patches and the fixed DINO.text patch-space target.

    The DINO.text visual tower applies its trained two-block vision head after
    the final DINO backbone layer. Comparing an intermediate raw backbone
    tensor directly with text therefore is not an aligned readout. Alignment
    fitting maps every candidate layer, including the final backbone layer, to
    the same fixed vision-head patch embedding space without using categories
    or segmentation labels.
    """
    visual_model = model.backbone.visual_model
    backbone = visual_model.backbone
    layer_indices = tuple(layer - 1 for layer in layers)
    outputs = backbone.get_intermediate_layers(
        image,
        n=layer_indices,
        norm=True,
        return_class_token=True,
        return_extra_tokens=True,
    )
    feature_layers = {
        layer: output[0][0].float()
        for layer, output in zip(layers, outputs)
    }
    final_position = tuple(layers).index(len(backbone.blocks))
    final_patches, final_class, final_extra = outputs[final_position]
    final_tokens = torch.cat(
        (final_class.unsqueeze(1), final_extra, final_patches),
        dim=1,
    )
    text_aligned_tokens = visual_model.head(final_tokens)
    prefix = 1 + backbone.n_storage_tokens
    text_aligned_patches = text_aligned_tokens[0, prefix:].float()
    if text_aligned_patches.shape != feature_layers[layers[final_position]].shape:
        raise RuntimeError("DINO.text vision-head patches do not match the backbone patch grid.")
    return feature_layers, text_aligned_patches


def normalized_image(path: Path, size: int, device: torch.device) -> Tensor:
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"))
    height, width = rgb.shape[:2]
    scale = size / min(height, width)
    resized = cv2.resize(rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_LINEAR)
    top = max((resized.shape[0] - size) // 2, 0)
    left = max((resized.shape[1] - size) // 2, 0)
    crop = np.ascontiguousarray(resized[top : top + size, left : left + size])
    tensor = torch.from_numpy(crop).permute(2, 0, 1).float().div_(255.0)
    mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = tensor.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0).to(device)


def fit_alignments(args: argparse.Namespace) -> dict[str, Any]:
    model, _, checkpoint_manifest = build_model(_model_args(args, window_size=args.crop_size))
    layers = validate_layers(model, args.layers)
    device = next(model.parameters()).device
    train_samples = discover_oem_samples(
        args.source_data_root,
        "train",
        max_images=args.max_train_images,
        allow_missing_images=args.allow_missing_source_images,
    )
    val_samples = discover_oem_samples(
        args.source_data_root,
        "val",
        max_images=args.max_val_images,
        allow_missing_images=args.allow_missing_source_images,
    )
    statistics = {layer: AlignmentStatistics(model.backbone.visual_model.backbone.embed_dim) for layer in layers}
    for index, sample in enumerate(train_samples, start=1):
        image = normalized_image(sample.image_path, args.crop_size, device)
        features, target = extract_alignment_features(model, image, layers)
        for layer in layers:
            source = features[layer]
            if source.shape[0] > args.max_patches_per_image:
                chosen = torch.linspace(0, source.shape[0] - 1, args.max_patches_per_image, device=source.device).long()
                statistics[layer].update(source[chosen], target[chosen])
            else:
                statistics[layer].update(source, target)
        if index % 50 == 0 or index == len(train_samples):
            print(json.dumps({"stage": "fit_statistics", "processed": index, "total": len(train_samples)}), flush=True)
    procrustes: dict[int, Tensor] = {}
    ridge_candidates: dict[int, dict[float, Tensor]] = {}
    reports: dict[str, Any] = {}
    for layer in layers:
        procrustes[layer], procrustes_report = statistics[layer].orthogonal_procrustes(
            solve_device=device,
        )
        ridge_solutions = statistics[layer].ridge_many(
            args.ridge_lambdas,
            solve_device=device,
        )
        ridge_candidates[layer] = {
            value: matrix
            for value, (matrix, _) in ridge_solutions.items()
        }
        ridge_reports = {
            str(value): asdict(report)
            for value, (_, report) in ridge_solutions.items()
        }
        reports[str(layer)] = {
            "raw": {"stored_parameters": 0, "fitted_patch_pairs": 0},
            "procrustes": asdict(procrustes_report),
            "ridge_candidates": ridge_reports,
        }
    validation_error = {
        layer: {value: 0.0 for value in args.ridge_lambdas}
        for layer in layers
    }
    validation_patches = 0
    for index, sample in enumerate(val_samples, start=1):
        image = normalized_image(sample.image_path, args.crop_size, device)
        features, target = extract_alignment_features(model, image, layers)
        target = F.normalize(target, dim=-1)
        validation_patches += int(target.shape[0])
        for layer in layers:
            for value, matrix in ridge_candidates[layer].items():
                aligned = apply_alignment(features[layer], matrix)
                validation_error[layer][value] += float(F.mse_loss(aligned, target, reduction="sum").item())
        if index % 50 == 0 or index == len(val_samples):
            print(json.dumps({"stage": "select_ridge", "processed": index, "total": len(val_samples)}), flush=True)
    selected_ridge: dict[int, Tensor] = {}
    selected_lambdas: dict[str, float] = {}
    for layer in layers:
        best = min(args.ridge_lambdas, key=lambda value: validation_error[layer][value])
        selected_ridge[layer] = ridge_candidates[layer][float(best)]
        selected_lambdas[str(layer)] = float(best)
    payload = {
        "layers": list(layers),
        "final_layer": len(model.backbone.visual_model.backbone.blocks),
        "feature_dim": next(iter(statistics.values())).feature_dim,
        "procrustes": procrustes,
        "ridge": selected_ridge,
        "manifest": {
            "method": "category-free layer-to-DINO.text patch-space alignment",
            "alignment_target": "fixed trained DINO.text vision-head patch embeddings from the final backbone layer",
            "source_dataset": "OpenEarthMap",
            "train_images": len(train_samples),
            "validation_images": len(val_samples),
            "validation_patch_pairs": validation_patches,
            "crop_size": args.crop_size,
            "selected_ridge_lambdas": selected_lambdas,
            "reports": reports,
            "checkpoint": checkpoint_manifest,
            "allow_missing_source_images": bool(args.allow_missing_source_images),
            "target_labels_used": False,
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    _write_json(output.with_suffix(".json"), payload["manifest"])
    return payload["manifest"]


def _window_starts(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def _edge_bank(
    features: Tensor,
    qkv: dict[str, Tensor],
    height: int,
    width: int,
    temperature: float,
) -> dict[str, tuple[Tensor, Tensor]]:
    bank = {
        "feature": grid_edge_affinity(features, height, width, temperature=temperature),
        "query": grid_edge_affinity(qkv["query"], height, width, temperature=temperature),
        "key": grid_edge_affinity(qkv["key"], height, width, temperature=temperature),
        "value": grid_edge_affinity(qkv["value"], height, width, temperature=temperature),
    }
    bank["qkv"] = combine_edge_affinities((bank["query"], bank["key"], bank["value"]))
    return bank


def _target_patch_ids(target: np.ndarray, height: int, width: int, class_count: int) -> Tensor:
    tensor = torch.from_numpy(target.astype(np.int64))
    valid = (tensor >= 0) & (tensor < class_count)
    one_hot = F.one_hot(tensor.clamp(0, class_count - 1), class_count).permute(2, 0, 1).float()
    one_hot *= valid.unsqueeze(0)
    pooled = F.interpolate(one_hot.unsqueeze(0), size=(height, width), mode="area")[0]
    support, labels = pooled.max(dim=0)
    return labels.masked_fill(support <= 0, -1)


def _update_region_semantic(
    matrix: np.ndarray,
    logits: Tensor,
    target_patch: Tensor,
) -> None:
    """Measure class recognition on GT interiors, independent of boundaries."""
    if logits.ndim != 3 or target_patch.shape != logits.shape[1:]:
        raise ValueError("Region semantic tensors have incompatible shapes.")
    class_count = logits.shape[0]
    target_patch = target_patch.to(logits.device)
    kernel = torch.ones((1, 1, 3, 3), device=logits.device)
    for class_index in range(class_count):
        mask = target_patch == class_index
        if not bool(mask.any()):
            continue
        neighbours = F.conv2d(mask.float()[None, None], kernel, padding=1)[0, 0]
        interior = mask & (neighbours >= 9)
        if not bool(interior.any()):
            # Tiny regions may have no 3x3 interior at patch resolution. They
            # remain valid recognition samples, but are explicitly flagged by
            # the interior-only definition through this fallback count.
            interior = mask
        pooled = logits[:, interior].float().mean(dim=1)
        predicted = int(pooled.argmax().item())
        matrix[class_index, predicted] += 1


def _add_window(accumulator: Tensor, values: Tensor, top: int, left: int, size: int) -> None:
    upsampled = F.interpolate(values.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)[0]
    accumulator[:, top : top + size, left : left + size] += upsampled.to(
        device=accumulator.device, dtype=accumulator.dtype
    )


def _oem_target_ids(mask_path: Path, evaluation_size: int) -> np.ndarray:
    with Image.open(mask_path) as source:
        raw = np.asarray(source).copy()
    if raw.ndim == 3:
        raw = raw[..., 0]
    raw = cv2.resize(raw, (evaluation_size, evaluation_size), interpolation=cv2.INTER_NEAREST)
    target = np.full(raw.shape, -1, dtype=np.int64)
    for raw_id, class_index in OEM_RAW_TO_CLASS_INDEX.items():
        target[raw == raw_id] = class_index
    unexpected = sorted(set(np.unique(raw).tolist()) - set(OEM_RAW_TO_CLASS_INDEX) - {0, 255})
    if unexpected:
        raise ValueError(f"Unexpected OpenEarthMap label IDs at {mask_path}: {unexpected}")
    return target


def evaluate_dataset(args: argparse.Namespace, dataset_name: str) -> dict[str, Any]:
    model, _, checkpoint_manifest = build_model(_model_args(args, window_size=args.window_size))
    layers = validate_layers(model, args.layers)
    alignments = torch.load(args.alignment_checkpoint, map_location="cpu", weights_only=False)
    if tuple(alignments["layers"]) != layers:
        raise ValueError("Alignment checkpoint layers do not match evaluation layers.")
    device = next(model.parameters()).device
    if args.start_index < 0:
        raise ValueError("start-index must be non-negative.")
    if dataset_name == "LoveDA":
        class_names = LOVE_DA_CLASS_NAMES
        all_samples = discover_samples(Path(args.data_root), None)

        def load_target(path: Path) -> np.ndarray:
            values = target_ids(path, args.evaluation_size, include_background=True).astype(np.int64)
            values[values == 255] = -1
            return values
    elif dataset_name == "OpenEarthMap":
        class_names = tuple(spec.name for spec in OEM_CLASSES)
        all_samples = discover_oem_samples(
            args.data_root,
            "val",
            allow_missing_images=args.allow_missing_source_images,
        )

        def load_target(path: Path) -> np.ndarray:
            return _oem_target_ids(path, args.evaluation_size)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    stop = None if args.max_images is None else args.start_index + args.max_images
    samples = all_samples[args.start_index : stop]
    if not samples:
        raise ValueError("The requested evaluation shard is empty.")
    text_features = model.build_text_embeddings([list(class_names)]).to(device)
    if args.semantic_seed_logit_scale is None:
        try:
            semantic_seed_logit_scale = float(model.backbone.logit_scale.exp().item())
        except (AttributeError, RuntimeError):
            semantic_seed_logit_scale = 1.0
    else:
        semantic_seed_logit_scale = float(args.semantic_seed_logit_scale)
    if semantic_seed_logit_scale <= 0:
        raise ValueError("semantic-seed-logit-scale must be positive.")
    semantic_confusions = {
        (method, layer): Confusion(class_names) for method in SEMANTIC_METHODS for layer in layers
    }
    spatial_confusions = {
        (kind, layer): Confusion(class_names) for kind in AFFINITY_KINDS for layer in layers
    }
    pair_confusions = {(semantic, spatial): Confusion(class_names) for semantic in layers for spatial in layers}
    fusion_confusion = Confusion(class_names)
    region_semantic_confusions = {
        (method, layer): np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        for method in SEMANTIC_METHODS
        for layer in layers
    }
    adaptive_confusions = {
        "unrestricted": Confusion(class_names),
        "dual_depth": Confusion(class_names),
        "same_depth": Confusion(class_names),
    }
    spatial_metrics: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    morphology: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    per_image_pair_scores: dict[tuple[int, int], list[float]] = defaultdict(list)
    seed_coverage = {
        str(layer): {
            "windows": 0,
            "windows_with_any_seed": 0,
            "seed_pixels": 0,
            "correct_seed_pixels": 0,
            "class_seed_pixels": [0 for _ in class_names],
            "class_correct_seed_pixels": [0 for _ in class_names],
        }
        for layer in layers
    }
    fused_seed_coverage = {
        "windows": 0,
        "windows_with_any_seed": 0,
        "seed_pixels": 0,
        "correct_seed_pixels": 0,
        "class_seed_pixels": [0 for _ in class_names],
        "class_correct_seed_pixels": [0 for _ in class_names],
    }
    started = time.perf_counter()
    for image_index, sample in enumerate(samples, start=1):
        image = image_tensor(sample.image_path, args.evaluation_size, device)
        target = load_target(sample.mask_path)
        semantic_acc = {
            key: torch.zeros(
                (len(class_names), args.evaluation_size, args.evaluation_size),
                device=device,
            )
            for key in semantic_confusions
        }
        spatial_acc = {
            key: torch.zeros(
                (len(class_names), args.evaluation_size, args.evaluation_size),
                device=device,
            )
            for key in spatial_confusions
        }
        pair_acc = {
            key: torch.zeros(
                (len(class_names), args.evaluation_size, args.evaluation_size),
                device=device,
            )
            for key in pair_confusions
        }
        fusion_acc = torch.zeros(
            (len(class_names), args.evaluation_size, args.evaluation_size),
            device=device,
        )
        counts = torch.zeros((args.evaluation_size, args.evaluation_size), device=device)
        starts = [
            (top, left)
            for top in _window_starts(args.evaluation_size, args.window_size, args.stride)
            for left in _window_starts(args.evaluation_size, args.window_size, args.stride)
        ]
        if args.window_batch_size < 1:
            raise ValueError("window-batch-size must be positive.")
        extracted_windows: list[
            tuple[int, int, dict[int, Tensor], dict[int, dict[str, Tensor]]]
        ] = []
        for offset in range(0, len(starts), args.window_batch_size):
            batch_starts = starts[offset : offset + args.window_batch_size]
            window_batch = torch.cat(
                [
                    image[:, :, top : top + args.window_size, left : left + args.window_size]
                    for top, left in batch_starts
                ],
                dim=0,
            )
            autocast = torch.autocast(device_type=device.type, enabled=device.type == "cuda")
            with autocast:
                feature_batches, qkv_batches = extract_layers_batched(
                    model,
                    window_batch,
                    layers,
                    capture_qkv=True,
                )
            for batch_index, (top, left) in enumerate(batch_starts):
                extracted_windows.append(
                    (
                        top,
                        left,
                        {layer: values[batch_index] for layer, values in feature_batches.items()},
                        {
                            layer: {kind: values[batch_index] for kind, values in layer_values.items()}
                            for layer, layer_values in qkv_batches.items()
                        },
                    )
                )
        for top, left, features, qkv in extracted_windows:
            counts[top : top + args.window_size, left : left + args.window_size] += 1.0
            grid_h = args.window_size // 16
            grid_w = args.window_size // 16
            target_patch = _target_patch_ids(
                target[top : top + args.window_size, left : left + args.window_size],
                grid_h,
                grid_w,
                len(class_names),
            ).to(device)
            gt_seeds = gt_seed_labels(target_patch, fraction=args.gt_seed_fraction)
            layer_logits: dict[tuple[str, int], Tensor] = {}
            edge_banks: dict[int, dict[str, tuple[Tensor, Tensor]]] = {}
            for layer in layers:
                matrices = {
                    "raw": None,
                    "procrustes": alignments["procrustes"][layer],
                    "ridge": alignments["ridge"][layer],
                }
                for method, matrix in matrices.items():
                    logits = cosine_logits(features[layer], text_features, matrix).transpose(0, 1).reshape(
                        len(class_names), grid_h, grid_w
                    )
                    layer_logits[(method, layer)] = logits
                    _update_region_semantic(
                        region_semantic_confusions[(method, layer)],
                        logits,
                        target_patch,
                    )
                    _add_window(semantic_acc[(method, layer)], logits, top, left, args.window_size)
                edge_banks[layer] = _edge_bank(
                    features[layer], qkv[layer], grid_h, grid_w, args.affinity_temperature
                )
            spatial_keys = [(kind, layer) for layer in layers for kind in AFFINITY_KINDS]
            spatial_outputs = seeded_grid_diffusion_batched(
                torch.zeros(
                    (len(spatial_keys), len(class_names), grid_h, grid_w),
                    device=device,
                ),
                torch.stack([edge_banks[layer][kind][0] for kind, layer in spatial_keys]),
                torch.stack([edge_banks[layer][kind][1] for kind, layer in spatial_keys]),
                seed_labels=gt_seeds.unsqueeze(0).expand(len(spatial_keys), -1, -1),
                iterations=args.diffusion_iterations,
                restart=args.diffusion_restart,
            )
            for key, structure_only in zip(spatial_keys, spatial_outputs):
                _add_window(spatial_acc[key], structure_only, top, left, args.window_size)
            semantic_seeds = {
                layer: semantic_seed_labels(
                    layer_logits[(args.pair_semantic_method, layer)],
                    confidence_threshold=args.semantic_seed_threshold,
                    logit_scale=semantic_seed_logit_scale,
                    minimum_seeds_per_class=args.semantic_seed_minimum,
                )
                for layer in layers
            }
            for layer, seeds in semantic_seeds.items():
                stats = seed_coverage[str(layer)]
                valid = seeds >= 0
                stats["windows"] += 1
                stats["windows_with_any_seed"] += int(valid.any().item())
                stats["seed_pixels"] += int(valid.sum().item())
                target_valid = target_patch >= 0
                correct = valid & target_valid & (seeds == target_patch)
                stats["correct_seed_pixels"] += int(correct.sum().item())
                counts_by_class = torch.bincount(
                    seeds[valid], minlength=len(class_names)
                ).tolist()
                correct_counts_by_class = torch.bincount(
                    seeds[correct], minlength=len(class_names)
                ).tolist()
                stats["class_seed_pixels"] = [
                    int(old) + int(new)
                    for old, new in zip(stats["class_seed_pixels"], counts_by_class)
                ]
                stats["class_correct_seed_pixels"] = [
                    int(old) + int(new)
                    for old, new in zip(
                        stats["class_correct_seed_pixels"], correct_counts_by_class
                    )
                ]
            pair_keys = list(pair_confusions)
            pair_outputs = seeded_grid_diffusion_batched(
                torch.stack(
                    [layer_logits[(args.pair_semantic_method, semantic)] for semantic, _ in pair_keys]
                ),
                torch.stack(
                    [edge_banks[spatial][args.pair_affinity][0] for _, spatial in pair_keys]
                ),
                torch.stack(
                    [edge_banks[spatial][args.pair_affinity][1] for _, spatial in pair_keys]
                ),
                seed_labels=torch.stack([semantic_seeds[semantic] for semantic, _ in pair_keys]),
                iterations=args.diffusion_iterations,
                restart=args.diffusion_restart,
            )
            for key, propagated in zip(pair_keys, pair_outputs):
                _add_window(pair_acc[key], propagated, top, left, args.window_size)
            fused_logits = torch.stack(
                [layer_logits[(args.pair_semantic_method, layer)] for layer in layers]
            ).mean(dim=0)
            fused_edges = combine_edge_affinities(
                [edge_banks[layer][args.pair_affinity] for layer in layers]
            )
            fused_seeds = semantic_seed_labels(
                fused_logits,
                confidence_threshold=args.semantic_seed_threshold,
                logit_scale=semantic_seed_logit_scale,
                minimum_seeds_per_class=args.semantic_seed_minimum,
            )
            valid_fused = fused_seeds >= 0
            fused_seed_coverage["windows"] += 1
            fused_seed_coverage["windows_with_any_seed"] += int(valid_fused.any().item())
            fused_seed_coverage["seed_pixels"] += int(valid_fused.sum().item())
            fused_correct = (
                valid_fused & (target_patch >= 0) & (fused_seeds == target_patch)
            )
            fused_seed_coverage["correct_seed_pixels"] += int(fused_correct.sum().item())
            fused_counts_by_class = torch.bincount(
                fused_seeds[valid_fused], minlength=len(class_names)
            ).tolist()
            fused_correct_counts_by_class = torch.bincount(
                fused_seeds[fused_correct], minlength=len(class_names)
            ).tolist()
            fused_seed_coverage["class_seed_pixels"] = [
                int(old) + int(new)
                for old, new in zip(
                    fused_seed_coverage["class_seed_pixels"], fused_counts_by_class
                )
            ]
            fused_seed_coverage["class_correct_seed_pixels"] = [
                int(old) + int(new)
                for old, new in zip(
                    fused_seed_coverage["class_correct_seed_pixels"],
                    fused_correct_counts_by_class,
                )
            ]
            fused = seeded_grid_diffusion(
                fused_logits,
                *fused_edges,
                seed_labels=fused_seeds,
                iterations=args.diffusion_iterations,
                restart=args.diffusion_restart,
            )
            _add_window(fusion_acc, fused, top, left, args.window_size)
        safe_counts = counts.clamp_min(1.0).unsqueeze(0)
        for key, accumulator in semantic_acc.items():
            prediction = (
                (accumulator / safe_counts).argmax(dim=0).cpu().numpy().astype(np.uint8)
            )
            semantic_confusions[key].update(prediction, target)
            _update_morphology(morphology, key, prediction, target, class_names)
        for key, accumulator in spatial_acc.items():
            prediction = (
                (accumulator / safe_counts).argmax(dim=0).cpu().numpy().astype(np.uint8)
            )
            spatial_confusions[key].update(prediction, target)
            for class_index, name in enumerate(class_names):
                valid = target >= 0
                metrics = binary_spatial_metrics(
                    np.logical_and(prediction == class_index, valid),
                    target == class_index,
                    boundary_tolerance=2,
                )
                for metric_name, value in asdict(metrics).items():
                    spatial_metrics[(key[0], key[1], f"{name}:{metric_name}")].append(value)
        image_pair_scores: dict[tuple[int, int], float] = {}
        image_pair_matrices: dict[tuple[int, int], np.ndarray] = {}
        for key, accumulator in pair_acc.items():
            prediction = (
                (accumulator / safe_counts).argmax(dim=0).cpu().numpy().astype(np.uint8)
            )
            matrix = _confusion_matrix(prediction, target, len(class_names))
            pair_confusions[key].matrix += matrix
            image_pair_matrices[key] = matrix
            image_metric = Confusion(class_names)
            image_metric.matrix = matrix
            score = image_metric.summary()["mean_iou_percent"]
            image_pair_scores[key] = score
            per_image_pair_scores[key].append(score)
        unrestricted_key = max(image_pair_scores, key=image_pair_scores.get)
        off_diagonal_keys = [key for key in pair_confusions if key[0] != key[1]]
        dual_key = max(off_diagonal_keys, key=image_pair_scores.get)
        same_key = max(
            ((layer, layer) for layer in layers), key=image_pair_scores.get
        )
        adaptive_confusions["unrestricted"].matrix += image_pair_matrices[unrestricted_key]
        adaptive_confusions["dual_depth"].matrix += image_pair_matrices[dual_key]
        adaptive_confusions["same_depth"].matrix += image_pair_matrices[same_key]
        fusion_prediction = (
            (fusion_acc / safe_counts).argmax(dim=0).cpu().numpy().astype(np.uint8)
        )
        fusion_confusion.update(fusion_prediction, target)
        if image_index % max(args.progress_every, 1) == 0 or image_index == len(samples):
            print(
                json.dumps(
                    {
                        "processed": image_index,
                        "total": len(samples),
                        "elapsed_seconds": round(time.perf_counter() - started, 2),
                    }
                ),
                flush=True,
            )
    semantic_results = {
        method: {str(layer): semantic_confusions[(method, layer)].summary() for layer in layers}
        for method in SEMANTIC_METHODS
    }
    region_semantic_results = {
        method: {
            str(layer): Confusion(class_names)
            for layer in layers
        }
        for method in SEMANTIC_METHODS
    }
    for key, matrix in region_semantic_confusions.items():
        method, layer = key
        region_semantic_results[method][str(layer)].matrix = matrix
    spatial_results = {
        kind: {
            str(layer): {
                **spatial_confusions[(kind, layer)].summary(),
                "mean_per_class_metrics": _mean_spatial_metrics(spatial_metrics, kind, layer, class_names),
            }
            for layer in layers
        }
        for kind in AFFINITY_KINDS
    }
    aggregate_pair_scores = {
        key: pair_confusions[key].summary()["mean_iou_percent"] for key in pair_confusions
    }
    pair_result = oracle_pair_matrix(aggregate_pair_scores, layers)
    per_image_unrestricted_values = [
        max(per_image_pair_scores[key][index] for key in pair_confusions)
        for index in range(len(samples))
    ]
    per_image_best_unrestricted = np.mean(per_image_unrestricted_values)
    off_diagonal_pairs = [key for key in pair_confusions if key[0] != key[1]]
    per_image_dual_values = [
        max(per_image_pair_scores[key][index] for key in off_diagonal_pairs)
        for index in range(len(samples))
    ]
    per_image_same_values = [
        max(per_image_pair_scores[(layer, layer)][index] for layer in layers)
        for index in range(len(samples))
    ]
    per_image_best_dual = np.mean(per_image_dual_values)
    per_image_best_same = np.mean(per_image_same_values)
    pair_result["per_image_oracle"] = {
        "adaptive_unrestricted_miou_percent": round(float(per_image_best_unrestricted), 4),
        "adaptive_dual_depth_miou_percent": round(float(per_image_best_dual), 4),
        "adaptive_same_depth_miou_percent": round(float(per_image_best_same), 4),
        "adaptive_dual_gap_points": round(float(per_image_best_dual - per_image_best_same), 4),
    }
    pair_result["dataset_oracle"] = {
        name: confusion.summary()
        for name, confusion in adaptive_confusions.items()
    }
    locked_fixed = _locked_fixed_results(
        args.locked_selection,
        pair_confusions,
        layers,
        args.pair_semantic_method,
        args.pair_affinity,
        args.semantic_seed_threshold,
        semantic_seed_logit_scale,
        args.semantic_seed_minimum,
    )
    result = {
        "status": "complete",
        "dataset": {
            "name": dataset_name,
            "data_root": str(Path(args.data_root).resolve()),
            "images": len(samples),
            "start_index": args.start_index,
            "classes": list(class_names),
            "labels_used_for_inference": False,
            "labels_used_for_spatial_oracle_seeds": True,
        },
        "config": {
            "layers": list(layers),
            "evaluation_size": args.evaluation_size,
            "window_size": args.window_size,
            "stride": args.stride,
            "window_batch_size": args.window_batch_size,
            "semantic_methods": list(SEMANTIC_METHODS),
            "affinity_kinds": list(AFFINITY_KINDS),
            "pair_semantic_method": args.pair_semantic_method,
            "pair_affinity": args.pair_affinity,
            "affinity_temperature": args.affinity_temperature,
            "diffusion_iterations": args.diffusion_iterations,
            "diffusion_restart": args.diffusion_restart,
            "semantic_seed_threshold": args.semantic_seed_threshold,
            "semantic_seed_logit_scale": semantic_seed_logit_scale,
            "semantic_seed_minimum": args.semantic_seed_minimum,
            "gt_seed_fraction": args.gt_seed_fraction,
            "upsampling": "bilinear for every condition",
            "window_blending": "uniform average for every condition",
        },
        "alignment_manifest": alignments["manifest"],
        "model_checkpoint": checkpoint_manifest,
        "semantic": semantic_results,
        "region_semantic": {
            method: {
                layer: confusion.summary()
                for layer, confusion in layer_results.items()
            }
            for method, layer_results in region_semantic_results.items()
        },
        "spatial": spatial_results,
        "oracle_pairs": pair_result,
        "locked_source_selection": locked_fixed,
        "static_multilayer_fusion": fusion_confusion.summary(),
        "semantic_seed_coverage": {
            layer: {
                **stats,
                "seed_fraction": round(
                    stats["seed_pixels"]
                    / max(stats["windows"] * (args.window_size // 16) ** 2, 1),
                    8,
                ),
                "seed_precision": round(
                    stats["correct_seed_pixels"] / max(stats["seed_pixels"], 1),
                    8,
                ),
                "window_seed_rate": round(
                    stats["windows_with_any_seed"] / max(stats["windows"], 1),
                    8,
                ),
            }
            for layer, stats in seed_coverage.items()
        },
        "fused_seed_coverage": {
            **fused_seed_coverage,
            "seed_fraction": round(
                fused_seed_coverage["seed_pixels"]
                / max(
                    fused_seed_coverage["windows"] * (args.window_size // 16) ** 2,
                    1,
                ),
                8,
            ),
            "seed_precision": round(
                fused_seed_coverage["correct_seed_pixels"]
                / max(fused_seed_coverage["seed_pixels"], 1),
                8,
            ),
            "window_seed_rate": round(
                fused_seed_coverage["windows_with_any_seed"]
                / max(fused_seed_coverage["windows"], 1),
                8,
            ),
        },
        "morphology_component_recall": _summarize_morphology(morphology),
        "merge_state": {
            "pair_confusions": {
                f"{semantic}:{spatial}": pair_confusions[(semantic, spatial)].matrix.tolist()
                for semantic, spatial in pair_confusions
            },
            "region_semantic_confusions": {
                f"{method}:{layer}": matrix.tolist()
                for (method, layer), matrix in region_semantic_confusions.items()
            },
            "spatial_metrics": {
                f"{kind}:{layer}:{metric}": {
                    "sum": float(np.sum(items)),
                    "count": len(items),
                }
                for (kind, layer, metric), items in spatial_metrics.items()
            },
            "morphology": {
                f"{method}:{layer}:{group}": {
                    "sum": float(np.sum(items)),
                    "count": len(items),
                }
                for (method, layer, group), items in morphology.items()
            },
            "per_image_oracle": {
                "unrestricted_sum": float(np.sum(per_image_unrestricted_values)),
                "dual_sum": float(np.sum(per_image_dual_values)),
                "same_sum": float(np.sum(per_image_same_values)),
                "count": len(samples),
            },
            "adaptive_confusions": {
                name: confusion.matrix.tolist()
                for name, confusion in adaptive_confusions.items()
            },
        },
        "timing_seconds": round(time.perf_counter() - started, 4),
        "interpretation_guardrails": [
            "Spatial-only scores use GT-derived seeds and are diagnostic upper bounds, not deployable OVSS.",
            "Oracle pair selection uses LoveDA labels only in explicitly named oracle summaries.",
            "The primary semantic readout uses a fixed bilinear decoder so layer effects are not mixed with CAFe retraining.",
            "Fixed and learned pair selection must be selected on source validation before target-test claims are made.",
        ],
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "results.json", result)
    _write_tables(output_dir, result, layers, class_names)
    return result


def _locked_fixed_results(
    source_path: str | None,
    pair_confusions: dict[tuple[int, int], Confusion],
    layers: Sequence[int],
    semantic_method: str,
    affinity: str,
    seed_threshold: float,
    seed_logit_scale: float,
    seed_minimum: int,
) -> dict[str, Any] | None:
    if source_path is None:
        return None
    path = Path(source_path).expanduser().resolve()
    source = json.loads(path.read_text(encoding="utf-8"))
    source_config = source.get("config", {})
    mismatches = []
    if tuple(source_config.get("layers", ())) != tuple(layers):
        mismatches.append("layers")
    if source_config.get("pair_semantic_method") != semantic_method:
        mismatches.append("pair_semantic_method")
    if source_config.get("pair_affinity") != affinity:
        mismatches.append("pair_affinity")
    if float(source_config.get("semantic_seed_threshold", float("nan"))) != float(seed_threshold):
        mismatches.append("semantic_seed_threshold")
    if float(source_config.get("semantic_seed_logit_scale", float("nan"))) != float(seed_logit_scale):
        mismatches.append("semantic_seed_logit_scale")
    if int(source_config.get("semantic_seed_minimum", -1)) != int(seed_minimum):
        mismatches.append("semantic_seed_minimum")
    if mismatches:
        raise ValueError(f"Locked source selection differs in: {', '.join(mismatches)}")
    oracle = source.get("oracle_pairs", {})
    same = oracle.get("best_same_depth")
    dual = oracle.get("best_off_diagonal_pair")
    if not isinstance(same, dict) or not isinstance(dual, dict):
        raise ValueError("Locked selection result lacks source best same/off-diagonal pairs.")
    same_layer = int(same["layer"])
    dual_pair = (int(dual["semantic_layer"]), int(dual["spatial_layer"]))
    if (same_layer, same_layer) not in pair_confusions or dual_pair not in pair_confusions:
        raise ValueError("Locked source pair is outside the active layer set.")
    return {
        "source_result": str(path),
        "fixed_same_depth": {
            "layer": same_layer,
            "target_metrics": pair_confusions[(same_layer, same_layer)].summary(),
        },
        "fixed_dual_depth": {
            "semantic_layer": dual_pair[0],
            "spatial_layer": dual_pair[1],
            "target_metrics": pair_confusions[dual_pair].summary(),
        },
    }


def _mean_spatial_metrics(
    values: dict[tuple[str, int, str], list[float]],
    kind: str,
    layer: int,
    class_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    result = {}
    for name in class_names:
        result[name] = {}
        for metric in ("region_iou", "boundary_f1", "connectivity_recall", "leakage_ratio"):
            items = values[(kind, layer, f"{name}:{metric}")]
            result[name][metric] = round(float(np.mean(items)), 6) if items else float("nan")
    return result


def _components(mask: np.ndarray) -> Iterable[np.ndarray]:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    for component_id in range(1, count):
        yield labels == component_id


def _update_morphology(
    output: dict[tuple[str, int, str], list[float]],
    key: tuple[str, int],
    prediction: np.ndarray,
    target: np.ndarray,
    class_names: Sequence[str],
) -> None:
    image_area = target.size
    for class_index, _ in enumerate(class_names):
        for component in _components(target == class_index):
            area = int(component.sum())
            if area < 1:
                continue
            boundary = component & (cv2.erode(component.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
            thickness = area / max(int(boundary.sum()), 1)
            if area <= image_area * 0.001:
                group = "small_object"
            elif area >= image_area * 0.05:
                group = "large_region"
            elif thickness <= 3.0:
                group = "thin_structure"
            else:
                group = "other"
            output[(key[0], key[1], group)].append(float((prediction[component] == class_index).mean()))


def _summarize_morphology(
    values: dict[tuple[str, int, str], list[float]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for (method, layer, group), items in values.items():
        output.setdefault(method, {}).setdefault(str(layer), {})[group] = {
            "components": len(items),
            "mean_correct_pixel_fraction": round(float(np.mean(items)), 6),
        }
    return output


def _write_tables(
    output_dir: Path,
    result: dict[str, Any],
    layers: Sequence[int],
    class_names: Sequence[str],
) -> None:
    semantic_lines = ["method,class," + ",".join(f"L{layer}" for layer in layers)]
    for method, layer_results in result["semantic"].items():
        semantic_lines.append(
            f"{method},mIoU," + ",".join(str(layer_results[str(layer)]["mean_iou_percent"]) for layer in layers)
        )
        for class_name in class_names:
            semantic_lines.append(
                f"{method},{class_name},"
                + ",".join(str(layer_results[str(layer)]["per_class_iou_percent"][class_name]) for layer in layers)
            )
    (output_dir / "semantic_heatmap.csv").write_text("\n".join(semantic_lines) + "\n", encoding="utf-8")
    spatial_lines = ["affinity,class_metric," + ",".join(f"L{layer}" for layer in layers)]
    for kind, layer_results in result["spatial"].items():
        for class_name in class_names:
            for metric in ("region_iou", "boundary_f1", "connectivity_recall", "leakage_ratio"):
                spatial_lines.append(
                    f"{kind},{class_name}:{metric},"
                    + ",".join(
                        str(layer_results[str(layer)]["mean_per_class_metrics"][class_name][metric])
                        for layer in layers
                    )
                )
    (output_dir / "spatial_heatmap.csv").write_text("\n".join(spatial_lines) + "\n", encoding="utf-8")
    pair_lines = ["semantic_layer," + ",".join(f"spatial_L{layer}" for layer in layers)]
    for semantic, row in zip(layers, result["oracle_pairs"]["matrix"]):
        pair_lines.append(f"L{semantic}," + ",".join(str(value) for value in row))
    (output_dir / "oracle_pair_matrix.csv").write_text("\n".join(pair_lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "fit-alignments":
        result = fit_alignments(args)
    elif args.command == "evaluate-loveda":
        result = evaluate_dataset(args, "LoveDA")
    else:
        result = evaluate_dataset(args, "OpenEarthMap")
    print(json.dumps(result, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
