#!/usr/bin/env python3
"""Diagnose distributional and spatial evidence in frozen DINOv3 token fields.

This is a label-oracle *diagnostic*, not an OVSS method.  GT masks only select
pure source regions and score a fixed linear probe.  The experiment asks
whether source-fitted token statistics distinguish hard RS classes beyond a
first-order token mean before any decoder is trained.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cafedino_locked_loveda import IMAGENET_MEAN, IMAGENET_STD, build_model  # noqa: E402
from dinotool.distribution_probe import (  # noqa: E402
    descriptor,
    fit_token_projection,
    match_classifier_dimension,
)
from dinotool.oem import OEM_CLASSES, OEM_RAW_TO_CLASS_INDEX, OemSample, discover_oem_samples  # noqa: E402


CONDITIONS = (
    "mean",
    "mean_variance",
    "mean_covariance",
    "mean_covariance_spatial",
    "mean_covariance_spatial_shuffled",
)


@dataclass(frozen=True)
class RegionWindow:
    split: str
    sample: OemSample
    class_index: int
    top: int
    left: int
    purity: float

    @property
    def key(self) -> str:
        return f"{self.sample.key}:{self.top}:{self.left}:{self.class_index}"


@dataclass(frozen=True)
class TokenRecord:
    key: str
    group: str
    label: int
    tokens: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bpe-path", required=True)
    parser.add_argument("--source-data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-mode", choices=("eval", "official-script"), default="eval")
    parser.add_argument("--feature-space", choices=("backbone", "text-aligned"), default="backbone")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=("rangeland", "agriculture land"),
        help="OEM class names. The first pilot should keep the grassland/cropland pair.",
    )
    parser.add_argument("--window-size", type=int, default=224)
    parser.add_argument("--window-stride", type=int, default=112)
    parser.add_argument("--minimum-purity", type=float, default=0.90)
    parser.add_argument("--max-windows-per-image-class", type=int, default=2)
    parser.add_argument("--max-windows-per-class", type=int, default=160)
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-eval-images", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pca-dimension", type=int, default=32)
    parser.add_argument("--covariance-dimension", type=int, default=12)
    parser.add_argument("--radial-bins", type=int, default=3)
    parser.add_argument("--pca-maximum-tokens", type=int, default=50000)
    parser.add_argument("--classifier-dimension", type=int, default=128)
    parser.add_argument("--classifier-c", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allow-missing-source-images", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.window_size < 16 or args.window_size % 16:
        raise ValueError("window-size must be a multiple of the DINO patch size (16).")
    if args.window_stride < 1 or args.max_windows_per_image_class < 1:
        raise ValueError("Window sampling controls must be positive.")
    if args.max_windows_per_class < 2 or not 0.0 < args.minimum_purity <= 1.0:
        raise ValueError("Purity or per-class window limit is invalid.")
    if args.batch_size < 1 or args.covariance_dimension > args.pca_dimension:
        raise ValueError("Batch or descriptor dimensions are invalid.")
    if args.classifier_c <= 0 or args.bootstrap_samples < 1:
        raise ValueError("Classifier or bootstrap controls are invalid.")


def _load_target(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as source:
        raw = np.asarray(source).copy()
    if raw.ndim == 3:
        raw = raw[..., 0]
    result = np.full(raw.shape, -1, dtype=np.int16)
    for raw_id, class_index in OEM_RAW_TO_CLASS_INDEX.items():
        result[raw == raw_id] = class_index
    unexpected = set(np.unique(raw).tolist()) - set(OEM_RAW_TO_CLASS_INDEX) - {0, 255}
    if unexpected:
        raise ValueError(f"Unexpected OpenEarthMap label IDs at {mask_path}: {sorted(unexpected)}")
    return result


def _starts(length: int, window: int, stride: int) -> list[int]:
    if length < window:
        return []
    result = list(range(0, length - window + 1, stride))
    if result and result[-1] != length - window:
        result.append(length - window)
    return result


def sample_windows(
    samples: Sequence[OemSample],
    *,
    split: str,
    class_indices: Sequence[int],
    window_size: int,
    stride: int,
    minimum_purity: float,
    per_image_class: int,
    maximum_per_class: int,
    seed: int,
) -> list[RegionWindow]:
    """Select pure, fixed-size regions without selecting by DINO features."""
    selected: list[RegionWindow] = []
    totals: Counter[int] = Counter()
    for sample_index, sample in enumerate(samples):
        target = _load_target(sample.mask_path)
        candidates: dict[int, list[RegionWindow]] = defaultdict(list)
        for top in _starts(target.shape[0], window_size, stride):
            for left in _starts(target.shape[1], window_size, stride):
                patch = target[top : top + window_size, left : left + window_size]
                for class_index in class_indices:
                    purity = float(np.mean(patch == class_index))
                    if purity >= minimum_purity:
                        candidates[class_index].append(
                            RegionWindow(split, sample, class_index, top, left, purity)
                        )
        for class_index in class_indices:
            values = candidates[class_index]
            if not values or totals[class_index] >= maximum_per_class:
                continue
            payload = f"{seed}:{split}:{sample.key}:{class_index}".encode("utf-8")
            local_seed = int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "little")
            order = np.random.default_rng(local_seed).permutation(len(values))
            remaining = maximum_per_class - totals[class_index]
            take = min(per_image_class, remaining, len(values))
            for index in order[:take]:
                selected.append(values[int(index)])
                totals[class_index] += 1
        if all(totals[index] >= maximum_per_class for index in class_indices):
            break
        if sample_index % 50 == 0:
            print(json.dumps({"sampling_split": split, "sample": sample_index + 1, "selected": dict(totals)}), flush=True)
    missing = [index for index in class_indices if totals[index] < 2]
    if missing:
        names = [OEM_CLASSES[index].name for index in missing]
        raise ValueError(f"Too few pure windows for classes: {names}; relax purity or increase source images.")
    return selected


def _crop_tensor(window: RegionWindow, window_size: int) -> Tensor:
    with Image.open(window.sample.image_path) as source:
        rgb = np.asarray(source.convert("RGB"))[
            window.top : window.top + window_size,
            window.left : window.left + window_size,
        ].copy()
    if rgb.shape[:2] != (window_size, window_size):
        raise RuntimeError(f"Invalid image crop for {window.key}")
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


@torch.inference_mode()
def extract_tokens(
    model: torch.nn.Module,
    windows: Sequence[RegionWindow],
    *,
    window_size: int,
    batch_size: int,
    feature_space: str,
) -> list[TokenRecord]:
    device = next(model.parameters()).device
    grid_size = window_size // 16
    records: list[TokenRecord] = []
    started = time.perf_counter()
    for start in range(0, len(windows), batch_size):
        batch_windows = windows[start : start + batch_size]
        images = torch.stack([_crop_tensor(window, window_size) for window in batch_windows]).to(
            device, non_blocking=True
        )
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            _, text_aligned, backbone = model.backbone.encode_image_with_patch_tokens(images, normalize=False)
        tokens = backbone if feature_space == "backbone" else text_aligned
        if tokens.shape[1] != grid_size * grid_size:
            raise RuntimeError(f"Unexpected token count: {tokens.shape[1]} != {grid_size * grid_size}")
        arrays = tokens.float().cpu().numpy().reshape(len(batch_windows), grid_size, grid_size, -1)
        records.extend(
            TokenRecord(
                key=window.key,
                group=window.sample.key,
                label=window.class_index,
                tokens=array,
            )
            for window, array in zip(batch_windows, arrays)
        )
        if start == 0 or start + len(batch_windows) == len(windows) or (start // batch_size + 1) % 10 == 0:
            print(json.dumps({"extracting": start + len(batch_windows), "total": len(windows), "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
    return records


def _describe_records(
    records: Sequence[TokenRecord],
    projection: Any,
    *,
    covariance_dimension: int,
    radial_bins: int,
    shuffle_seed: int,
) -> dict[str, np.ndarray]:
    result: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS}
    for record in records:
        projected = projection.transform(record.tokens)
        for condition in CONDITIONS:
            result[condition].append(
                descriptor(
                    condition,
                    projected,
                    covariance_dimension=covariance_dimension,
                    radial_bins=radial_bins,
                    shuffle_key=record.key,
                    shuffle_seed=shuffle_seed,
                )
            )
    return {name: np.stack(values).astype(np.float32) for name, values in result.items()}


def _classification_summary(target: np.ndarray, prediction: np.ndarray, class_indices: Sequence[int]) -> dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

    labels = np.asarray(class_indices, dtype=np.int64)
    return {
        "balanced_accuracy": round(float(balanced_accuracy_score(target, prediction)) * 100.0, 4),
        "macro_f1": round(float(f1_score(target, prediction, labels=labels, average="macro", zero_division=0)) * 100.0, 4),
        "accuracy": round(float(np.mean(target == prediction)) * 100.0, 4),
        "confusion_matrix": confusion_matrix(target, prediction, labels=labels).tolist(),
        "per_class_recall_percent": {
            OEM_CLASSES[index].name: round(
                float(np.mean(prediction[target == index] == index)) * 100.0,
                4,
            )
            for index in labels
            if bool(np.any(target == index))
        },
    }


def _group_bootstrap_delta(
    target: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    groups: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap image groups; crop-level independence is never assumed."""
    from sklearn.metrics import balanced_accuracy_score

    groups_array = np.asarray(groups)
    labels = np.unique(target)
    by_label: dict[int, list[np.ndarray]] = {}
    for label in labels:
        label_rows = np.flatnonzero(target == label)
        label_groups = np.unique(groups_array[label_rows])
        by_label[int(label)] = [
            label_rows[groups_array[label_rows] == group] for group in label_groups
        ]
        if not by_label[int(label)]:
            raise ValueError("Every evaluated class requires at least one source image group.")
    generator = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        rows = np.concatenate(
            [
                np.concatenate(
                    [
                        group_rows[value]
                        for value in generator.integers(0, len(group_rows), size=len(group_rows))
                    ]
                )
                for group_rows in by_label.values()
            ]
        )
        reference_score = balanced_accuracy_score(target[rows], reference[rows])
        candidate_score = balanced_accuracy_score(target[rows], candidate[rows])
        deltas[index] = (candidate_score - reference_score) * 100.0
    return {
        "mean_delta_points": round(float(deltas.mean()), 4),
        "ci95_low_points": round(float(np.quantile(deltas, 0.025)), 4),
        "ci95_high_points": round(float(np.quantile(deltas, 0.975)), 4),
    }


def _fit_and_score(
    train_values: np.ndarray,
    train_target: np.ndarray,
    eval_values: np.ndarray,
    eval_target: np.ndarray,
    class_indices: Sequence[int],
    *,
    classifier_dimension: int,
    classifier_c: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_matched = match_classifier_dimension(
        train_values, output_dimension=classifier_dimension, seed=seed
    )
    eval_matched = match_classifier_dimension(
        eval_values, output_dimension=classifier_dimension, seed=seed
    )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=classifier_c,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        ),
    )
    classifier.fit(train_matched, train_target)
    prediction = classifier.predict(eval_matched).astype(np.int64)
    summary = _classification_summary(eval_target, prediction, class_indices)
    summary["raw_descriptor_dimension"] = int(train_values.shape[1])
    summary["classifier_feature_dimension"] = classifier_dimension
    summary["classifier_trainable_parameters"] = int(
        classifier_dimension * len(class_indices) + len(class_indices)
    )
    return prediction, summary


def _split_label_counts(records: Iterable[TokenRecord]) -> dict[str, int]:
    counts = Counter(record.label for record in records)
    return {OEM_CLASSES[index].name: int(value) for index, value in sorted(counts.items())}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    class_lookup = {spec.name: index for index, spec in enumerate(OEM_CLASSES)}
    requested_names = tuple(args.classes)
    if len(set(requested_names)) != len(requested_names) or any(name not in class_lookup for name in requested_names):
        raise ValueError(f"classes must be unique OEM names: {sorted(class_lookup)}")
    class_indices = tuple(class_lookup[name] for name in requested_names)
    train_samples = discover_oem_samples(
        args.source_data_root,
        args.train_split,
        max_images=args.max_train_images,
        allow_missing_images=args.allow_missing_source_images,
    )
    eval_samples = discover_oem_samples(
        args.source_data_root,
        args.eval_split,
        max_images=args.max_eval_images,
        allow_missing_images=args.allow_missing_source_images,
    )
    train_windows = sample_windows(
        train_samples,
        split=args.train_split,
        class_indices=class_indices,
        window_size=args.window_size,
        stride=args.window_stride,
        minimum_purity=args.minimum_purity,
        per_image_class=args.max_windows_per_image_class,
        maximum_per_class=args.max_windows_per_class,
        seed=args.seed,
    )
    eval_windows = sample_windows(
        eval_samples,
        split=args.eval_split,
        class_indices=class_indices,
        window_size=args.window_size,
        stride=args.window_stride,
        minimum_purity=args.minimum_purity,
        per_image_class=args.max_windows_per_image_class,
        maximum_per_class=args.max_windows_per_class,
        seed=args.seed + 1,
    )
    model_args = argparse.Namespace(
        official_root=args.official_root,
        checkpoint=args.checkpoint,
        bpe_path=args.bpe_path,
        device=args.device,
        model_mode=args.model_mode,
        window_size=args.window_size,
    )
    model, _, checkpoint_manifest = build_model(model_args)
    train_records = extract_tokens(
        model,
        train_windows,
        window_size=args.window_size,
        batch_size=args.batch_size,
        feature_space=args.feature_space,
    )
    eval_records = extract_tokens(
        model,
        eval_windows,
        window_size=args.window_size,
        batch_size=args.batch_size,
        feature_space=args.feature_space,
    )
    projection = fit_token_projection(
        [record.tokens for record in train_records],
        dimension=args.pca_dimension,
        maximum_tokens=args.pca_maximum_tokens,
        seed=args.seed,
    )
    train_descriptors = _describe_records(
        train_records,
        projection,
        covariance_dimension=args.covariance_dimension,
        radial_bins=args.radial_bins,
        shuffle_seed=args.seed,
    )
    eval_descriptors = _describe_records(
        eval_records,
        projection,
        covariance_dimension=args.covariance_dimension,
        radial_bins=args.radial_bins,
        shuffle_seed=args.seed,
    )
    train_target = np.asarray([record.label for record in train_records], dtype=np.int64)
    eval_target = np.asarray([record.label for record in eval_records], dtype=np.int64)
    eval_groups = [record.group for record in eval_records]
    results: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for index, condition in enumerate(CONDITIONS):
        prediction, summary = _fit_and_score(
            train_descriptors[condition],
            train_target,
            eval_descriptors[condition],
            eval_target,
            class_indices,
            classifier_dimension=args.classifier_dimension,
            classifier_c=args.classifier_c,
            seed=args.seed + index,
        )
        predictions[condition] = prediction
        results[condition] = summary
    reference = predictions["mean"]
    for index, condition in enumerate(CONDITIONS):
        if condition == "mean":
            continue
        results[condition]["vs_mean_group_bootstrap"] = _group_bootstrap_delta(
            eval_target,
            reference,
            predictions[condition],
            eval_groups,
            samples=args.bootstrap_samples,
            seed=args.seed + 100 + index,
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    payload = {
        "status": "complete",
        "purpose": "GT-region diagnostic only; not a deployable OVSS model or target-domain score.",
        "dataset": {
            "name": "OpenEarthMap",
            "data_root": str(Path(args.source_data_root).expanduser().resolve()),
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "classes": list(requested_names),
            "labels_used_for_region_selection": True,
            "labels_used_for_probe_scoring": True,
            "labels_used_for_dino_or_descriptor_fitting": False,
            "group_definition": "original OEM image path; this initial split is not claimed geographic-disjoint",
        },
        "sampling": {
            "window_size": args.window_size,
            "window_stride": args.window_stride,
            "minimum_purity": args.minimum_purity,
            "max_windows_per_image_class": args.max_windows_per_image_class,
            "max_windows_per_class": args.max_windows_per_class,
            "train_windows": len(train_records),
            "eval_windows": len(eval_records),
            "train_class_counts": _split_label_counts(train_records),
            "eval_class_counts": _split_label_counts(eval_records),
            "train_source_images": len({record.group for record in train_records}),
            "eval_source_images": len({record.group for record in eval_records}),
        },
        "token_source": args.feature_space,
        "projection": {
            "method": "PCA fitted from source-train tokens without labels",
            "dimension": projection.dimension,
            "explained_variance_ratio_sum": round(float(projection.explained_variance_ratio.sum()), 6),
            "pca_maximum_tokens": args.pca_maximum_tokens,
        },
        "descriptor_controls": {
            "conditions": list(CONDITIONS),
            "covariance_dimension": args.covariance_dimension,
            "radial_bins": args.radial_bins,
            "spatial_shuffle": "deterministic shuffle in both train and eval; mean/covariance stay unchanged",
        },
        "classifier": {
            "type": "standardized logistic regression",
            "classifier_dimension": args.classifier_dimension,
            "C": args.classifier_c,
            "class_weight": "balanced",
            "selection": "all hyperparameters fixed before evaluation; no eval-label tuning",
        },
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "unit": "class-stratified original source image path",
            "metric": "balanced accuracy delta versus mean",
        },
        "checkpoint": checkpoint_manifest,
        "results": results,
    }
    _write_json(output_dir / "results.json", payload)
    np.savez_compressed(
        output_dir / "eval_predictions.npz",
        target=eval_target,
        groups=np.asarray(eval_groups),
        **predictions,
    )
    print(json.dumps({"status": "complete", "output": str(output_dir / "results.json"), "results": results}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
