#!/usr/bin/env python3
"""Strict distributional-field diagnosis with low-level appearance controls.

This is deliberately a *diagnostic*, not an OVSS method. It tests whether
frozen DINOv3 token statistics retain class evidence beyond token means and
beyond conventional RGB/LBP/HOG controls. PCA and every classifier are fit on
OpenEarthMap source-train regions only. A LoveDA target run may use its labels
only to select/scored diagnostic regions, never to fit transformations or
choose hyperparameters.
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
from typing import Any, Callable, Sequence

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
from dinotool.distribution_probe import descriptor, fit_token_projection, match_classifier_dimension  # noqa: E402
from dinotool.handcrafted_probe import lbp_histogram, rgb_lbp_hog, rgb_moments_histogram, spatial_hog  # noqa: E402
from dinotool.oem import OEM_CLASSES, OEM_RAW_TO_CLASS_INDEX, discover_oem_samples  # noqa: E402


DINO_CONDITIONS = (
    "dinov3_mean",
    "dinov3_mean_variance",
    "dinov3_mean_covariance",
    "dinov3_mean_covariance_spatial",
    "dinov3_mean_covariance_spatial_shuffled",
)
CONTROL_CONDITIONS = ("rgb_moments_histogram", "lbp_histogram", "spatial_hog", "rgb_lbp_hog")
FUSION_CONDITION = "dinov3_mean_covariance_plus_rgb_lbp_hog"
CONDITIONS = DINO_CONDITIONS + CONTROL_CONDITIONS + (FUSION_CONDITION,)

# LoveDA uses 1..7 for background, building, road, water, barren, forest,
# agriculture. Only foreground classes having an unambiguous OEM counterpart
# are admitted to cross-domain classification.
LOVEDA_RAW_TO_OEM = {2: 7, 3: 3, 4: 5, 5: 0, 6: 4, 7: 6}
PAIR_PRESETS = (
    ("rangeland", "agriculture land"),
    ("bareland", "developed space"),
    ("road", "building"),
    ("tree", "agriculture land"),
)


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path
    key: str


@dataclass(frozen=True)
class RegionWindow:
    split: str
    sample: Sample
    class_index: int
    top: int
    left: int
    purity: float

    @property
    def key(self) -> str:
        return f"{self.sample.key}:{self.top}:{self.left}:{self.class_index}"


@dataclass(frozen=True)
class RegionRecord:
    key: str
    group: str
    label: int
    tokens: np.ndarray
    rgb: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bpe-path", required=True)
    parser.add_argument("--source-data-root", required=True, help="OpenEarthMap source root.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-dataset", choices=("oem", "loveda"), default="oem")
    parser.add_argument("--eval-data-root", help="OEM or LoveDA evaluation root; defaults to source root for OEM.")
    parser.add_argument("--source-train-split", default="train")
    parser.add_argument("--source-eval-split", default="val")
    parser.add_argument("--loveda-subset", default="val", choices=("train", "val", "all"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-mode", choices=("eval", "official-script"), default="eval")
    parser.add_argument("--feature-space", choices=("backbone", "text-aligned"), default="backbone")
    parser.add_argument("--classes", nargs="+", default=tuple(spec.name for spec in OEM_CLASSES))
    parser.add_argument("--window-size", type=int, default=224)
    parser.add_argument("--window-stride", type=int, default=112)
    parser.add_argument("--minimum-purity", type=float, default=0.85)
    parser.add_argument("--max-windows-per-image-class", type=int, default=1)
    parser.add_argument("--max-windows-per-class", type=int, default=240)
    parser.add_argument("--minimum-windows-per-class", type=int, default=96)
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-eval-images", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pca-dimension", type=int, default=32)
    parser.add_argument("--covariance-dimension", type=int, default=12)
    parser.add_argument("--radial-bins", type=int, default=3)
    parser.add_argument("--pca-maximum-tokens", type=int, default=100000)
    parser.add_argument("--classifier-dimension", type=int, default=128)
    parser.add_argument("--classifier-c", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--allow-missing-source-images", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.window_size < 16 or args.window_size % 16:
        raise ValueError("window-size must be a DINO patch-size multiple (16).")
    if not 0.0 < args.minimum_purity <= 1.0:
        raise ValueError("minimum-purity must be in (0, 1].")
    if min(args.max_windows_per_image_class, args.max_windows_per_class, args.minimum_windows_per_class) < 1:
        raise ValueError("Window limits must be positive.")
    if args.minimum_windows_per_class > args.max_windows_per_class:
        raise ValueError("minimum-windows-per-class cannot exceed max-windows-per-class.")
    if args.covariance_dimension > args.pca_dimension or args.pca_dimension < 1:
        raise ValueError("PCA/covariance dimensions are invalid.")
    if args.classifier_dimension < 1 or args.classifier_c <= 0 or args.bootstrap_samples < 1:
        raise ValueError("Classifier/bootstrap settings are invalid.")


def _load_raw_mask(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        raw = np.asarray(source).copy()
    if raw.ndim == 3:
        if raw.shape[2] < 3 or not np.array_equal(raw[..., :3], np.repeat(raw[..., :1], 3, axis=2)):
            raise ValueError(f"Mask must contain scalar IDs: {path}")
        raw = raw[..., 0]
    return raw


def load_oem_target(path: Path) -> np.ndarray:
    raw = _load_raw_mask(path)
    result = np.full(raw.shape, -1, dtype=np.int16)
    for raw_id, class_index in OEM_RAW_TO_CLASS_INDEX.items():
        result[raw == raw_id] = class_index
    unexpected = set(np.unique(raw).tolist()) - set(OEM_RAW_TO_CLASS_INDEX) - {0, 255}
    if unexpected:
        raise ValueError(f"Unexpected OEM label IDs at {path}: {sorted(unexpected)}")
    return result


def load_loveda_target(path: Path) -> np.ndarray:
    raw = _load_raw_mask(path)
    result = np.full(raw.shape, -1, dtype=np.int16)
    for raw_id, class_index in LOVEDA_RAW_TO_OEM.items():
        result[raw == raw_id] = class_index
    unexpected = set(np.unique(raw).tolist()) - set(LOVEDA_RAW_TO_OEM) - {0, 1, 255}
    if unexpected:
        raise ValueError(f"Unexpected LoveDA label IDs at {path}: {sorted(unexpected)}")
    return result


def discover_source(args: argparse.Namespace) -> list[Sample]:
    samples = discover_oem_samples(
        args.source_data_root,
        args.source_train_split,
        max_images=args.max_train_images,
        allow_missing_images=args.allow_missing_source_images,
    )
    return [Sample(item.image_path, item.mask_path, item.key) for item in samples]


def discover_evaluation(args: argparse.Namespace) -> tuple[list[Sample], Callable[[Path], np.ndarray], str]:
    if args.eval_dataset == "oem":
        root = args.eval_data_root or args.source_data_root
        samples = discover_oem_samples(
            root,
            args.source_eval_split,
            max_images=args.max_eval_images,
            allow_missing_images=args.allow_missing_source_images,
        )
        return [Sample(item.image_path, item.mask_path, item.key) for item in samples], load_oem_target, "OpenEarthMap"

    from dinotool.loveda import discover_loveda_samples

    if not args.eval_data_root:
        raise ValueError("eval-data-root is required for a LoveDA cross-domain run.")
    root = Path(args.eval_data_root).expanduser().resolve()
    entries = discover_loveda_samples(root)
    root_parts = {part.casefold() for part in root.parts}
    if args.loveda_subset != "all" and args.loveda_subset not in root_parts:
        entries = [
            item
            for item in entries
            if args.loveda_subset in {part.casefold() for part in Path(item.key).parts}
        ]
    if not entries:
        raise ValueError(f"No LoveDA samples matched subset '{args.loveda_subset}' below {root}.")
    if args.max_eval_images is not None:
        entries = entries[: args.max_eval_images]
    return [Sample(item.image_path, item.mask_path, item.key) for item in entries], load_loveda_target, "LoveDA"


def _starts(length: int, window: int, stride: int) -> list[int]:
    if length < window:
        return []
    values = list(range(0, length - window + 1, stride))
    if values[-1] != length - window:
        values.append(length - window)
    return values


def _rng_seed(seed: int, *parts: object) -> int:
    payload = ":".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def sample_windows(
    samples: Sequence[Sample],
    *,
    split: str,
    target_loader: Callable[[Path], np.ndarray],
    class_indices: Sequence[int],
    window_size: int,
    stride: int,
    minimum_purity: float,
    per_image_class: int,
    maximum_per_class: int,
    minimum_per_class: int,
    seed: int,
) -> list[RegionWindow]:
    """Sample across all images before capping, avoiding filename-prefix bias."""
    pools: dict[int, list[RegionWindow]] = defaultdict(list)
    for sample_index, sample in enumerate(samples, start=1):
        target = target_loader(sample.mask_path)
        local: dict[int, list[RegionWindow]] = defaultdict(list)
        for top in _starts(target.shape[0], window_size, stride):
            for left in _starts(target.shape[1], window_size, stride):
                patch = target[top : top + window_size, left : left + window_size]
                for class_index in class_indices:
                    purity = float(np.mean(patch == class_index))
                    if purity >= minimum_purity:
                        local[class_index].append(RegionWindow(split, sample, class_index, top, left, purity))
        for class_index, windows in local.items():
            order = np.random.default_rng(_rng_seed(seed, split, sample.key, class_index)).permutation(len(windows))
            pools[class_index].extend(windows[int(index)] for index in order[:per_image_class])
        if sample_index == 1 or sample_index % 100 == 0 or sample_index == len(samples):
            print(json.dumps({"sampling_split": split, "sample": sample_index, "total": len(samples), "candidate_windows": {str(key): len(value) for key, value in pools.items()}}), flush=True)
    selected: list[RegionWindow] = []
    insufficient: list[str] = []
    for class_index in class_indices:
        windows = pools[class_index]
        order = np.random.default_rng(_rng_seed(seed, split, "global", class_index)).permutation(len(windows))
        chosen = [windows[int(index)] for index in order[:maximum_per_class]]
        if len(chosen) < minimum_per_class:
            insufficient.append(f"{OEM_CLASSES[class_index].name}={len(chosen)}")
        selected.extend(chosen)
    if insufficient:
        raise ValueError("Insufficient high-purity original-image regions: " + ", ".join(insufficient))
    return selected


def crop_rgb(window: RegionWindow, window_size: int) -> np.ndarray:
    with Image.open(window.sample.image_path) as source:
        rgb = np.asarray(source.convert("RGB"))[window.top : window.top + window_size, window.left : window.left + window_size].copy()
    if rgb.shape != (window_size, window_size, 3):
        raise RuntimeError(f"Invalid crop for {window.key}: {rgb.shape}")
    return rgb


def normalized_tensor(rgb: np.ndarray) -> Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


@torch.inference_mode()
def extract_records(
    model: torch.nn.Module,
    windows: Sequence[RegionWindow],
    *,
    window_size: int,
    batch_size: int,
    feature_space: str,
) -> list[RegionRecord]:
    device = next(model.parameters()).device
    grid_size = window_size // 16
    records: list[RegionRecord] = []
    started = time.perf_counter()
    for start in range(0, len(windows), batch_size):
        batch_windows = windows[start : start + batch_size]
        rgbs = [crop_rgb(window, window_size) for window in batch_windows]
        images = torch.stack([normalized_tensor(rgb) for rgb in rgbs]).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            _, text_aligned, backbone = model.backbone.encode_image_with_patch_tokens(images, normalize=False)
        token_tensor = backbone if feature_space == "backbone" else text_aligned
        if token_tensor.shape[1] != grid_size * grid_size:
            raise RuntimeError(f"Unexpected token count {token_tensor.shape[1]} for a {grid_size}x{grid_size} grid.")
        token_arrays = token_tensor.float().cpu().numpy().reshape(len(batch_windows), grid_size, grid_size, -1)
        records.extend(
            RegionRecord(window.key, window.sample.key, window.class_index, tokens, rgb)
            for window, tokens, rgb in zip(batch_windows, token_arrays, rgbs)
        )
        completed = start + len(batch_windows)
        if completed == len(windows) or start == 0 or (completed // batch_size) % 10 == 0:
            print(json.dumps({"extracting": completed, "total": len(windows), "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
    return records


def describe_records(records: Sequence[RegionRecord], projection: Any, *, covariance_dimension: int, radial_bins: int, seed: int) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS}
    for record in records:
        projected = projection.transform(record.tokens)
        dino_mean = descriptor("mean", projected, covariance_dimension=covariance_dimension, radial_bins=radial_bins)
        dino_variance = descriptor("mean_variance", projected, covariance_dimension=covariance_dimension, radial_bins=radial_bins)
        dino_covariance = descriptor("mean_covariance", projected, covariance_dimension=covariance_dimension, radial_bins=radial_bins)
        dino_spatial = descriptor("mean_covariance_spatial", projected, covariance_dimension=covariance_dimension, radial_bins=radial_bins)
        dino_shuffled = descriptor("mean_covariance_spatial_shuffled", projected, covariance_dimension=covariance_dimension, radial_bins=radial_bins, shuffle_key=record.key, shuffle_seed=seed)
        handcrafted = rgb_lbp_hog(record.rgb)
        values["dinov3_mean"].append(dino_mean)
        values["dinov3_mean_variance"].append(dino_variance)
        values["dinov3_mean_covariance"].append(dino_covariance)
        values["dinov3_mean_covariance_spatial"].append(dino_spatial)
        values["dinov3_mean_covariance_spatial_shuffled"].append(dino_shuffled)
        values["rgb_moments_histogram"].append(rgb_moments_histogram(record.rgb))
        values["lbp_histogram"].append(lbp_histogram(record.rgb))
        values["spatial_hog"].append(spatial_hog(record.rgb))
        values["rgb_lbp_hog"].append(handcrafted)
        values[FUSION_CONDITION].append(np.concatenate((dino_covariance, handcrafted)).astype(np.float32))
    return {name: np.stack(rows).astype(np.float32) for name, rows in values.items()}


def classification_summary(target: np.ndarray, prediction: np.ndarray, class_indices: Sequence[int]) -> dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

    labels = np.asarray(class_indices, dtype=np.int64)
    return {
        "balanced_accuracy": round(float(balanced_accuracy_score(target, prediction)) * 100.0, 4),
        "macro_f1": round(float(f1_score(target, prediction, labels=labels, average="macro", zero_division=0)) * 100.0, 4),
        "accuracy": round(float(np.mean(target == prediction)) * 100.0, 4),
        "confusion_matrix": confusion_matrix(target, prediction, labels=labels).tolist(),
        "per_class_recall_percent": {OEM_CLASSES[index].name: round(float(np.mean(prediction[target == index] == index)) * 100.0, 4) for index in labels},
    }


def bootstrap_delta(target: np.ndarray, reference: np.ndarray, candidate: np.ndarray, groups: Sequence[str], *, samples: int, seed: int) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score

    groups_array = np.asarray(groups)
    by_label: dict[int, list[np.ndarray]] = {}
    for label in np.unique(target):
        rows = np.flatnonzero(target == label)
        by_label[int(label)] = [rows[groups_array[rows] == group] for group in np.unique(groups_array[rows])]
    generator = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        rows = np.concatenate([np.concatenate([group_rows[value] for value in generator.integers(0, len(group_rows), size=len(group_rows))]) for group_rows in by_label.values()])
        deltas[index] = (balanced_accuracy_score(target[rows], candidate[rows]) - balanced_accuracy_score(target[rows], reference[rows])) * 100.0
    return {"mean_delta_points": round(float(deltas.mean()), 4), "ci95_low_points": round(float(np.quantile(deltas, 0.025)), 4), "ci95_high_points": round(float(np.quantile(deltas, 0.975)), 4)}


def fit_predict(
    train_values: np.ndarray,
    train_target: np.ndarray,
    eval_values: np.ndarray,
    *,
    classifier_dimension: int,
    classifier_c: float,
    feature_projection_seed: int,
    classifier_seed: int,
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Conditions with equal raw dimensionality, especially the original and
    # shuffled spatial descriptors, must use *exactly the same* label-free
    # feature map. Otherwise random projection variation itself can masquerade
    # as a spatial-organization effect.
    train_matched = match_classifier_dimension(
        train_values, output_dimension=classifier_dimension, seed=feature_projection_seed
    )
    eval_matched = match_classifier_dimension(
        eval_values, output_dimension=classifier_dimension, seed=feature_projection_seed
    )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=classifier_c, class_weight="balanced", max_iter=3000, random_state=classifier_seed),
    )
    classifier.fit(train_matched, train_target)
    return classifier.predict(eval_matched).astype(np.int64)


def evaluate_task(
    train_descriptors: dict[str, np.ndarray],
    train_records: Sequence[RegionRecord],
    eval_descriptors: dict[str, np.ndarray],
    eval_records: Sequence[RegionRecord],
    class_indices: Sequence[int],
    *,
    classifier_dimension: int,
    classifier_c: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train_rows = np.asarray([record.label in class_indices for record in train_records])
    eval_rows = np.asarray([record.label in class_indices for record in eval_records])
    train_target = np.asarray([record.label for record in train_records], dtype=np.int64)[train_rows]
    eval_target = np.asarray([record.label for record in eval_records], dtype=np.int64)[eval_rows]
    eval_groups = [record.group for record, keep in zip(eval_records, eval_rows) if keep]
    results: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for condition_index, condition in enumerate(CONDITIONS):
        feature_projection_seed = _rng_seed(seed, "classifier-feature-map", train_descriptors[condition].shape[1])
        prediction = fit_predict(
            train_descriptors[condition][train_rows],
            train_target,
            eval_descriptors[condition][eval_rows],
            classifier_dimension=classifier_dimension,
            classifier_c=classifier_c,
            feature_projection_seed=feature_projection_seed,
            classifier_seed=seed + condition_index,
        )
        predictions[condition] = prediction
        result = classification_summary(eval_target, prediction, class_indices)
        result["raw_descriptor_dimension"] = int(train_descriptors[condition].shape[1])
        result["classifier_feature_dimension"] = classifier_dimension
        result["classifier_trainable_parameters"] = int(classifier_dimension * len(class_indices) + len(class_indices))
        results[condition] = result
    dino_reference = predictions["dinov3_mean"]
    control_reference = predictions["rgb_lbp_hog"]
    for condition_index, condition in enumerate(CONDITIONS):
        if condition != "dinov3_mean":
            results[condition]["vs_dinov3_mean_group_bootstrap"] = bootstrap_delta(eval_target, dino_reference, predictions[condition], eval_groups, samples=bootstrap_samples, seed=seed + 100 + condition_index)
    results[FUSION_CONDITION]["vs_rgb_lbp_hog_group_bootstrap"] = bootstrap_delta(eval_target, control_reference, predictions[FUSION_CONDITION], eval_groups, samples=bootstrap_samples, seed=seed + 300)
    return {"class_names": [OEM_CLASSES[index].name for index in class_indices], "samples": int(len(eval_target)), "source_images": int(len(set(eval_groups))), "results": results}, predictions


def record_counts(records: Sequence[RegionRecord]) -> dict[str, int]:
    counts = Counter(record.label for record in records)
    return {OEM_CLASSES[index].name: int(count) for index, count in sorted(counts.items())}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    lookup = {spec.name: index for index, spec in enumerate(OEM_CLASSES)}
    requested_names = tuple(args.classes)
    if len(set(requested_names)) != len(requested_names) or any(name not in lookup for name in requested_names):
        raise ValueError(f"classes must be unique OEM class names: {sorted(lookup)}")
    class_indices = tuple(lookup[name] for name in requested_names)
    if args.eval_dataset == "loveda":
        unsupported = [name for name in requested_names if lookup[name] not in set(LOVEDA_RAW_TO_OEM.values())]
        if unsupported:
            raise ValueError(f"LoveDA has no unambiguous label counterpart for: {unsupported}")

    train_samples = discover_source(args)
    eval_samples, eval_target_loader, eval_dataset_name = discover_evaluation(args)
    train_windows = sample_windows(train_samples, split="source_train", target_loader=load_oem_target, class_indices=class_indices, window_size=args.window_size, stride=args.window_stride, minimum_purity=args.minimum_purity, per_image_class=args.max_windows_per_image_class, maximum_per_class=args.max_windows_per_class, minimum_per_class=args.minimum_windows_per_class, seed=args.seed)
    eval_windows = sample_windows(eval_samples, split="evaluation", target_loader=eval_target_loader, class_indices=class_indices, window_size=args.window_size, stride=args.window_stride, minimum_purity=args.minimum_purity, per_image_class=args.max_windows_per_image_class, maximum_per_class=args.max_windows_per_class, minimum_per_class=args.minimum_windows_per_class, seed=args.seed + 1)

    model_args = argparse.Namespace(official_root=args.official_root, checkpoint=args.checkpoint, bpe_path=args.bpe_path, device=args.device, model_mode=args.model_mode, window_size=args.window_size)
    model, _, checkpoint_manifest = build_model(model_args)
    train_records = extract_records(model, train_windows, window_size=args.window_size, batch_size=args.batch_size, feature_space=args.feature_space)
    eval_records = extract_records(model, eval_windows, window_size=args.window_size, batch_size=args.batch_size, feature_space=args.feature_space)
    projection = fit_token_projection([record.tokens for record in train_records], dimension=args.pca_dimension, maximum_tokens=args.pca_maximum_tokens, seed=args.seed)
    train_descriptors = describe_records(train_records, projection, covariance_dimension=args.covariance_dimension, radial_bins=args.radial_bins, seed=args.seed)
    eval_descriptors = describe_records(eval_records, projection, covariance_dimension=args.covariance_dimension, radial_bins=args.radial_bins, seed=args.seed)

    multiclass, predictions = evaluate_task(train_descriptors, train_records, eval_descriptors, eval_records, class_indices, classifier_dimension=args.classifier_dimension, classifier_c=args.classifier_c, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    pairwise: dict[str, Any] = {}
    for pair_index, pair in enumerate(PAIR_PRESETS):
        if all(name in requested_names for name in pair):
            pair_indices = tuple(lookup[name] for name in pair)
            result, _ = evaluate_task(train_descriptors, train_records, eval_descriptors, eval_records, pair_indices, classifier_dimension=args.classifier_dimension, classifier_c=args.classifier_c, bootstrap_samples=args.bootstrap_samples, seed=args.seed + 1000 * (pair_index + 1))
            pairwise[" vs ".join(pair)] = result

    output_dir = Path(args.output_dir).expanduser().resolve()
    payload = {
        "status": "complete",
        "purpose": "Pre-registered GT-region diagnostic; not a deployable OVSS score. Evaluation labels only select high-purity windows and score fixed source-fitted probes.",
        "protocol": {
            "source_dataset": "OpenEarthMap",
            "source_split": args.source_train_split,
            "evaluation_dataset": eval_dataset_name,
            "evaluation_split": args.source_eval_split if args.eval_dataset == "oem" else args.loveda_subset,
            "classes": list(requested_names),
            "target_labels_used_for_any_fit_or_selection": False,
            "source_labels_used_for_pca_or_descriptor_fitting": False,
            "source_labels_used_for_linear_probe": True,
            "group_unit": "original image path",
            "image_group_bootstrap": "class-stratified; crops are not treated as independent",
        },
        "sampling": {
            "window_size": args.window_size,
            "window_stride": args.window_stride,
            "minimum_purity": args.minimum_purity,
            "max_windows_per_image_class": args.max_windows_per_image_class,
            "max_windows_per_class": args.max_windows_per_class,
            "minimum_windows_per_class": args.minimum_windows_per_class,
            "source_train_windows": len(train_records),
            "evaluation_windows": len(eval_records),
            "source_train_class_counts": record_counts(train_records),
            "evaluation_class_counts": record_counts(eval_records),
            "source_train_images": len({record.group for record in train_records}),
            "evaluation_images": len({record.group for record in eval_records}),
            "selection": "all original images scanned before deterministic class-wise cap; at most one window per image/class",
        },
        "dino": {"feature_space": args.feature_space, "pca": {"fit": "source-train tokens only, label-free", "dimension": projection.dimension, "explained_variance_ratio_sum": round(float(projection.explained_variance_ratio.sum()), 6)}},
        "controls": {
            "conditions": list(CONDITIONS),
            "handcrafted": "fixed RGB moments+histogram, 8-neighbour LBP histogram, and 4x4x9 HOG; no learned parameters",
            "spatial_shuffle": "deterministic token-position shuffle in source train and evaluation; preserves token means/covariance",
            "capacity": f"Every probe receives {args.classifier_dimension} features and has the same logistic-regression parameter count for a task.",
        },
        "classifier": {"type": "standardized logistic regression", "C": args.classifier_c, "class_weight": "balanced", "feature_dimension": args.classifier_dimension, "selection": "all settings fixed before evaluation"},
        "bootstrap": {"samples": args.bootstrap_samples, "metric": "balanced-accuracy difference"},
        "checkpoint": checkpoint_manifest,
        "multiclass": multiclass,
        "pairwise": pairwise,
    }
    write_json(output_dir / "results.json", payload)
    np.savez_compressed(output_dir / "multiclass_predictions.npz", target=np.asarray([record.label for record in eval_records]), groups=np.asarray([record.group for record in eval_records]), **predictions)
    print(json.dumps({"status": "complete", "output": str(output_dir / "results.json"), "multiclass": multiclass["results"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
