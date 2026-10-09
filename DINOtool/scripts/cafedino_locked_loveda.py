#!/usr/bin/env python3
"""Evaluate the published CAFe-DINO checkpoint on an untouched LoveDA split.

This is an adapter around the official CAFe-DINO implementation, not a new
decoder. It fixes every inference setting before loading LoveDA masks:

* official 224-pixel windows with 112-pixel stride;
* the paper's 512-pixel resized validation protocol by default;
* either paper-primary foreground-only evaluation or the supplemental
  background-inclusive protocol;
* a caller-supplied background threshold only when background is included,
  never selected on LoveDA labels;
* the official prompt vocabulary (including ``tree`` and ``farm``).

For controlled vocabulary-sensitivity experiments, ``--vocabulary-config``
can replace that fixed one-prompt-per-class vocabulary. The default remains
bit-for-bit the original prompt construction. A configuration can either
average synonymous prompts *within* each semantic class or make every prompt
an independent competing label and merge their probabilities back to the
seven LoveDA classes. The latter intentionally measures interference from a
large dynamic vocabulary rather than prompt ensembling.

The published checkpoint contains the full CAFe-DINO model state. The adapter
therefore initializes the official architecture without network downloads and
then requires every checkpoint key to load exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


OFFICIAL_SOURCE_COMMIT = "f6704c1b76137543328567c8f8b49a2dc318824c"
OFFICIAL_WEIGHT_REVISION = "eb13e051ca4d5729415e47fd7e069871cda2711a"
LOVE_DA_CLASS_NAMES = (
    "background",
    "building",
    "road",
    "water",
    "barren",
    "tree",
    "farm",
)
LOVE_DA_FOREGROUND_CLASS_NAMES = LOVE_DA_CLASS_NAMES[1:]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class LoveDASample:
    key: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class PromptVocabulary:
    """Prompt aliases associated with immutable LoveDA semantic classes."""

    aggregation: str
    class_names: tuple[str, ...]
    prompts_by_class: tuple[tuple[str, ...], ...]
    source_path: str

    @property
    def prompt_count(self) -> int:
        return sum(len(prompts) for prompts in self.prompts_by_class)

    def record(self) -> dict[str, Any]:
        return {
            "aggregation": self.aggregation,
            "class_names": list(self.class_names),
            "prompts_by_class": [list(prompts) for prompts in self.prompts_by_class],
            "prompt_count": self.prompt_count,
            "source_path": self.source_path,
        }


class ConfusionMatrix:
    def __init__(self, class_names: tuple[str, ...]) -> None:
        self.class_names = class_names
        self.matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        self.ignored_pixels = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shape {prediction.shape} does not match target shape {target.shape}."
            )
        valid = (target >= 0) & (target < len(self.class_names))
        self.ignored_pixels += int((~valid).sum())
        if not bool(valid.any()):
            return
        predicted = prediction[valid]
        if np.any((predicted < 0) | (predicted >= len(self.class_names))):
            raise ValueError("Prediction contains an invalid class ID on a labeled LoveDA pixel.")
        indices = target[valid].astype(np.int64) * len(self.class_names) + predicted.astype(np.int64)
        self.matrix += np.bincount(indices, minlength=self.matrix.size).reshape(self.matrix.shape)

    def summary(self) -> dict[str, Any]:
        intersection = np.diag(self.matrix)
        target_pixels = self.matrix.sum(axis=1)
        predicted_pixels = self.matrix.sum(axis=0)
        union = target_pixels + predicted_pixels - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.full(len(self.class_names), np.nan, dtype=np.float64),
            where=union > 0,
        )
        labeled_pixels = int(target_pixels.sum())
        pixel_accuracy = float(intersection.sum() / labeled_pixels) if labeled_pixels else float("nan")
        return {
            "mean_iou": float(np.nanmean(iou)),
            "mean_iou_percent": round(float(np.nanmean(iou)) * 100.0, 4),
            "pixel_accuracy": pixel_accuracy,
            "pixel_accuracy_percent": round(pixel_accuracy * 100.0, 4),
            "labeled_pixels": labeled_pixels,
            "ignored_pixels": self.ignored_pixels,
            "per_class": [
                {
                    "id": index,
                    "name": name,
                    "iou": None if np.isnan(iou[index]) else float(iou[index]),
                    "iou_percent": None if np.isnan(iou[index]) else round(float(iou[index]) * 100.0, 4),
                    "intersection_pixels": int(intersection[index]),
                    "union_pixels": int(union[index]),
                    "target_pixels": int(target_pixels[index]),
                    "predicted_pixels": int(predicted_pixels[index]),
                }
                for index, name in enumerate(self.class_names)
            ],
            "confusion_matrix": self.matrix.tolist(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", required=True, help="Pinned official DINO_Soars source checkout.")
    parser.add_argument("--checkpoint", required=True, help="Published or externally selected CAFe-DINO checkpoint.")
    parser.add_argument("--bpe-path", required=True, help="Local DINO BPE vocabulary used by DINO.text.")
    parser.add_argument("--data-root", required=True, help="LoveDA validation directory containing images/ and masks/.")
    parser.add_argument("--output-dir", required=True, help="Fresh or resumable output directory.")
    parser.add_argument(
        "--vocabulary-config",
        help=(
            "Optional JSON vocabulary experiment. Its classes must be the seven LoveDA classes in the official "
            "order, with nonempty prompt lists. aggregation is 'mean' or 'compete'."
        ),
    )
    parser.add_argument(
        "--background-threshold",
        type=float,
        default=0.6,
        help="Frozen probability below which a pixel becomes background. Used only with --include-background.",
    )
    background_group = parser.add_mutually_exclusive_group()
    background_group.add_argument(
        "--include-background",
        dest="include_background",
        action="store_true",
        help="Use the supplemental seven-class paper protocol with a background prompt.",
    )
    background_group.add_argument(
        "--exclude-background",
        dest="include_background",
        action="store_false",
        help="Use the paper's primary six-class protocol: ignore background labels and omit its prompt.",
    )
    parser.set_defaults(include_background=True)
    parser.add_argument(
        "--evaluation-size",
        type=int,
        default=512,
        help="Square evaluation size. 512 reproduces the paper's strided script; 0 keeps native resolution.",
    )
    parser.add_argument(
        "--model-mode",
        choices=("eval", "official-script"),
        default="eval",
        help=(
            "Model mode. 'eval' is the standard deterministic inference mode. "
            "'official-script' preserves the published validate_strided_with_bg.py module modes for reproduction."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PyTorch seed recorded for reproducibility; matches the published validation script by default.",
    )
    parser.add_argument("--window-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--max-images", type=int, default=None, help="Deterministic smoke-test limit.")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.background_threshold <= 1.0:
        raise ValueError("background-threshold must be in [0, 1].")
    if args.evaluation_size != 0 and args.evaluation_size < 16:
        raise ValueError("evaluation-size must be zero or at least 16.")
    if args.window_size < 16 or args.stride < 1:
        raise ValueError("window-size must be at least 16 and stride must be positive.")
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("max-images must be positive when supplied.")


def load_prompt_vocabulary(
    config_path: str | None,
    semantic_class_names: tuple[str, ...],
) -> PromptVocabulary | None:
    """Load and validate an externally supplied, class-preserving prompt bank.

    The config always declares the seven class rows. Foreground-only evaluation
    removes the background row after validation, preventing an accidental
    reordering of the label/metric mapping.
    """

    if config_path is None:
        return None
    path = Path(config_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid vocabulary JSON at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Vocabulary config must be a JSON object.")
    aggregation = payload.get("aggregation", "mean")
    if aggregation not in {"mean", "compete"}:
        raise ValueError("Vocabulary aggregation must be 'mean' or 'compete'.")
    rows = payload.get("classes")
    if not isinstance(rows, list) or len(rows) != len(LOVE_DA_CLASS_NAMES):
        raise ValueError(
            f"Vocabulary config must contain exactly {len(LOVE_DA_CLASS_NAMES)} ordered LoveDA class rows."
        )
    prompts_by_name: dict[str, tuple[str, ...]] = {}
    for expected_name, row in zip(LOVE_DA_CLASS_NAMES, rows):
        if not isinstance(row, dict) or row.get("name") != expected_name:
            raise ValueError(
                "Vocabulary class rows must use the official order: "
                f"{LOVE_DA_CLASS_NAMES}. Expected {expected_name!r}."
            )
        raw_prompts = row.get("prompts", row.get("synonyms"))
        if not isinstance(raw_prompts, list) or not raw_prompts:
            raise ValueError(f"Vocabulary class {expected_name!r} needs a nonempty prompts list.")
        prompts: list[str] = []
        seen: set[str] = set()
        for value in raw_prompts:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Vocabulary class {expected_name!r} has an empty/non-text prompt.")
            prompt = value.strip()
            if prompt.casefold() not in seen:
                prompts.append(prompt)
                seen.add(prompt.casefold())
        if not prompts:
            raise ValueError(f"Vocabulary class {expected_name!r} has no unique text prompts.")
        prompts_by_name[expected_name] = tuple(prompts)
    return PromptVocabulary(
        aggregation=aggregation,
        class_names=semantic_class_names,
        prompts_by_class=tuple(prompts_by_name[name] for name in semantic_class_names),
        source_path=str(path),
    )


def discover_samples(data_root: Path, max_images: int | None) -> list[LoveDASample]:
    image_dir = data_root / "images"
    mask_dir = data_root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected LoveDA images/ and masks/ below {data_root}.")
    images = {path.name: path for path in image_dir.glob("*.png")}
    masks = {path.name: path for path in mask_dir.glob("*.png")}
    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        raise ValueError(
            f"LoveDA image/mask mismatch: {len(missing_masks)} images lack masks and "
            f"{len(missing_images)} masks lack images."
        )
    samples = [LoveDASample(key=name, image_path=images[name], mask_path=masks[name]) for name in sorted(images)]
    if not samples:
        raise ValueError(f"No PNG image/mask pairs found below {data_root}.")
    return samples if max_images is None else samples[:max_images]


def _import_official_modules(official_root: Path) -> tuple[type[torch.nn.Module], Any, Any]:
    cafe_root = official_root / "CAFe_DINO"
    anyup_root = official_root / "anyup"
    if not cafe_root.is_dir() or not anyup_root.is_dir() or not (official_root / "dinov3").is_dir():
        raise FileNotFoundError("official-root does not contain CAFe_DINO/, anyup/, and dinov3/.")
    for path in (str(cafe_root), str(anyup_root), str(official_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from anyup.model import AnyUp
    from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
    from modeling.cafedino import CAFe_DINO

    return CAFe_DINO, AnyUp, dinov3_vitl16_dinotxt_tet1280d20h24l


def build_model(args: argparse.Namespace) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    official_root = Path(args.official_root).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    bpe_path = Path(args.bpe_path).expanduser().resolve()
    if not checkpoint_path.is_file() or not bpe_path.is_file():
        raise FileNotFoundError("checkpoint and bpe-path must both reference existing files.")
    CAFe_DINO, AnyUp, dinotxt_factory = _import_official_modules(official_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    torch.set_float32_matmul_precision("high")

    # The published checkpoint contains all CAFe-DINO parameters, including
    # DINO.text and AnyUp. Initializing without pretraining avoids any network
    # fetch and makes strict checkpoint loading an integrity check.
    backbone, tokenizer = dinotxt_factory(pretrained=False, bpe_path_or_url=str(bpe_path))
    # The upstream validation script loads AnyUp separately and immediately
    # switches it to eval mode before attaching it to the CAFe-DINO module.
    upsampler = AnyUp().eval()
    model = CAFe_DINO(
        backbone,
        tokenizer,
        upsampler,
        input_resolution=(args.window_size // 16, args.window_size // 16),
        device=str(device),
        aggregator_dim=64,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError("CAFe-DINO checkpoint must contain a mapping at key 'model'.")
    clean_state = {
        str(key).replace("_orig_mod.", "", 1): value
        for key, value in state.items()
    }
    incompatible = model.load_state_dict(clean_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Published checkpoint does not match the pinned official architecture. "
            f"Missing={incompatible.missing_keys}; unexpected={incompatible.unexpected_keys}."
        )
    model.to(device).requires_grad_(False)
    if args.model_mode == "eval":
        model.eval()
    # The published validation script never invokes model.eval().  In this
    # compatibility mode preserve the constructor and AnyUp module modes.
    elif args.model_mode != "official-script":
        raise ValueError(f"Unsupported model mode: {args.model_mode}")
    stat = checkpoint_path.stat()
    manifest = {
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "official_weight_revision": OFFICIAL_WEIGHT_REVISION,
        "checkpoint": {"path": str(checkpoint_path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "bpe": {"path": str(bpe_path), "bytes": bpe_path.stat().st_size},
        "device": str(device),
        "checkpoint_model_keys": len(clean_state),
    }
    return model, tokenizer, manifest


def _resize_rgb(image: np.ndarray, evaluation_size: int) -> np.ndarray:
    if evaluation_size == 0:
        return image
    return cv2.resize(image, (evaluation_size, evaluation_size), interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask: np.ndarray, evaluation_size: int) -> np.ndarray:
    if evaluation_size == 0:
        return mask
    return cv2.resize(mask, (evaluation_size, evaluation_size), interpolation=cv2.INTER_NEAREST)


def image_tensor(image_path: Path, evaluation_size: int, device: torch.device) -> torch.Tensor:
    with Image.open(image_path) as source:
        rgb = np.asarray(source.convert("RGB"))
    rgb = np.ascontiguousarray(_resize_rgb(rgb, evaluation_size))
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0).to(device, non_blocking=True)


def target_ids(mask_path: Path, evaluation_size: int, *, include_background: bool) -> np.ndarray:
    with Image.open(mask_path) as source:
        raw = np.asarray(source).copy()
    if raw.ndim == 3:
        if raw.shape[2] < 3 or not np.array_equal(raw[..., :3], np.repeat(raw[..., :1], 3, axis=2)):
            raise ValueError(f"LoveDA mask must be scalar IDs, got shape {raw.shape} at {mask_path}.")
        raw = raw[..., 0]
    raw = _resize_mask(raw, evaluation_size)
    result = np.full(raw.shape, 255, dtype=np.uint8)
    if include_background:
        # This project's LoveDA export uses IDs 1..7 for background through farm.
        valid = (raw >= 1) & (raw <= len(LOVE_DA_CLASS_NAMES))
        result[valid] = raw[valid].astype(np.uint8) - 1
    else:
        # The paper's primary result excludes background.  In the local export,
        # raw ID 1 is background and raw IDs 2..7 are the six foreground classes.
        valid = (raw >= 2) & (raw <= len(LOVE_DA_CLASS_NAMES))
        result[valid] = raw[valid].astype(np.uint8) - 2
    return result


@torch.inference_mode()
def strided_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    text_features: torch.Tensor,
    *,
    window_size: int,
    stride: int,
) -> torch.Tensor:
    _, _, height, width = image.shape
    classes = int(text_features.shape[0])
    scores = torch.zeros((classes, height, width), device=image.device, dtype=torch.float32)
    counts = torch.zeros((height, width), device=image.device, dtype=torch.float32)
    vertical = max(height - window_size + stride - 1, 0) // stride + 1
    horizontal = max(width - window_size + stride - 1, 0) // stride + 1
    for row in range(vertical):
        for column in range(horizontal):
            top = row * stride
            left = column * stride
            bottom = min(top + window_size, height)
            right = min(left + window_size, width)
            top = max(bottom - window_size, 0)
            left = max(right - window_size, 0)
            with torch.autocast(device_type=image.device.type, enabled=image.device.type == "cuda"):
                window_scores = model(image[:, :, top:bottom, left:right], text_features, pre_text_emb=True)
            scores[:, top:bottom, left:right] += window_scores.squeeze(0).float()
            counts[top:bottom, left:right] += 1.0
    return scores / counts.clamp_min(1.0)


@torch.inference_mode()
def build_vocabulary_text_features(
    model: torch.nn.Module,
    class_names: tuple[str, ...],
    vocabulary: PromptVocabulary | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build CAFe-DINO text features and an optional alias-to-class mapping.

    ``None`` deliberately uses the exact original CAFe-DINO code path. For a
    mean vocabulary, each class feature is the arithmetic mean of its prompt
    embeddings, matching the author's multi-list averaging convention. In
    competing mode CAFe-DINO receives every alias separately; the returned
    mapping is applied only after its dense logits become probabilities.
    """

    if vocabulary is None:
        return model.build_text_embeddings([list(class_names)]).to(device), None
    if vocabulary.class_names != class_names:
        raise ValueError("Vocabulary semantic classes do not match the active LoveDA protocol.")
    if vocabulary.aggregation == "mean":
        features = [
            model.build_text_embeddings([[prompt] for prompt in prompts]).squeeze(0)
            for prompts in vocabulary.prompts_by_class
        ]
        return torch.stack(features, dim=0).to(device), None
    flat_prompts: list[str] = []
    parents: list[int] = []
    for parent, prompts in enumerate(vocabulary.prompts_by_class):
        flat_prompts.extend(prompts)
        parents.extend([parent] * len(prompts))
    features = model.build_text_embeddings([flat_prompts]).to(device)
    parent_indices = torch.tensor(parents, device=device, dtype=torch.long)
    return features, parent_indices


@torch.inference_mode()
def predict_sample(
    model: torch.nn.Module,
    text_features: torch.Tensor,
    image: torch.Tensor,
    *,
    window_size: int,
    stride: int,
    background_threshold: float | None,
    prompt_parent_indices: torch.Tensor | None = None,
    semantic_class_count: int | None = None,
) -> np.ndarray:
    logits = strided_inference(
        model,
        image,
        text_features,
        window_size=window_size,
        stride=stride,
    )
    probabilities = torch.softmax(logits, dim=0)
    if prompt_parent_indices is not None:
        if semantic_class_count is None:
            raise ValueError("semantic_class_count is required for a competing vocabulary.")
        if prompt_parent_indices.numel() != probabilities.shape[0]:
            raise ValueError("Competing vocabulary mapping does not match the number of CAFe-DINO prompt logits.")
        merged = torch.zeros(
            (semantic_class_count, *probabilities.shape[1:]),
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        probabilities = merged.index_add(0, prompt_parent_indices, probabilities)
    confidence, labels = probabilities.max(dim=0)
    if background_threshold is not None:
        labels = labels.masked_fill(confidence < background_threshold, 0)
    return labels.to(torch.uint8).cpu().numpy()


def _save_prediction(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.png")
    Image.fromarray(values, mode="L").save(temporary)
    temporary.replace(path)


def _load_prediction(path: Path, expected_shape: tuple[int, int], class_count: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        values = np.asarray(Image.open(path)).copy()
    except OSError:
        return None
    if values.ndim != 2 or values.shape != expected_shape or np.any(values >= class_count):
        return None
    return values.astype(np.uint8, copy=False)


def _json_digest(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    samples = discover_samples(data_root, args.max_images)
    class_names = LOVE_DA_CLASS_NAMES if args.include_background else LOVE_DA_FOREGROUND_CLASS_NAMES
    vocabulary = load_prompt_vocabulary(args.vocabulary_config, class_names)
    background_threshold = args.background_threshold if args.include_background else None
    ground_truth_mapping = (
        "raw IDs 1..7 map to class IDs 0..6; raw 0 and 255 are ignored"
        if args.include_background
        else "raw IDs 2..7 map to foreground class IDs 0..5; raw 0, 1 (background), and 255 are ignored"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions"
    signature = {
        "method": "official CAFe-DINO published checkpoint",
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "official_weight_revision": OFFICIAL_WEIGHT_REVISION,
        "data_root": str(data_root),
        "sample_count": len(samples),
        "sample_keys_sha256": _json_digest(sample.key for sample in samples),
        "classes": list(class_names),
        "vocabulary": (
            {
                "aggregation": "official-single-prompt",
                "class_names": list(class_names),
                "prompts_by_class": [[name] for name in class_names],
                "prompt_count": len(class_names),
                "source_path": None,
            }
            if vocabulary is None
            else vocabulary.record()
        ),
        "include_background": args.include_background,
        "evaluation_size": args.evaluation_size,
        "window_size": args.window_size,
        "stride": args.stride,
        "background_threshold": background_threshold,
        "model_mode": args.model_mode,
        "seed": args.seed,
        "ground_truth_mapping": ground_truth_mapping,
        "protocol": (
            "Masks are read only after each prediction is persisted, for final metric accumulation; no LoveDA "
            "label affects model loading or inference. "
            + (
                "The background threshold is fixed before predictions."
                if args.include_background
                else "Background labels and the background prompt are excluded, matching the paper's primary protocol."
            )
        ),
    }
    signature_path = output_dir / "signature.json"
    if signature_path.is_file():
        if json.loads(signature_path.read_text(encoding="utf-8")) != signature:
            raise ValueError("Existing output directory has a different locked CAFe-DINO evaluation signature.")
    else:
        _write_json(signature_path, signature)

    model, tokenizer, checkpoints = build_model(args)
    # The upstream validation script seeds immediately after model creation.
    # Keeping the same placement matters when reproducing its train-mode path.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = next(model.parameters()).device
    with torch.inference_mode():
        text_features, prompt_parent_indices = build_vocabulary_text_features(
            model,
            class_names,
            vocabulary,
            device,
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    confusion = ConfusionMatrix(class_names)
    new_images = 0
    resumed_images = 0

    for index, sample in enumerate(samples, start=1):
        with Image.open(sample.image_path) as source:
            raw_width, raw_height = source.size
        expected_shape = (
            (args.evaluation_size, args.evaluation_size)
            if args.evaluation_size
            else (raw_height, raw_width)
        )
        prediction_path = prediction_dir / sample.key
        prediction = _load_prediction(prediction_path, expected_shape, len(class_names))
        if prediction is None:
            tensor = image_tensor(sample.image_path, args.evaluation_size, device)
            prediction = predict_sample(
                model,
                text_features,
                tensor,
                window_size=args.window_size,
                stride=args.stride,
                background_threshold=background_threshold,
                prompt_parent_indices=prompt_parent_indices,
                semantic_class_count=len(class_names),
            )
            _save_prediction(prediction_path, prediction)
            new_images += 1
        else:
            resumed_images += 1

        # This is deliberately after prediction persistence. Target labels only
        # contribute to the immutable benchmark metric.
        confusion.update(
            prediction,
            target_ids(sample.mask_path, args.evaluation_size, include_background=args.include_background),
        )
        if index % max(args.progress_every, 1) == 0 or index == len(samples):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            _write_json(
                output_dir / "results.json",
                {
                    "status": "complete" if index == len(samples) else "running",
                    "dataset": {
                        "name": "LoveDA",
                        "split": "validation",
                        "data_root": str(data_root),
                        "total_images": len(samples),
                        "processed_images": index,
                    },
                    "method": signature["method"],
                    "metrics": confusion.summary(),
                    "config": signature,
                    "checkpoints": checkpoints,
                    "timing": {
                        "wall_seconds": round(time.perf_counter() - started, 4),
                        "new_images": new_images,
                        "resumed_images": resumed_images,
                    },
                    "outputs": {
                        "root": str(output_dir),
                        "predictions": str(prediction_dir),
                        "signature": str(signature_path),
                    },
                    "peak_cuda_memory_mb": (
                        round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
                        if device.type == "cuda"
                        else None
                    ),
                },
            )
            print(
                json.dumps(
                    {
                        "processed": index,
                        "total": len(samples),
                        "new_images": new_images,
                        "mean_iou_percent": confusion.summary()["mean_iou_percent"],
                    }
                ),
                flush=True,
            )
    return json.loads((output_dir / "results.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    run(parse_args())
