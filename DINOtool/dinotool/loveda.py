from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import GSUPConfig, InferenceConfig, TLPConfig, dataclass_dict
from .gsup import GaussianSplatUpsampler
from .inference import (
    ProbabilityAccumulator,
    global_anchor_weight,
    hann_blend_window,
    pad_tile,
    pil_to_tensor,
    tile_starts,
)
from .model import DINOTextSegmenter
from .ov_adapter import model_checkpoint_manifest
from .prompts import ClassSpec, serialize_class_specs
from .raster import RasterSource
from .tlp import text_aware_laplacian_propagation


LOVEDA_CLASS_NAMES: tuple[str, ...] = (
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agricultural",
)
SUPPORTED_MODES = ("baseline", "tlp", "dinosplat", "dinosplat-sat")


@dataclass(frozen=True)
class LoveDASample:
    image_path: Path
    mask_path: Path
    key: str


def loveda_labeled_split(samples: Sequence[LoveDASample]) -> str:
    """Return the split represented by a local labeled LoveDA mirror."""

    observed: set[str] = set()
    for sample in samples:
        parts = {part.casefold() for part in Path(sample.key).parts}
        if "train" in parts:
            observed.add("train")
        if "val" in parts or "validation" in parts:
            observed.add("validation")
    if len(observed) == 1:
        return next(iter(observed))
    return "unknown" if not observed else "mixed"


def default_loveda_classes() -> list[ClassSpec]:
    synonyms = (
        ("background", "other land cover", "miscellaneous terrain"),
        ("building", "rooftop", "house"),
        ("road", "street", "paved roadway"),
        ("water", "river", "lake", "pond"),
        ("barren", "barren land", "bare soil", "exposed earth"),
        ("forest", "woodland", "trees"),
        ("agricultural", "agricultural land", "farmland", "cropland", "cultivated field"),
    )
    return [ClassSpec(name, aliases) for name, aliases in zip(LOVEDA_CLASS_NAMES, synonyms)]


def discover_loveda_samples(data_root: str | Path) -> list[LoveDASample]:
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    pairs: list[tuple[Path, Path]] = []
    for image_dir in sorted(path for path in root.rglob("images_png") if path.is_dir()):
        mask_dir = image_dir.parent / "masks_png"
        if mask_dir.is_dir():
            pairs.append((image_dir, mask_dir))
    if not pairs:
        for image_dir in sorted(path for path in root.rglob("images") if path.is_dir()):
            mask_dir = image_dir.parent / "masks"
            if mask_dir.is_dir():
                pairs.append((image_dir, mask_dir))
    if not pairs:
        raise ValueError(
            f"No LoveDA image/mask directories found below {root}. Expected "
            "images_png + masks_png or images + masks."
        )

    samples: list[LoveDASample] = []
    seen_keys: set[str] = set()
    for image_dir, mask_dir in pairs:
        images = {path.name: path for path in image_dir.glob("*.png")}
        masks = {path.name: path for path in mask_dir.glob("*.png")}
        missing_masks = sorted(images.keys() - masks.keys())
        missing_images = sorted(masks.keys() - images.keys())
        if missing_masks or missing_images:
            raise ValueError(
                f"Unpaired LoveDA files in {image_dir.parent}: "
                f"{len(missing_masks)} images without masks, {len(missing_images)} masks without images."
            )
        parent = image_dir.parent.relative_to(root)
        for name in sorted(images, key=_natural_name_key):
            key = (parent / name).as_posix() if parent.parts else name
            if key in seen_keys:
                raise ValueError(f"Duplicate LoveDA sample key: {key}")
            seen_keys.add(key)
            samples.append(LoveDASample(images[name], masks[name], key))
    if not samples:
        raise ValueError(f"No paired LoveDA PNG files found below {root}.")
    return sorted(samples, key=lambda sample: _natural_name_key(sample.key))


class LoveDAConfusionMatrix:
    def __init__(self) -> None:
        self.matrix = np.zeros((len(LOVEDA_CLASS_NAMES), len(LOVEDA_CLASS_NAMES)), dtype=np.int64)
        self.ignored_pixels = 0

    def update(self, prediction: np.ndarray, raw_target: np.ndarray) -> None:
        target = _target_to_ids(raw_target)
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction shape {prediction.shape} does not match target {target.shape}.")
        invalid_target = ~np.isin(target, np.array((0, 1, 2, 3, 4, 5, 6, 7, 255)))
        if invalid_target.any():
            values = np.unique(target[invalid_target]).tolist()
            raise ValueError(f"Unexpected LoveDA ground-truth IDs: {values}")
        valid = (target >= 1) & (target <= len(LOVEDA_CLASS_NAMES))
        invalid_prediction = valid & ((prediction < 0) | (prediction >= len(LOVEDA_CLASS_NAMES)))
        if invalid_prediction.any():
            values = np.unique(prediction[invalid_prediction]).tolist()
            raise ValueError(f"Unexpected prediction IDs on labeled pixels: {values}")
        self.ignored_pixels += int((~valid).sum())
        mapped_target = target[valid].astype(np.int64) - 1
        mapped_prediction = prediction[valid].astype(np.int64)
        bins = mapped_target * len(LOVEDA_CLASS_NAMES) + mapped_prediction
        self.matrix += np.bincount(bins, minlength=self.matrix.size).reshape(self.matrix.shape)

    def summary(self) -> dict[str, object]:
        intersection = np.diag(self.matrix)
        target_pixels = self.matrix.sum(axis=1)
        predicted_pixels = self.matrix.sum(axis=0)
        union = target_pixels + predicted_pixels - intersection
        valid_classes = union > 0
        iou = np.divide(
            intersection,
            union,
            out=np.full(intersection.shape, np.nan, dtype=np.float64),
            where=valid_classes,
        )
        labeled_pixels = int(target_pixels.sum())
        pixel_accuracy = float(intersection.sum() / labeled_pixels) if labeled_pixels else float("nan")
        mean_iou = float(np.nanmean(iou)) if valid_classes.any() else float("nan")
        foreground_iou = iou[1:]
        valid_foreground = valid_classes[1:]
        foreground_mean_iou = (
            float(np.nanmean(foreground_iou)) if valid_foreground.any() else float("nan")
        )
        frequency = np.divide(
            target_pixels,
            labeled_pixels,
            out=np.zeros(target_pixels.shape, dtype=np.float64),
            where=labeled_pixels > 0,
        )
        frequency_weighted_iou = float(np.nansum(frequency * iou))
        per_class = []
        for index, name in enumerate(LOVEDA_CLASS_NAMES):
            value = None if not valid_classes[index] else float(iou[index])
            per_class.append(
                {
                    "id": index,
                    "name": name,
                    "iou": value,
                    "iou_percent": None if value is None else round(value * 100.0, 4),
                    "intersection_pixels": int(intersection[index]),
                    "union_pixels": int(union[index]),
                    "target_pixels": int(target_pixels[index]),
                    "predicted_pixels": int(predicted_pixels[index]),
                }
            )
        return {
            "mean_iou": mean_iou,
            "mean_iou_percent": round(mean_iou * 100.0, 4),
            "foreground_mean_iou": foreground_mean_iou,
            "foreground_mean_iou_percent": round(foreground_mean_iou * 100.0, 4),
            "pixel_accuracy": pixel_accuracy,
            "pixel_accuracy_percent": round(pixel_accuracy * 100.0, 4),
            "frequency_weighted_iou": frequency_weighted_iou,
            "frequency_weighted_iou_percent": round(frequency_weighted_iou * 100.0, 4),
            "labeled_pixels": labeled_pixels,
            "ignored_pixels": self.ignored_pixels,
            "per_class": per_class,
            "confusion_matrix": self.matrix.tolist(),
        }


class LoveDABenchmark:
    def __init__(
        self,
        model: DINOTextSegmenter,
        inference_config: InferenceConfig,
        tlp_config: TLPConfig,
        gsup_config: GSUPConfig,
        modes: Sequence[str],
    ) -> None:
        ordered_modes = tuple(dict.fromkeys(modes))
        if not ordered_modes or any(mode not in SUPPORTED_MODES for mode in ordered_modes):
            raise ValueError(f"LoveDA modes must be selected from {SUPPORTED_MODES}.")
        inference_config.validate(model.patch_size)
        tlp_config.validate()
        gsup_config.validate()
        if "dinosplat-sat" in ordered_modes and model.satellite is None:
            raise ValueError("dinosplat-sat requires a model with the SAT-493M encoder loaded.")
        self.model = model
        self.inference_config = inference_config
        self.tlp_config = tlp_config
        self.gsup_config = gsup_config
        self.modes = ordered_modes
        self.upsampler = (
            GaussianSplatUpsampler(gsup_config)
            if any(mode in {"dinosplat", "dinosplat-sat"} for mode in ordered_modes)
            else None
        )

    def run(
        self,
        data_root: str | Path,
        classes: Sequence[ClassSpec],
        output_dir: str | Path,
        max_images: int | None = None,
        progress_every: int = 10,
    ) -> dict[str, object]:
        if tuple(spec.name for spec in classes) != LOVEDA_CLASS_NAMES:
            raise ValueError(f"LoveDA class order must be exactly {LOVEDA_CLASS_NAMES}.")
        samples = discover_loveda_samples(data_root)
        if max_images is not None:
            if max_images < 1:
                raise ValueError("max_images must be positive when provided.")
            samples = samples[:max_images]
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / ".accumulators").mkdir(exist_ok=True)
        signature = self._signature(data_root, samples, classes)
        _ensure_signature(output_dir / "benchmark_config.json", signature)

        matrices = {mode: LoveDAConfusionMatrix() for mode in self.modes}
        diagnostics = _BenchmarkDiagnostics()
        result_path = output_dir / "results.json"
        started = time.perf_counter()
        resumed_images = 0
        new_images = 0
        text_features = self.model.encode_text(classes)
        if self.model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.model.device)

        for index, sample in enumerate(samples, start=1):
            with Image.open(sample.mask_path) as target_image:
                raw_target = np.asarray(target_image).copy()
            target_shape = _target_to_ids(raw_target).shape
            predictions: dict[str, np.ndarray] = {}
            for mode in self.modes:
                prediction_path = output_dir / "predictions" / mode / sample.key
                prediction = _load_prediction(prediction_path, target_shape)
                if prediction is not None:
                    predictions[mode] = prediction
            if len(predictions) == len(self.modes):
                resumed_images += 1
            else:
                generated, image_diagnostics = self._predict_image(
                    sample.image_path,
                    text_features,
                    output_dir / ".accumulators",
                )
                diagnostics.add(image_diagnostics)
                new_images += 1
                predictions = generated
                for mode, prediction in predictions.items():
                    if prediction.shape != target_shape:
                        raise ValueError(
                            f"Prediction shape {prediction.shape} for {sample.key} does not match {target_shape}."
                        )
                    _save_prediction(output_dir / "predictions" / mode / sample.key, prediction)
            for mode, prediction in predictions.items():
                matrices[mode].update(prediction, raw_target)

            if index % max(progress_every, 1) == 0 or index == len(samples):
                result = self._make_result(
                    status="running" if index < len(samples) else "complete",
                    data_root=data_root,
                    samples=samples,
                    processed=index,
                    matrices=matrices,
                    diagnostics=diagnostics,
                    elapsed=time.perf_counter() - started,
                    resumed_images=resumed_images,
                    new_images=new_images,
                    output_dir=output_dir,
                    classes=classes,
                )
                _write_json_atomic(result_path, result)
                print(
                    json.dumps(
                        {
                            "processed": index,
                            "total": len(samples),
                            "new_images": new_images,
                            "mIoU_percent": {
                                mode: result["metrics"][mode]["mean_iou_percent"] for mode in self.modes
                            },
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
        return result

    def _predict_image(
        self,
        image_path: Path,
        text_features: torch.Tensor,
        work_dir: Path,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        started = time.perf_counter()
        tile_count = 0
        tlp_calls = 0
        tlp_iterations = 0
        tlp_residual = 0.0
        gsup_calls = 0
        gsup_initial = 0.0
        gsup_final = 0.0
        with RasterSource(image_path, self.inference_config.bands) as source:
            metadata = source.metadata
            global_anchor = self._global_anchor(source) if self.inference_config.use_global_anchor else None
            starts_x = tile_starts(metadata.width, self.inference_config.tile_size, self.inference_config.overlap)
            starts_y = tile_starts(metadata.height, self.inference_config.tile_size, self.inference_config.overlap)
            blend = hann_blend_window(self.inference_config.tile_size)
            with ExitStack() as stack:
                accumulators = {
                    mode: stack.enter_context(
                        ProbabilityAccumulator(
                            len(LOVEDA_CLASS_NAMES),
                            metadata.height,
                            metadata.width,
                            self.inference_config.max_in_memory_mb,
                            work_dir,
                        )
                    )
                    for mode in self.modes
                }
                for top in starts_y:
                    for left in starts_x:
                        actual_w = min(self.inference_config.tile_size, metadata.width - left)
                        actual_h = min(self.inference_config.tile_size, metadata.height - top)
                        tile = source.read_window(left, top, actual_w, actual_h)
                        rgb = pad_tile(pil_to_tensor(tile, self.model.device), self.inference_config.tile_size)
                        patch_features, tile_anchor = self.model.encode_image(rgb)
                        baseline_logits = self.model.similarity_logits(patch_features, text_features)
                        normal_tlp_logits = None
                        satellite_tlp_logits = None
                        if any(mode in {"tlp", "dinosplat"} for mode in self.modes):
                            grid_rgb = F.interpolate(rgb, size=baseline_logits.shape[-2:], mode="area")
                            normal_tlp_logits, tlp_diag = text_aware_laplacian_propagation(
                                baseline_logits,
                                grid_rgb,
                                text_features,
                                self.tlp_config,
                            )
                            tlp_calls += 1
                            tlp_iterations += tlp_diag.iterations
                            tlp_residual += tlp_diag.relative_residual
                        if "dinosplat-sat" in self.modes:
                            grid_rgb = F.interpolate(rgb, size=baseline_logits.shape[-2:], mode="area")
                            structure = self.model.encode_satellite_structure(rgb)
                            satellite_tlp_logits, tlp_diag = text_aware_laplacian_propagation(
                                baseline_logits,
                                grid_rgb,
                                text_features,
                                self.tlp_config,
                                structure,
                            )
                            tlp_calls += 1
                            tlp_iterations += tlp_diag.iterations
                            tlp_residual += tlp_diag.relative_residual

                        parameters = None
                        if self.upsampler is not None:
                            parameters, gsup_diag = self.upsampler.fit(rgb, baseline_logits.shape[-2:])
                            gsup_calls += 1
                            gsup_initial += gsup_diag.initial_reconstruction_l1
                            gsup_final += gsup_diag.final_reconstruction_l1

                        global_weight = global_anchor_weight(
                            tile_anchor,
                            global_anchor,
                            left,
                            top,
                            actual_w,
                            actual_h,
                            metadata.width,
                            metadata.height,
                            temperature=self.inference_config.global_anchor_temperature,
                            spatial_sigma=self.inference_config.global_anchor_sigma,
                        )
                        weights = blend[:actual_h, :actual_w] * global_weight
                        for mode in self.modes:
                            if mode == "baseline":
                                high_logits = F.interpolate(
                                    baseline_logits,
                                    size=rgb.shape[-2:],
                                    mode="bilinear",
                                    align_corners=False,
                                )
                            elif mode == "tlp":
                                assert normal_tlp_logits is not None
                                high_logits = F.interpolate(
                                    normal_tlp_logits,
                                    size=rgb.shape[-2:],
                                    mode="bilinear",
                                    align_corners=False,
                                )
                            elif mode == "dinosplat":
                                assert normal_tlp_logits is not None and parameters is not None
                                high_logits = self.upsampler.upsample(normal_tlp_logits, rgb, parameters)
                            else:
                                assert satellite_tlp_logits is not None and parameters is not None
                                high_logits = self.upsampler.upsample(satellite_tlp_logits, rgb, parameters)
                            probabilities = torch.softmax(
                                high_logits.float() / self.inference_config.output_temperature,
                                dim=1,
                            )
                            accumulators[mode].add(
                                probabilities[0, :, :actual_h, :actual_w].cpu().numpy(),
                                weights,
                                left,
                                top,
                            )
                            del high_logits, probabilities
                        tile_count += 1
                        del rgb, patch_features, baseline_logits
                predictions = {mode: accumulator.finalize(None)[0] for mode, accumulator in accumulators.items()}

        elapsed = time.perf_counter() - started
        return predictions, {
            "images": 1.0,
            "tiles": float(tile_count),
            "inference_seconds": elapsed,
            "tlp_calls": float(tlp_calls),
            "tlp_iterations": float(tlp_iterations),
            "tlp_residual": tlp_residual,
            "gsup_calls": float(gsup_calls),
            "gsup_initial": gsup_initial,
            "gsup_final": gsup_final,
        }

    def _global_anchor(self, source: RasterSource) -> torch.Tensor:
        preview = source.read_preview(self.inference_config.tile_size)
        rgb = pil_to_tensor(preview, self.model.device)
        if rgb.shape[-2:] != (self.inference_config.tile_size, self.inference_config.tile_size):
            rgb = F.interpolate(
                rgb,
                size=(self.inference_config.tile_size, self.inference_config.tile_size),
                mode="bilinear",
                align_corners=False,
            )
        _, anchor = self.model.encode_image(rgb)
        return anchor

    def _signature(
        self,
        data_root: str | Path,
        samples: Sequence[LoveDASample],
        classes: Sequence[ClassSpec],
    ) -> dict[str, object]:
        sample_digest = hashlib.sha256("\n".join(sample.key for sample in samples).encode("utf-8")).hexdigest()
        return {
            "dataset": f"LoveDA {loveda_labeled_split(samples)}",
            "data_root": str(Path(data_root).resolve()),
            "data_provenance": _load_data_provenance(data_root),
            "sample_count": len(samples),
            "sample_keys_sha256": sample_digest,
            "ground_truth_mapping": "0 and 255 ignored; IDs 1..7 mapped to predictions 0..6",
            "classes": serialize_class_specs(classes),
            "modes": list(self.modes),
            "config": {
                "inference": dataclass_dict(self.inference_config),
                "tlp": asdict(self.tlp_config),
                "gsup": asdict(self.gsup_config),
            },
            "checkpoints": model_checkpoint_manifest(self.model),
        }

    def _make_result(
        self,
        status: str,
        data_root: str | Path,
        samples: Sequence[LoveDASample],
        processed: int,
        matrices: dict[str, LoveDAConfusionMatrix],
        diagnostics: "_BenchmarkDiagnostics",
        elapsed: float,
        resumed_images: int,
        new_images: int,
        output_dir: Path,
        classes: Sequence[ClassSpec],
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "status": status,
            "dataset": {
                "name": "LoveDA",
                "split": loveda_labeled_split(samples),
                "data_root": str(Path(data_root).resolve()),
                "total_images": len(samples),
                "processed_images": processed,
                "ground_truth_mapping": "0 and 255 ignored; IDs 1..7 mapped to predictions 0..6",
                "provenance": _load_data_provenance(data_root),
            },
            "method": getattr(self.model, "method_name", "DinoSplat-OV reimplementation"),
            "modes": list(self.modes),
            "classes": serialize_class_specs(classes),
            "metrics": {mode: matrices[mode].summary() for mode in self.modes},
            "timing": {
                "wall_seconds": round(elapsed, 4),
                "new_inference_seconds": round(diagnostics.inference_seconds, 4),
                "new_images": new_images,
                "resumed_images": resumed_images,
                "new_images_per_second": (
                    round(new_images / diagnostics.inference_seconds, 6)
                    if diagnostics.inference_seconds > 0
                    else None
                ),
            },
            "diagnostics": diagnostics.summary(),
            "config": {
                "inference": dataclass_dict(self.inference_config),
                "tlp": asdict(self.tlp_config),
                "gsup": asdict(self.gsup_config),
            },
            "checkpoints": model_checkpoint_manifest(self.model),
            "outputs": {
                "root": str(output_dir),
                "predictions": {mode: str(output_dir / "predictions" / mode) for mode in self.modes},
                "results": str(output_dir / "results.json"),
            },
            "protocol_note": (
                "Local evaluation uses whichever labeled LoveDA split is present in the supplied mirror. "
                "The official hidden validation/test labels are not public and require server-side evaluation. "
                "Predictions are shared-feature, single-scale, Hann-blended sliding-window outputs with no "
                "confidence rejection."
            ),
        }
        if self.model.device.type == "cuda":
            result["peak_cuda_memory_mb"] = round(
                torch.cuda.max_memory_allocated(self.model.device) / (1024 * 1024),
                2,
            )
        return result


class _BenchmarkDiagnostics:
    def __init__(self) -> None:
        self.images = 0
        self.tiles = 0
        self.inference_seconds = 0.0
        self.tlp_calls = 0
        self.tlp_iterations = 0
        self.tlp_residual = 0.0
        self.gsup_calls = 0
        self.gsup_initial = 0.0
        self.gsup_final = 0.0

    def add(self, values: dict[str, float]) -> None:
        for name in (
            "images",
            "tiles",
            "inference_seconds",
            "tlp_calls",
            "tlp_iterations",
            "tlp_residual",
            "gsup_calls",
            "gsup_initial",
            "gsup_final",
        ):
            setattr(self, name, getattr(self, name) + values[name])

    def summary(self) -> dict[str, float | int | None]:
        return {
            "new_images": int(self.images),
            "tiles": int(self.tiles),
            "mean_image_seconds": round(self.inference_seconds / self.images, 4) if self.images else None,
            "mean_tile_seconds": round(self.inference_seconds / self.tiles, 4) if self.tiles else None,
            "mean_tlp_iterations": round(self.tlp_iterations / self.tlp_calls, 4) if self.tlp_calls else None,
            "mean_tlp_relative_residual": round(self.tlp_residual / self.tlp_calls, 8) if self.tlp_calls else None,
            "mean_gsup_initial_l1": round(self.gsup_initial / self.gsup_calls, 8) if self.gsup_calls else None,
            "mean_gsup_final_l1": round(self.gsup_final / self.gsup_calls, 8) if self.gsup_calls else None,
        }


def _target_to_ids(raw_target: np.ndarray) -> np.ndarray:
    if raw_target.ndim == 2:
        return raw_target
    if raw_target.ndim == 3 and raw_target.shape[2] in {3, 4}:
        first = raw_target[..., 0]
        if np.array_equal(raw_target[..., :3], np.repeat(first[..., None], 3, axis=2)):
            return first
    raise ValueError(f"LoveDA masks must contain scalar IDs, got shape {raw_target.shape}.")


def _natural_name_key(value: str) -> tuple[str, int, int | str]:
    stem = Path(value).stem
    if stem.isdigit():
        return (str(Path(value).parent), 0, int(stem))
    return (str(Path(value).parent), 1, value)


def _load_prediction(path: Path, expected_shape: tuple[int, int]) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        prediction = np.asarray(Image.open(path)).copy()
    except (OSError, ValueError):
        return None
    if prediction.ndim != 2 or prediction.shape != expected_shape:
        return None
    return prediction


def _save_prediction(path: Path, prediction: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.png")
    Image.fromarray(prediction.astype(np.uint8), mode="L").save(temporary)
    temporary.replace(path)


def _load_data_provenance(data_root: str | Path) -> dict[str, object] | None:
    root = Path(data_root).resolve()
    candidates = [root / "provenance.json", root.parent / "provenance.json"]
    for path in candidates:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            payload["local_path"] = str(path)
            payload["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            return payload
    return None


def _ensure_signature(path: Path, signature: dict[str, object]) -> None:
    normalized = json.loads(json.dumps(signature, ensure_ascii=True))
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise ValueError(
                f"Existing benchmark configuration at {path} differs from this run. "
                "Use a new output directory to avoid mixing predictions."
            )
        return
    _write_json_atomic(path, normalized)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)
