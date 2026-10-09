from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .config import CheckpointConfig
from .inference import pil_to_tensor
from .model import DINOTextSegmenter, checkpoint_manifest
from .palette import indexed_image, make_palette
from .prompts import ClassSpec, serialize_class_specs


Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class FastDenseConfig:
    """Latency-oriented settings for one-pass DINO dense OVSS."""

    input_resolution: int = 512
    output_temperature: float = 0.07
    confidence_threshold: float | None = None
    image_cache_size: int = 1

    def validate(self, patch_size: int = DINOTextSegmenter.patch_size) -> None:
        if self.input_resolution < patch_size or self.input_resolution % patch_size:
            raise ValueError(f"input_resolution must be divisible by patch size {patch_size}.")
        if self.output_temperature <= 0:
            raise ValueError("output_temperature must be positive.")
        if self.confidence_threshold is not None and not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1].")
        if self.image_cache_size < 0:
            raise ValueError("image_cache_size must be non-negative.")


@dataclass
class _DenseImageEntry:
    patch_features: torch.Tensor
    resized_size: tuple[int, int]


class FastDenseSession:
    """Persistent, SAM3-free DINO dense open-vocabulary segmentation.

    The image encoder runs once per image. All requested classes share one
    text-embedding batch and one ``[patches] x [classes]`` similarity product.
    This is intentionally a separate path from the higher-quality SAM3+DINO
    worker: there is no mutable SAM3 prompt state, region verifier, sliding
    window, TLP, or per-class decoder call.
    """

    def __init__(
        self,
        checkpoints: CheckpointConfig,
        config: FastDenseConfig | None = None,
        *,
        device: str = "cuda",
        amp: bool = True,
        cost_aggregation_checkpoint: str | Path | None = None,
    ) -> None:
        self.config = config or FastDenseConfig()
        self.config.validate()
        if cost_aggregation_checkpoint:
            # The optional decoder is useful for a quality comparison, but it
            # is deliberately opt-in because its attention blocks add latency.
            from .cost_aggregation import CostAggregatedDINOTextSegmenter

            self.model = CostAggregatedDINOTextSegmenter(
                checkpoints,
                cost_aggregation_checkpoint,
                device=device,
                amp=amp,
            )
            self.quality_decoder = str(Path(cost_aggregation_checkpoint).expanduser().resolve())
        else:
            self.model = DINOTextSegmenter(checkpoints, device=device, amp=amp, use_satellite=False)
            self.quality_decoder = None
        self._text_cache: dict[tuple[tuple[str, tuple[str, ...]], ...], torch.Tensor] = {}
        self._class_text_cache: dict[tuple[str, tuple[str, ...]], torch.Tensor] = {}
        self._image_cache: OrderedDict[str, _DenseImageEntry] = OrderedDict()

    @property
    def device(self) -> torch.device:
        return self.model.device

    def close(self) -> None:
        self._image_cache.clear()
        self._text_cache.clear()
        self._class_text_cache.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def warmup(self) -> None:
        """Initialize model kernels before a persistent worker accepts requests."""

        side = self.config.input_resolution
        image = Image.new("RGB", (side, side), color=(96, 112, 88))
        self.segment(
            image,
            (
                ClassSpec("background", ("background", "other")),
                ClassSpec("terrain", ("terrain", "land cover")),
            ),
        )

    def segment(
        self,
        image: Image.Image,
        classes: Sequence[ClassSpec],
        *,
        image_key: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        normalized_classes, background_added = _normalize_classes(classes)
        started = time.perf_counter()
        entry, image_cache_hit, image_prepare_seconds = self._image_entry(image, image_key)

        routing_started = time.perf_counter()
        text_features, text_cache_hit, text_class_cache_hits = self._text_features(normalized_classes)
        routing_seconds = time.perf_counter() - routing_started

        logits_started = time.perf_counter()
        with torch.inference_mode(), _model_autocast(self.model):
            logits = self.model.similarity_logits(entry.patch_features, text_features).float()
            logits = F.interpolate(
                logits,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )
            probabilities = torch.softmax(logits / self.config.output_temperature, dim=1)
            confidence, labels = probabilities.max(dim=1)
        logits_seconds = time.perf_counter() - logits_started

        confidence_map = confidence[0]
        label_map = labels[0]
        if self.config.confidence_threshold is not None:
            label_map = label_map.clone()
            label_map[confidence_map < self.config.confidence_threshold] = 255
        labels_np = label_map.to(torch.uint8).cpu().numpy()
        confidence_np = confidence_map.mul(255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        metadata: dict[str, object] = {
            "method": "DINOv3 dino.txt one-pass dense OVSS",
            "quality_decoder": getattr(self, "quality_decoder", None),
            "background_added": background_added,
            "input_class_count": len(normalized_classes),
            "patch_grid": [int(entry.patch_features.shape[-2]), int(entry.patch_features.shape[-1])],
            "dino_input_size": [entry.resized_size[0], entry.resized_size[1]],
            "cache": {
                "image_hit": image_cache_hit,
                "text_hit": text_cache_hit,
                "text_class_hits": text_class_cache_hits,
                "image_entries": len(self._image_cache),
            },
            "timing_seconds": {
                "image_prepare": round(image_prepare_seconds, 4),
                "class_text": round(routing_seconds, 4),
                "dense_logits_and_upscale": round(logits_seconds, 4),
                "total": round(time.perf_counter() - started, 4),
            },
        }
        return labels_np, confidence_np, metadata

    def run_file(
        self,
        image_path: str | Path,
        classes: Sequence[ClassSpec],
        output_dir: str | Path,
        *,
        image_key: str | None = None,
    ) -> dict[str, object]:
        request_started = time.perf_counter()
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as source:
            image = source.convert("RGB")
        stat = path.stat()
        resolved_key = image_key or f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        labels, confidence, diagnostics = self.segment(image, classes, image_key=resolved_key)
        normalized_classes, _ = _normalize_classes(classes)

        labels_path = output / "labels.png"
        confidence_path = output / "confidence.png"
        color_path = output / "labels_color.png"
        legend_path = output / "legend.json"
        _save_uint8(labels_path, labels)
        _save_uint8(confidence_path, confidence)
        colors = make_palette(len(normalized_classes))
        indexed_image(labels, colors).save(color_path)
        legend = {
            str(index): {"name": spec.name, "color": list(colors[index])}
            for index, spec in enumerate(normalized_classes)
        }
        legend["255"] = {"name": "unknown", "color": [0, 0, 0]}
        legend_path.write_text(json.dumps(legend, indent=2, ensure_ascii=True), encoding="utf-8")

        timing = diagnostics.get("timing_seconds")
        if isinstance(timing, dict):
            timing["end_to_end"] = round(time.perf_counter() - request_started, 4)
        result: dict[str, object] = {
            "image": str(path),
            "width": image.width,
            "height": image.height,
            "classes": serialize_class_specs(normalized_classes),
            "class_id_mapping": [
                {"id": index, "name": spec.name}
                for index, spec in enumerate(normalized_classes)
            ],
            "diagnostics": diagnostics,
            "config": {"fast_dense": asdict(self.config)},
            "checkpoints": checkpoint_manifest(self.model.checkpoints),
            "outputs": {
                "labels": str(labels_path),
                "confidence": str(confidence_path),
                "labels_color": str(color_path),
                "legend": str(legend_path),
                "metadata": str(output / "metadata.json"),
            },
            "protocol_note": (
                "Training-free one-pass DINOv3 dense OVSS by default. The optional cost-aggregation decoder, "
                "when supplied, is trained only on external data and remains open-vocabulary."
            ),
        }
        if self.device.type == "cuda":
            result["peak_cuda_memory_mb"] = round(
                torch.cuda.max_memory_allocated(self.device) / (1024 * 1024), 2
            )
        _write_json_atomic(output / "metadata.json", result)
        return result

    def _text_features(self, classes: Sequence[ClassSpec]) -> tuple[torch.Tensor, bool, int]:
        key = tuple((spec.name, spec.synonyms) for spec in classes)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached, True, len(classes)

        missing = [
            (class_key, spec)
            for class_key, spec in zip(key, classes)
            if class_key not in self._class_text_cache
        ]
        class_cache_hits = len(classes) - len(missing)
        if missing:
            encoded = self.model.encode_text([spec for _, spec in missing])
            if encoded.ndim != 2 or encoded.shape[0] != len(missing):
                raise RuntimeError("DINO text encoder returned an unexpected class feature shape.")
            for (class_key, _), feature in zip(missing, encoded):
                self._class_text_cache[class_key] = feature.detach().clone()
        features = torch.stack([self._class_text_cache[class_key] for class_key in key], dim=0)
        self._text_cache[key] = features
        return features, False, class_cache_hits

    def _image_entry(
        self,
        image: Image.Image,
        image_key: str | None,
    ) -> tuple[_DenseImageEntry, bool, float]:
        if image_key is not None and image_key in self._image_cache:
            entry = self._image_cache.pop(image_key)
            self._image_cache[image_key] = entry
            return entry, True, 0.0

        image = image.convert("RGB")
        started = time.perf_counter()
        resized = _resize_for_dino(image, self.config.input_resolution)
        rgb = pil_to_tensor(resized, self.device)
        patch_features, _ = self.model.encode_image(rgb)
        entry = _DenseImageEntry(
            patch_features=patch_features.float(),
            resized_size=(resized.width, resized.height),
        )
        elapsed = time.perf_counter() - started
        if image_key is not None and self.config.image_cache_size:
            self._image_cache[image_key] = entry
            while len(self._image_cache) > self.config.image_cache_size:
                self._image_cache.popitem(last=False)
        return entry, False, elapsed


def _normalize_classes(classes: Sequence[ClassSpec]) -> tuple[tuple[ClassSpec, ...], bool]:
    if not classes:
        raise ValueError("At least one foreground class is required.")
    background = [spec for spec in classes if spec.name.casefold() == "background"]
    if len(background) > 1:
        raise ValueError("Only one class may be named 'background'.")
    foreground = [spec for spec in classes if spec.name.casefold() != "background"]
    if not foreground:
        raise ValueError("At least one non-background class is required.")
    if background:
        return tuple(background + foreground), False
    return (ClassSpec("background", ("background", "other")), *foreground), True


def _resize_for_dino(image: Image.Image, target_long_side: int) -> Image.Image:
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("DINO dense inference requires a non-empty image.")
    scale = target_long_side / max(width, height)
    resized_width = max(16, int(round(width * scale / 16.0)) * 16)
    resized_height = max(16, int(round(height * scale / 16.0)) * 16)
    if (resized_width, resized_height) == image.size:
        return image
    return image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)


def _model_autocast(model: object):
    use_amp = bool(getattr(model, "use_amp", False))
    device = getattr(model, "device", None)
    if use_amp and isinstance(device, torch.device) and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _save_uint8(path: Path, values: np.ndarray) -> None:
    if values.ndim != 2 or values.dtype != np.uint8:
        raise ValueError("OVSS output must be a two-dimensional uint8 array.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.png")
    Image.fromarray(values, mode="L").save(temporary)
    temporary.replace(path)


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.json")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)
