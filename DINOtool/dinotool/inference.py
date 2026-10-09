from __future__ import annotations

import json
import math
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from .config import GSUPConfig, InferenceConfig, TLPConfig, dataclass_dict
from .gsup import GSUPDiagnostics, GaussianSplatUpsampler
from .model import DINOTextSegmenter
from .ov_adapter import model_checkpoint_manifest
from .palette import colorize, indexed_image, make_palette
from .prompts import ClassSpec, serialize_class_specs
from .raster import RasterMetadata, RasterSource, write_geotiff
from .tlp import TLPDiagnostics, text_aware_laplacian_propagation


Image.MAX_IMAGE_PIXELS = None


class InferenceRunner:
    def __init__(
        self,
        model: DINOTextSegmenter,
        inference_config: InferenceConfig,
        tlp_config: TLPConfig,
        gsup_config: GSUPConfig,
    ) -> None:
        inference_config.validate(model.patch_size)
        tlp_config.validate()
        gsup_config.validate()
        self.model = model
        self.inference_config = inference_config
        self.tlp_config = tlp_config
        self.gsup_config = gsup_config
        self.upsampler = GaussianSplatUpsampler(gsup_config) if inference_config.use_gsup else None

    def run(
        self,
        image_path: str | Path,
        classes: Sequence[ClassSpec],
        output_dir: str | Path,
    ) -> dict[str, object]:
        if not classes:
            raise ValueError("At least one class is required.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        diagnostics = _Diagnostics()
        if self.model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.model.device)

        text_features = self.model.encode_text(classes)
        with RasterSource(image_path, self.inference_config.bands) as source:
            metadata = source.metadata
            global_anchor = self._global_anchor(source) if self.inference_config.use_global_anchor else None
            starts_x = tile_starts(metadata.width, self.inference_config.tile_size, self.inference_config.overlap)
            starts_y = tile_starts(metadata.height, self.inference_config.tile_size, self.inference_config.overlap)
            blend = hann_blend_window(self.inference_config.tile_size)
            with ProbabilityAccumulator(
                len(classes),
                metadata.height,
                metadata.width,
                self.inference_config.max_in_memory_mb,
                output_dir,
            ) as accumulator:
                for top in starts_y:
                    for left in starts_x:
                        actual_w = min(self.inference_config.tile_size, metadata.width - left)
                        actual_h = min(self.inference_config.tile_size, metadata.height - top)
                        tile = source.read_window(left, top, actual_w, actual_h)
                        rgb = pil_to_tensor(tile, self.model.device)
                        rgb = pad_tile(rgb, self.inference_config.tile_size)
                        tile_probabilities, tile_anchor, tlp_diag, gsup_diag, elapsed = self._segment_tile(
                            rgb, text_features
                        )
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
                        accumulator.add(
                            tile_probabilities[0, :, :actual_h, :actual_w].float().cpu().numpy(),
                            weights,
                            left,
                            top,
                        )
                        diagnostics.add(tlp_diag, gsup_diag, elapsed)
                        del tile_probabilities, rgb
                labels, confidence = accumulator.finalize(self.inference_config.confidence_threshold)

            outputs = save_outputs(output_dir, labels, confidence, metadata, source, classes)

        total_seconds = time.perf_counter() - started
        metadata_path = output_dir / "metadata.json"
        outputs["metadata"] = str(metadata_path)
        result: dict[str, object] = {
            "method": getattr(self.model, "method_name", "DinoSplat-OV reimplementation"),
            "mode": self.inference_config.mode,
            "image": str(Path(image_path)),
            "width": metadata.width,
            "height": metadata.height,
            "classes": serialize_class_specs(classes),
            "tiles": diagnostics.tiles,
            "timing_seconds": {
                "tile_inference": round(diagnostics.tile_seconds, 4),
                "total": round(total_seconds, 4),
            },
            "diagnostics": diagnostics.summary(),
            "config": {
                "inference": dataclass_dict(self.inference_config),
                "tlp": asdict(self.tlp_config),
                "gsup": asdict(self.gsup_config),
            },
            "checkpoints": model_checkpoint_manifest(self.model),
            "outputs": outputs,
            "implementation_note": getattr(
                self.model,
                "implementation_note",
                (
                    "TLP follows arXiv:2608.03023 equations 2-7. GSUP follows equations 8-12; "
                    "paper-unspecified optimizer settings are recorded above. dinosplat-sat is an experimental extension."
                ),
            ),
        }
        if self.model.device.type == "cuda":
            result["peak_cuda_memory_mb"] = round(
                torch.cuda.max_memory_allocated(self.model.device) / (1024 * 1024), 2
            )
        metadata_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
        return result

    def _segment_tile(
        self,
        rgb: Tensor,
        text_features: Tensor,
    ) -> tuple[Tensor, Tensor, TLPDiagnostics | None, GSUPDiagnostics | None, float]:
        started = time.perf_counter()
        patch_features, anchor = self.model.encode_image(rgb)
        logits = self.model.similarity_logits(patch_features, text_features)
        tlp_diag = None
        if self.inference_config.use_tlp:
            grid_rgb = F.interpolate(rgb, size=logits.shape[-2:], mode="area")
            structure = None
            if self.inference_config.use_satellite:
                structure = self.model.encode_satellite_structure(rgb)
            logits, tlp_diag = text_aware_laplacian_propagation(
                logits,
                grid_rgb,
                text_features,
                self.tlp_config,
                structure,
            )
        gsup_diag = None
        if self.upsampler is not None:
            parameters, gsup_diag = self.upsampler.fit(rgb, logits.shape[-2:])
            logits = self.upsampler.upsample(logits, rgb, parameters)
        else:
            logits = F.interpolate(logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        probabilities = torch.softmax(logits.float() / self.inference_config.output_temperature, dim=1)
        return probabilities, anchor, tlp_diag, gsup_diag, time.perf_counter() - started

    def _global_anchor(self, source: RasterSource) -> Tensor:
        preview = source.read_preview(self.inference_config.tile_size)
        rgb = pil_to_tensor(preview, self.model.device)
        rgb = pad_to_multiple(rgb, self.model.patch_size)
        _, anchor = self.model.encode_image(rgb)
        return anchor


class ProbabilityAccumulator:
    def __init__(
        self,
        classes: int,
        height: int,
        width: int,
        max_in_memory_mb: int,
        work_dir: Path,
    ) -> None:
        self.classes = classes
        self.height = height
        self.width = width
        required = (classes + 1) * height * width * np.dtype(np.float32).itemsize
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        if required <= max_in_memory_mb * 1024 * 1024:
            self.probabilities = np.zeros((classes, height, width), dtype=np.float32)
            self.normalizer = np.zeros((height, width), dtype=np.float32)
        else:
            self._temporary = tempfile.TemporaryDirectory(prefix=".dinotool-", dir=work_dir)
            root = Path(self._temporary.name)
            self.probabilities = np.memmap(
                root / "probabilities.dat", mode="w+", dtype=np.float32, shape=(classes, height, width)
            )
            self.normalizer = np.memmap(
                root / "normalizer.dat", mode="w+", dtype=np.float32, shape=(height, width)
            )
            self.probabilities[:] = 0
            self.normalizer[:] = 0

    def add(self, probabilities: np.ndarray, weights: np.ndarray, left: int, top: int) -> None:
        height, width = weights.shape
        self.probabilities[:, top : top + height, left : left + width] += probabilities * weights[None]
        self.normalizer[top : top + height, left : left + width] += weights

    def finalize(self, confidence_threshold: float | None,
                 background_index: int = 255) -> tuple[np.ndarray, np.ndarray]:
        labels = np.empty((self.height, self.width), dtype=np.uint8)
        confidence = np.empty((self.height, self.width), dtype=np.uint8)
        rows_per_chunk = max(1, min(1024, (64 * 1024 * 1024) // max(self.classes * self.width * 4, 1)))
        for top in range(0, self.height, rows_per_chunk):
            bottom = min(top + rows_per_chunk, self.height)
            normalizer = np.maximum(self.normalizer[top:bottom], 1e-8)
            probabilities = self.probabilities[:, top:bottom] / normalizer[None]
            class_ids = probabilities.argmax(axis=0).astype(np.uint8)
            confidence_float = probabilities.max(axis=0)
            if confidence_threshold is not None:
                class_ids[confidence_float < confidence_threshold] = background_index
            labels[top:bottom] = class_ids
            confidence[top:bottom] = np.clip(confidence_float * 255.0, 0, 255).astype(np.uint8)
        return labels, confidence

    def close(self) -> None:
        if isinstance(self.probabilities, np.memmap):
            self.probabilities.flush()
        if isinstance(self.normalizer, np.memmap):
            self.normalizer.flush()
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __enter__(self) -> "ProbabilityAccumulator":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class _Diagnostics:
    def __init__(self) -> None:
        self.tiles = 0
        self.tile_seconds = 0.0
        self.tlp_iterations = 0
        self.tlp_residual = 0.0
        self.gsup_initial = 0.0
        self.gsup_final = 0.0

    def add(
        self,
        tlp: TLPDiagnostics | None,
        gsup: GSUPDiagnostics | None,
        elapsed: float,
    ) -> None:
        self.tiles += 1
        self.tile_seconds += elapsed
        if tlp is not None:
            self.tlp_iterations += tlp.iterations
            self.tlp_residual += tlp.relative_residual
        if gsup is not None:
            self.gsup_initial += gsup.initial_reconstruction_l1
            self.gsup_final += gsup.final_reconstruction_l1

    def summary(self) -> dict[str, float]:
        count = max(self.tiles, 1)
        return {
            "mean_tile_seconds": round(self.tile_seconds / count, 4),
            "mean_tlp_iterations": round(self.tlp_iterations / count, 3),
            "mean_tlp_relative_residual": round(self.tlp_residual / count, 7),
            "mean_gsup_initial_l1": round(self.gsup_initial / count, 7),
            "mean_gsup_final_l1": round(self.gsup_final / count, 7),
        }


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def hann_blend_window(size: int) -> np.ndarray:
    axis = np.hanning(size).astype(np.float32)
    axis = np.maximum(axis, 0.05)
    return np.outer(axis, axis).astype(np.float32)


def global_anchor_weight(
    tile_anchor: Tensor,
    global_anchor: Tensor | None,
    left: int,
    top: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
    temperature: float = 0.07,
    spatial_sigma: float = 0.5,
) -> float:
    if global_anchor is None:
        return 1.0
    similarity = float(F.cosine_similarity(tile_anchor, global_anchor).mean().item())
    semantic = math.exp(max(-30.0, min(0.0, (similarity - 1.0) / temperature)))
    center_x = (left + width / 2) / image_width
    center_y = (top + height / 2) / image_height
    distance_squared = (center_x - 0.5) ** 2 + (center_y - 0.5) ** 2
    spatial = math.exp(-distance_squared / (2 * spatial_sigma**2))
    return max(semantic * spatial, 1e-6)


def pil_to_tensor(image: Image.Image, device: torch.device) -> Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0


def pad_tile(rgb: Tensor, tile_size: int) -> Tensor:
    pad_h = tile_size - rgb.shape[-2]
    pad_w = tile_size - rgb.shape[-1]
    if pad_h < 0 or pad_w < 0:
        raise ValueError("Input tile is larger than tile_size.")
    if pad_h or pad_w:
        rgb = F.pad(rgb, (0, pad_w, 0, pad_h), mode="replicate")
    return rgb


def pad_to_multiple(rgb: Tensor, multiple: int) -> Tensor:
    target_h = math.ceil(rgb.shape[-2] / multiple) * multiple
    target_w = math.ceil(rgb.shape[-1] / multiple) * multiple
    if (target_h, target_w) != rgb.shape[-2:]:
        rgb = F.pad(rgb, (0, target_w - rgb.shape[-1], 0, target_h - rgb.shape[-2]), mode="replicate")
    return rgb


def save_outputs(
    output_dir: Path,
    labels: np.ndarray,
    confidence: np.ndarray,
    metadata: RasterMetadata,
    source: RasterSource,
    classes: Sequence[ClassSpec],
) -> dict[str, str]:
    colors = make_palette(len(classes))
    labels_path = output_dir / "labels.png"
    confidence_path = output_dir / "confidence.png"
    color_path = output_dir / "labels_color.png"
    overlay_path = output_dir / "preview_overlay.png"
    indexed_image(labels, colors).save(labels_path)
    Image.fromarray(confidence, mode="L").save(confidence_path)
    color_array = colorize(labels, colors)
    Image.fromarray(color_array, mode="RGB").save(color_path)

    preview = source.read_preview(1600)
    preview_labels = Image.fromarray(color_array, mode="RGB").resize(preview.size, Image.Resampling.NEAREST)
    Image.blend(preview.convert("RGB"), preview_labels, 0.48).save(overlay_path)
    outputs = {
        "labels_png": str(labels_path),
        "confidence_png": str(confidence_path),
        "labels_color_png": str(color_path),
        "preview_overlay_png": str(overlay_path),
    }
    if metadata.is_geotiff:
        labels_tif = output_dir / "labels.tif"
        confidence_tif = output_dir / "confidence.tif"
        write_geotiff(labels_tif, labels, metadata, nodata=255)
        write_geotiff(confidence_tif, confidence, metadata)
        outputs["labels_geotiff"] = str(labels_tif)
        outputs["confidence_geotiff"] = str(confidence_tif)
    legend = {
        str(index): {"name": spec.name, "color": list(colors[index])}
        for index, spec in enumerate(classes)
    }
    legend["255"] = {"name": "unknown", "color": [0, 0, 0]}
    legend_path = output_dir / "legend.json"
    legend_path.write_text(json.dumps(legend, indent=2), encoding="utf-8")
    outputs["legend"] = str(legend_path)
    return outputs
