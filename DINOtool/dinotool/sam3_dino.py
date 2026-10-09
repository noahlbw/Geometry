from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .config import CheckpointConfig, TLPConfig
from .inference import pil_to_tensor
from .loveda import LOVEDA_CLASS_NAMES, LoveDAConfusionMatrix, default_loveda_classes, discover_loveda_samples
from .model import DINOTextSegmenter, checkpoint_manifest
from .prompts import ClassSpec, serialize_class_specs
from .region_verifier import DINORegionVerifier, RegionVerifierConfig
from .sam3 import (
    OFFICIAL_LOVEDA_PROMPTS,
    Sam3Config,
    Sam3GeometryConfig,
    Sam3ImageComponents,
    Sam3PromptComponents,
    SegEarthOV3Predictor,
    _ensure_signature,
    _load_prediction,
    _save_prediction,
    _write_json_atomic,
)
from .tlp import text_aware_laplacian_propagation


@dataclass(frozen=True)
class Sam3PromptBank:
    """Deterministic, constrained SAM3 prompt variants grouped by class."""

    query_words: tuple[str, ...]
    query_class_indices: tuple[int, ...]
    source: str

    def signature(self) -> dict[str, object]:
        return {
            "source": self.source,
            "query_words": list(self.query_words),
            "query_class_indices": list(self.query_class_indices),
        }


@dataclass(frozen=True)
class Sam3DinoFusionConfig:
    """Strictly training-free SAM3-first region-verification configuration."""

    dino_input_resolution: int = 1024
    use_constrained_prompts: bool = False
    maximum_prompts_per_class: int = 5
    use_pgrf: bool = False
    geometry: Sam3GeometryConfig = field(default_factory=lambda: Sam3GeometryConfig(enabled=False))
    region_verifier: RegionVerifierConfig = field(default_factory=RegionVerifierConfig)
    use_dino_tlp: bool = False
    use_satellite_structure: bool = False
    legacy_pixel_fusion: bool = False

    # Retained only to reproduce prior posterior-blending experiments.
    sam3_posterior_temperature: float = 0.25
    dino_posterior_temperature: float = 0.07
    semantic_strength: float = 0.0
    structure_strength: float = 0.0
    uncertainty_power: float = 1.0

    def validate(self) -> None:
        if self.dino_input_resolution < 16 or self.dino_input_resolution % 16:
            raise ValueError("DINO input resolution must be a multiple of 16 and at least 16.")
        if self.maximum_prompts_per_class < 1:
            raise ValueError("maximum_prompts_per_class must be positive.")
        if self.sam3_posterior_temperature <= 0 or self.dino_posterior_temperature <= 0:
            raise ValueError("Fusion posterior temperatures must be positive.")
        if not 0.0 <= self.semantic_strength <= 1.0:
            raise ValueError("DINO semantic_strength must be in [0, 1].")
        if not 0.0 <= self.structure_strength <= 1.0:
            raise ValueError("DINO structure_strength must be in [0, 1].")
        if self.uncertainty_power < 0:
            raise ValueError("DINO uncertainty_power must be non-negative.")
        self.geometry.validate()
        self.region_verifier.validate()

    def signature(self) -> dict[str, object]:
        return asdict(self)


def uncertainty_gated_fuse(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    strength: float,
    uncertainty_power: float,
) -> torch.Tensor:
    """Legacy posterior blend retained for reproducibility only."""

    if reference.ndim != 4 or candidate.shape != reference.shape:
        raise ValueError("Fusion posteriors must have the same BCHW shape.")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("Fusion strength must be in [0, 1].")
    if uncertainty_power < 0:
        raise ValueError("Fusion uncertainty_power must be non-negative.")
    if strength == 0.0:
        return reference
    uncertainty = (1.0 - reference.amax(dim=1, keepdim=True)).clamp_(0.0, 1.0)
    weight = strength * uncertainty.pow(uncertainty_power)
    return reference + weight * (candidate - reference)


class Sam3DinoPredictor:
    """SAM3 masks with optional experiments and DINO region verification.

    SAM3 remains responsible for mask generation. DINOv3 contributes global
    class evidence and region-level semantic verification only; it never
    paints or blends a dense DINO posterior into the final segmentation.
    """

    def __init__(
        self,
        sam3_config: Sam3Config,
        dino_checkpoints: CheckpointConfig,
        fusion_config: Sam3DinoFusionConfig,
        tlp_config: TLPConfig,
    ) -> None:
        fusion_config.validate()
        tlp_config.validate()
        self.sam3 = SegEarthOV3Predictor(sam3_config)
        self.config = sam3_config
        self.fusion_config = fusion_config
        self.tlp_config = tlp_config
        self.device = self.sam3.device
        # An ablation without the region verifier must remain a real SAM3-only
        # execution path.  Do not load DINO weights merely to build its prompt
        # bank in that case.
        self.dino_checkpoints = dino_checkpoints
        self.dino: DINOTextSegmenter | None = None
        self.dino_classes = default_loveda_classes()
        self.text_features: torch.Tensor | None = None
        if fusion_config.region_verifier.enabled or fusion_config.legacy_pixel_fusion:
            self.dino = DINOTextSegmenter(
                dino_checkpoints,
                device=str(self.device),
                amp=sam3_config.amp,
                use_satellite=fusion_config.use_satellite_structure,
            )
            self.text_features = self.dino.encode_text(self.dino_classes)
            if self.text_features.shape[0] != len(LOVEDA_CLASS_NAMES):
                raise RuntimeError("DINO text feature count does not match the LoveDA class contract.")
        self.prompt_bank = _build_prompt_bank(
            self.sam3.query_words,
            self.sam3.query_indices.detach().cpu().tolist(),
            self.dino_classes,
            enabled=fusion_config.use_constrained_prompts,
            maximum_per_class=fusion_config.maximum_prompts_per_class,
        )
        self.region_verifier = DINORegionVerifier(fusion_config.region_verifier)
        self.last_diagnostics: dict[str, object] = {}

    def predict(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        image = image.convert("RGB")
        if self.fusion_config.legacy_pixel_fusion:
            return self._predict_legacy_pixel_fusion(image)

        components = self.sam3.predict_components(
            image,
            query_words=self.prompt_bank.query_words,
            query_class_indices=self.prompt_bank.query_class_indices,
            geometry=self.fusion_config.geometry,
        )
        if self.fusion_config.use_pgrf:
            sam_scores = _pgrf_class_scores(components, len(LOVEDA_CLASS_NAMES))
        else:
            sam_scores = _class_probabilities_from_baseline(components, len(LOVEDA_CLASS_NAMES))
        maximum, labels = sam_scores.max(dim=0)
        region_diagnostics: dict[str, int] | None = None

        if self.fusion_config.region_verifier.enabled:
            if self.text_features is None:
                raise RuntimeError("DINO region verification was enabled without DINO text features.")
            dino_logits, patch_features, anchor = self._dino_region_inputs(image)
            labels, diagnostics = self.region_verifier.verify(
                sam_scores,
                patch_features,
                self.text_features,
                anchor,
                dino_logits=dino_logits,
            )
            region_diagnostics = diagnostics.summary()

        labels = labels.clone()
        labels[maximum < self.config.probability_threshold] = 0
        self.last_diagnostics = {
            "prompt_count": len(self.prompt_bank.query_words),
            "geometry_attempted": components.geometry_attempted,
            "geometry_accepted": components.geometry_accepted,
            "region_verification": region_diagnostics,
            "mode": "sam3-pgrf-dino-region" if self.fusion_config.use_pgrf else "sam3-official-dino-region",
        }
        return (
            labels.to(torch.uint8).cpu().numpy(),
            maximum.mul(255.0).clamp(0, 255).to(torch.uint8).cpu().numpy(),
        )

    def _dino_region_inputs(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dino is None or self.text_features is None:
            raise RuntimeError("DINO region inputs were requested while DINO is not loaded.")
        dino_image = _resize_for_dino(image, self.fusion_config.dino_input_resolution)
        rgb = pil_to_tensor(dino_image, self.dino.device)
        patch_features, anchor = self.dino.encode_image(rgb)
        dino_logits = self.dino.similarity_logits(patch_features, self.text_features)
        if self.fusion_config.use_dino_tlp:
            structure = self.dino.encode_satellite_structure(rgb) if self.dino.satellite is not None else None
            dino_logits, _ = text_aware_laplacian_propagation(
                dino_logits,
                F.interpolate(rgb, size=dino_logits.shape[-2:], mode="area"),
                self.text_features,
                self.tlp_config,
                structure,
            )
        return dino_logits.squeeze(0).float(), patch_features.float(), anchor.squeeze(0).float()

    def _predict_legacy_pixel_fusion(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Old implementation for a controlled historical ablation only."""

        sam3_scores = self.sam3.predict_logits(image).float().clamp(1e-4, 1.0 - 1e-4)
        sam3_probabilities = torch.softmax(
            torch.logit(sam3_scores).unsqueeze(0) / self.fusion_config.sam3_posterior_temperature,
            dim=1,
        )
        fused = sam3_probabilities
        dino_logits, patch_features, _ = self._dino_region_inputs(image)
        if self.fusion_config.semantic_strength > 0.0:
            candidate = torch.softmax(
                F.interpolate(
                    dino_logits.unsqueeze(0),
                    size=sam3_scores.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                / self.fusion_config.dino_posterior_temperature,
                dim=1,
            )
            fused = uncertainty_gated_fuse(
                fused,
                candidate,
                strength=self.fusion_config.semantic_strength,
                uncertainty_power=self.fusion_config.uncertainty_power,
            )
        maximum, labels = fused.squeeze(0).max(dim=0)
        labels[sam3_scores.max(dim=0).values < self.config.probability_threshold] = 0
        self.last_diagnostics = {"mode": "legacy-pixel-fusion", "patch_grid": list(patch_features.shape[-2:])}
        return (
            labels.to(torch.uint8).cpu().numpy(),
            maximum.mul(255.0).clamp(0, 255).to(torch.uint8).cpu().numpy(),
        )


def _build_prompt_bank(
    baseline_words: Sequence[str],
    baseline_indices: Sequence[int],
    classes: Sequence[ClassSpec],
    *,
    enabled: bool,
    maximum_per_class: int,
) -> Sam3PromptBank:
    if len(baseline_words) != len(baseline_indices):
        raise ValueError("Baseline SAM3 prompt words and indices must have matching lengths.")
    grouped: list[list[str]] = [[] for _ in classes]
    for word, class_index in zip(baseline_words, baseline_indices):
        if class_index < 0 or class_index >= len(classes):
            raise ValueError("Baseline SAM3 prompt class index is outside the class specification range.")
        grouped[class_index].append(word)
    output_words: list[str] = []
    output_indices: list[int] = []
    for class_index, spec in enumerate(classes):
        baseline = list(dict.fromkeys(word.strip() for word in grouped[class_index] if word.strip()))
        candidates = list(baseline)
        if enabled:
            candidates.extend(spec.synonyms)
            canonical = spec.name
            candidates.extend(
                (
                    f"aerial view of {canonical}",
                    f"overhead view of {canonical}",
                    f"{canonical} seen from above",
                )
            )
        # The validated SegEarth aliases are always retained.  The configured
        # cap only limits newly introduced aerial prompt variants when a class
        # already has more official aliases than the requested cap.
        unique_candidates = list(dict.fromkeys(word.strip() for word in candidates if word.strip()))
        selected = baseline + [
            candidate
            for candidate in unique_candidates
            if candidate not in baseline
        ][: max(0, maximum_per_class - len(baseline))]
        if not selected:
            raise ValueError(f"No SAM3 prompt remains for class '{spec.name}'.")
        output_words.extend(selected)
        output_indices.extend([class_index] * len(selected))
    return Sam3PromptBank(
        query_words=tuple(output_words),
        query_class_indices=tuple(output_indices),
        source="constrained-aerial" if enabled else "official-segearth",
    )


def _pgrf_prompt_score(components: Sam3PromptComponents) -> torch.Tensor:
    # Sam3Processor exposes both maps after sigmoid despite their historical
    # ``*_logits`` names.  A second sigmoid collapses the score range and
    # invalidates the official 0.5 probability threshold.
    semantic = components.semantic_logits.clamp(0.0, 1.0)
    if components.instance_mask_logits.shape[0]:
        object_scores = components.instance_scores.clamp(0.0, 1.0)
        instance = (components.instance_mask_logits.clamp(0.0, 1.0) * object_scores[:, None, None]).amax(dim=0)
        maximum_instance_score = object_scores.max()
    else:
        instance = torch.zeros_like(semantic)
        maximum_instance_score = semantic.new_zeros(())
    agreement = 1.0 - (semantic - torch.maximum(semantic, instance)).abs()
    residual_gate = (maximum_instance_score * agreement).clamp(0.0, 1.0)
    presence = components.presence_score.clamp(0.0, 1.0)
    return (presence * (semantic + residual_gate * (instance - semantic).clamp_min(0.0))).clamp(0.0, 1.0)


def _pgrf_class_scores(components: Sam3ImageComponents, classes: int) -> torch.Tensor:
    height, width = components.class_logits.shape[-2:]
    scores = torch.zeros((classes, height, width), device=components.class_logits.device)
    for output in components.prompt_outputs:
        scores[output.class_index] = torch.maximum(scores[output.class_index], _pgrf_prompt_score(output))
    return scores


def _class_probabilities_from_baseline(components: Sam3ImageComponents, classes: int) -> torch.Tensor:
    if components.class_logits.shape[0] != classes:
        raise ValueError("SAM3 class component count does not match the expected class count.")
    return components.class_logits.clamp(0.0, 1.0)


class Sam3DinoLoveDABenchmark:
    """Resumable full LoveDA benchmark for SAM3 + DINO region verification."""

    def __init__(self, predictor: Sam3DinoPredictor) -> None:
        self.predictor = predictor

    def _method_name(self) -> str:
        aggregation = "SAM3 PGRF" if self.predictor.fusion_config.use_pgrf else "SAM3 official dual-head"
        return f"{aggregation} + DINOv3 region verification"

    def run(
        self,
        data_root: str | Path,
        output_dir: str | Path,
        *,
        max_images: int | None = None,
        start_index: int = 0,
        progress_every: int = 10,
    ) -> dict[str, object]:
        if max_images is not None and max_images < 1:
            raise ValueError("max_images must be positive when provided.")
        if start_index < 0:
            raise ValueError("start_index must be non-negative.")
        if progress_every < 1:
            raise ValueError("progress_every must be positive.")
        samples = discover_loveda_samples(data_root)
        if start_index >= len(samples):
            raise ValueError(f"start_index {start_index} is outside {len(samples)} LoveDA samples.")
        samples = samples[start_index:]
        if max_images is not None:
            samples = samples[:max_images]
        output_dir = Path(output_dir).expanduser().resolve()
        predictions_dir = output_dir / "predictions" / "sam3-dino-region"
        output_dir.mkdir(parents=True, exist_ok=True)
        _ensure_signature(output_dir / "signature.json", self._signature(data_root, samples, start_index))
        if self.predictor.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.predictor.device)

        started = time.perf_counter()
        matrix = LoveDAConfusionMatrix()
        new_images = 0
        resumed_images = 0
        inference_seconds = 0.0
        diagnostics_totals: dict[str, int] = {}
        for index, sample in enumerate(samples, start=1):
            with Image.open(sample.mask_path) as mask_image:
                target = np.asarray(mask_image).copy()
            prediction_path = predictions_dir / sample.key
            prediction = _load_prediction(prediction_path, target.shape)
            if prediction is None:
                with Image.open(sample.image_path) as image:
                    image_started = time.perf_counter()
                    prediction, _ = self.predictor.predict(image)
                    inference_seconds += time.perf_counter() - image_started
                _save_prediction(prediction_path, prediction)
                _accumulate_region_diagnostics(diagnostics_totals, self.predictor.last_diagnostics)
                new_images += 1
            else:
                resumed_images += 1
            matrix.update(prediction, target)
            if index % progress_every == 0 or index == len(samples):
                _write_json_atomic(
                    output_dir / "results.json",
                    self._result(
                        "running" if index < len(samples) else "complete",
                        data_root,
                        len(samples),
                        index,
                        matrix,
                        new_images,
                        resumed_images,
                        inference_seconds,
                        time.perf_counter() - started,
                        output_dir,
                        diagnostics_totals,
                    ),
                )
        return self._result(
            "complete",
            data_root,
            len(samples),
            len(samples),
            matrix,
            new_images,
            resumed_images,
            inference_seconds,
            time.perf_counter() - started,
            output_dir,
            diagnostics_totals,
        )

    def _signature(
        self,
        data_root: str | Path,
        samples: Sequence[Any],
        start_index: int,
    ) -> dict[str, object]:
        sample_digest = hashlib.sha256("\n".join(sample.key for sample in samples).encode("utf-8")).hexdigest()
        return {
            "method": self._method_name(),
            "dataset": "LoveDA validation",
            "data_root": str(Path(data_root).expanduser().resolve()),
            "sample_count": len(samples),
            "source_start_index": start_index,
            "sample_keys_sha256": sample_digest,
            "ground_truth_mapping": "0 and 255 ignored; IDs 1..7 mapped to predictions 0..6",
            "sam3_prompt_lines": list(OFFICIAL_LOVEDA_PROMPTS),
            "prompt_bank": self.predictor.prompt_bank.signature(),
            "dino_classes": serialize_class_specs(self.predictor.dino_classes),
            "sam3": self.predictor.config.signature(),
            "dino": checkpoint_manifest(self.predictor.dino_checkpoints),
            "fusion": self.predictor.fusion_config.signature(),
            "tlp": asdict(self.predictor.tlp_config),
        }

    def _result(
        self,
        status: str,
        data_root: str | Path,
        total_images: int,
        processed_images: int,
        matrix: LoveDAConfusionMatrix,
        new_images: int,
        resumed_images: int,
        inference_seconds: float,
        wall_seconds: float,
        output_dir: Path,
        diagnostics_totals: dict[str, int],
    ) -> dict[str, object]:
        sam3_config = asdict(self.predictor.config) | {
            "root": str(self.predictor.config.root),
            "checkpoint_path": str(self.predictor.config.checkpoint_path),
            "bpe_path": str(self.predictor.config.bpe_path),
            "class_file": str(self.predictor.config.class_file),
        }
        result: dict[str, object] = {
            "status": status,
            "dataset": {
                "name": "LoveDA",
                "split": "validation",
                "data_root": str(Path(data_root).expanduser().resolve()),
                "total_images": total_images,
                "processed_images": processed_images,
                "ground_truth_mapping": "0 and 255 ignored; IDs 1..7 mapped to predictions 0..6",
            },
            "method": self._method_name(),
            "classes": list(LOVEDA_CLASS_NAMES),
            "sam3_prompt_lines": list(OFFICIAL_LOVEDA_PROMPTS),
            "prompt_bank": self.predictor.prompt_bank.signature(),
            "dino_classes": serialize_class_specs(self.predictor.dino_classes),
            "metrics": matrix.summary(),
            "timing": {
                "wall_seconds": round(wall_seconds, 4),
                "new_inference_seconds": round(inference_seconds, 4),
                "new_images": new_images,
                "resumed_images": resumed_images,
                "mean_new_image_seconds": round(inference_seconds / new_images, 4) if new_images else None,
            },
            "diagnostics": diagnostics_totals,
            "config": {
                "sam3": sam3_config,
                "fusion": self.predictor.fusion_config.signature(),
                "tlp": asdict(self.predictor.tlp_config),
            },
            "checkpoints": {
                "sam3": self.predictor.config.signature(),
                "dino": checkpoint_manifest(self.predictor.dino_checkpoints),
            },
            "outputs": {
                "root": str(output_dir),
                "predictions": str(output_dir / "predictions" / "sam3-dino-region"),
                "results": str(output_dir / "results.json"),
            },
            "protocol_note": (
                "Strictly training-free inference: SAM3 and DINOv3/DINO.text/SAT remain frozen. "
                "DINO is used for class evidence and region verification, not pixel posterior blending. "
                "Ground-truth IDs 0 and 255 are ignored."
            ),
        }
        if self.predictor.device.type == "cuda":
            result["peak_cuda_memory_mb"] = round(
                torch.cuda.max_memory_allocated(self.predictor.device) / (1024 * 1024), 2
            )
        return result


def _accumulate_region_diagnostics(total: dict[str, int], diagnostics: dict[str, object]) -> None:
    total["images"] = total.get("images", 0) + 1
    total["geometry_attempted"] = total.get("geometry_attempted", 0) + int(diagnostics.get("geometry_attempted", 0))
    total["geometry_accepted"] = total.get("geometry_accepted", 0) + int(diagnostics.get("geometry_accepted", 0))
    region = diagnostics.get("region_verification")
    if isinstance(region, dict):
        for key, value in region.items():
            total[f"region_{key}"] = total.get(f"region_{key}", 0) + int(value)


def _resize_for_dino(image: Image.Image, target_long_side: int) -> Image.Image:
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("DINO fusion requires a non-empty image.")
    scale = target_long_side / max(width, height)
    resized_width = max(16, int(round(width * scale / 16.0)) * 16)
    resized_height = max(16, int(round(height * scale / 16.0)) * 16)
    if (resized_width, resized_height) == image.size:
        return image
    return image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
