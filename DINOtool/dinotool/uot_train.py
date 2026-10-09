from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .config import CheckpointConfig
from .model import DINOTextSegmenter, checkpoint_manifest
from .oem import (
    OEM_CLASSES,
    OPEN_VOCABULARY_PRESERVATION_CLASSES,
    OpenEarthMapDataset,
    discover_oem_samples,
    oem_class_histogram,
    oem_source_manifest,
)
from .uot import (
    UOTOutput,
    UOTPrototypeConfig,
    UOTPrototypeDecoder,
    load_uot_checkpoint,
    make_uot_checkpoint,
)
from .ov_train import (
    _SegmentationConfusion,
    _append_json_line,
    _atomic_torch_save,
    _class_balanced_weights,
    _cosine_factor,
    _ensure_disjoint_splits,
    _ensure_signature,
    _make_loader,
    _seed_everything,
    _segmentation_loss,
    _write_json_atomic,
)


@dataclass(frozen=True)
class OemUOTTrainingConfig:
    """External-only training configuration for the UOT open-vocabulary decoder."""

    data_root: str
    output_dir: str
    train_split: str = "train"
    val_split: str = "val"
    crop_size: int = 512
    batch_size: int = 8
    epochs: int = 30
    learning_rate: float = 2e-4
    visual_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 300
    num_workers: int = 8
    num_modes: int = 4
    epsilon: float = 0.08
    marginal_relaxation: float = 0.5
    unknown_prior: float = 0.20
    unknown_cost: float = 0.80
    mode_offset_scale: float = 0.08
    base_score_weight: float = 0.35
    transport_score_weight: float = 1.0
    sinkhorn_iterations: int = 15
    mode_diversity_weight: float = 0.02
    output_temperature: float = 0.07
    dice_weight: float = 0.5
    class_balance_power: float = 0.5
    class_balance_max_weight: float = 4.0
    label_smoothing: float = 0.02
    open_vocabulary_preservation_weight: float = 0.15
    unknown_supervision_weight: float = 0.10
    gradient_clip: float = 1.0
    train_last_visual_blocks: int = 2
    seed: int = 3407
    max_train_images: int | None = None
    max_val_images: int | None = None
    allow_missing_source_images: bool = False
    amp: bool = True
    device: str = "cuda"
    resume_checkpoint: str | None = None

    def decoder_config(self, feature_dim: int) -> UOTPrototypeConfig:
        return UOTPrototypeConfig(
            feature_dim=feature_dim,
            num_modes=self.num_modes,
            epsilon=self.epsilon,
            marginal_relaxation=self.marginal_relaxation,
            unknown_prior=self.unknown_prior,
            unknown_cost=self.unknown_cost,
            mode_offset_scale=self.mode_offset_scale,
            base_score_weight=self.base_score_weight,
            transport_score_weight=self.transport_score_weight,
            sinkhorn_iterations=self.sinkhorn_iterations,
            mode_diversity_weight=self.mode_diversity_weight,
        )

    def validate(self) -> None:
        root = Path(self.data_root).expanduser().resolve()
        if "loveda" in {part.casefold() for part in root.parts}:
            raise ValueError("UOT training must not use a LoveDA path; LoveDA is evaluation-only.")
        if self.allow_missing_source_images and not (root / "xbd_files.csv").is_file():
            raise ValueError("allow_missing_source_images requires the official OpenEarthMap_wo_xBD archive.")
        if self.crop_size < 16 or self.crop_size % 16:
            raise ValueError("crop_size must be a multiple of 16 and at least 16.")
        if self.batch_size < 1 or self.epochs < 1 or self.num_workers < 0:
            raise ValueError("batch_size and epochs must be positive; num_workers must be non-negative.")
        if self.learning_rate <= 0 or self.visual_learning_rate < 0 or self.weight_decay < 0:
            raise ValueError("Learning rates and weight_decay are invalid.")
        if self.warmup_steps < 0 or self.train_last_visual_blocks < 0:
            raise ValueError("warmup_steps and train_last_visual_blocks must be non-negative.")
        if self.output_temperature <= 0 or not 0 <= self.dice_weight <= 1:
            raise ValueError("output_temperature must be positive and dice_weight must be in [0, 1].")
        if not 0 <= self.class_balance_power <= 1 or self.class_balance_max_weight < 1:
            raise ValueError("Class balancing configuration is invalid.")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1).")
        if self.open_vocabulary_preservation_weight < 0 or self.unknown_supervision_weight < 0:
            raise ValueError("Preservation and unknown weights must be non-negative.")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive.")
        for name, value in (("max_train_images", self.max_train_images), ("max_val_images", self.max_val_images)):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when provided.")
        self.decoder_config(feature_dim=16).validate()


def train_oem_uot(checkpoints: CheckpointConfig, config: OemUOTTrainingConfig) -> dict[str, Any]:
    """Train the multi-prototype UOT decoder using OpenEarthMap only."""

    config.validate()
    _seed_everything(config.seed)
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_samples = discover_oem_samples(
        config.data_root,
        config.train_split,
        max_images=config.max_train_images,
        allow_missing_images=config.allow_missing_source_images,
    )
    val_samples = discover_oem_samples(
        config.data_root,
        config.val_split,
        max_images=config.max_val_images,
        allow_missing_images=config.allow_missing_source_images,
    )
    _ensure_disjoint_splits(train_samples, val_samples)
    source_dataset = {
        "train": oem_source_manifest(config.data_root, config.train_split, train_samples, allow_missing_images=config.allow_missing_source_images),
        "validation": oem_source_manifest(config.data_root, config.val_split, val_samples, allow_missing_images=config.allow_missing_source_images),
    }

    extractor = DINOTextSegmenter(checkpoints, device=config.device, amp=config.amp, use_satellite=False)
    if config.train_last_visual_blocks:
        extractor.enable_visual_finetuning(config.train_last_visual_blocks)
    decoder_config = config.decoder_config(extractor.text_feature_dim)
    decoder = UOTPrototypeDecoder(decoder_config).to(extractor.device)
    source_text = extractor.encode_text(OEM_CLASSES).detach()
    preservation_text = extractor.encode_text(OPEN_VOCABULARY_PRESERVATION_CLASSES).detach()
    class_weights = _class_balanced_weights(
        oem_class_histogram(train_samples), power=config.class_balance_power, maximum=config.class_balance_max_weight
    ).to(extractor.device)
    signature = {
        "method": UOTPrototypeDecoder.__name__,
        "protocol": "External OpenEarthMap supervision only; LoveDA is evaluation-only and never used for selection.",
        "training_config": _signature_training_config(config),
        "decoder_config": decoder_config.as_dict(),
        "source_dataset": source_dataset,
        "source_classes": [{"name": spec.name, "synonyms": list(spec.synonyms)} for spec in OEM_CLASSES],
        "preservation_classes": [{"name": spec.name, "synonyms": list(spec.synonyms)} for spec in OPEN_VOCABULARY_PRESERVATION_CLASSES],
        "source_class_pixel_counts": {
            spec.name: int(count) for spec, count in zip(OEM_CLASSES, oem_class_histogram(train_samples))
        },
        "dino_checkpoints": checkpoint_manifest(checkpoints),
    }
    _ensure_signature(output_dir / "training_config.json", signature, allow_existing=config.resume_checkpoint is not None)
    train_loader = _make_loader(OpenEarthMapDataset(train_samples, crop_size=config.crop_size, training=True), config, shuffle=True)
    val_loader = _make_loader(OpenEarthMapDataset(val_samples, crop_size=config.crop_size, training=False), config, shuffle=False)
    parameter_groups: list[dict[str, Any]] = [{"params": decoder.parameters(), "base_lr": config.learning_rate}]
    visual_parameters = extractor.visual_tuning_parameters()
    if visual_parameters:
        parameter_groups.append({"params": visual_parameters, "base_lr": config.visual_learning_rate})
    optimizer = torch.optim.AdamW(parameter_groups, lr=config.learning_rate, weight_decay=config.weight_decay)
    start_epoch, best_miou, global_step = _resume_if_requested(
        config, extractor, decoder, decoder_config, source_dataset, optimizer
    )
    total_steps = max(len(train_loader) * config.epochs, 1)
    history_path = output_dir / "history.jsonl"
    if start_epoch == 1 and history_path.exists() and config.resume_checkpoint is None:
        raise ValueError(f"{history_path} already exists. Choose a new output directory or pass --resume-checkpoint.")
    started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        train_summary, global_step = _run_epoch(
            extractor, decoder, source_text, preservation_text, class_weights, train_loader, optimizer, config, global_step, total_steps, training=True
        )
        validation, _ = _run_epoch(
            extractor, decoder, source_text, preservation_text, class_weights, val_loader, optimizer, config, global_step, total_steps, training=False
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "train": train_summary,
            "validation": validation,
        }
        _append_json_line(history_path, record)
        payload = make_uot_checkpoint(
            decoder=decoder,
            source_dataset=source_dataset,
            source_classes=OEM_CLASSES,
            training_config=_signature_training_config(config),
            dino_checkpoints=checkpoints,
            epoch=epoch,
            validation=validation,
            optimizer_state_dict={"optimizer": optimizer.state_dict(), "global_step": global_step, "best_miou": best_miou},
            visual_tuning_state_dict=extractor.visual_tuning_state_dict() if visual_parameters else None,
            visual_tuning_manifest=extractor.visual_tuning_manifest,
        )
        _atomic_torch_save(output_dir / "last.pt", payload)
        if validation["mean_iou"] > best_miou:
            best_miou = float(validation["mean_iou"])
            payload["optimizer_state_dict"]["best_miou"] = best_miou
            _atomic_torch_save(output_dir / "best.pt", payload)
            inference_payload = dict(payload)
            inference_payload["optimizer_state_dict"] = None
            _atomic_torch_save(output_dir / "best_inference.pt", inference_payload)
        _write_json_atomic(
            output_dir / "summary.json",
            {
                "status": "running" if epoch < config.epochs else "complete",
                "method": signature["method"],
                "protocol": signature["protocol"],
                "epoch": epoch,
                "best_validation_mean_iou": best_miou,
                "latest": record,
                "outputs": {
                    "best_checkpoint": str(output_dir / "best.pt"),
                    "best_inference_checkpoint": str(output_dir / "best_inference.pt"),
                    "last_checkpoint": str(output_dir / "last.pt"),
                    "history": str(history_path),
                    "training_config": str(output_dir / "training_config.json"),
                },
            },
        )
    return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))


def _run_epoch(
    extractor: DINOTextSegmenter,
    decoder: UOTPrototypeDecoder,
    source_text: Tensor,
    preservation_text: Tensor,
    class_weights: Tensor,
    loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    config: OemUOTTrainingConfig,
    global_step: int,
    total_steps: int,
    *,
    training: bool,
) -> tuple[dict[str, Any], int]:
    decoder.train(training)
    extractor.set_visual_tuning_training(training)
    metrics = _SegmentationConfusion(len(OEM_CLASSES))
    losses: list[float] = []
    source_losses: list[float] = []
    preservation_losses: list[float] = []
    unknown_losses: list[float] = []
    diversity_losses: list[float] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for rgb, target in loader:
            rgb = rgb.to(extractor.device, non_blocking=True)
            target = target.to(extractor.device, non_blocking=True)
            if training:
                _set_learning_rates(optimizer, _cosine_factor(global_step, total_steps, config.warmup_steps))
                optimizer.zero_grad(set_to_none=True)
            patch_features, _ = extractor.encode_image_for_cost_aggregation(
                rgb, train_visual_backbone=training and config.train_last_visual_blocks > 0
            )
            output = decoder(patch_features, source_text, return_diagnostics=True)
            if not isinstance(output, UOTOutput):
                raise RuntimeError("UOT decoder did not return its structured output.")
            logits = F.interpolate(output.logits.float(), size=rgb.shape[-2:], mode="bilinear", align_corners=False)
            source_loss = _segmentation_loss(
                logits / config.output_temperature,
                target,
                dice_weight=config.dice_weight,
                class_weights=class_weights,
                label_smoothing=config.label_smoothing,
            )
            frozen_scores = DINOTextSegmenter.similarity_logits(patch_features, preservation_text).detach()
            adapted_scores = decoder(patch_features, preservation_text)
            preservation_loss = F.smooth_l1_loss(adapted_scores.float(), frozen_scores.float())
            ignored = F.interpolate(
                (target == 255).unsqueeze(1).to(dtype=output.unknown_probability.dtype),
                size=output.unknown_probability.shape[-2:],
                mode="area",
            ).squeeze(1) > 0.5
            unknown_loss = (
                -torch.log(output.unknown_probability.clamp_min(1e-5))[ignored].mean()
                if bool(ignored.any())
                else output.logits.new_zeros(())
            )
            diversity_loss = decoder.diversity_loss()
            loss = (
                source_loss
                + config.open_vocabulary_preservation_weight * preservation_loss
                + config.unknown_supervision_weight * unknown_loss
                + diversity_loss
            )
            if training:
                loss.backward()
                parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
                torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
                optimizer.step()
                global_step += 1
            with torch.no_grad():
                metrics.update(logits.argmax(dim=1), target)
            losses.append(float(loss.detach().item()))
            source_losses.append(float(source_loss.detach().item()))
            preservation_losses.append(float(preservation_loss.detach().item()))
            unknown_losses.append(float(unknown_loss.detach().item()))
            diversity_losses.append(float(diversity_loss.detach().item()))
    summary = metrics.summary()
    summary.update(
        {
            "loss": _mean(losses),
            "source_loss": _mean(source_losses),
            "open_vocabulary_preservation_loss": _mean(preservation_losses),
            "unknown_loss": _mean(unknown_losses),
            "mode_diversity_loss": _mean(diversity_losses),
        }
    )
    if training:
        summary["learning_rates"] = [round(float(group["lr"]), 10) for group in optimizer.param_groups]
    return summary, global_step


def _resume_if_requested(
    config: OemUOTTrainingConfig,
    extractor: DINOTextSegmenter,
    decoder: UOTPrototypeDecoder,
    decoder_config: UOTPrototypeConfig,
    source_dataset: dict[str, Any],
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float, int]:
    if config.resume_checkpoint is None:
        return 1, float("-inf"), 0
    payload = load_uot_checkpoint(config.resume_checkpoint)
    if payload["decoder_config"] != decoder_config.as_dict():
        raise ValueError("The resume checkpoint's UOT architecture differs from this requested run.")
    if payload.get("source_dataset") != source_dataset:
        raise ValueError("The resume checkpoint's external data signature differs from this requested run.")
    decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
    extractor.load_visual_tuning_state_dict(payload.get("visual_tuning_state_dict"))
    state = payload.get("optimizer_state_dict")
    if not isinstance(state, dict) or "optimizer" not in state:
        raise ValueError("The resume checkpoint does not include optimizer state.")
    optimizer.load_state_dict(state["optimizer"])
    return int(payload["epoch"]) + 1, float(state.get("best_miou", float("-inf"))), int(state.get("global_step", 0))


def _set_learning_rates(optimizer: torch.optim.Optimizer, factor: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(group["base_lr"]) * factor


def _mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 6) if values else float("nan")


def _signature_training_config(config: OemUOTTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("resume_checkpoint", None)
    return payload
