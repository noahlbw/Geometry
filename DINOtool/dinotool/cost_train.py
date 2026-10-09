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
from .cost_aggregation import (
    CostAggregationConfig,
    CostAggregationDecoder,
    load_cost_aggregation_checkpoint,
    make_cost_aggregation_checkpoint,
)
from .model import DINOTextSegmenter, checkpoint_manifest
from .oem import (
    OEM_CLASSES,
    OPEN_VOCABULARY_PRESERVATION_CLASSES,
    OpenEarthMapDataset,
    discover_oem_samples,
    oem_class_histogram,
    oem_source_manifest,
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
class OemCostTrainingConfig:
    """External-only training configuration for the class-agnostic DINO decoder."""

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
    hidden_dim: int = 128
    context_blocks: int = 6
    attention_heads: int = 8
    window_size: int = 7
    dropout: float = 0.1
    train_last_visual_blocks: int = 2
    output_temperature: float = 0.07
    dice_weight: float = 0.5
    class_balance_power: float = 0.5
    class_balance_max_weight: float = 4.0
    label_smoothing: float = 0.02
    open_vocabulary_preservation_weight: float = 0.15
    ignored_cost_preservation_weight: float = 0.1
    gradient_clip: float = 1.0
    seed: int = 3407
    max_train_images: int | None = None
    max_val_images: int | None = None
    allow_missing_source_images: bool = False
    amp: bool = True
    device: str = "cuda"
    resume_checkpoint: str | None = None

    def validate(self) -> None:
        root = Path(self.data_root).expanduser().resolve()
        if "loveda" in {part.casefold() for part in root.parts}:
            raise ValueError("Cost-aggregation training must not use a LoveDA path. LoveDA is evaluation-only.")
        if self.allow_missing_source_images and not (root / "xbd_files.csv").is_file():
            raise ValueError(
                "allow_missing_source_images is reserved for the official OpenEarthMap_wo_xBD archive."
            )
        if self.crop_size < 16 or self.crop_size % 16:
            raise ValueError("crop_size must be a multiple of 16 and at least 16.")
        if self.batch_size < 1 or self.epochs < 1 or self.num_workers < 0:
            raise ValueError("batch_size and epochs must be positive; num_workers must be non-negative.")
        if self.learning_rate <= 0 or self.visual_learning_rate < 0 or self.weight_decay < 0:
            raise ValueError("Learning rates and weight_decay are invalid.")
        if self.warmup_steps < 0 or self.train_last_visual_blocks < 0:
            raise ValueError("warmup_steps and train_last_visual_blocks must be non-negative.")
        CostAggregationConfig(
            feature_dim=16,
            hidden_dim=self.hidden_dim,
            context_blocks=self.context_blocks,
            attention_heads=self.attention_heads,
            window_size=self.window_size,
            dropout=self.dropout,
        ).validate()
        if self.output_temperature <= 0 or not 0 <= self.dice_weight <= 1:
            raise ValueError("output_temperature must be positive and dice_weight must be in [0, 1].")
        if not 0 <= self.class_balance_power <= 1 or self.class_balance_max_weight < 1:
            raise ValueError("Class balancing configuration is invalid.")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1).")
        if self.open_vocabulary_preservation_weight < 0 or self.ignored_cost_preservation_weight < 0:
            raise ValueError("Preservation weights must be non-negative.")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive.")
        for name, value in (("max_train_images", self.max_train_images), ("max_val_images", self.max_val_images)):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when provided.")


def train_oem_cost_aggregation(
    checkpoints: CheckpointConfig,
    config: OemCostTrainingConfig,
) -> dict[str, Any]:
    """Train a DINOv3 dino.txt cost decoder on OpenEarthMap only.

    The decoder is shared over classes and always receives text embeddings at
    runtime.  It therefore remains usable with a vocabulary different from the
    eight OpenEarthMap source concepts.
    """
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
        "train": oem_source_manifest(
            config.data_root,
            config.train_split,
            train_samples,
            allow_missing_images=config.allow_missing_source_images,
        ),
        "validation": oem_source_manifest(
            config.data_root,
            config.val_split,
            val_samples,
            allow_missing_images=config.allow_missing_source_images,
        ),
    }

    extractor = DINOTextSegmenter(checkpoints, device=config.device, amp=config.amp, use_satellite=False)
    if config.train_last_visual_blocks:
        extractor.enable_visual_finetuning(config.train_last_visual_blocks)
    decoder_config = CostAggregationConfig(
        feature_dim=extractor.text_feature_dim,
        hidden_dim=config.hidden_dim,
        context_blocks=config.context_blocks,
        attention_heads=config.attention_heads,
        window_size=config.window_size,
        dropout=config.dropout,
    )
    decoder = CostAggregationDecoder(decoder_config).to(extractor.device)
    source_text = extractor.encode_text(OEM_CLASSES).detach()
    preservation_text = extractor.encode_text(OPEN_VOCABULARY_PRESERVATION_CLASSES).detach()
    if source_text.shape != (len(OEM_CLASSES), decoder_config.feature_dim):
        raise RuntimeError("DINO source text prototype dimensions do not match the cost decoder.")
    class_weights = _class_balanced_weights(
        oem_class_histogram(train_samples),
        power=config.class_balance_power,
        maximum=config.class_balance_max_weight,
    ).to(extractor.device)

    signature = {
        "method": "DINOv3 dino.txt class-agnostic cost-aggregation OVSS decoder",
        "protocol": (
            "External OpenEarthMap supervision only. LoveDA is excluded from training, checkpoint selection, "
            "hyperparameter selection, and pseudo-label generation."
        ),
        "training_config": _signature_training_config(config),
        "decoder_config": asdict(decoder_config),
        "visual_tuning": extractor.visual_tuning_manifest,
        "source_dataset": source_dataset,
        "source_classes": [{"name": spec.name, "synonyms": list(spec.synonyms)} for spec in OEM_CLASSES],
        "preservation_classes": [
            {"name": spec.name, "synonyms": list(spec.synonyms)}
            for spec in OPEN_VOCABULARY_PRESERVATION_CLASSES
        ],
        "source_class_pixel_counts": {
            spec.name: int(count)
            for spec, count in zip(OEM_CLASSES, oem_class_histogram(train_samples))
        },
        "dino_checkpoints": checkpoint_manifest(checkpoints),
    }
    _ensure_signature(output_dir / "training_config.json", signature, allow_existing=config.resume_checkpoint is not None)

    train_loader = _make_loader(
        OpenEarthMapDataset(train_samples, crop_size=config.crop_size, training=True),
        config,
        shuffle=True,
    )
    val_loader = _make_loader(
        OpenEarthMapDataset(val_samples, crop_size=config.crop_size, training=False),
        config,
        shuffle=False,
    )
    parameter_groups: list[dict[str, Any]] = [{"params": decoder.parameters(), "base_lr": config.learning_rate}]
    visual_parameters = extractor.visual_tuning_parameters()
    if visual_parameters:
        parameter_groups.append({"params": visual_parameters, "base_lr": config.visual_learning_rate})
    optimizer = torch.optim.AdamW(parameter_groups, lr=config.learning_rate, weight_decay=config.weight_decay)
    start_epoch, best_miou, global_step = _resume_if_requested(
        config,
        extractor,
        decoder,
        decoder_config,
        source_dataset,
        optimizer,
    )
    total_steps = max(len(train_loader) * config.epochs, 1)
    history_path = output_dir / "history.jsonl"
    if start_epoch == 1 and history_path.exists() and config.resume_checkpoint is None:
        raise ValueError(f"{history_path} already exists. Choose a new output directory or pass --resume-checkpoint.")

    started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        train_summary, global_step = _train_epoch(
            extractor,
            decoder,
            source_text,
            preservation_text,
            class_weights,
            train_loader,
            optimizer,
            config,
            global_step,
            total_steps,
        )
        validation = _evaluate(
            extractor,
            decoder,
            source_text,
            preservation_text,
            class_weights,
            val_loader,
            config,
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "train": train_summary,
            "validation": validation,
        }
        _append_json_line(history_path, record)
        payload = make_cost_aggregation_checkpoint(
            decoder=decoder,
            source_dataset=source_dataset,
            source_classes=OEM_CLASSES,
            training_config=_signature_training_config(config),
            dino_checkpoints=checkpoints,
            epoch=epoch,
            validation=validation,
            optimizer_state_dict={
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "best_miou": best_miou,
            },
            visual_tuning_state_dict=(extractor.visual_tuning_state_dict() if visual_parameters else None),
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


def _train_epoch(
    extractor: DINOTextSegmenter,
    decoder: CostAggregationDecoder,
    source_text: Tensor,
    preservation_text: Tensor,
    class_weights: Tensor,
    loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    config: OemCostTrainingConfig,
    global_step: int,
    total_steps: int,
) -> tuple[dict[str, float], int]:
    decoder.train()
    extractor.set_visual_tuning_training(True)
    losses: list[float] = []
    source_losses: list[float] = []
    preservation_losses: list[float] = []
    ignored_losses: list[float] = []
    metrics = _SegmentationConfusion(len(OEM_CLASSES))
    for rgb, target in loader:
        rgb = rgb.to(extractor.device, non_blocking=True)
        target = target.to(extractor.device, non_blocking=True)
        _set_learning_rates(optimizer, _cosine_factor(global_step, total_steps, config.warmup_steps))
        optimizer.zero_grad(set_to_none=True)
        outputs = _cost_outputs(extractor, decoder, source_text, rgb, config, train=True)
        source_loss = _segmentation_loss(
            outputs.logits,
            target,
            dice_weight=config.dice_weight,
            class_weights=class_weights,
            label_smoothing=config.label_smoothing,
        )
        preservation_loss = _open_vocabulary_cost_preservation_loss(
            decoder,
            outputs.patch_features,
            preservation_text,
        )
        ignored_loss = _ignored_cost_preservation_loss(
            decoder,
            outputs.patch_features,
            preservation_text,
            target,
        )
        loss = (
            source_loss
            + config.open_vocabulary_preservation_weight * preservation_loss
            + config.ignored_cost_preservation_weight * ignored_loss
        )
        loss.backward()
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
        optimizer.step()
        with torch.no_grad():
            metrics.update(outputs.logits.argmax(dim=1), target)
        losses.append(float(loss.detach().item()))
        source_losses.append(float(source_loss.detach().item()))
        preservation_losses.append(float(preservation_loss.detach().item()))
        ignored_losses.append(float(ignored_loss.detach().item()))
        global_step += 1
    return _summarize_epoch(metrics, losses, source_losses, preservation_losses, ignored_losses, optimizer), global_step


@torch.inference_mode()
def _evaluate(
    extractor: DINOTextSegmenter,
    decoder: CostAggregationDecoder,
    source_text: Tensor,
    preservation_text: Tensor,
    class_weights: Tensor,
    loader: DataLoader[tuple[Tensor, Tensor]],
    config: OemCostTrainingConfig,
) -> dict[str, Any]:
    decoder.eval()
    extractor.set_visual_tuning_training(False)
    losses: list[float] = []
    source_losses: list[float] = []
    preservation_losses: list[float] = []
    ignored_losses: list[float] = []
    metrics = _SegmentationConfusion(len(OEM_CLASSES))
    for rgb, target in loader:
        rgb = rgb.to(extractor.device, non_blocking=True)
        target = target.to(extractor.device, non_blocking=True)
        outputs = _cost_outputs(extractor, decoder, source_text, rgb, config, train=False)
        source_loss = _segmentation_loss(
            outputs.logits,
            target,
            dice_weight=config.dice_weight,
            class_weights=class_weights,
            label_smoothing=config.label_smoothing,
        )
        preservation_loss = _open_vocabulary_cost_preservation_loss(
            decoder,
            outputs.patch_features,
            preservation_text,
        )
        ignored_loss = _ignored_cost_preservation_loss(
            decoder,
            outputs.patch_features,
            preservation_text,
            target,
        )
        losses.append(
            float(
                (
                    source_loss
                    + config.open_vocabulary_preservation_weight * preservation_loss
                    + config.ignored_cost_preservation_weight * ignored_loss
                ).item()
            )
        )
        source_losses.append(float(source_loss.item()))
        preservation_losses.append(float(preservation_loss.item()))
        ignored_losses.append(float(ignored_loss.item()))
        metrics.update(outputs.logits.argmax(dim=1), target)
    return _summarize_epoch(metrics, losses, source_losses, preservation_losses, ignored_losses, None)


@dataclass(frozen=True)
class _CostOutputs:
    logits: Tensor
    patch_features: Tensor


def _cost_outputs(
    extractor: DINOTextSegmenter,
    decoder: CostAggregationDecoder,
    source_text: Tensor,
    rgb: Tensor,
    config: OemCostTrainingConfig,
    *,
    train: bool,
) -> _CostOutputs:
    patch_features, _ = extractor.encode_image_for_cost_aggregation(
        rgb,
        train_visual_backbone=train and config.train_last_visual_blocks > 0,
    )
    with torch.autocast(
        device_type=extractor.device.type,
        dtype=torch.bfloat16,
        enabled=config.amp and extractor.device.type == "cuda",
    ):
        logits = decoder(patch_features, source_text)
    logits = F.interpolate(logits.float(), size=rgb.shape[-2:], mode="bilinear", align_corners=False)
    return _CostOutputs(logits=logits / config.output_temperature, patch_features=patch_features)


def _open_vocabulary_cost_preservation_loss(
    decoder: CostAggregationDecoder,
    patch_features: Tensor,
    preservation_text: Tensor,
) -> Tensor:
    """Retain DINO scores for concepts outside the OpenEarthMap label space."""
    frozen_scores = DINOTextSegmenter.similarity_logits(patch_features, preservation_text).detach()
    adapted_scores = decoder(patch_features, preservation_text)
    return F.smooth_l1_loss(adapted_scores.float(), frozen_scores.float())


def _ignored_cost_preservation_loss(
    decoder: CostAggregationDecoder,
    patch_features: Tensor,
    preservation_text: Tensor,
    target: Tensor,
) -> Tensor:
    ignored = F.interpolate(
        (target == 255).unsqueeze(1).to(dtype=patch_features.dtype),
        size=patch_features.shape[-2:],
        mode="area",
    ).squeeze(1) > 0.5
    if not bool(ignored.any()):
        return patch_features.new_zeros(())
    frozen_scores = DINOTextSegmenter.similarity_logits(patch_features, preservation_text).detach()
    adapted_scores = decoder(patch_features, preservation_text)
    return F.smooth_l1_loss(adapted_scores.permute(0, 2, 3, 1)[ignored], frozen_scores.permute(0, 2, 3, 1)[ignored])


def _summarize_epoch(
    metrics: _SegmentationConfusion,
    losses: list[float],
    source_losses: list[float],
    preservation_losses: list[float],
    ignored_losses: list[float],
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    summary = metrics.summary()
    summary["loss"] = round(float(np.mean(losses)), 6) if losses else float("nan")
    summary["source_loss"] = round(float(np.mean(source_losses)), 6) if source_losses else float("nan")
    summary["open_vocabulary_preservation_loss"] = (
        round(float(np.mean(preservation_losses)), 6) if preservation_losses else float("nan")
    )
    summary["ignored_cost_preservation_loss"] = (
        round(float(np.mean(ignored_losses)), 6) if ignored_losses else float("nan")
    )
    if optimizer is not None:
        summary["learning_rates"] = [round(float(group["lr"]), 10) for group in optimizer.param_groups]
    return summary


def _set_learning_rates(optimizer: torch.optim.Optimizer, factor: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(group["base_lr"]) * factor


def _resume_if_requested(
    config: OemCostTrainingConfig,
    extractor: DINOTextSegmenter,
    decoder: CostAggregationDecoder,
    decoder_config: CostAggregationConfig,
    source_dataset: dict[str, Any],
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float, int]:
    if config.resume_checkpoint is None:
        return 1, float("-inf"), 0
    payload = load_cost_aggregation_checkpoint(config.resume_checkpoint)
    if payload["decoder_config"] != asdict(decoder_config):
        raise ValueError("The resume checkpoint's cost-decoder architecture differs from this requested run.")
    if payload.get("source_dataset") != source_dataset:
        raise ValueError("The resume checkpoint's external data signature differs from this requested run.")
    decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
    extractor.load_visual_tuning_state_dict(payload.get("visual_tuning_state_dict"))
    state = payload.get("optimizer_state_dict")
    if not isinstance(state, dict) or "optimizer" not in state:
        raise ValueError("The resume checkpoint does not include optimizer state.")
    optimizer.load_state_dict(state["optimizer"])
    return int(payload["epoch"]) + 1, float(state.get("best_miou", float("-inf"))), int(state.get("global_step", 0))


def _signature_training_config(config: OemCostTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("resume_checkpoint", None)
    return payload
