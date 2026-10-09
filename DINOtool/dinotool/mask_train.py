from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .config import CheckpointConfig
from .mask_ov import (
    MaskOVConfig,
    MaskOVDINOTextSegmenter,
    MultiScaleMaskQueryDecoder,
    make_mask_ov_checkpoint,
    load_mask_ov_checkpoint,
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
from .ov_adapter import DenseAdapterConfig, DenseTextAdapter, load_adapter_checkpoint
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
class MaskOemTrainingConfig:
    """External-only training configuration for the DINO mask-query decoder."""

    data_root: str
    output_dir: str
    train_split: str = "train"
    val_split: str = "val"
    crop_size: int = 768
    batch_size: int = 4
    epochs: int = 20
    learning_rate: float = 1e-4
    visual_learning_rate: float = 5e-6
    weight_decay: float = 0.01
    warmup_steps: int = 300
    num_workers: int = 8
    adapter_hidden_dim: int = 512
    adapter_context_blocks: int = 4
    fusion_dim: int = 256
    num_queries: int = 64
    attention_heads: int = 8
    query_layers: int = 2
    dropout: float = 0.1
    feature_layers: tuple[int, ...] = (5, 11, 17, 23)
    residual_scale: float = 0.05
    output_temperature: float = 0.07
    dice_weight: float = 0.5
    class_balance_power: float = 0.5
    class_balance_max_weight: float = 4.0
    label_smoothing: float = 0.02
    query_mask_weight: float = 1.0
    query_class_weight: float = 0.25
    semantic_mask_weight: float = 1.0
    open_vocabulary_preservation_weight: float = 0.15
    ignored_feature_preservation_weight: float = 0.1
    gradient_clip: float = 1.0
    train_last_visual_blocks: int = 0
    seed: int = 3407
    max_train_images: int | None = None
    max_val_images: int | None = None
    allow_missing_source_images: bool = False
    amp: bool = True
    device: str = "cuda"
    init_adapter_checkpoint: str | None = None
    resume_checkpoint: str | None = None

    def validate(self) -> None:
        root = Path(self.data_root).expanduser().resolve()
        if "loveda" in {part.casefold() for part in root.parts}:
            raise ValueError("Mask OV training must not use a LoveDA path. LoveDA is evaluation-only.")
        if self.allow_missing_source_images and not (root / "xbd_files.csv").is_file():
            raise ValueError(
                "allow_missing_source_images is reserved for the official OpenEarthMap_wo_xBD archive, "
                "which must include xbd_files.csv."
            )
        if self.crop_size < 16 or self.crop_size % 16:
            raise ValueError("crop_size must be a multiple of 16 and at least 16.")
        if self.batch_size < 1 or self.epochs < 1 or self.num_workers < 0:
            raise ValueError("batch_size and epochs must be positive; num_workers must be non-negative.")
        if self.learning_rate <= 0 or self.visual_learning_rate < 0 or self.weight_decay < 0:
            raise ValueError("Learning rates and weight_decay are invalid.")
        if self.warmup_steps < 0 or self.train_last_visual_blocks < 0:
            raise ValueError("warmup_steps and train_last_visual_blocks must be non-negative.")
        if self.adapter_hidden_dim < 16 or self.adapter_context_blocks < 0:
            raise ValueError("Adapter dimensions or depth are invalid.")
        MaskOVConfig(
            feature_dim=16,
            adapter_hidden_dim=self.adapter_hidden_dim,
            adapter_context_blocks=self.adapter_context_blocks,
            fusion_dim=self.fusion_dim,
            num_queries=self.num_queries,
            attention_heads=self.attention_heads,
            query_layers=self.query_layers,
            dropout=self.dropout,
            feature_layers=self.feature_layers,
            residual_scale=self.residual_scale,
        ).validate()
        if self.output_temperature <= 0 or not 0 <= self.dice_weight <= 1:
            raise ValueError("output_temperature must be positive and dice_weight must be in [0, 1].")
        if not 0 <= self.class_balance_power <= 1 or self.class_balance_max_weight < 1:
            raise ValueError("Class balancing configuration is invalid.")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1).")
        for name, value in (
            ("query_mask_weight", self.query_mask_weight),
            ("query_class_weight", self.query_class_weight),
            ("semantic_mask_weight", self.semantic_mask_weight),
            ("open_vocabulary_preservation_weight", self.open_vocabulary_preservation_weight),
            ("ignored_feature_preservation_weight", self.ignored_feature_preservation_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive.")
        for name, value in (("max_train_images", self.max_train_images), ("max_val_images", self.max_val_images)):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when provided.")
        if self.init_adapter_checkpoint and self.resume_checkpoint:
            raise ValueError("Choose either init_adapter_checkpoint or resume_checkpoint, not both.")


@dataclass(frozen=True)
class _MaskOutputs:
    logits: Tensor
    adapted_features: Tensor
    base_features: Tensor
    source_decoder: Any
    preservation_decoder: Any


def train_oem_mask_ov(checkpoints: CheckpointConfig, config: MaskOemTrainingConfig) -> dict[str, Any]:
    """Train a class-agnostic query mask decoder on OpenEarthMap only.

    The text encoder stays frozen.  The decoder receives arbitrary text
    embeddings at every forward pass, so the source CE teaches spatial masks
    without turning the checkpoint into an eight-class closed-set head.
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

    extractor = DINOTextSegmenter(checkpoints, device=config.device, amp=config.amp, use_satellite=True)
    adapter, adapter_config = _make_adapter(extractor, config)
    mask_config = _make_mask_config(extractor, config, adapter_config)
    decoder = MultiScaleMaskQueryDecoder(mask_config).to(extractor.device)
    visual_state: dict[str, dict[str, Tensor]] | None = None

    if config.resume_checkpoint:
        payload = load_mask_ov_checkpoint(config.resume_checkpoint)
        if payload.get("source_dataset") != source_dataset:
            raise ValueError("The resume checkpoint's external data signature differs from this requested run.")
        if payload.get("mask_config") != asdict(mask_config) or payload.get("adapter_config") != asdict(adapter_config):
            raise ValueError("The resume checkpoint architecture differs from this requested run.")
        adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
        decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
        visual_state = payload.get("visual_tuning_state_dict")
        _load_visual_tuning_state(extractor, visual_state)
    elif config.init_adapter_checkpoint:
        payload = load_adapter_checkpoint(config.init_adapter_checkpoint)
        adapter.load_state_dict(payload["adapter_state_dict"], strict=True)

    if config.train_last_visual_blocks and not extractor.visual_tuning_parameters():
        extractor.enable_visual_finetuning(config.train_last_visual_blocks)
    if extractor.visual_tuning_parameters():
        extractor.set_visual_tuning_training(True)

    source_text = extractor.encode_text(OEM_CLASSES).detach()
    preservation_text = extractor.encode_text(OPEN_VOCABULARY_PRESERVATION_CLASSES).detach()
    if source_text.shape != (len(OEM_CLASSES), adapter_config.feature_dim):
        raise RuntimeError("DINO source text prototype dimensions do not match the mask adapter.")
    class_counts = oem_class_histogram(train_samples)
    class_weights = _class_balanced_weights(
        class_counts,
        power=config.class_balance_power,
        maximum=config.class_balance_max_weight,
    ).to(extractor.device)

    signature = {
        "method": MaskOVDINOTextSegmenter.method_name,
        "protocol": (
            "External OpenEarthMap supervision only. LoveDA is excluded from training, checkpoint selection, "
            "hyperparameter selection, and pseudo-label generation."
        ),
        "training_config": _signature_training_config(config),
        "mask_config": asdict(mask_config),
        "adapter_config": asdict(adapter_config),
        "visual_tuning": extractor.visual_tuning_manifest,
        "source_dataset": source_dataset,
        "source_classes": [{"name": spec.name, "synonyms": list(spec.synonyms)} for spec in OEM_CLASSES],
        "preservation_classes": [
            {"name": spec.name, "synonyms": list(spec.synonyms)}
            for spec in OPEN_VOCABULARY_PRESERVATION_CLASSES
        ],
        "source_class_pixel_counts": {
            spec.name: int(count) for spec, count in zip(OEM_CLASSES, class_counts)
        },
        "source_class_weights": {
            spec.name: round(float(weight), 6)
            for spec, weight in zip(OEM_CLASSES, class_weights.cpu())
        },
        "dino_checkpoints": checkpoint_manifest(checkpoints),
    }
    _ensure_signature(
        output_dir / "training_config.json",
        signature,
        allow_existing=config.resume_checkpoint is not None,
    )

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
    parameter_groups: list[dict[str, Any]] = [
        {"params": adapter.parameters(), "base_lr": config.learning_rate},
        {"params": decoder.parameters(), "base_lr": config.learning_rate},
    ]
    visual_parameters = extractor.visual_tuning_parameters()
    if visual_parameters:
        parameter_groups.append({"params": visual_parameters, "base_lr": config.visual_learning_rate})
    optimizer = torch.optim.AdamW(parameter_groups, lr=config.learning_rate, weight_decay=config.weight_decay)
    start_epoch, best_miou, global_step = _resume_if_requested(
        config,
        adapter,
        decoder,
        optimizer,
    )
    total_steps = max(len(train_loader) * config.epochs, 1)
    history_path = output_dir / "history.jsonl"
    if start_epoch == 1 and history_path.exists() and config.resume_checkpoint is None:
        raise ValueError(f"{history_path} already exists. Choose a new output directory or pass --resume-checkpoint.")

    started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        train_summary, global_step = _run_epoch(
            extractor,
            adapter,
            decoder,
            source_text,
            preservation_text,
            class_weights,
            train_loader,
            optimizer,
            config,
            global_step,
            total_steps,
            training=True,
        )
        validation, _ = _run_epoch(
            extractor,
            adapter,
            decoder,
            source_text,
            preservation_text,
            class_weights,
            val_loader,
            optimizer,
            config,
            global_step,
            total_steps,
            training=False,
        )
        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "train": train_summary,
            "validation": validation,
        }
        _append_json_line(history_path, epoch_record)
        payload = make_mask_ov_checkpoint(
            adapter=adapter,
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
            visual_tuning_state_dict=extractor.visual_tuning_state_dict() or None,
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
        summary = {
            "status": "running" if epoch < config.epochs else "complete",
            "method": MaskOVDINOTextSegmenter.method_name,
            "protocol": signature["protocol"],
            "epoch": epoch,
            "best_validation_mean_iou": best_miou,
            "latest": epoch_record,
            "outputs": {
                "best_checkpoint": str(output_dir / "best.pt"),
                "best_inference_checkpoint": str(output_dir / "best_inference.pt"),
                "last_checkpoint": str(output_dir / "last.pt"),
                "history": str(history_path),
                "training_config": str(output_dir / "training_config.json"),
            },
        }
        _write_json_atomic(output_dir / "summary.json", summary)
    return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))


def _make_adapter(extractor: DINOTextSegmenter, config: MaskOemTrainingConfig) -> tuple[DenseTextAdapter, DenseAdapterConfig]:
    if config.init_adapter_checkpoint:
        payload = load_adapter_checkpoint(config.init_adapter_checkpoint)
        adapter_config = DenseAdapterConfig(**payload["adapter_config"])
    else:
        adapter_config = DenseAdapterConfig(
            feature_dim=extractor.text_feature_dim,
            hidden_dim=config.adapter_hidden_dim,
            context_blocks=config.adapter_context_blocks,
            use_satellite_features=True,
        )
    adapter_config.validate()
    if adapter_config.feature_dim != extractor.text_feature_dim or not adapter_config.use_satellite_features:
        raise ValueError("Mask OV requires a DINO.text adapter with SAT features and the current feature dimension.")
    return DenseTextAdapter(adapter_config).to(extractor.device), adapter_config


def _make_mask_config(
    extractor: DINOTextSegmenter,
    config: MaskOemTrainingConfig,
    adapter_config: DenseAdapterConfig,
) -> MaskOVConfig:
    mask_config = MaskOVConfig(
        feature_dim=extractor.text_feature_dim,
        adapter_hidden_dim=adapter_config.hidden_dim,
        adapter_context_blocks=adapter_config.context_blocks,
        fusion_dim=config.fusion_dim,
        num_queries=config.num_queries,
        attention_heads=config.attention_heads,
        query_layers=config.query_layers,
        dropout=config.dropout,
        feature_layers=tuple(config.feature_layers),
        residual_scale=config.residual_scale,
    )
    mask_config.validate()
    return mask_config


def _load_visual_tuning_state(
    extractor: DINOTextSegmenter,
    state: dict[str, dict[str, Tensor]] | None,
) -> None:
    if not state:
        return
    extractor.load_visual_tuning_state_dict(state)
    blocks = extractor.model.visual_model.backbone.blocks
    for index in extractor.visual_tuning_manifest["tuned_backbone_blocks"]:
        for parameter in blocks[int(index)].parameters():
            parameter.requires_grad_(True)


def _run_epoch(
    extractor: DINOTextSegmenter,
    adapter: DenseTextAdapter,
    decoder: MultiScaleMaskQueryDecoder,
    source_text: Tensor,
    preservation_text: Tensor,
    class_weights: Tensor,
    loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    config: MaskOemTrainingConfig,
    global_step: int,
    total_steps: int,
    *,
    training: bool,
) -> tuple[dict[str, Any], int]:
    adapter.train(training)
    decoder.train(training)
    if extractor.visual_tuning_parameters():
        extractor.set_visual_tuning_training(training)
    losses: list[float] = []
    source_losses: list[float] = []
    semantic_losses: list[float] = []
    query_mask_losses: list[float] = []
    query_class_losses: list[float] = []
    preservation_losses: list[float] = []
    ignored_losses: list[float] = []
    metrics = _SegmentationConfusion(len(OEM_CLASSES))
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for rgb, target in loader:
            rgb = rgb.to(extractor.device, non_blocking=True)
            target = target.to(extractor.device, non_blocking=True)
            if training:
                _set_learning_rate(
                    optimizer,
                    config.learning_rate * _cosine_factor(global_step, total_steps, config.warmup_steps),
                )
                optimizer.zero_grad(set_to_none=True)
            outputs = _mask_outputs(
                extractor,
                adapter,
                decoder,
                source_text,
                preservation_text,
                rgb,
                config,
                train_visual_backbone=training and bool(extractor.visual_tuning_parameters()),
            )
            source_loss = _segmentation_loss(
                outputs.logits,
                target,
                dice_weight=config.dice_weight,
                class_weights=class_weights,
                label_smoothing=config.label_smoothing,
            )
            target_low = _downsample_target(target, outputs.source_decoder.mask_logits.shape[-2:])
            semantic_loss = _semantic_mask_loss(
                outputs.source_decoder.semantic_mask,
                target_low,
                class_weights,
                config.dice_weight,
            )
            query_mask_loss, query_class_loss = _query_losses(
                outputs.source_decoder.mask_logits,
                outputs.source_decoder.class_scores,
                target_low,
            )
            preservation_loss = _preservation_loss(
                outputs.base_features,
                outputs.adapted_features,
                outputs.preservation_decoder.residual_logits,
                preservation_text,
            )
            ignored_loss = _ignored_feature_loss(outputs.base_features, outputs.adapted_features, target)
            loss = (
                source_loss
                + config.semantic_mask_weight * semantic_loss
                + config.query_mask_weight * query_mask_loss
                + config.query_class_weight * query_class_loss
                + config.open_vocabulary_preservation_weight * preservation_loss
                + config.ignored_feature_preservation_weight * ignored_loss
            )
            if training:
                loss.backward()
                parameters = list(adapter.parameters()) + list(decoder.parameters()) + extractor.visual_tuning_parameters()
                torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
                optimizer.step()
                global_step += 1
            with torch.no_grad():
                metrics.update(outputs.logits.argmax(dim=1), target)
            losses.append(float(loss.detach().item()))
            source_losses.append(float(source_loss.detach().item()))
            semantic_losses.append(float(semantic_loss.detach().item()))
            query_mask_losses.append(float(query_mask_loss.detach().item()))
            query_class_losses.append(float(query_class_loss.detach().item()))
            preservation_losses.append(float(preservation_loss.detach().item()))
            ignored_losses.append(float(ignored_loss.detach().item()))
    summary = metrics.summary()
    summary.update(
        {
            "loss": _mean(losses),
            "source_loss": _mean(source_losses),
            "semantic_mask_loss": _mean(semantic_losses),
            "query_mask_loss": _mean(query_mask_losses),
            "query_class_loss": _mean(query_class_losses),
            "open_vocabulary_preservation_loss": _mean(preservation_losses),
            "ignored_feature_preservation_loss": _mean(ignored_losses),
            "learning_rate": round(float(optimizer.param_groups[0]["lr"]), 10),
        }
    )
    return summary, global_step


def _mask_outputs(
    extractor: DINOTextSegmenter,
    adapter: DenseTextAdapter,
    decoder: MultiScaleMaskQueryDecoder,
    source_text: Tensor,
    preservation_text: Tensor,
    rgb: Tensor,
    config: MaskOemTrainingConfig,
    *,
    train_visual_backbone: bool,
) -> _MaskOutputs:
    multiscale, _ = extractor.encode_image_multiscale_for_mask(
        rgb,
        layers=decoder.config.feature_layers,
        train_visual_backbone=train_visual_backbone,
    )
    base_features = multiscale[-1]
    satellite_features = extractor.encode_satellite_structure_for_adapter(rgb)
    with torch.autocast(
        device_type=extractor.device.type,
        dtype=torch.bfloat16,
        enabled=config.amp and extractor.device.type == "cuda",
    ):
        adapted_features = adapter(base_features, satellite_features)
        source_decoder = decoder(multiscale, satellite_features, _rgb_grid(rgb, base_features), source_text)
        preservation_decoder = decoder(
            multiscale,
            satellite_features,
            _rgb_grid(rgb, base_features),
            preservation_text,
        )
        source_logits = torch.einsum(
            "bdhw,cd->bchw",
            adapted_features,
            F.normalize(source_text, dim=-1),
        )
        source_logits = source_logits + source_decoder.residual_logits
    source_logits = F.interpolate(source_logits.float(), size=rgb.shape[-2:], mode="bilinear", align_corners=False)
    return _MaskOutputs(
        logits=source_logits / config.output_temperature,
        adapted_features=adapted_features,
        base_features=base_features,
        source_decoder=source_decoder,
        preservation_decoder=preservation_decoder,
    )


def _rgb_grid(rgb: Tensor, features: Tensor) -> Tensor:
    return F.interpolate(rgb.to(features.device), size=features.shape[-2:], mode="area")


def _downsample_target(target: Tensor, shape: tuple[int, int]) -> Tensor:
    return F.interpolate(target.unsqueeze(1).float(), size=shape, mode="nearest").squeeze(1).to(torch.long)


def _semantic_mask_loss(
    semantic_mask: Tensor,
    target: Tensor,
    class_weights: Tensor,
    dice_weight: float,
) -> Tensor:
    probabilities = semantic_mask.clamp(min=1e-5, max=1.0)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-5)
    return _segmentation_loss(
        torch.log(probabilities).float(),
        target,
        dice_weight=dice_weight,
        class_weights=class_weights,
        label_smoothing=0.0,
    )


def _query_losses(mask_logits: Tensor, class_scores: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Soft DETR-style assignment without a fixed class-specific query head."""
    valid = target != 255
    target_safe = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(target_safe, num_classes=class_scores.shape[-1]).permute(0, 3, 1, 2).to(mask_logits.dtype)
    valid_mask = valid.unsqueeze(1).to(mask_logits.dtype)
    class_probabilities = torch.softmax(class_scores.float(), dim=-1).to(mask_logits.dtype)
    query_targets = torch.einsum("bqc,bchw->bqhw", class_probabilities, one_hot)
    query_mask_loss = F.binary_cross_entropy_with_logits(mask_logits, query_targets, reduction="none")
    query_mask_loss = (query_mask_loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0) / mask_logits.shape[1]

    mask_probabilities = torch.sigmoid(mask_logits) * valid_mask
    area = mask_probabilities.sum(dim=(2, 3), keepdim=False).clamp_min(1e-4)
    target_distribution = torch.einsum("bqhw,bchw->bqc", mask_probabilities, one_hot) / area.unsqueeze(-1)
    query_class_loss = -(target_distribution.detach() * F.log_softmax(class_scores.float(), dim=-1)).sum(dim=-1)
    active = area > 1e-4
    return query_mask_loss, query_class_loss[active].mean() if bool(active.any()) else mask_logits.new_zeros(())


def _preservation_loss(
    base_features: Tensor,
    adapted_features: Tensor,
    residual_logits: Tensor,
    preservation_text: Tensor,
) -> Tensor:
    base_scores = torch.einsum("bdhw,cd->bchw", base_features, F.normalize(preservation_text, dim=-1))
    adapted_scores = torch.einsum("bdhw,cd->bchw", adapted_features, F.normalize(preservation_text, dim=-1))
    return F.smooth_l1_loss(adapted_scores + residual_logits, base_scores.detach())


def _ignored_feature_loss(base_features: Tensor, adapted_features: Tensor, target: Tensor) -> Tensor:
    ignored = F.interpolate(
        (target == 255).unsqueeze(1).to(dtype=base_features.dtype),
        size=base_features.shape[-2:],
        mode="area",
    ).squeeze(1) > 0.5
    if not bool(ignored.any()):
        return base_features.new_zeros(())
    cosine_distance = 1.0 - (base_features * adapted_features).sum(dim=1)
    return cosine_distance[ignored].mean()


def _resume_if_requested(
    config: MaskOemTrainingConfig,
    adapter: DenseTextAdapter,
    decoder: MultiScaleMaskQueryDecoder,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float, int]:
    if config.resume_checkpoint is None:
        return 1, float("-inf"), 0
    payload = load_mask_ov_checkpoint(config.resume_checkpoint)
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
    state = payload.get("optimizer_state_dict")
    if not isinstance(state, dict) or "optimizer" not in state:
        raise ValueError("The resume checkpoint does not include optimizer state.")
    optimizer.load_state_dict(state["optimizer"])
    return int(payload["epoch"]) + 1, float(state.get("best_miou", float("-inf"))), int(state.get("global_step", 0))


def _set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    factor = learning_rate / max(float(optimizer.param_groups[0].get("base_lr", learning_rate)), 1e-12)
    for group in optimizer.param_groups:
        base_lr = float(group.get("base_lr", learning_rate))
        group["lr"] = base_lr * factor


def _signature_training_config(config: MaskOemTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("resume_checkpoint", None)
    return payload


def _mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 6) if values else float("nan")
