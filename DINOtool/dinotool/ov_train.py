from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
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
from .ov_adapter import DenseAdapterConfig, DenseTextAdapter, load_adapter_checkpoint, make_adapter_checkpoint


@dataclass(frozen=True)
class OemTrainingConfig:
    data_root: str
    output_dir: str
    train_split: str = "train"
    val_split: str = "val"
    crop_size: int = 512
    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    num_workers: int = 8
    hidden_dim: int = 256
    context_blocks: int = 2
    output_temperature: float = 0.07
    dice_weight: float = 0.5
    class_balance_power: float = 0.5
    class_balance_max_weight: float = 4.0
    label_smoothing: float = 0.02
    open_vocabulary_preservation_weight: float = 0.25
    ignored_feature_preservation_weight: float = 0.1
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
            raise ValueError("OpenEarthMap training must not use a LoveDA path. LoveDA is evaluation-only in this protocol.")
        if self.allow_missing_source_images and not (root / "xbd_files.csv").is_file():
            raise ValueError(
                "allow_missing_source_images is reserved for the official OpenEarthMap_wo_xBD archive, "
                "which must include xbd_files.csv."
            )
        if self.crop_size < 16 or self.crop_size % 16:
            raise ValueError("crop_size must be a multiple of 16 and at least 16.")
        if self.batch_size < 1 or self.epochs < 1 or self.num_workers < 0:
            raise ValueError("batch_size and epochs must be positive; num_workers must be non-negative.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if self.warmup_steps < 0 or self.hidden_dim < 16 or self.context_blocks < 0:
            raise ValueError("warmup_steps, hidden_dim, or context_blocks is invalid.")
        if self.output_temperature <= 0 or not 0 <= self.dice_weight <= 1:
            raise ValueError("output_temperature must be positive and dice_weight must be in [0, 1].")
        if not 0 <= self.class_balance_power <= 1 or self.class_balance_max_weight < 1:
            raise ValueError("class balance power must be in [0, 1] and maximum class weight must be at least 1.")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1).")
        if self.open_vocabulary_preservation_weight < 0 or self.ignored_feature_preservation_weight < 0:
            raise ValueError("preservation weights must be non-negative.")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive.")
        for name, value in (("max_train_images", self.max_train_images), ("max_val_images", self.max_val_images)):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when provided.")


def train_oem_adapter(
    checkpoints: CheckpointConfig,
    config: OemTrainingConfig,
) -> dict[str, Any]:
    """Train only a DINO feature bridge using OpenEarthMap; LoveDA is never loaded."""
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

    extractor = DINOTextSegmenter(
        checkpoints,
        device=config.device,
        amp=config.amp,
        use_satellite=True,
    )
    adapter_config = DenseAdapterConfig(
        feature_dim=extractor.text_feature_dim,
        hidden_dim=config.hidden_dim,
        context_blocks=config.context_blocks,
        use_satellite_features=True,
    )
    adapter = DenseTextAdapter(adapter_config).to(extractor.device)
    text_features = extractor.encode_text(OEM_CLASSES).detach()
    preservation_text_features = extractor.encode_text(OPEN_VOCABULARY_PRESERVATION_CLASSES).detach()
    if text_features.shape != (len(OEM_CLASSES), adapter_config.feature_dim):
        raise RuntimeError("DINO text prototype dimensions do not match the adapter configuration.")
    if preservation_text_features.shape != (len(OPEN_VOCABULARY_PRESERVATION_CLASSES), adapter_config.feature_dim):
        raise RuntimeError("Open-vocabulary preservation prototypes do not match the adapter configuration.")
    class_counts = oem_class_histogram(train_samples)
    class_weights = _class_balanced_weights(
        class_counts,
        power=config.class_balance_power,
        maximum=config.class_balance_max_weight,
    ).to(extractor.device)

    signature = {
        "method": "DINOv3-SAT plus dino.txt dense open-vocabulary adapter training",
        "protocol": (
            "External OpenEarthMap supervision only. LoveDA is excluded from training, checkpoint selection, "
            "and hyperparameter selection."
        ),
        "training_config": _signature_training_config(config),
        "adapter_config": asdict(adapter_config),
        "source_dataset": source_dataset,
        "source_classes": [{"name": spec.name, "synonyms": list(spec.synonyms)} for spec in OEM_CLASSES],
        "open_vocabulary_preservation_classes": [
            {"name": spec.name, "synonyms": list(spec.synonyms)} for spec in OPEN_VOCABULARY_PRESERVATION_CLASSES
        ],
        "source_class_pixel_counts": {spec.name: int(count) for spec, count in zip(OEM_CLASSES, class_counts)},
        "source_class_weights": {spec.name: round(float(weight), 6) for spec, weight in zip(OEM_CLASSES, class_weights.cpu())},
        "dino_checkpoints": checkpoint_manifest(checkpoints),
    }
    _ensure_signature(output_dir / "training_config.json", signature, allow_existing=config.resume_checkpoint is not None)

    train_dataset = OpenEarthMapDataset(train_samples, crop_size=config.crop_size, training=True)
    val_dataset = OpenEarthMapDataset(val_samples, crop_size=config.crop_size, training=False)
    train_loader = _make_loader(train_dataset, config, shuffle=True)
    val_loader = _make_loader(val_dataset, config, shuffle=False)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    start_epoch, best_miou, global_step = _resume_if_requested(
        config,
        adapter,
        optimizer,
        adapter_config,
        source_dataset,
    )
    total_steps = max(len(train_loader) * config.epochs, 1)
    history_path = output_dir / "history.jsonl"
    if start_epoch == 1 and history_path.exists() and config.resume_checkpoint is None:
        raise ValueError(f"{history_path} already exists. Choose a new output directory or pass --resume-checkpoint.")

    started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        train_summary, global_step = _train_epoch(
            extractor,
            adapter,
            text_features,
            preservation_text_features,
            class_weights,
            train_loader,
            optimizer,
            config,
            global_step,
            total_steps,
        )
        validation = _evaluate(
            extractor,
            adapter,
            text_features,
            preservation_text_features,
            class_weights,
            val_loader,
            config,
        )
        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "train": train_summary,
            "validation": validation,
        }
        _append_json_line(history_path, epoch_record)
        payload = make_adapter_checkpoint(
            adapter=adapter,
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
        )
        _atomic_torch_save(output_dir / "last.pt", payload)
        if validation["mean_iou"] > best_miou:
            best_miou = float(validation["mean_iou"])
            payload["optimizer_state_dict"]["best_miou"] = best_miou
            _atomic_torch_save(output_dir / "best.pt", payload)
        summary = {
            "status": "running" if epoch < config.epochs else "complete",
            "method": "DINOv3-SAT plus dino.txt dense open-vocabulary adapter training",
            "protocol": signature["protocol"],
            "epoch": epoch,
            "best_validation_mean_iou": best_miou,
            "latest": epoch_record,
            "outputs": {
                "best_checkpoint": str(output_dir / "best.pt"),
                "last_checkpoint": str(output_dir / "last.pt"),
                "history": str(history_path),
                "training_config": str(output_dir / "training_config.json"),
            },
        }
        _write_json_atomic(output_dir / "summary.json", summary)
    return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))


def _train_epoch(
    extractor: DINOTextSegmenter,
    adapter: DenseTextAdapter,
    text_features: Tensor,
    preservation_text_features: Tensor,
    class_weights: Tensor,
    loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    config: OemTrainingConfig,
    global_step: int,
    total_steps: int,
) -> tuple[dict[str, float], int]:
    adapter.train()
    losses: list[float] = []
    source_losses: list[float] = []
    preservation_losses: list[float] = []
    ignored_losses: list[float] = []
    metrics = _SegmentationConfusion(len(OEM_CLASSES))
    for rgb, target in loader:
        rgb = rgb.to(extractor.device, non_blocking=True)
        target = target.to(extractor.device, non_blocking=True)
        _set_learning_rate(optimizer, config.learning_rate * _cosine_factor(global_step, total_steps, config.warmup_steps))
        optimizer.zero_grad(set_to_none=True)
        outputs = _adapter_outputs(extractor, adapter, text_features, rgb, config)
        source_loss = _segmentation_loss(
            outputs.logits,
            target,
            dice_weight=config.dice_weight,
            class_weights=class_weights,
            label_smoothing=config.label_smoothing,
        )
        preservation_loss = _open_vocabulary_preservation_loss(
            outputs.base_features,
            outputs.patch_features,
            preservation_text_features,
        )
        ignored_loss = _ignored_feature_preservation_loss(
            outputs.base_features,
            outputs.patch_features,
            target,
        )
        loss = (
            source_loss
            + config.open_vocabulary_preservation_weight * preservation_loss
            + config.ignored_feature_preservation_weight * ignored_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), config.gradient_clip)
        optimizer.step()
        with torch.no_grad():
            metrics.update(outputs.logits.argmax(dim=1), target)
        losses.append(float(loss.detach().item()))
        source_losses.append(float(source_loss.detach().item()))
        preservation_losses.append(float(preservation_loss.detach().item()))
        ignored_losses.append(float(ignored_loss.detach().item()))
        global_step += 1
    summary = metrics.summary()
    summary["loss"] = round(float(np.mean(losses)), 6) if losses else float("nan")
    summary["source_loss"] = round(float(np.mean(source_losses)), 6) if source_losses else float("nan")
    summary["open_vocabulary_preservation_loss"] = (
        round(float(np.mean(preservation_losses)), 6) if preservation_losses else float("nan")
    )
    summary["ignored_feature_preservation_loss"] = (
        round(float(np.mean(ignored_losses)), 6) if ignored_losses else float("nan")
    )
    summary["learning_rate"] = round(float(optimizer.param_groups[0]["lr"]), 10)
    return summary, global_step


@torch.inference_mode()
def _evaluate(
    extractor: DINOTextSegmenter,
    adapter: DenseTextAdapter,
    text_features: Tensor,
    preservation_text_features: Tensor,
    class_weights: Tensor,
    loader: DataLoader[tuple[Tensor, Tensor]],
    config: OemTrainingConfig,
) -> dict[str, Any]:
    adapter.eval()
    losses: list[float] = []
    source_losses: list[float] = []
    preservation_losses: list[float] = []
    ignored_losses: list[float] = []
    metrics = _SegmentationConfusion(len(OEM_CLASSES))
    for rgb, target in loader:
        rgb = rgb.to(extractor.device, non_blocking=True)
        target = target.to(extractor.device, non_blocking=True)
        outputs = _adapter_outputs(extractor, adapter, text_features, rgb, config)
        source_loss = _segmentation_loss(
            outputs.logits,
            target,
            dice_weight=config.dice_weight,
            class_weights=class_weights,
            label_smoothing=config.label_smoothing,
        )
        preservation_loss = _open_vocabulary_preservation_loss(
            outputs.base_features,
            outputs.patch_features,
            preservation_text_features,
        )
        ignored_loss = _ignored_feature_preservation_loss(
            outputs.base_features,
            outputs.patch_features,
            target,
        )
        losses.append(
            float(
                (
                    source_loss
                    + config.open_vocabulary_preservation_weight * preservation_loss
                    + config.ignored_feature_preservation_weight * ignored_loss
                ).item()
            )
        )
        source_losses.append(float(source_loss.item()))
        preservation_losses.append(float(preservation_loss.item()))
        ignored_losses.append(float(ignored_loss.item()))
        metrics.update(outputs.logits.argmax(dim=1), target)
    summary = metrics.summary()
    summary["loss"] = round(float(np.mean(losses)), 6) if losses else float("nan")
    summary["source_loss"] = round(float(np.mean(source_losses)), 6) if source_losses else float("nan")
    summary["open_vocabulary_preservation_loss"] = (
        round(float(np.mean(preservation_losses)), 6) if preservation_losses else float("nan")
    )
    summary["ignored_feature_preservation_loss"] = (
        round(float(np.mean(ignored_losses)), 6) if ignored_losses else float("nan")
    )
    return summary


@dataclass(frozen=True)
class _AdapterOutputs:
    logits: Tensor
    base_features: Tensor
    patch_features: Tensor


def _adapter_outputs(
    extractor: DINOTextSegmenter,
    adapter: DenseTextAdapter,
    text_features: Tensor,
    rgb: Tensor,
    config: OemTrainingConfig,
) -> _AdapterOutputs:
    # The two DINO branches are frozen but intentionally use no_grad, rather
    # than inference_mode, because adapter convolutions need their inputs during backward.
    lvd_features, _ = extractor.encode_image_for_adapter(rgb)
    satellite_features = extractor.encode_satellite_structure_for_adapter(rgb)
    with torch.autocast(
        device_type=extractor.device.type,
        dtype=torch.bfloat16,
        enabled=config.amp and extractor.device.type == "cuda",
    ):
        patch_features = adapter(lvd_features, satellite_features)
        logits = torch.einsum("bdhw,cd->bchw", patch_features, text_features)
    logits = F.interpolate(logits.float(), size=rgb.shape[-2:], mode="bilinear", align_corners=False)
    return _AdapterOutputs(
        logits=logits / config.output_temperature,
        base_features=lvd_features,
        patch_features=patch_features,
    )


def _segmentation_loss(
    logits: Tensor,
    target: Tensor,
    *,
    dice_weight: float,
    class_weights: Tensor | None = None,
    label_smoothing: float = 0.0,
) -> Tensor:
    cross_entropy = F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        ignore_index=255,
        label_smoothing=label_smoothing,
    )
    if dice_weight == 0:
        return cross_entropy
    valid = target != 255
    if not bool(valid.any()):
        raise ValueError("A training batch contains no labeled OpenEarthMap pixels.")
    target_safe = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(target_safe, num_classes=logits.shape[1]).permute(0, 3, 1, 2).to(logits.dtype)
    valid_mask = valid.unsqueeze(1).to(logits.dtype)
    probabilities = torch.softmax(logits, dim=1) * valid_mask
    one_hot = one_hot * valid_mask
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    present = one_hot.sum(dim=(0, 2, 3)) > 0
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice_loss = 1.0 - dice[present].mean()
    return (1.0 - dice_weight) * cross_entropy + dice_weight * dice_loss


def _open_vocabulary_preservation_loss(
    base_features: Tensor,
    patch_features: Tensor,
    preservation_text_features: Tensor,
) -> Tensor:
    """Keep source adaptation from erasing frozen dino.txt concepts outside OEM."""

    if base_features.shape != patch_features.shape:
        raise ValueError("Base and adapted features must have identical shapes.")
    base_scores = torch.einsum("bdhw,cd->bchw", base_features, preservation_text_features)
    adapted_scores = torch.einsum("bdhw,cd->bchw", patch_features, preservation_text_features)
    return F.smooth_l1_loss(adapted_scores, base_scores.detach())


def _ignored_feature_preservation_loss(base_features: Tensor, patch_features: Tensor, target: Tensor) -> Tensor:
    """Freeze unknown/unlabeled source regions instead of letting source CE drift them."""

    ignored = F.interpolate(
        (target == 255).unsqueeze(1).to(dtype=base_features.dtype),
        size=base_features.shape[-2:],
        mode="area",
    ).squeeze(1) > 0.5
    if not bool(ignored.any()):
        return base_features.new_zeros(())
    cosine_distance = 1.0 - (base_features * patch_features).sum(dim=1)
    return cosine_distance[ignored].mean()


def _class_balanced_weights(counts: np.ndarray, *, power: float, maximum: float) -> Tensor:
    if counts.ndim != 1 or counts.shape[0] != len(OEM_CLASSES):
        raise ValueError("OpenEarthMap class counts must have one value per source class.")
    if power == 0:
        return torch.ones(len(OEM_CLASSES), dtype=torch.float32)
    positive = counts > 0
    if not bool(positive.any()):
        raise ValueError("OpenEarthMap training split has no labeled source pixels.")
    counts_float = torch.from_numpy(counts.astype(np.float64, copy=False))
    weights = torch.ones(len(OEM_CLASSES), dtype=torch.float64)
    observed_counts = counts_float[torch.from_numpy(positive)]
    observed_weights = (observed_counts.mean() / observed_counts).pow(power)
    weights[torch.from_numpy(positive)] = observed_weights / observed_weights.mean()
    return weights.clamp(min=1.0 / maximum, max=maximum).to(dtype=torch.float32)


class _SegmentationConfusion:
    def __init__(self, classes: int) -> None:
        self.classes = classes
        self.matrix = torch.zeros((classes, classes), dtype=torch.int64)

    def update(self, prediction: Tensor, target: Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target shapes must match for metrics.")
        valid = target != 255
        if not bool(valid.any()):
            return
        prediction = prediction[valid].to(torch.int64).cpu()
        target = target[valid].to(torch.int64).cpu()
        if int(prediction.min()) < 0 or int(prediction.max()) >= self.classes:
            raise ValueError("Prediction contains a class outside the source vocabulary.")
        encoded = target * self.classes + prediction
        self.matrix += torch.bincount(encoded, minlength=self.classes * self.classes).reshape(self.classes, self.classes)

    def summary(self) -> dict[str, Any]:
        diagonal = self.matrix.diag().to(torch.float64)
        total = self.matrix.sum().to(torch.float64)
        union = self.matrix.sum(dim=1).to(torch.float64) + self.matrix.sum(dim=0).to(torch.float64) - diagonal
        iou = torch.where(union > 0, diagonal / union, torch.full_like(union, float("nan")))
        return {
            "mean_iou": round(float(torch.nanmean(iou).item()), 6),
            "pixel_accuracy": round(float((diagonal.sum() / total).item()) if total else 0.0, 6),
            "labeled_pixels": int(total.item()),
            "per_class_iou": {
                spec.name: (round(float(score.item()), 6) if not torch.isnan(score) else None)
                for spec, score in zip(OEM_CLASSES, iou)
            },
        }


def _make_loader(
    dataset: OpenEarthMapDataset,
    config: OemTrainingConfig,
    *,
    shuffle: bool,
) -> DataLoader[tuple[Tensor, Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(config.seed + (1 if shuffle else 2))
    return DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
        persistent_workers=config.num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _resume_if_requested(
    config: OemTrainingConfig,
    adapter: DenseTextAdapter,
    optimizer: torch.optim.Optimizer,
    adapter_config: DenseAdapterConfig,
    source_dataset: dict[str, Any],
) -> tuple[int, float, int]:
    if config.resume_checkpoint is None:
        return 1, float("-inf"), 0
    payload = load_adapter_checkpoint(config.resume_checkpoint)
    if payload["adapter_config"] != asdict(adapter_config):
        raise ValueError("The resume checkpoint's adapter architecture differs from this requested run.")
    if payload.get("source_dataset") != source_dataset:
        raise ValueError("The resume checkpoint's external data signature differs from this requested run.")
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    state = payload.get("optimizer_state_dict")
    if not isinstance(state, dict) or "optimizer" not in state:
        raise ValueError("The resume checkpoint does not include optimizer state.")
    optimizer.load_state_dict(state["optimizer"])
    return int(payload["epoch"]) + 1, float(state.get("best_miou", float("-inf"))), int(state.get("global_step", 0))


def _ensure_disjoint_splits(train_samples: list[Any], val_samples: list[Any]) -> None:
    overlap = {sample.key for sample in train_samples} & {sample.key for sample in val_samples}
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"OpenEarthMap train and validation splits overlap, for example: {preview}")


def _cosine_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1e-4)
    progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
    return max(0.5 * (1.0 + math.cos(math.pi * progress)), 1e-4)


def _set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _jsonable_config(config: OemTrainingConfig) -> dict[str, Any]:
    return asdict(config)


def _signature_training_config(config: OemTrainingConfig) -> dict[str, Any]:
    payload = _jsonable_config(config)
    payload.pop("resume_checkpoint", None)
    return payload


def _ensure_signature(path: Path, payload: dict[str, Any], *, allow_existing: bool) -> None:
    normalized = json.loads(json.dumps(payload, ensure_ascii=True))
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise ValueError(f"Existing training configuration differs at {path}. Use a new output directory.")
        if not allow_existing:
            raise ValueError(f"Training output already exists at {path}. Pass --resume-checkpoint to continue it.")
        return
    _write_json_atomic(path, normalized)


def _append_json_line(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
