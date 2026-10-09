from __future__ import annotations

"""Multi-prototype unbalanced optimal-transport DINO OVSS decoder.

The module deliberately keeps the DINO text/image interface class agnostic.  A
prompted class is represented by several shared visual modes anchored at its
text embedding.  A relaxed (unbalanced) transport layer assigns patch mass to
those modes and to an explicit unknown sink, so absent classes are not forced
to explain every pixel.
"""

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import CheckpointConfig
from .model import DINOTextSegmenter, checkpoint_manifest
from .prompts import ClassSpec


UOT_FORMAT_VERSION = 1


class UOTOutput:
    """Structured decoder output used by training and diagnostics."""

    def __init__(self, logits: Tensor, unknown_probability: Tensor, mode_mass: Tensor) -> None:
        self.logits = logits
        self.unknown_probability = unknown_probability
        self.mode_mass = mode_mass


@dataclass(frozen=True)
class UOTPrototypeConfig:
    """Architecture and numerical settings for :class:`UOTPrototypeDecoder`."""

    feature_dim: int
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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive.")
        if self.num_modes < 1:
            raise ValueError("num_modes must be positive.")
        if self.epsilon <= 0 or self.marginal_relaxation <= 0:
            raise ValueError("epsilon and marginal_relaxation must be positive.")
        if not 0 < self.unknown_prior < 1:
            raise ValueError("unknown_prior must be in (0, 1).")
        if self.unknown_cost < 0 or self.mode_offset_scale < 0:
            raise ValueError("unknown_cost and mode_offset_scale must be non-negative.")
        if self.base_score_weight < 0 or self.transport_score_weight < 0:
            raise ValueError("score weights must be non-negative.")
        if self.base_score_weight + self.transport_score_weight <= 0:
            raise ValueError("At least one score weight must be positive.")
        if self.sinkhorn_iterations < 1:
            raise ValueError("sinkhorn_iterations must be positive.")
        if self.mode_diversity_weight < 0:
            raise ValueError("mode_diversity_weight must be non-negative.")


class UOTPrototypeDecoder(nn.Module):
    """Text-anchored multi-prototype decoder with unbalanced Sinkhorn routing.

    Inputs are DINO text-aligned patch features ``[B, D, H, W]`` and arbitrary
    prompted text features ``[C, D]``.  The transport problem is solved over
    patch mass, not over a fixed source classifier, so the class vocabulary can
    change at inference time.
    """

    def __init__(self, config: UOTPrototypeConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        # Shared offsets are deliberately small and text-anchored.  The random
        # seed is controlled by the training entry point for reproducibility.
        self.mode_offsets = nn.Parameter(torch.empty(config.num_modes, config.feature_dim))
        nn.init.normal_(self.mode_offsets, mean=0.0, std=0.02)
        self.mode_bias = nn.Parameter(torch.zeros(config.num_modes))
        self.unknown_bias = nn.Parameter(torch.zeros(()))

    def prototypes(self, text_features: Tensor) -> Tensor:
        self._validate_text(text_features)
        text = F.normalize(text_features.float(), dim=-1)
        offsets = self.config.mode_offset_scale * torch.tanh(self.mode_offsets)
        prototypes = F.normalize(text[:, None, :] + offsets[None, :, :], dim=-1)
        return prototypes

    def forward(
        self,
        patch_features: Tensor,
        text_features: Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | UOTOutput:
        self._validate_inputs(patch_features, text_features)
        original_dtype = patch_features.dtype
        visual = F.normalize(patch_features.float(), dim=1)
        text = F.normalize(text_features.float(), dim=-1)
        batch, _, height, width = visual.shape
        pixels = visual.permute(0, 2, 3, 1).reshape(batch, height * width, -1)
        prototypes = self.prototypes(text)
        similarities = torch.einsum("bnd,ckd->bnck", pixels, prototypes)
        base_similarity = torch.einsum("bnd,cd->bnc", pixels, text)

        # Cost is kept in float32 even when the DINO backbone uses bfloat16;
        # log-domain Sinkhorn is noticeably more stable for large vocabularies.
        known_cost = 1.0 - similarities - self.mode_bias[None, None, None, :]
        unknown_cost = (
            torch.as_tensor(self.config.unknown_cost, device=visual.device, dtype=visual.dtype)
            - self.unknown_bias
        )
        unknown_cost = unknown_cost.expand(batch, pixels.shape[1], 1)
        cost = torch.cat((known_cost.reshape(batch, pixels.shape[1], -1), unknown_cost), dim=-1)
        log_plan = unbalanced_sinkhorn_log_plan(
            cost,
            epsilon=self.config.epsilon,
            marginal_relaxation=self.config.marginal_relaxation,
            unknown_prior=self.config.unknown_prior,
            iterations=self.config.sinkhorn_iterations,
        )
        row_log_mass = torch.logsumexp(log_plan, dim=-1, keepdim=True)
        row_log_probability = log_plan - row_log_mass
        known_log_probability = torch.logsumexp(
            row_log_probability[..., : text.shape[0] * self.config.num_modes].reshape(
                batch, pixels.shape[1], text.shape[0], self.config.num_modes
            ),
            dim=-1,
        )
        unknown_probability = row_log_probability[..., -1].exp()
        # Subtracting a uniform-class reference makes the transport term have a
        # scale comparable to cosine similarity while retaining unknown mass.
        transport_score = known_log_probability + torch.log(torch.as_tensor(text.shape[0], device=visual.device))
        logits = (
            self.config.base_score_weight * base_similarity
            + self.config.transport_score_weight * transport_score
        )
        logits = logits.transpose(1, 2).reshape(batch, text.shape[0], height, width)
        mode_mass = row_log_probability[..., : text.shape[0] * self.config.num_modes].exp().reshape(
            batch, pixels.shape[1], text.shape[0], self.config.num_modes
        )
        mode_mass = mode_mass.permute(0, 2, 3, 1).reshape(batch, text.shape[0], self.config.num_modes, height, width)
        unknown_probability = unknown_probability.reshape(batch, height, width)
        if return_diagnostics:
            return UOTOutput(logits=logits.to(dtype=original_dtype), unknown_probability=unknown_probability, mode_mass=mode_mass)
        return logits.to(dtype=original_dtype)

    def diversity_loss(self) -> Tensor:
        """Penalize collapse of shared mode offsets without forcing separation from text."""

        if self.config.num_modes < 2 or self.config.mode_diversity_weight == 0:
            return self.mode_offsets.new_zeros(())
        offsets = F.normalize(self.mode_offsets, dim=-1)
        gram = offsets @ offsets.t()
        mask = ~torch.eye(self.config.num_modes, device=gram.device, dtype=torch.bool)
        return self.config.mode_diversity_weight * gram[mask].pow(2).mean()

    def _validate_text(self, text_features: Tensor) -> None:
        if text_features.ndim != 2 or text_features.shape[1] != self.config.feature_dim:
            raise ValueError("text_features must have shape [classes, feature_dim].")
        if text_features.shape[0] < 1:
            raise ValueError("At least one text feature is required.")

    def _validate_inputs(self, patch_features: Tensor, text_features: Tensor) -> None:
        if patch_features.ndim != 4 or patch_features.shape[1] != self.config.feature_dim:
            raise ValueError("patch_features must have shape [B, feature_dim, H, W].")
        self._validate_text(text_features)
        if patch_features.shape[-1] < 1 or patch_features.shape[-2] < 1:
            raise ValueError("patch_features must have a non-empty spatial grid.")


def unbalanced_sinkhorn_log_plan(
    cost: Tensor,
    *,
    epsilon: float,
    marginal_relaxation: float,
    unknown_prior: float,
    iterations: int,
) -> Tensor:
    """Return a differentiable log transport plan for patch-to-prototype OT.

    KL-relaxed marginals yield the unbalanced Sinkhorn updates.  All inputs are
    finite float tensors; the result has shape ``[B, N, M]`` and is normalized
    neither per row nor per column (the caller chooses the desired posterior).
    """

    if cost.ndim != 3:
        raise ValueError("cost must have shape [batch, pixels, support].")
    if cost.shape[1] < 1 or cost.shape[2] < 2:
        raise ValueError("cost must contain at least one pixel and one known plus unknown support.")
    if epsilon <= 0 or marginal_relaxation <= 0 or iterations < 1:
        raise ValueError("Sinkhorn parameters are invalid.")
    if not 0 < unknown_prior < 1:
        raise ValueError("unknown_prior must be in (0, 1).")
    cost = cost.float()
    batch, pixels, support = cost.shape
    known = support - 1
    log_kernel = (-cost / epsilon).clamp(min=-80.0, max=80.0)
    log_a = cost.new_full((batch, pixels), -math.log(float(pixels)))
    known_prior = (1.0 - unknown_prior) / known
    log_b = cost.new_full((batch, support), math.log(known_prior))
    log_b[:, -1] = math.log(unknown_prior)
    relaxation = marginal_relaxation / (marginal_relaxation + epsilon)
    log_u = cost.new_zeros((batch, pixels))
    log_v = cost.new_zeros((batch, support))
    for _ in range(iterations):
        log_u = relaxation * (log_a - torch.logsumexp(log_kernel + log_v[:, None, :], dim=-1))
        log_v = relaxation * (log_b - torch.logsumexp(log_kernel + log_u[:, :, None], dim=1))
    return log_u[:, :, None] + log_kernel + log_v[:, None, :]


class UOTAggregatedDINOTextSegmenter:
    """DINOv3/dino.txt model with an externally trained UOT decoder."""

    patch_size = DINOTextSegmenter.patch_size
    method_name = "DINOv3 dino.txt multi-prototype unbalanced-OT OVSS"
    implementation_note = (
        "DINOv3 dino.txt remains open-vocabulary. A source-trained, class-agnostic "
        "multi-prototype unbalanced transport decoder models class-internal modes "
        "and routes unmatched patch mass to an unknown sink."
    )

    def __init__(
        self,
        checkpoints: CheckpointConfig,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        amp: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        payload = load_uot_checkpoint(self.checkpoint_path)
        config = UOTPrototypeConfig(**payload["decoder_config"])
        self.base = DINOTextSegmenter(checkpoints, device=device, amp=amp, use_satellite=False)
        visual_tuning = payload.get("visual_tuning_state_dict")
        if visual_tuning is not None:
            self.base.load_visual_tuning_state_dict(visual_tuning)
        if self.base.text_feature_dim != config.feature_dim:
            raise ValueError(
                "UOT checkpoint feature dimension does not match dino.txt: "
                f"{config.feature_dim} != {self.base.text_feature_dim}."
            )
        self.decoder = UOTPrototypeDecoder(config).to(self.base.device)
        self.decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
        self.decoder.eval().requires_grad_(False)
        self.decoder_config = config
        self.decoder_metadata = payload

    @property
    def checkpoints(self) -> CheckpointConfig:
        return self.base.checkpoints

    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def satellite(self) -> None:
        return None

    @property
    def visual_tuning_manifest(self) -> dict[str, object]:
        return self.base.visual_tuning_manifest

    def encode_text(self, classes: Sequence[ClassSpec], batch_size: int = 64) -> Tensor:
        return self.base.encode_text(classes, batch_size=batch_size)

    def encode_image(self, rgb: Tensor) -> tuple[Tensor, Tensor]:
        return self.base.encode_image(rgb)

    def encode_image_for_cost_aggregation(
        self, rgb: Tensor, *, train_visual_backbone: bool = False
    ) -> tuple[Tensor, Tensor]:
        return self.base.encode_image_for_cost_aggregation(rgb, train_visual_backbone=train_visual_backbone)

    def encode_satellite_structure(self, rgb: Tensor) -> Tensor:
        raise RuntimeError("The UOT DINO-only model does not load the SAT structure encoder.")

    def similarity_logits(self, patch_features: Tensor, text_features: Tensor) -> Tensor:
        with torch.inference_mode(), self.base._autocast():
            return self.decoder(patch_features, text_features).float()

    def similarity_logits_with_unknown(self, patch_features: Tensor, text_features: Tensor) -> UOTOutput:
        with torch.inference_mode(), self.base._autocast():
            output = self.decoder(patch_features, text_features, return_diagnostics=True)
        if not isinstance(output, UOTOutput):
            raise RuntimeError("UOT decoder did not return diagnostics.")
        return UOTOutput(output.logits.float(), output.unknown_probability.float(), output.mode_mass.float())


def make_uot_checkpoint(
    *,
    decoder: UOTPrototypeDecoder,
    source_dataset: dict[str, Any],
    source_classes: Sequence[ClassSpec],
    training_config: dict[str, Any],
    dino_checkpoints: CheckpointConfig,
    epoch: int,
    validation: dict[str, Any],
    optimizer_state_dict: dict[str, Any] | None = None,
    visual_tuning_state_dict: dict[str, dict[str, Tensor]] | None = None,
    visual_tuning_manifest: dict[str, object] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": UOT_FORMAT_VERSION,
        "method": UOTAggregatedDINOTextSegmenter.method_name,
        "decoder_config": decoder.config.as_dict(),
        "decoder_state_dict": {key: value.detach().cpu() for key, value in decoder.state_dict().items()},
        "source_dataset": source_dataset,
        "source_classes": [{"name": spec.name, "synonyms": list(spec.synonyms)} for spec in source_classes],
        "training_config": training_config,
        "dino_checkpoints": checkpoint_manifest(dino_checkpoints),
        "epoch": epoch,
        "validation": validation,
        "optimizer_state_dict": optimizer_state_dict,
        "visual_tuning_state_dict": visual_tuning_state_dict,
        "visual_tuning_manifest": visual_tuning_manifest,
    }


def load_uot_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != UOT_FORMAT_VERSION:
        raise ValueError(f"Unsupported UOT checkpoint at {path}.")
    if not isinstance(payload.get("decoder_config"), dict) or not isinstance(payload.get("decoder_state_dict"), dict):
        raise ValueError(f"UOT checkpoint at {path} is missing architecture or weights.")
    UOTPrototypeConfig(**payload["decoder_config"]).validate()
    return payload


def uot_checkpoint_manifest(path: str | Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if payload is None:
        payload = load_uot_checkpoint(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "sha256": digest.hexdigest(),
        "format_version": payload["format_version"],
        "decoder_config": payload["decoder_config"],
        "source_dataset": payload.get("source_dataset"),
        "training_epoch": payload.get("epoch"),
        "validation": payload.get("validation"),
    }
