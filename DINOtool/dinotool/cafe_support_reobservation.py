"""Parallel semantic evidence and ownership with fine-scale reobservation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SupportReobservationConfig:
    arm: str = "support_reobservation"
    rounds: int = 2
    detail_scale: int = 2
    hidden_dim: int = 32
    support_weight: float = 0.05
    propagation_init: float = 0.25
    feedback_init: float = 0.50
    residual_init: float = 0.10

    def validate(self) -> None:
        if self.arm != "support_reobservation":
            raise ValueError("Unknown support-reobservation arm")
        if self.rounds < 1 or self.detail_scale < 2 or self.hidden_dim < 4:
            raise ValueError("Need at least one round, two-scale observations, and hidden_dim >= 4")
        if self.hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by four")
        for value in (self.support_weight, self.propagation_init, self.feedback_init, self.residual_init):
            if not 0 < value < 1:
                raise ValueError("Loss and initialization weights must lie in (0, 1)")


def _inverse_sigmoid(value: float) -> float:
    return math.log(value / (1.0 - value))


def _cost_tokens(value: Tensor) -> Tensor:
    batch, dim, queries, height, width = value.shape
    return value.permute(0, 2, 1, 3, 4).reshape(batch * queries, dim, height, width)


def _restore_cost(value: Tensor, batch: int, queries: int) -> Tensor:
    _, dim, height, width = value.shape
    return value.reshape(batch, queries, dim, height, width).permute(0, 2, 1, 3, 4).contiguous()


def _resize_cost(value: Tensor, size: tuple[int, int]) -> Tensor:
    batch, _, queries, _, _ = value.shape
    resized = F.interpolate(_cost_tokens(value), size=size, mode="bilinear", align_corners=False)
    return _restore_cost(resized, batch, queries)


def _local_values(value: Tensor) -> tuple[Tensor, Tensor]:
    """Return row-major 3x3 cost neighbors and their in-image validity."""
    if value.ndim != 5:
        raise ValueError("Expected [B,D,Q,H,W] local value")
    batch, dim, queries, height, width = value.shape
    flat = value.reshape(batch, dim * queries, height, width)
    neighbors = F.unfold(flat, kernel_size=3, padding=1)
    neighbors = neighbors.reshape(batch, dim, queries, 9, height, width)
    valid = F.unfold(value.new_ones(batch, 1, height, width), kernel_size=3, padding=1)
    return neighbors, valid.reshape(batch, 9, height, width).bool()


def _local_maps(value: Tensor) -> tuple[Tensor, Tensor]:
    """Return row-major 3x3 neighbors for [B,C,H,W] maps."""
    if value.ndim != 4:
        raise ValueError("Expected [B,C,H,W] local map")
    batch, channels, height, width = value.shape
    neighbors = F.unfold(value, kernel_size=3, padding=1)
    neighbors = neighbors.reshape(batch, channels, 9, height, width)
    valid = F.unfold(value.new_ones(batch, 1, height, width), kernel_size=3, padding=1)
    return neighbors, valid.reshape(batch, 9, height, width).bool()


class SemanticEvidenceBranch(nn.Module):
    """Update per-query evidence without consuming current-round support."""

    def __init__(self, cost_dim: int, text_dim: int, hidden_dim: int, residual_init: float) -> None:
        super().__init__()
        self.cost_dim = cost_dim
        self.current = nn.Conv2d(cost_dim, hidden_dim, 1, bias=False)
        self.initial = nn.Conv2d(cost_dim, hidden_dim, 1, bias=False)
        self.text = nn.Linear(text_dim, 2 * hidden_dim)
        self.update = nn.Sequential(
            nn.GroupNorm(4, hidden_dim), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, 1), nn.GELU(),
            nn.Conv2d(hidden_dim, cost_dim, 1, bias=False),
        )
        self.gain = nn.Parameter(torch.tensor(_inverse_sigmoid(residual_init)))
        nn.init.normal_(self.update[-1].weight, std=0.01 / math.sqrt(hidden_dim))

    def forward(self, state: Tensor, initial: Tensor, text: Tensor) -> Tensor:
        if state.shape != initial.shape or state.ndim != 5 or state.shape[1] != self.cost_dim:
            raise ValueError("Semantic branch expects matching [B,D,Q,H,W] states")
        batch, _, queries, _, _ = state.shape
        if text.shape[0] != queries:
            raise ValueError("Text and cost query axes differ")
        current = _cost_tokens(state)
        reference = _cost_tokens(initial)
        scale, bias = self.text(F.normalize(text, dim=-1)).chunk(2, dim=-1)
        scale = scale[None].expand(batch, -1, -1).reshape(batch * queries, -1, 1, 1)
        bias = bias[None].expand(batch, -1, -1).reshape(batch * queries, -1, 1, 1)
        hidden = self.current(current) + self.initial(reference)
        hidden = hidden * (1.0 + 0.5 * torch.tanh(scale)) + bias
        updated = current + torch.sigmoid(self.gain) * self.update(hidden)
        return _restore_cost(updated, batch, queries)


class EvidenceOwnershipBranch(nn.Module):
    """Predict a query-permutation-invariant 3x3 evidence ownership field."""

    def __init__(self, cost_dim: int, visual_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.state_readout = nn.Linear(cost_dim, 1, bias=False)
        self.visual = nn.Sequential(
            nn.Conv2d(visual_dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(4, hidden_dim), nn.GELU(),
        )
        self.predict = nn.Sequential(
            nn.Conv2d(hidden_dim + 4 + 9, hidden_dim, 3, padding=1),
            nn.GroupNorm(4, hidden_dim), nn.GELU(),
            nn.Conv2d(hidden_dim, 9, 1),
        )
        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)
        with torch.no_grad():
            self.predict[-1].bias[4] = 1.0

    def forward(self, visual: Tensor, state: Tensor, prior: Tensor) -> tuple[Tensor, Tensor]:
        if visual.ndim != 4 or state.ndim != 5 or prior.ndim != 4:
            raise ValueError("Invalid ownership inputs")
        batch, _, queries, height, width = state.shape
        if visual.shape[0] != batch or visual.shape[-2:] != (height, width):
            raise ValueError("Visual and semantic states must share the coarse grid")
        if prior.shape != (batch, 9, height, width):
            raise ValueError("Ownership prior must be [B,9,H,W]")
        tokens = state.permute(0, 2, 3, 4, 1)
        scores = self.state_readout(tokens).squeeze(-1)
        probabilities = scores.float().softmax(dim=1).to(dtype=scores.dtype)
        entropy = -(probabilities.float().clamp_min(1e-8).log() * probabilities.float()).sum(1, keepdim=True)
        entropy = entropy / math.log(max(queries, 2))
        summary = torch.cat((
            scores.mean(1, keepdim=True),
            scores.var(1, keepdim=True, unbiased=False).sqrt(),
            scores.amax(1, keepdim=True),
            entropy.to(dtype=scores.dtype),
        ), dim=1)
        logits = self.predict(torch.cat((self.visual(F.normalize(visual, dim=1)), summary, prior), dim=1))
        _, valid = _local_maps(summary[:, :1])
        logits = logits.masked_fill(~valid, -1e4)
        return logits, logits.softmax(dim=1)


class FineEvidenceReader(nn.Module):
    """Conditionally read actual fine-grid image/text evidence."""

    def __init__(self, cost_dim: int, visual_dim: int, text_dim: int,
                 hidden_dim: int, residual_init: float) -> None:
        super().__init__()
        self.cost_dim = cost_dim
        self.observed = nn.Conv2d(cost_dim, hidden_dim, 1, bias=False)
        self.context = nn.Conv2d(cost_dim, hidden_dim, 1, bias=False)
        self.difference = nn.Conv2d(cost_dim, hidden_dim, 1, bias=False)
        self.visual = nn.Conv2d(visual_dim, hidden_dim, 1, bias=False)
        self.text = nn.Linear(text_dim, 2 * hidden_dim)
        self.mixer = nn.Sequential(
            nn.GroupNorm(4, hidden_dim), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, 1), nn.GELU(),
        )
        self.evidence_gate = nn.Conv2d(hidden_dim + 1, 1, 1)
        self.residual = nn.Conv2d(hidden_dim, cost_dim, 1, bias=False)
        self.residual_gain = nn.Parameter(torch.tensor(_inverse_sigmoid(residual_init)))
        nn.init.zeros_(self.evidence_gate.weight)
        nn.init.zeros_(self.evidence_gate.bias)
        nn.init.normal_(self.residual.weight, std=0.01 / math.sqrt(hidden_dim))

    def forward(self, observed: Tensor, coarse: Tensor, visual: Tensor,
                text: Tensor, disagreement: Tensor) -> tuple[Tensor, Tensor]:
        if observed.shape != coarse.shape or observed.ndim != 5:
            raise ValueError("Fine reader expects matching observed/context costs")
        batch, _, queries, height, width = observed.shape
        if visual.shape[0] != batch or visual.shape[-2:] != (height, width):
            raise ValueError("Fine visual evidence must share the fine grid")
        if disagreement.shape != (batch, 1, height, width):
            raise ValueError("Disagreement must be [B,1,H,W]")
        observed_flat, coarse_flat = _cost_tokens(observed), _cost_tokens(coarse)
        visual_features = self.visual(F.normalize(visual, dim=1))
        visual_features = visual_features[:, None].expand(-1, queries, -1, -1, -1)
        visual_features = visual_features.reshape(batch * queries, -1, height, width)
        scale, bias = self.text(F.normalize(text, dim=-1)).chunk(2, dim=-1)
        scale = scale[None].expand(batch, -1, -1).reshape(batch * queries, -1, 1, 1)
        bias = bias[None].expand(batch, -1, -1).reshape(batch * queries, -1, 1, 1)
        hidden = self.observed(observed_flat) + self.context(coarse_flat)
        hidden = hidden + self.difference(observed_flat - coarse_flat) + visual_features
        hidden = hidden * (1.0 + 0.5 * torch.tanh(scale)) + bias
        hidden = hidden + self.mixer(hidden)
        disagreement_flat = disagreement[:, None].expand(-1, queries, -1, -1, -1)
        disagreement_flat = disagreement_flat.reshape(batch * queries, 1, height, width)
        gate = torch.sigmoid(self.evidence_gate(torch.cat((hidden, disagreement_flat), dim=1)))
        value = coarse_flat + gate * (observed_flat - coarse_flat)
        value = value + torch.sigmoid(self.residual_gain) * self.residual(hidden)
        return _restore_cost(value, batch, queries), gate.reshape(batch, queries, height, width)


def propagate_evidence(value: Tensor, weights: Tensor) -> Tensor:
    neighbors, valid = _local_values(value)
    if weights.shape != valid.shape:
        raise ValueError("Support weights must match the local stencil")
    weights = weights * valid.to(dtype=weights.dtype)
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
    return (neighbors * weights[:, None, None]).sum(3)


def soft_patch_labels(target: Tensor, classes: int, height: int, width: int) -> tuple[Tensor, Tensor]:
    """Return soft patch labels and fractional valid coverage."""
    if target.ndim != 3 or min(classes, height, width) < 1:
        raise ValueError("Invalid target grid")
    image_height, image_width = target.shape[-2:]
    if image_height % height or image_width % width:
        raise ValueError("Target must divide the coarse grid exactly")
    valid = target != 255
    labels = target.clamp(0, classes - 1)
    one_hot = F.one_hot(labels, classes).permute(0, 3, 1, 2).float()
    one_hot = one_hot * valid[:, None]
    kernel = (image_height // height, image_width // width)
    counts = F.avg_pool2d(one_hot, kernel, kernel)
    coverage = F.avg_pool2d(valid[:, None].float(), kernel, kernel)
    proportions = counts / coverage.clamp_min(1e-6)
    return proportions, coverage


def ownership_supervision(logits: Tensor, target: Tensor, classes: int) -> Tensor:
    """Match ownership to soft same-class transfer targets."""
    if logits.ndim != 4 or logits.shape[1] != 9:
        raise ValueError("Ownership logits must be [B,9,H,W]")
    _, _, height, width = logits.shape
    proportions, coverage = soft_patch_labels(target, classes, height, width)
    proportions = proportions.to(device=logits.device)
    coverage = coverage.to(device=logits.device)
    neighbors, valid = _local_maps(proportions)
    neighbor_coverage, _ = _local_maps(coverage)
    agreement = (proportions[:, :, None] * neighbors).sum(1)
    transfer = agreement * neighbor_coverage[:, 0] * valid.to(dtype=agreement.dtype)
    transfer[:, 4] = transfer[:, 4].clamp_min(coverage[:, 0])
    distribution = transfer / transfer.sum(1, keepdim=True).clamp_min(1e-6)
    values = -(distribution * logits.float().log_softmax(dim=1)).sum(1)
    center_weight = coverage[:, 0]
    return (values * center_weight).sum() / center_weight.sum().clamp_min(1.0)


class CafeSupportReobservation(nn.Module):
    """Frozen CAFe observations with trainable semantic/support feedback."""

    def __init__(self, cafe: nn.Module, config: SupportReobservationConfig) -> None:
        super().__init__()
        config.validate()
        self.cafe = cafe
        self.config = config
        self.cost_dim = int(cafe.aggregator_dim)
        self.cafe.requires_grad_(False)
        self.semantic = SemanticEvidenceBranch(
            self.cost_dim, 1024, config.hidden_dim, config.residual_init
        )
        self.ownership = EvidenceOwnershipBranch(self.cost_dim, 1024, config.hidden_dim)
        self.reader = FineEvidenceReader(
            self.cost_dim, 1024, 1024, config.hidden_dim, config.residual_init
        )
        self.propagation_gain = nn.Parameter(torch.tensor(_inverse_sigmoid(config.propagation_init)))
        self.feedback_gain = nn.Parameter(torch.tensor(_inverse_sigmoid(config.feedback_init)))
        self.train(False)

    @property
    def checkpoint_format(self) -> str:
        return "cafe_support_reobservation_v1"

    def train(self, mode: bool = True) -> "CafeSupportReobservation":
        nn.Module.train(self, mode)
        self.cafe.eval()
        for module in (self.semantic, self.ownership, self.reader):
            module.train(mode)
        return self

    def _encode(self, images: Tensor) -> Tensor:
        batch, _, image_height, image_width = images.shape
        height, width = image_height // 16, image_width // 16
        with torch.no_grad():
            _, _, tokens = self.cafe.backbone.encode_image_with_patch_tokens(images, normalize=False)
        if tokens.shape[1:] != (height * width, 1024):
            raise RuntimeError("Unexpected DINO patch-token contract")
        return tokens.transpose(1, 2).reshape(batch, 1024, height, width)

    def _initial_state(self, visual: Tensor, text: Tensor) -> Tensor:
        batch, _, height, width = visual.shape
        queries = len(text)
        with torch.no_grad():
            raw = torch.einsum(
                "bdhw,qd->bqhw", F.normalize(visual, dim=1), F.normalize(text, dim=-1)
            )
            cost = self.cafe.corr_embed(raw.reshape(batch * queries, 1, height, width))
        return cost.reshape(batch, queries, self.cost_dim, height, width).transpose(1, 2).contiguous()

    def _cost_logits(self, cost: Tensor) -> Tensor:
        batch, _, queries, height, width = cost.shape
        logits = self.cafe.reduce_d(_cost_tokens(cost))
        return logits.reshape(batch, queries, height, width)

    def _decode(self, images: Tensor, cost: Tensor, count: int) -> Tensor:
        batch, _, _, height, width = cost.shape
        image_height, image_width = images.shape[-2:]
        selected = cost[:, :, :count].contiguous().reshape(
            batch, self.cost_dim * count, height, width
        )
        upsampled = self.cafe.upsampler(images, selected)
        features = upsampled.reshape(batch, self.cost_dim, count, image_height, image_width)
        features = features.transpose(1, 2).reshape(
            batch * count, self.cost_dim, image_height, image_width
        )
        return self.cafe.reduce_d(features).reshape(batch, count, image_height, image_width)

    @staticmethod
    def _support_prior(probabilities: Tensor) -> Tensor:
        neighbors, valid = _local_maps(probabilities)
        agreement = (probabilities[:, :, None] * neighbors).sum(1).clamp_min(1e-6)
        return agreement.log().masked_fill(~valid, 0.0)

    def forward(self, images: Tensor, text: Tensor, *, output_count: int | None = None,
                ownership_target: Tensor | None = None) -> dict[str, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3 or any(
            size % 16 for size in images.shape[-2:]
        ):
            raise ValueError("Expected RGB BCHW detail images divisible by 16")
        if text.ndim != 2 or text.shape[1] != 1024 or not len(text):
            raise ValueError("Expected nonempty [queries,1024] text embeddings")
        scale = self.config.detail_scale
        if any(size % scale for size in images.shape[-2:]):
            raise ValueError("Detail image must divide the context scale exactly")
        context_size = (images.shape[-2] // scale, images.shape[-1] // scale)
        if any(size % 16 for size in context_size):
            raise ValueError("Context observation must remain divisible by 16")
        count = len(text) if output_count is None else output_count
        if not 1 <= count <= len(text):
            raise ValueError("Invalid output_count")
        if ownership_target is not None and ownership_target.shape != images.shape[:1] + images.shape[-2:]:
            raise ValueError("Ownership target must match the detail image grid")

        context_images = F.interpolate(images, size=context_size, mode="bilinear", align_corners=False)
        coarse_visual = self._encode(context_images)
        fine_visual = self._encode(images)
        coarse_initial = self._initial_state(coarse_visual, text)
        fine_initial = self._initial_state(fine_visual, text)
        state = coarse_initial
        prior = state.new_zeros(state.shape[0], 9, state.shape[-2], state.shape[-1])
        ownership_losses, entropies, revisions, feedback_deltas = [], [], [], []
        fine = fine_initial

        for _ in range(self.config.rounds):
            category = self.semantic(state, coarse_initial, text)
            support_logits, support = self.ownership(coarse_visual, state, prior)
            neighborhood = propagate_evidence(category, support)
            propagated = category + torch.sigmoid(self.propagation_gain) * (neighborhood - category)

            category_probability = self._cost_logits(category).float().softmax(1)
            neighborhood_probability = self._cost_logits(neighborhood).float().softmax(1)
            disagreement = (category_probability - neighborhood_probability).abs().sum(1, keepdim=True)
            disagreement = F.interpolate(
                disagreement, size=fine_initial.shape[-2:], mode="bilinear", align_corners=False
            ).to(dtype=propagated.dtype)
            fine_context = _resize_cost(propagated, fine_initial.shape[-2:])
            fine, _ = self.reader(fine_initial, fine_context, fine_visual, text, disagreement)
            feedback = F.adaptive_avg_pool2d(
                _cost_tokens(fine).float(), propagated.shape[-2:]
            ).to(dtype=propagated.dtype)
            feedback = _restore_cost(feedback, propagated.shape[0], propagated.shape[2])
            next_state = propagated + torch.sigmoid(self.feedback_gain) * (feedback - propagated)

            fine_probability = self._cost_logits(fine).float().softmax(1)
            coarse_probability = F.adaptive_avg_pool2d(fine_probability, propagated.shape[-2:])
            prior = self._support_prior(coarse_probability).to(dtype=propagated.dtype)
            state = next_state

            if ownership_target is not None:
                ownership_losses.append(
                    ownership_supervision(support_logits, ownership_target, len(text))
                )
            entropies.append(
                -(support.float().clamp_min(1e-8).log() * support.float()).sum(1).mean()
            )
            revisions.append((fine.float() - fine_context.float()).abs().mean())
            feedback_deltas.append((next_state.float() - propagated.float()).abs().mean())

        result = {
            "logits": self._decode(images, fine[:, :, :count].contiguous(), count),
            "support_entropy": torch.stack(entropies).mean(),
            "fine_revision": torch.stack(revisions).mean(),
            "feedback_delta": torch.stack(feedback_deltas).mean(),
        }
        if ownership_target is not None:
            result["ownership_loss"] = torch.stack(ownership_losses).mean()
        return result

    def optimizer_groups(self, new_lr: float, head_lr: float,
                         visual_lr: float) -> list[dict[str, Any]]:
        del head_lr, visual_lr
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        return [dict(params=parameters, role="new", lr=new_lr, initial_lr=new_lr)]

    def adapted_state_dict(self) -> dict[str, Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if not name.startswith("cafe.")
        }

    def load_adapted_state_dict(self, state: dict[str, Tensor]) -> None:
        expected, actual = set(self.adapted_state_dict()), set(state)
        if actual != expected:
            raise ValueError(
                f"Support-reobservation checkpoint keys differ: missing={expected - actual}, "
                f"unexpected={actual - expected}"
            )
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(
            not key.startswith("cafe.") for key in result.missing_keys
        ):
            raise RuntimeError("Incomplete support-reobservation inference state")

    def architecture(self) -> dict[str, Any]:
        return dict(
            name="DINO-SupportReobservation-v1",
            **asdict(self.config),
            topology=("parallel category evidence and query-shared 3x3 evidence ownership; "
                      "fine DINO observation revises both for the next shared round"),
            observation=("same source crop at native detail sampling plus an area-matched "
                         "downsampled context view; no interpolated pseudo-detail"),
            supervision="source segmentation plus one soft same-class ownership objective",
            text_frozen=True,
            dino_frozen=True,
            cafe_frozen=True,
            anyup_frozen=True,
            source_training="locked COCO-Stuff 2017 only; external datasets excluded",
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            frozen_parameters=sum(p.numel() for p in self.parameters() if not p.requires_grad),
        )


def support_reobservation_loss(output: dict[str, Tensor], target: Tensor, *,
                               smoothing: float = 0.1, support_weight: float = 0.05
                               ) -> tuple[Tensor, dict[str, Tensor]]:
    if "ownership_loss" not in output:
        raise ValueError("Ownership target is required during training")
    valid = target != 255
    if valid.any():
        segmentation = F.cross_entropy(
            output["logits"].float(), target, ignore_index=255, label_smoothing=smoothing
        )
    else:
        segmentation = output["logits"].sum() * 0
    ownership = output["ownership_loss"]
    total = segmentation + support_weight * ownership
    return total, {
        "ce_loss": segmentation,
        "ownership_loss": ownership,
        "support_entropy": output["support_entropy"],
        "fine_revision": output["fine_revision"],
        "feedback_delta": output["feedback_delta"],
    }


def config_from_checkpoint(payload: dict) -> SupportReobservationConfig:
    architecture = payload.get("architecture", {})
    if (payload.get("format") != "cafe_support_reobservation_v1" or
            architecture.get("name") != "DINO-SupportReobservation-v1"):
        raise ValueError("Not a DINO-SupportReobservation-v1 checkpoint")
    config = SupportReobservationConfig(
        **{item.name: architecture[item.name] for item in fields(SupportReobservationConfig)}
    )
    config.validate()
    return config
