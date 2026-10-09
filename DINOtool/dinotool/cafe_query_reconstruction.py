"""DINO-RC with query-conditioned, evidence-preserving reconstruction."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .cafe_relational_parallel import (
    CafeRelationalParallel,
    RelationalParallelConfig,
    relational_parallel_loss,
)
from .relational_parallel import relation_supervision


@dataclass(frozen=True)
class QueryReconstructionConfig:
    arm: str = "query_reconstruction"
    stages: int = 6
    relation_dim: int = 32
    spatial_weight: float = 0.05
    class_weight: float = 0.05
    solver_steps: int = 4
    solver_step_size: float = 2.0 / 2.85
    relation_weight: float = 0.05
    corrected_class_attention: bool = True
    guide_dim: int = 16
    residual_scale: float = 0.10
    consistency_weight: float = 0.05

    def validate(self) -> None:
        if self.arm != "query_reconstruction":
            raise ValueError("Unknown query reconstruction arm")
        if min(self.stages, self.relation_dim, self.solver_steps, self.guide_dim) < 1:
            raise ValueError("Stages and reconstruction dimensions must be positive")
        if self.guide_dim % 4:
            raise ValueError("guide_dim must be divisible by four")
        if min(self.spatial_weight, self.class_weight, self.solver_step_size,
               self.relation_weight, self.residual_scale, self.consistency_weight) <= 0:
            raise ValueError("Reconstruction and supervision weights must be positive")

    def relational(self) -> RelationalParallelConfig:
        return RelationalParallelConfig(
            stages=self.stages,
            relation_dim=self.relation_dim,
            spatial_weight=self.spatial_weight,
            class_weight=self.class_weight,
            solver_steps=self.solver_steps,
            solver_step_size=self.solver_step_size,
            relation_weight=self.relation_weight,
            corrected_class_attention=self.corrected_class_attention,
        )


class EvidencePreservingQueryReconstructor(nn.Module):
    """Lift ``[B,D,Q,h,w]`` costs while preserving each patch mean exactly."""

    def __init__(self, cost_dim: int, text_dim: int, guide_dim: int, residual_scale: float) -> None:
        super().__init__()
        if min(cost_dim, text_dim, guide_dim) < 1 or residual_scale <= 0:
            raise ValueError("Invalid query reconstruction dimensions")
        self.cost_dim = int(cost_dim)
        self.guide_dim = int(guide_dim)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, guide_dim, 3, padding=1, bias=False),
            nn.GroupNorm(4, guide_dim), nn.GELU(),
            nn.Conv2d(guide_dim, guide_dim, 3, padding=1, groups=guide_dim, bias=False),
            nn.Conv2d(guide_dim, guide_dim, 1),
        )
        self.cost_encoder = nn.Conv2d(cost_dim, guide_dim, 1, bias=False)
        self.text_modulation = nn.Linear(text_dim, 2 * guide_dim)
        self.mixer = nn.Sequential(
            nn.GroupNorm(4, guide_dim), nn.GELU(),
            nn.Conv2d(guide_dim, guide_dim, 3, padding=1, groups=guide_dim, bias=False),
            nn.Conv2d(guide_dim, 2 * guide_dim, 1), nn.GELU(),
            nn.Conv2d(2 * guide_dim, guide_dim, 1),
        )
        self.residual_projection = nn.Conv2d(guide_dim, cost_dim, 1, bias=False)
        self.residual_gain = nn.Parameter(torch.tensor(math.atanh(min(residual_scale, 0.99))))
        nn.init.normal_(self.residual_projection.weight, std=0.01 / math.sqrt(guide_dim))

    @staticmethod
    def _patch_projection(value: Tensor, low_size: tuple[int, int]) -> Tensor:
        mean = F.adaptive_avg_pool2d(value.float(), low_size).to(dtype=value.dtype)
        return value - F.interpolate(mean, size=value.shape[-2:], mode="nearest")

    @staticmethod
    def _evidence_base(cost: Tensor, output_size: tuple[int, int]) -> Tensor:
        low_size = cost.shape[-2:]
        base = F.interpolate(cost, size=output_size, mode="bilinear", align_corners=False)
        observed = F.adaptive_avg_pool2d(base.float(), low_size).to(dtype=base.dtype)
        return base + F.interpolate(cost - observed, size=output_size, mode="nearest")

    def forward(self, images: Tensor, cost: Tensor, text: Tensor) -> tuple[Tensor, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected RGB BCHW images")
        if cost.ndim != 5 or cost.shape[0] != images.shape[0] or cost.shape[1] != self.cost_dim:
            raise ValueError("Expected cost [B,D,Q,h,w] sharing the image batch")
        if text.ndim != 2 or text.shape[0] != cost.shape[2]:
            raise ValueError("Text embeddings must share the cost query axis")
        batch, dim, queries, low_h, low_w = cost.shape
        output_size = images.shape[-2:]
        if output_size[0] % low_h or output_size[1] % low_w:
            raise ValueError("High-resolution grid must be an integer multiple of the cost grid")

        flat_cost = cost.permute(0, 2, 1, 3, 4).reshape(batch * queries, dim, low_h, low_w)
        base = self._evidence_base(flat_cost, output_size)
        image = self.image_encoder(images)
        image = image[:, None].expand(-1, queries, -1, -1, -1).reshape(
            batch * queries, self.guide_dim, *output_size
        )
        coarse = F.interpolate(self.cost_encoder(flat_cost), size=output_size, mode="bilinear", align_corners=False)
        scale, bias = self.text_modulation(F.normalize(text, dim=-1)).chunk(2, dim=-1)
        scale = scale[None].expand(batch, -1, -1).reshape(batch * queries, self.guide_dim, 1, 1)
        bias = bias[None].expand(batch, -1, -1).reshape(batch * queries, self.guide_dim, 1, 1)
        context = coarse + image * (1.0 + 0.5 * torch.tanh(scale)) + bias
        residual = self.residual_projection(context + self.mixer(context))
        residual = self._patch_projection(residual, (low_h, low_w))
        features = base + torch.tanh(self.residual_gain) * residual
        pooled = F.adaptive_avg_pool2d(features.float(), (low_h, low_w))
        preservation_error = (pooled - flat_cost.float()).square().mean()
        return features, preservation_error


class CafeQueryReconstruction(CafeRelationalParallel):
    """DINO relational reconstruction with query-conditioned high-res lifting."""

    def __init__(self, cafe: nn.Module, config: QueryReconstructionConfig) -> None:
        config.validate()
        super().__init__(cafe, config.relational())
        self.query_config = config
        self.reconstructor = EvidencePreservingQueryReconstructor(
            self.cost_dim, 1024, config.guide_dim, config.residual_scale
        )
        self.train(False)

    @property
    def checkpoint_format(self) -> str:
        return "cafe_query_reconstruction_v1"

    def _query_decode(self, images: Tensor, cost: Tensor, text: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, dim, queries, low_h, low_w = cost.shape
        features, preservation_error = self.reconstructor(images, cost, text)
        logits = self.cafe.reduce_d(features).reshape(batch, queries, *images.shape[-2:])
        coarse = cost.permute(0, 2, 1, 3, 4).reshape(batch * queries, dim, low_h, low_w)
        coarse_logits = self.cafe.reduce_d(coarse).reshape(batch, queries, low_h, low_w)
        pooled_logits = F.adaptive_avg_pool2d(logits.float(), (low_h, low_w))
        consistency = F.smooth_l1_loss(pooled_logits, coarse_logits.float())
        return logits, consistency, preservation_error

    def forward(self, images: Tensor, text: Tensor, *, output_count: int | None = None,
                relation_target: Tensor | None = None) -> dict[str, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 16 or any(
            size % 16 for size in images.shape[-2:]
        ):
            raise ValueError("Expected RGB BCHW images divisible by 16")
        if text.ndim != 2 or text.shape[1] != 1024 or not len(text):
            raise ValueError("Expected nonempty [queries, 1024] text embeddings")
        count = len(text) if output_count is None else output_count
        if not 1 <= count <= len(text):
            raise ValueError("Invalid output_count")
        if relation_target is not None and relation_target.shape != images.shape[:1] + images.shape[-2:]:
            raise ValueError("Relation target must match the input image grid")
        height, width = images.shape[-2] // 16, images.shape[-1] // 16
        self._set_aggregator_resolution(height, width, images.device)
        visual, _ = self._encode(images)
        cost, text_guidance, visual_guidance = self._initial_state(visual, text)
        spatial_losses, class_losses = [], []
        for spatial, channel, fusion in zip(self.spatial, self.channel, self.fusion):
            spatial_cost = spatial(cost, visual_guidance)
            semantic_cost = channel(cost, text_guidance)
            cost, edge_relation, class_relation = fusion(spatial_cost, semantic_cost, cost, visual, text)
            if relation_target is not None:
                spatial_loss, class_loss = relation_supervision(
                    edge_relation, class_relation, relation_target, self.relation_readout
                )
                spatial_losses.append(spatial_loss)
                class_losses.append(class_loss)
        logits, consistency, preservation_error = self._query_decode(
            images, cost[:, :, :count].contiguous(), text[:count]
        )
        result = {
            "logits": logits,
            "coarse_consistency_loss": consistency,
            "feature_preservation_error": preservation_error,
        }
        if relation_target is not None:
            spatial_loss = torch.stack(spatial_losses).mean()
            class_loss = torch.stack(class_losses).mean()
            result.update(
                relation_loss=0.5 * (spatial_loss + class_loss),
                spatial_relation_loss=spatial_loss,
                class_relation_loss=class_loss,
            )
        return result

    def architecture(self) -> dict[str, Any]:
        return dict(
            name="DINO-QueryReconstruction-v1",
            **asdict(self.query_config),
            topology=("DINO relational low-resolution reconstruction followed by shared "
                      "RGB/cost/text-conditioned high-resolution residual transport"),
            reconstruction_contract=("bilinear evidence base corrected to exact DINO-patch means; "
                                     "query-conditioned residual has zero mean in every patch"),
            consistency="pooled high-resolution logits preserve low-resolution relational logits",
            class_attention=("corrected query-axis SDPA" if self.query_config.corrected_class_attention
                             else "published CAFe implementation"),
            text_frozen=True,
            anyup_used=False,
            reduce_d_trainable=True,
            tuned_visual_indices=list(self.tuned_indices),
            source_training="locked COCO-Stuff 2017 only; external datasets excluded from training and selection",
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            frozen_parameters=sum(p.numel() for p in self.parameters() if not p.requires_grad),
        )


def query_reconstruction_loss(output: dict[str, Tensor], target: Tensor, teacher_logits: Tensor, *,
                              smoothing: float = 0.1, kd_weight: float = 0.02,
                              relation_weight: float = 0.05,
                              consistency_weight: float = 0.05) -> tuple[Tensor, dict[str, Tensor]]:
    base, terms = relational_parallel_loss(
        output, target, teacher_logits, smoothing=smoothing,
        kd_weight=kd_weight, relation_weight=relation_weight,
    )
    consistency = output["coarse_consistency_loss"]
    terms["coarse_consistency_loss"] = consistency
    terms["feature_preservation_error"] = output["feature_preservation_error"]
    return base + consistency_weight * consistency, terms


def config_from_checkpoint(payload: dict) -> QueryReconstructionConfig:
    architecture = payload.get("architecture", {})
    if (payload.get("format") != "cafe_query_reconstruction_v1" or
            architecture.get("name") != "DINO-QueryReconstruction-v1"):
        raise ValueError("Not a DINO-QueryReconstruction-v1 checkpoint")
    config = QueryReconstructionConfig(
        **{item.name: architecture[item.name] for item in fields(QueryReconstructionConfig)}
    )
    config.validate()
    return config
