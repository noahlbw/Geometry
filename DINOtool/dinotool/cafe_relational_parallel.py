"""CAFe-DINO decoder with task-relational parallel cost reconstruction."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn

from .cafe_pca import CafePCA, PCADINOConfig
from .cafe_ped import ped_loss
from .relational_parallel import (
    AnchoredCostReconstruction,
    ClassRelationHead,
    SpatialRelationHead,
    relation_supervision,
)


@dataclass(frozen=True)
class RelationalParallelConfig:
    arm: str = "relational_parallel"
    stages: int = 6
    relation_dim: int = 32
    spatial_weight: float = 0.05
    class_weight: float = 0.05
    solver_steps: int = 4
    solver_step_size: float = 2.0 / 2.85
    relation_weight: float = 0.05
    corrected_class_attention: bool = True

    def validate(self) -> None:
        if self.arm != "relational_parallel":
            raise ValueError("Unknown relational parallel arm")
        if min(self.stages, self.relation_dim, self.solver_steps) < 1:
            raise ValueError("Stages, relation dimension, and solver steps must be positive")
        if min(self.spatial_weight, self.class_weight, self.solver_step_size, self.relation_weight) <= 0:
            raise ValueError("Relation and solver weights must be positive")


class RelationalParallelStage(nn.Module):
    """Per-stage relation heads and fixed-step anchored reconstruction."""

    def __init__(self, cost_dim: int, relation_dim: int, spatial_weight: float,
                 class_weight: float, solver_steps: int, solver_step_size: float) -> None:
        super().__init__()
        self.spatial_relation = SpatialRelationHead(cost_dim, 1024, relation_dim)
        self.class_relation = ClassRelationHead(cost_dim, 1024, relation_dim)
        self.reconstruct = AnchoredCostReconstruction(
            spatial_weight=spatial_weight,
            class_weight=class_weight,
            iterations=solver_steps,
            step_size=solver_step_size,
        )

    def forward(self, spatial: Tensor, semantic: Tensor, previous: Tensor, visual: Tensor, text: Tensor):
        edge_relation, edge_weight = self.spatial_relation(spatial, visual)
        class_relation = self.class_relation(semantic, text)
        anchor = spatial + semantic - previous
        cost = self.reconstruct(anchor, edge_relation, edge_weight, class_relation)
        return cost, edge_relation, class_relation


class CafeRelationalParallel(CafePCA):
    """Parallel CAFe branches whose relations jointly reconstruct the next cost state."""

    def __init__(self, cafe: nn.Module, config: RelationalParallelConfig) -> None:
        config.validate()
        super().__init__(
            cafe,
            PCADINOConfig(
                arm="parallel",
                stages=config.stages,
                corrected_class_attention=config.corrected_class_attention,
            ),
        )
        self.config = config
        self.fusion = nn.ModuleList([
            RelationalParallelStage(
                self.cost_dim,
                config.relation_dim,
                config.spatial_weight,
                config.class_weight,
                config.solver_steps,
                config.solver_step_size,
            )
            for _ in range(config.stages)
        ])
        self.relation_readout = nn.Parameter(torch.empty(self.cost_dim))
        nn.init.normal_(self.relation_readout, std=1.0 / self.cost_dim ** 0.5)
        self.train(False)

    @property
    def checkpoint_format(self) -> str:
        return "cafe_relational_parallel_v1"

    def forward(self, images: Tensor, text: Tensor, *, output_count: int | None = None,
                relation_target: Tensor | None = None) -> dict[str, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 16 or any(size % 16 for size in images.shape[-2:]):
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
                    edge_relation,
                    class_relation,
                    relation_target,
                    self.relation_readout,
                )
                spatial_losses.append(spatial_loss)
                class_losses.append(class_loss)
        result = {"logits": self._decode(images, cost, count)}
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
            name="DINO-RelationalParallel-v1",
            **asdict(self.config),
            topology=(
                "parallel Spatial(C)||Class(C); spatial edge and query-relative observations "
                "jointly reconstruct an anchored multi-channel cost field"
            ),
            class_attention="corrected query-axis SDPA" if self.config.corrected_class_attention else "published CAFe implementation",
            fusion_contract="R=Spatial(C)+Class(C)-C; four fixed FP32 Richardson reconstruction steps",
            spatial_relation="four canonical undirected 8-neighborhood edges with signed D-channel observations",
            class_relation="query-centered potential field; no explicit Q^2 pair tensor",
            relation_supervision="source-label patch proportions supervise a shared scalar projection of consumed g/h fields",
            text_frozen=True,
            anyup_frozen=True,
            reduce_d_trainable=True,
            tuned_visual_indices=list(self.tuned_indices),
            source_training="locked COCO/OEM; LoveDA excluded from training and selection",
            trainable_parameters=sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            frozen_parameters=sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad),
        )


def relational_parallel_loss(output: dict[str, Tensor], target: Tensor, teacher_logits: Tensor, *, smoothing: float = 0.1,
                             kd_weight: float = 0.02, relation_weight: float = 0.05) -> tuple[Tensor, dict[str, Tensor]]:
    """Locked CE/KL loss plus source-only relation supervision consumed by fusion."""
    if "relation_loss" not in output:
        raise ValueError("Relation target must be supplied during relational-parallel training")
    base, terms = ped_loss(output, target, teacher_logits, smoothing=smoothing, kd_weight=kd_weight)
    relation = output["relation_loss"]
    terms["region_loss"] = output["spatial_relation_loss"]
    terms["affinity_loss"] = output["class_relation_loss"]
    return base + relation_weight * relation, terms


def config_from_checkpoint(payload: dict) -> RelationalParallelConfig:
    architecture = payload.get("architecture", {})
    if payload.get("format") != "cafe_relational_parallel_v1" or architecture.get("name") != "DINO-RelationalParallel-v1":
        raise ValueError("Not a DINO-RelationalParallel-v1 checkpoint")
    config = RelationalParallelConfig(**{field.name: architecture[field.name] for field in fields(RelationalParallelConfig)})
    config.validate()
    return config
