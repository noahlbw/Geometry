"""DINO cost decoder with parallel shared-support pair-comparison fusion."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch
from torch import Tensor, nn

from .cafe_pca import CafePCA, PCADINOConfig
from .pair_fusion import PairComparisonFusion


@dataclass(frozen=True)
class CafePairConfig:
    arm: str = "pair_fusion"
    stages: int = 6
    relation_dim: int = 32
    kernel_size: int = 3
    pair_chunk: int = 16
    recompute_pairs: bool = True
    corrected_class_attention: bool = True

    def validate(self) -> None:
        if self.arm != "pair_fusion":
            raise ValueError("Unknown pair-fusion arm")
        if min(self.stages, self.relation_dim, self.pair_chunk) < 1:
            raise ValueError("Stages, relation_dim, and pair_chunk must be positive")
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError("Use an odd neighborhood kernel of at least three")


class CafePair(CafePCA):
    """Reuse CAFe encoding, trainability and AnyUp; replace only fusion."""

    def __init__(self, cafe: nn.Module, config: CafePairConfig) -> None:
        config.validate()
        super().__init__(cafe, PCADINOConfig(arm="parallel", stages=config.stages,
                         corrected_class_attention=config.corrected_class_attention))
        self.config = config
        self.fusion = nn.ModuleList([
            PairComparisonFusion(self.cost_dim, relation_dim=config.relation_dim,
                                 kernel_size=config.kernel_size, pair_chunk=config.pair_chunk,
                                 recompute_pairs=config.recompute_pairs)
            for _ in range(config.stages)
        ])
        self.train(False)

    @property
    def checkpoint_format(self) -> str:
        return "cafe_pair_v1"

    def forward(self, images: Tensor, text: Tensor, *, output_count: int | None = None,
                return_aux: bool = False) -> dict[str, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 16 or any(s % 16 for s in images.shape[-2:]):
            raise ValueError("Expected RGB BCHW images divisible by 16")
        if text.ndim != 2 or text.shape[1] != 1024 or not len(text):
            raise ValueError("Expected nonempty Q x 1024 DINO.text embeddings")
        count = len(text) if output_count is None else output_count
        if not 1 <= count <= len(text):
            raise ValueError("Invalid output_count")
        height, width = images.shape[-2] // 16, images.shape[-1] // 16
        self._set_aggregator_resolution(height, width, images.device)
        visual, _ = self._encode(images)
        cost, text_guidance, visual_guidance = self._initial_state(visual, text)
        update_norms = []
        for spatial, channel, fusion in zip(self.spatial, self.channel, self.fusion):
            spatial_cost = spatial(cost, visual_guidance)
            semantic_cost = channel(cost, text_guidance)
            cost = fusion(spatial_cost, semantic_cost, visual, text)
            if return_aux:
                update_norms.append((cost.detach().float() - semantic_cost.detach().float()).square().mean().sqrt())
        result = {"logits": self._decode(images, cost, count)}
        if return_aux:
            result["fusion_update_rms"] = torch.stack(update_norms)
        return result

    def architecture(self) -> dict:
        return dict(
            name="DINO-PairFusion-v1", **asdict(self.config),
            topology="parallel Spatial(C)||Class(C); symmetric local support and antisymmetric class-pair updates",
            class_attention="corrected query-axis SDPA" if self.config.corrected_class_attention else "published CAFe implementation",
            pair_selection="all unordered pairs; all query classes retained",
            aggregation_cost="quadratic in query count; chunked with optional activation recomputation",
            text_frozen=True, anyup_frozen=True, reduce_d_trainable=True,
            tuned_visual_indices=list(self.tuned_indices),
            source_training="locked COCO/OEM; LoveDA excluded from training and selection",
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            frozen_parameters=sum(p.numel() for p in self.parameters() if not p.requires_grad),
        )


def config_from_checkpoint(payload: dict) -> CafePairConfig:
    architecture = payload.get("architecture", {})
    if payload.get("format") != "cafe_pair_v1" or architecture.get("name") != "DINO-PairFusion-v1":
        raise ValueError("Not a DINO-PairFusion-v1 checkpoint")
    config = CafePairConfig(**{field.name: architecture[field.name] for field in fields(CafePairConfig)})
    config.validate()
    return config
