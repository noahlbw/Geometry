"""CAFe with explicit mask states and query-conditioned region composition."""
from __future__ import annotations

from dataclasses import asdict, fields

import torch
from torch import Tensor, nn

from .cafe_ped import CafePED, ped_loss
from .cafe_vc import CafeVC
from .parallel_evidence import ParallelEvidenceConfig
from .region_assembly import RegionAssemblyConfig, RegionAssemblyStage, SharedMaskProposals, membership_loss


class CafeRegionAssembly(CafePED):
    """Reuse PED's trainability/checkpoint contracts and CAFe's serial base."""

    def __init__(self, cafe: nn.Module, config: RegionAssemblyConfig) -> None:
        config.validate()
        if len(cafe.aggregator) < config.stages or len(cafe.aggregator) % config.stages:
            raise ValueError("Assembly stages must partition all official CAFe blocks evenly")
        super().__init__(cafe, ParallelEvidenceConfig(arm="plain", stages=len(cafe.aggregator)))
        self.config = config
        self.proposals = SharedMaskProposals(config)
        self.assembly = nn.ModuleList([RegionAssemblyStage(config, self.cost_dim) for _ in range(config.stages)])
        self.train(False)

    @property
    def checkpoint_format(self) -> str:
        return "cafe_region_assembly_v1"

    def train(self, mode: bool = True) -> "CafeRegionAssembly":
        nn.Module.train(self, mode)
        self.cafe.eval()
        old_active = mode and not self.warmup
        for name in ("aggregator", "corr_embed", "text_guidance_projection", "vis_guidance_projection"):
            getattr(self.cafe, name).train(old_active)
        for index in self.tuned_indices:
            self.cafe.backbone.visual_model.backbone.blocks[index].train(old_active)
        return self

    def _set_aggregator_resolution(self, height: int, width: int, device: torch.device) -> None:
        CafeVC._set_aggregator_resolution(self, height, width, device)

    def forward(self, images: Tensor, text: Tensor, *, output_count: int | None = None,
                return_aux: bool = False, return_regions: bool = False) -> dict:
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 16 or any(s % 16 for s in images.shape[-2:]):
            raise ValueError("Expected RGB BCHW images divisible by 16")
        if text.ndim != 2 or text.shape[1] != 1024 or not len(text):
            raise ValueError("Expected nonempty Q x 1024 DINO.text embeddings")
        count = len(text) if output_count is None else output_count
        if not 1 <= count <= len(text) or (return_aux and count != len(text)):
            raise ValueError("Auxiliary supervision requires the entire query set")
        b, _, ih, iw = images.shape
        h, w = ih // 16, iw // 16
        self._set_aggregator_resolution(h, w, images.device)
        x, y = self._encode(images)
        initial, tg, vg = self._initial_state(x, text)
        pixels, semantics, proposal_logits, xy = self.proposals(x, y)
        masks = proposal_logits[:, None].expand(-1, len(text), -1, -1)
        cost, auxiliary, traces = initial, [], []
        group = len(self.cafe.aggregator) // self.config.stages
        for index, stage in enumerate(self.assembly):
            for layer in self.cafe.aggregator[index * group:(index + 1) * group]:
                cost = layer(cost, tg, vg)
            cost, masks, members, detail = stage(pixels, semantics, masks, cost, initial, text, xy,
                                                return_regions=return_regions)
            if return_aux:
                auxiliary.append(members)
            if return_regions:
                traces.append(detail)
        selected = cost[:, :, :count].contiguous().reshape(b, self.cost_dim * count, h, w)
        up = self.cafe.upsampler(images, selected)
        features = up.reshape(b, self.cost_dim, count, ih, iw).transpose(1, 2).reshape(b * count, self.cost_dim, ih, iw)
        result = {"logits": self.cafe.reduce_d(features).reshape(b, count, ih, iw)}
        if return_aux:
            result["membership_logits"] = auxiliary
            result["proposal_logits"] = proposal_logits
            result["proposal_grid"] = (h, w)
        if return_regions:
            result.update(region_traces=traces, proposal_logits=proposal_logits, final_membership=masks.float().softmax(-1))
        return result

    def architecture(self) -> dict:
        return dict(name="CAFe-RegionAssembly-v1", **asdict(self.config),
                    proposal_backend="pytorch_masked_transformer_v1_not_full_mask2former",
                    grouping="dense_soft_child_parent_assignment",
                    source_training="locked_COCO_OEM_with_membership_auxiliary",
                    upsampling="original multichannel AnyUp then reduce_d",
                    tuned_visual_blocks=list(self.tuned_indices),
                    aggregator_blocks=len(self.cafe.aggregator),
                    trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    frozen_parameters=sum(p.numel() for p in self.parameters() if not p.requires_grad))


def config_from_checkpoint(payload: dict) -> RegionAssemblyConfig:
    architecture = payload.get("architecture", {})
    if payload.get("format") != "cafe_region_assembly_v1" or architecture.get("name") != "CAFe-RegionAssembly-v1":
        raise ValueError("Not a CAFe region assembly v1 checkpoint")
    config = RegionAssemblyConfig(**{item.name: architecture[item.name] for item in fields(RegionAssemblyConfig)})
    config.validate()
    return config


def assembly_loss(output: dict, target: Tensor, teacher_logits: Tensor, *, smoothing: float = 0.1,
                  kd_weight: float = 0.02, membership_weight: float = 0.1) -> tuple[Tensor, dict]:
    if membership_weight < 0:
        raise ValueError("Membership weight must be nonnegative")
    loss, terms = ped_loss(output, target, teacher_logits.detach(), smoothing=smoothing, kd_weight=kd_weight)
    members = membership_loss(output["membership_logits"], target, output["logits"].shape[1])
    terms["region_loss"] = members
    return loss + membership_weight * members, terms
