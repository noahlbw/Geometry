"""CAFe-DINO wrapper for the controlled PSDR readout probe."""
from __future__ import annotations

from dataclasses import asdict, fields

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .cafe_ped import CafePED, ped_loss
from .parallel_evidence import ParallelEvidenceConfig, VisualEvidenceField
from .pixel_query_evidence import PixelQueryEvidenceConfig, PixelQueryEvidenceReader, SharedGridEvidenceBank


class CafePSDRProbe(CafePED):
    """Keep CAFe's dense decoder fixed while testing only evidence addressing."""

    def __init__(self, cafe: nn.Module, config: PixelQueryEvidenceConfig) -> None:
        config.validate()
        super().__init__(cafe, ParallelEvidenceConfig(arm="plain", stages=len(cafe.aggregator)))
        self.psdr_config = config
        self.evidence_field = VisualEvidenceField(config.dim, config.visual_blocks)
        self.evidence_bank = SharedGridEvidenceBank(config.grids, config.samples_per_region)
        self.evidence_reader = PixelQueryEvidenceReader(self.cost_dim, config)
        self.evidence_field.requires_grad_(True)
        self.evidence_reader.requires_grad_(True)
        self.train(False)

    @property
    def checkpoint_format(self) -> str:
        return "cafe_psdr_probe_v1"

    def train(self, mode: bool = True) -> "CafePSDRProbe":
        super().train(mode)
        # CafePED.__init__ calls self.train(False) before this subclass has
        # attached the probe modules.
        if hasattr(self, "evidence_field"):
            self.evidence_field.train(mode)
            self.evidence_bank.train(mode)
            self.evidence_reader.train(mode)
        return self

    def _decode(self, images: Tensor, cost: Tensor, count: int) -> Tensor:
        batch, _, _, height, width = cost.shape
        image_height, image_width = images.shape[-2:]
        selected = cost[:, :, :count].contiguous().reshape(batch, self.cost_dim * count, height, width)
        up = self.cafe.upsampler(images, selected)
        features = up.reshape(batch, self.cost_dim, count, image_height, image_width)
        features = features.transpose(1, 2).reshape(batch * count, self.cost_dim, image_height, image_width)
        return self.cafe.reduce_d(features).reshape(batch, count, image_height, image_width)

    def forward(self, images: Tensor, text: Tensor, *, output_count: int | None = None, return_utility: bool = False) -> dict[str, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 16 or any(size % 16 for size in images.shape[-2:]):
            raise ValueError("Expected RGB BCHW images divisible by 16")
        if text.ndim != 2 or text.shape[1] != 1024 or not len(text):
            raise ValueError("Expected nonempty Q x 1024 text embeddings")
        count = len(text) if output_count is None else output_count
        if not 1 <= count <= len(text) or (return_utility and count != len(text)):
            raise ValueError("Utility labels require the complete query set")
        batch, _, image_height, image_width = images.shape
        height, width = image_height // 16, image_width // 16
        self._set_aggregator_resolution(height, width, images.device)
        x, y = self._encode(images)
        initial, text_guidance, visual_guidance = self._initial_state(x, text)
        cost = self._plain(initial, text_guidance, visual_guidance)
        field, bank = self.evidence_field(x, y), None
        bank = self.evidence_bank(field)
        final = None
        for round_index in range(self.psdr_config.rounds):
            final = self.evidence_reader(cost, initial, field, bank, text,
                                         return_candidates=return_utility and round_index + 1 == self.psdr_config.rounds)
            previous, cost = cost, final["cost"]
        result = {"logits": self._decode(images, cost, count)}
        if return_utility:
            # The candidate outcome labels are detached complete decoder losses;
            # only the address-ranking scores receive the utility gradients.
            with torch.no_grad():
                reference = self._decode(images, previous, count)
                candidates = torch.stack([self._decode(images, candidate, count) for candidate in final["candidate_costs"]])
            result.update(utility_scores=final["utility_scores"], utility_reference_logits=reference,
                          utility_candidate_logits=candidates, utility_patch_shape=(height, width))
        return result

    def optimizer_groups(self, new_lr: float, head_lr: float, visual_lr: float):
        groups = super().optimizer_groups(new_lr, head_lr, visual_lr)
        for group in groups:
            if group["role"] == "new":
                return groups
        raise RuntimeError("PSDR probe reader must be present in the new-parameter optimizer group")

    def architecture(self) -> dict:
        return dict(name="CAFe-PSDR-Probe-v1", **asdict(self.psdr_config),
                    bank="shared_fixed_multiscale_grid; no target labels or category-specific proposals",
                    reader="recipient-pixel update from bounded query-addressed evidence",
                    utility_label="source-only post-read output BCE improvement when enabled",
                    original_aggregator="published CAFe serial aggregation retained",
                    tuned_visual_indices=list(self.tuned_indices), text_frozen=True, anyup_frozen=True,
                    trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    frozen_parameters=sum(p.numel() for p in self.parameters() if not p.requires_grad))


def config_from_checkpoint(payload: dict) -> PixelQueryEvidenceConfig:
    architecture = payload.get("architecture", {})
    if payload.get("format") != "cafe_psdr_probe_v1" or architecture.get("name") != "CAFe-PSDR-Probe-v1":
        raise ValueError("Not a CAFe-PSDR-Probe-v1 checkpoint")
    config = PixelQueryEvidenceConfig(**{item.name: architecture[item.name] for item in fields(PixelQueryEvidenceConfig)})
    config.validate()
    return config


def _binary_loss(logits: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    valid = target != 255
    labels = F.one_hot(target.masked_fill(~valid, 0), logits.shape[1]).permute(0, 3, 1, 2).to(logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction="none")
    return loss, valid


def psdr_probe_loss(output: dict, target: Tensor, teacher_logits: Tensor, *, smoothing: float = 0.1,
                    kd_weight: float = 0.02, utility_weight: float = 0.0) -> tuple[Tensor, dict]:
    loss, terms = ped_loss(output, target, teacher_logits, smoothing=smoothing, kd_weight=kd_weight)
    utility = loss * 0
    if utility_weight:
        if not {"utility_scores", "utility_reference_logits", "utility_candidate_logits"} <= set(output):
            raise ValueError("Utility training requires candidate outcome predictions")
        reference, valid = _binary_loss(output["utility_reference_logits"], target)
        candidates = torch.stack([_binary_loss(item, target)[0] for item in output["utility_candidate_logits"]])
        gain = reference[None] - candidates
        patch_shape = tuple(output["utility_patch_shape"])
        valid_mass = F.adaptive_avg_pool2d(valid[:, None].float(), patch_shape).squeeze(1)
        pooled = F.adaptive_avg_pool2d((gain * valid[None, :, None]).reshape(-1, gain.shape[2], *gain.shape[-2:]), patch_shape)
        pooled = pooled.reshape(gain.shape[0], gain.shape[1], gain.shape[2], *patch_shape) / valid_mass[None, :, None].clamp_min(1e-6)
        target_distribution = (pooled.permute(1, 2, 3, 4, 0).detach() / 0.25).softmax(-1)
        log_score = output["utility_scores"].reshape(*output["utility_scores"].shape[:2], *patch_shape, -1).log_softmax(-1)
        utility = -(target_distribution * log_score).sum(-1)
        utility = (utility * valid_mass[:, None]).sum() / (valid_mass.sum() * utility.shape[1]).clamp_min(1)
        loss = loss + utility_weight * utility
    terms["utility_loss"] = utility
    return loss, terms
