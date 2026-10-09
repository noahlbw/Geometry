"""Offline VIP-style alias distillation for frozen DINO.text experiments.

This module implements the paper's two data-free selection signals on a fixed
unlabeled image collection: visual-grounding consistency after attention random
walk and class-posterior certainty in an alias activation region. It deliberately
does not change image features or learn any parameter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .hypothesis_readout import uniform_subset_scores


@dataclass(frozen=True)
class VIPDistillationConfig:
    affinity_power: float = 2.0
    random_walk_steps: int = 2
    activation_threshold: float = 0.4
    alias_temperature: float = 0.07

    def validate(self) -> None:
        if self.affinity_power < 1 or self.random_walk_steps < 1:
            raise ValueError("VIP affinity power and random-walk steps must be positive.")
        if not 0 < self.activation_threshold < 1 or self.alias_temperature <= 0:
            raise ValueError("VIP threshold and temperature must be positive.")


def multilayer_attention_affinity(backbone: Any, rgb: Tensor) -> Tensor:
    """Return the mean patch-to-patch DINOv3 attention relation for one tile.

    The A800 DINOv3 build does not expose attention tensors through
    ``get_intermediate_layers``. We therefore replay the frozen backbone's own
    block sequence and read its Q/K probabilities immediately before each
    untouched attention block executes. No parameter or model output changes.
    """
    backbone._validate_rgb(rgb)
    normalized = (rgb.to(backbone.device, non_blocking=True) - backbone._imagenet_mean) / backbone._imagenet_std
    visual = backbone.model.visual_model
    with torch.inference_mode(), backbone._autocast():
        model = visual.backbone
        tokens, (height, width) = model.prepare_tokens_with_masks(normalized)
        patch_count = height * width
        if tokens.shape[0] != 1 or tokens.shape[1] < patch_count:
            raise ValueError("VIP distillation expects one regular image tile.")
        prefix = tokens.shape[1] - patch_count
        aggregate = None
        for block in model.blocks:
            rope = model.rope_embed(H=height, W=width) if model.rope_embed is not None else None
            attention = block.attn
            qkv = attention.qkv(block.norm1(tokens))
            channels = attention.qkv.in_features
            qkv = qkv.reshape(1, tokens.shape[1], 3, attention.num_heads, channels // attention.num_heads)
            query, key, _ = (value.transpose(1, 2) for value in torch.unbind(qkv, dim=2))
            if rope is not None:
                query, key = attention.apply_rope(query, key, rope)
            logits = (query.float() @ key.float().transpose(-1, -2)) * attention.scale
            patch_attention = torch.softmax(logits[..., prefix:, prefix:], dim=-1).mean(dim=1)[0]
            aggregate = patch_attention if aggregate is None else aggregate + patch_attention
            tokens = block(tokens, rope)
            if isinstance(tokens, (tuple, list)):
                tokens = tokens[0]
    if aggregate is None:
        raise RuntimeError("DINOv3 backbone contains no attention blocks.")
    return aggregate / len(model.blocks)


def random_walk_relation(attention: Tensor, config: VIPDistillationConfig) -> Tensor:
    """Construct the row-stochastic random-walk operator from VIP Eq. 4."""
    config.validate()
    if attention.ndim != 2 or attention.shape[0] != attention.shape[1]:
        raise ValueError("attention must be square [N,N].")
    relation = attention.clamp_min(0).pow(config.affinity_power)
    relation = relation / relation.sum(-1, keepdim=True).clamp_min(torch.finfo(relation.dtype).eps)
    walk = relation
    for _ in range(config.random_walk_steps - 1):
        walk = walk @ relation
    return walk


def normalized_alias_activation(alias_scores: Tensor, valid: Tensor) -> Tensor:
    """Normalize each alias map to [0,1] over valid patch positions only."""
    if alias_scores.ndim != 3 or valid.shape != alias_scores.shape[:2]:
        raise ValueError("Expected alias_scores [B,N,M] and valid [B,N].")
    lower = alias_scores.masked_fill(~valid[..., None], float("inf")).amin(1, keepdim=True)
    upper = alias_scores.masked_fill(~valid[..., None], float("-inf")).amax(1, keepdim=True)
    normalized = (alias_scores - lower) / (upper - lower).clamp_min(1e-6)
    return torch.where(valid[..., None], normalized.clamp(0, 1), torch.zeros_like(normalized))


def gar_alias_text_margin(bank, temperature: float) -> Tensor:
    """Current GAR's text-only discriminativeness term."""
    text = F.normalize(bank.features.float(), dim=-1)
    prototypes = []
    for class_index in range(bank.class_count):
        canonical = (bank.parent_indices == class_index) & bank.canonical_mask
        prototypes.append(text[canonical].squeeze(0))
    prototypes = torch.stack(prototypes)
    similarity = text @ prototypes.T
    own = similarity.gather(1, bank.parent_indices[:, None]).squeeze(1)
    if bank.class_count == 1:
        return own
    competitors = similarity.masked_fill(
        F.one_hot(bank.parent_indices, bank.class_count).bool(), float("-inf")
    )
    return own - temperature * torch.logsumexp(competitors / temperature, dim=1)


def gar_geometry_alias_reliability(local: Tensor, geometry: Tensor, bank, valid: Tensor,
                                   alias_temperature: float) -> Tensor:
    """The current GAR reliability rule, isolated for a matched ablation."""
    supported = geometry @ local
    variance = (geometry @ local.square() - supported.square()).clamp_min(0).sqrt()
    subset = torch.ones(local.shape[-1], dtype=torch.bool, device=local.device)
    class_scores = uniform_subset_scores(
        supported, bank.parent_indices, bank.class_count, subset, alias_temperature
    )
    if bank.class_count == 1:
        competition = torch.zeros_like(local)
    else:
        class_competitors = class_scores[..., None, :].expand(
            local.shape[0], local.shape[1], local.shape[-1], bank.class_count
        ).masked_fill(
            F.one_hot(bank.parent_indices, bank.class_count)[None, None].bool(),
            float("-inf"),
        )
        competition = alias_temperature * torch.logsumexp(class_competitors / alias_temperature, dim=-1)
    return (0.5 * (local + supported) - variance + gar_alias_text_margin(bank, alias_temperature)[None, None] - competition) * valid[..., None]


def gar_geometry_alias_weights(reliability: Tensor, bank, valid: Tensor, *, temperature: float,
                               uniform_prior: float, canonical_prior: float) -> Tensor:
    """Current GAR's coverage-preserving local alias weighting."""
    if uniform_prior < 0 or canonical_prior < 0 or uniform_prior + canonical_prior > 1:
        raise ValueError("Invalid GAR alias priors.")
    weights = torch.zeros_like(reliability)
    geometry_mass = 1.0 - uniform_prior - canonical_prior
    for class_index in range(bank.class_count):
        members = bank.parent_indices == class_index
        count = int(members.sum())
        values = reliability[..., members]
        relative = torch.softmax(torch.where(valid[..., None], values, torch.zeros_like(values)) / temperature, dim=-1)
        uniform = torch.full_like(relative, 1.0 / count)
        canonical = (members & bank.canonical_mask)[members].to(relative.dtype)[None, None].expand_as(relative)
        weights[..., members] = geometry_mass * relative + uniform_prior * uniform + canonical_prior * canonical
    return weights * valid[..., None]


def gar_weighted_alias_scores(aliases: Tensor, bank, weights: Tensor, temperature: float) -> Tensor:
    """Current GAR's class-normalized weighted log-mean-exp aggregator."""
    output = []
    for class_index in range(bank.class_count):
        members = bank.parent_indices == class_index
        member_weights = weights[..., members]
        log_weights = torch.where(member_weights > 0, member_weights.log(), torch.full_like(member_weights, float("-inf")))
        score = temperature * torch.logsumexp(aliases[..., members] / temperature + log_weights, dim=-1)
        output.append(torch.where(member_weights.sum(-1) > 0, score, torch.zeros_like(score)))
    return torch.stack(output, dim=-1)


class VIPAliasAccumulator:
    """Accumulate paper-style alias scores, averaging first within each image."""

    def __init__(self, alias_count: int, *, device: torch.device) -> None:
        if alias_count < 1:
            raise ValueError("alias_count must be positive.")
        self.alias_count = alias_count
        self.device = device
        self.intersection = torch.zeros(alias_count, device=device, dtype=torch.float64)
        self.union = torch.zeros(alias_count, device=device, dtype=torch.float64)
        self.entropy_sum = torch.zeros(alias_count, device=device, dtype=torch.float64)
        self.active_count = torch.zeros(alias_count, device=device, dtype=torch.float64)
        self.vg_sum = torch.zeros(alias_count, device=device, dtype=torch.float64)
        self.sc_sum = torch.zeros(alias_count, device=device, dtype=torch.float64)
        self.observed_images = torch.zeros(alias_count, device=device, dtype=torch.int64)

    def update_tile(
        self,
        alias_scores: Tensor,
        attention: Tensor,
        parents: Tensor,
        class_count: int,
        valid: Tensor,
        config: VIPDistillationConfig,
    ) -> None:
        config.validate()
        if alias_scores.shape != (1, attention.shape[0], self.alias_count):
            raise ValueError("Alias score shape does not match attention or alias count.")
        activation = normalized_alias_activation(alias_scores.float(), valid)[0]
        walk = random_walk_relation(attention.float(), config)
        propagated = (walk @ activation).clamp(0, 1)
        high = (activation >= config.activation_threshold) & valid[0, :, None]
        active = high.sum(0).to(torch.float64)
        overlap = (activation * propagated * high).sum(0).to(torch.float64)
        union = (activation + propagated - activation * propagated).mul(high).sum(0).to(torch.float64)
        all_aliases = torch.ones(self.alias_count, dtype=torch.bool, device=alias_scores.device)
        class_scores = uniform_subset_scores(
            alias_scores, parents, class_count, all_aliases, config.alias_temperature
        )
        posterior = torch.softmax(class_scores / config.alias_temperature, dim=-1).clamp_min(1e-8)
        entropy = -(posterior * posterior.log()).sum(-1)[0] / torch.log(
            torch.tensor(float(class_count), device=posterior.device)
        )
        self.intersection += overlap
        self.union += union
        self.entropy_sum += (high.to(entropy.dtype) * entropy[:, None]).sum(0).to(torch.float64)
        self.active_count += active

    def finalize_image(self) -> None:
        present = self.active_count > 0
        self.vg_sum[present] += self.intersection[present] / self.union[present].clamp_min(1e-12)
        self.sc_sum[present] += self.entropy_sum[present] / self.active_count[present]
        self.observed_images[present] += 1
        self.intersection.zero_()
        self.union.zero_()
        self.entropy_sum.zero_()
        self.active_count.zero_()

    def merge(self, other: "VIPAliasAccumulator") -> None:
        if self.alias_count != other.alias_count:
            raise ValueError("Cannot merge alias accumulators of different widths.")
        self.vg_sum += other.vg_sum.to(self.device)
        self.sc_sum += other.sc_sum.to(self.device)
        self.observed_images += other.observed_images.to(self.device)

    def report(self, bank) -> dict[str, object]:
        observed = self.observed_images.clamp_min(1).to(torch.float64)
        vg = self.vg_sum / observed
        sc = self.sc_sum / observed
        selected = torch.zeros(self.alias_count, dtype=torch.bool, device=self.device)
        entries: list[dict[str, object]] = []
        for class_index in range(bank.class_count):
            members = bank.parent_indices == class_index
            canonical = members & bank.canonical_mask
            canonical_index = int(canonical.nonzero(as_tuple=False).item())
            selected[canonical] = True
            for alias_index in members.nonzero(as_tuple=False).flatten().tolist():
                keep = alias_index == canonical_index or (
                    bool(self.observed_images[alias_index])
                    and bool(self.observed_images[canonical_index])
                    and bool(vg[alias_index] > vg[canonical_index])
                    and bool(sc[alias_index] < sc[canonical_index])
                )
                selected[alias_index] = keep
                entries.append({
                    "class_index": class_index,
                    "alias_index": alias_index,
                    "alias": bank.alias_names[alias_index],
                    "canonical": alias_index == canonical_index,
                    "observed_images": int(self.observed_images[alias_index]),
                    "visual_grounding_score": round(float(vg[alias_index]), 8),
                    "semantic_certainty_entropy": round(float(sc[alias_index]), 8),
                    "selected": keep,
                })
        counts = [int((selected & (bank.parent_indices == index)).sum()) for index in range(bank.class_count)]
        return {
            "selected_mask": selected.cpu().tolist(),
            "selected_counts_per_class": counts,
            "aliases": entries,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "alias_count": self.alias_count,
            "vg_sum": self.vg_sum.cpu().tolist(),
            "sc_sum": self.sc_sum.cpu().tolist(),
            "observed_images": self.observed_images.cpu().tolist(),
        }

    @classmethod
    def from_state_dict(cls, value: dict[str, object], *, device: torch.device) -> "VIPAliasAccumulator":
        result = cls(int(value["alias_count"]), device=device)
        result.vg_sum.copy_(torch.tensor(value["vg_sum"], device=device, dtype=torch.float64))
        result.sc_sum.copy_(torch.tensor(value["sc_sum"], device=device, dtype=torch.float64))
        result.observed_images.copy_(torch.tensor(value["observed_images"], device=device, dtype=torch.int64))
        return result


def selected_mask_from_report(report: dict[str, object], *, device: torch.device) -> Tensor:
    selected = torch.tensor(report["selected_mask"], device=device, dtype=torch.bool)
    if selected.ndim != 1 or not bool(selected.any()):
        raise ValueError("Invalid VIP distillation selection report.")
    return selected
