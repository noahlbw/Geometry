"""Frozen context innovations with semantic-structural graph regularization.

The graph acts on a correction, not on the local semantic field. Centering
removes the unidentifiable per-alias scene offset before reconstructing it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .contextual_phrase_readout import (
    ContextualPhraseSegmenter, masked_geometry, sample_overview_aliases,
)
from .hypothesis_readout import uniform_subset_scores
from .tcpr import _tokens_to_map


METHODS = (
    "G_all20_uniform", "StructuredAliasUnion", "GAR_WeightedLocal",
    "GAR_BoundedUnion", "GAR_CenteredInnovation", "GAR_GCR",
)


@dataclass(frozen=True)
class GroundedContextConfig:
    context_extent: int = 1024
    graph_strength: float = 1.0
    solver_steps: int = 12
    alias_topk: int = 8
    alias_selection_temperature: float = 0.07
    canonical_alias_prior: float = 0.20
    local_uniform_prior: float = 0.35
    local_alias_temperature: float = 0.20

    def validate(self) -> None:
        if self.context_extent < 512 or self.context_extent % 16:
            raise ValueError("Context extent must be patch aligned and at least 512.")
        if self.graph_strength < 0 or self.solver_steps < 1 or self.alias_topk < 1:
            raise ValueError("Invalid graph, solver, or alias-selection configuration.")
        if min(self.alias_selection_temperature, self.local_alias_temperature) <= 0:
            raise ValueError("Alias-selection temperatures must be positive.")
        if not 0 <= self.canonical_alias_prior <= 1:
            raise ValueError("Canonical alias prior must be in [0, 1].")
        if not 0 <= self.local_uniform_prior <= 1 - self.canonical_alias_prior:
            raise ValueError("Local uniform prior must leave non-negative Geometry weight.")


def context_box(height: int, width: int, top: int, left: int,
                tile_size: int, extent: int) -> tuple[int, int, int]:
    """Select a bounded square field of view without stretching the image."""
    side = min(extent, max(height, width))
    y = min(max(top + tile_size // 2 - side // 2, 0), max(height - side, 0))
    x = min(max(left + tile_size // 2 - side // 2, 0), max(width - side, 0))
    return y, x, side


def context_crop(image: Tensor, box: tuple[int, int, int]) -> Tensor:
    top, left, side = box
    crop = image[:, top:top + side, left:left + side].unsqueeze(0)
    return F.pad(crop, (0, side - crop.shape[-1], 0, side - crop.shape[-2]), mode="replicate")


def center_innovation(residual: Tensor, valid: Tensor) -> Tensor:
    weight = valid[..., None].to(residual.dtype)
    count = weight.sum(1, keepdim=True).clamp_min(1)
    return (residual - (residual * weight).sum(1, keepdim=True) / count) * weight


def normalized_logmeanexp(values: Tensor, temperature: float) -> Tensor:
    """A count-normalized soft evidence union over the final axis."""
    if values.shape[-1] < 1 or temperature <= 0:
        raise ValueError("Evidence union requires values and a positive temperature.")
    return temperature * (
        torch.logsumexp(values / temperature, dim=-1)
        - torch.log(values.new_tensor(float(values.shape[-1])))
    )


def alias_text_margin(bank, temperature: float) -> Tensor:
    """Text-only alias discriminativeness against canonical class prototypes."""
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


def geometry_alias_reliability(local: Tensor, geometry: Tensor, bank, valid: Tensor,
                               alias_temperature: float) -> tuple[Tensor, Tensor]:
    """Score each alias by visual support, structural consistency, and competition."""
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
        competition = alias_temperature * torch.logsumexp(
            class_competitors / alias_temperature, dim=-1
        )
    text_margin = alias_text_margin(bank, alias_temperature)[None, None]
    reliability = 0.5 * (local + supported) - variance + text_margin - competition
    return reliability * valid[..., None], class_scores


def geometry_alias_selection(reliability: Tensor, bank, valid: Tensor, *, topk: int,
                             temperature: float, canonical_prior: float) -> tuple[Tensor, Tensor]:
    """Admit Geometry-supported aliases per tile and weight them per patch."""
    if reliability.ndim != 3 or valid.shape != reliability.shape[:2]:
        raise ValueError("Expected reliability [B,N,M] and valid [B,N].")
    batch, _, aliases = reliability.shape
    evidence = temperature * torch.logsumexp(
        reliability.masked_fill(~valid[..., None], float("-inf")) / temperature,
        dim=1,
    )
    evidence = evidence - temperature * valid.sum(1, keepdim=True).clamp_min(1).float().log()
    admitted = torch.zeros((batch, aliases), dtype=torch.bool, device=reliability.device)
    weights = torch.zeros_like(reliability)
    for class_index in range(bank.class_count):
        members = bank.parent_indices == class_index
        count = int(members.sum())
        selected_count = min(topk, count)
        local_indices = evidence[:, members].topk(selected_count, dim=-1).indices
        member_indices = members.nonzero(as_tuple=False).squeeze(1)
        admitted.scatter_(1, member_indices[local_indices], True)
        canonical = members & bank.canonical_mask
        admitted[:, canonical] = True
        masked = reliability[..., members].masked_fill(~admitted[:, None, members], float("-inf"))
        # A padded query has no admitted alias. Avoid softmax(-inf, ..., -inf),
        # whose NaNs would otherwise reach the evidence graph through 0 * NaN.
        safe_masked = torch.where(valid[..., None], masked, torch.zeros_like(masked))
        soft = torch.softmax(safe_masked / temperature, dim=-1)
        canonical_weights = canonical[members].to(soft.dtype)[None, None].expand_as(soft)
        weights[..., members] = (
            (1.0 - canonical_prior) * soft + canonical_prior * canonical_weights
        )
    return admitted, weights * valid[..., None]


def geometry_alias_weights(reliability: Tensor, bank, valid: Tensor, *, temperature: float,
                           uniform_prior: float, canonical_prior: float) -> Tensor:
    """Continuously reweight every alias for the local classifier.

    This is deliberately not Top-K selection. The uniform prior protects rare
    but correct descriptions, the canonical prior keeps the class name present,
    and the remaining mass is assigned by Geometry-grounded reliability.
    """
    if reliability.ndim != 3 or valid.shape != reliability.shape[:2]:
        raise ValueError("Expected reliability [B,N,M] and valid [B,N].")
    if temperature <= 0 or uniform_prior < 0 or canonical_prior < 0:
        raise ValueError("Invalid local alias weighting configuration.")
    if uniform_prior + canonical_prior > 1:
        raise ValueError("Local alias priors exceed one.")
    weights = torch.zeros_like(reliability)
    geometry_mass = 1.0 - uniform_prior - canonical_prior
    for class_index in range(bank.class_count):
        members = bank.parent_indices == class_index
        count = int(members.sum())
        values = reliability[..., members]
        safe_values = torch.where(valid[..., None], values, torch.zeros_like(values))
        relative = torch.softmax(safe_values / temperature, dim=-1)
        uniform = torch.full_like(relative, 1.0 / count)
        canonical = (members & bank.canonical_mask)[members].to(relative.dtype)
        canonical = canonical[None, None].expand_as(relative)
        weights[..., members] = (
            geometry_mass * relative + uniform_prior * uniform + canonical_prior * canonical
        )
    return weights * valid[..., None]


def weighted_alias_scores(aliases: Tensor, bank, weights: Tensor, temperature: float) -> Tensor:
    """Aggregate only admitted aliases, preserving per-class normalized weights."""
    if aliases.shape != weights.shape:
        raise ValueError("Alias evidence and alias weights must have the same shape.")
    output = []
    for class_index in range(bank.class_count):
        members = bank.parent_indices == class_index
        member_weights = weights[..., members]
        log_weights = torch.where(
            member_weights > 0,
            member_weights.log(),
            torch.full_like(member_weights, float("-inf")),
        )
        output.append(temperature * torch.logsumexp(
            aliases[..., members] / temperature + log_weights,
            dim=-1,
        ))
        # Invalid padded tokens have zero total alias mass. Their score is never
        # used for classification, but it must stay finite before graph masking.
        output[-1] = torch.where(
            member_weights.sum(dim=-1) > 0, output[-1], torch.zeros_like(output[-1])
        )
    return torch.stack(output, dim=-1)


def evidence_graph(geometry: Tensor, local_scores: Tensor, valid: Tensor,
                   temperature: float) -> Tensor:
    """Symmetric graph jointly supported by visual structure and text posteriors."""
    safe_scores = torch.where(valid[..., None], local_scores, torch.zeros_like(local_scores))
    posterior = torch.softmax(safe_scores / temperature, dim=-1).sqrt()
    compatibility = posterior @ posterior.transpose(-1, -2)
    graph = 0.5 * (geometry + geometry.transpose(-1, -2)) * compatibility
    graph = graph * (valid[:, :, None] & valid[:, None, :])
    diagonal = torch.eye(graph.shape[-1], device=graph.device, dtype=torch.bool)[None]
    graph = graph.masked_fill(diagonal, 0.0)
    # One scalar per image preserves symmetry and bounds every degree by one.
    scale = graph.sum(-1).amax(-1, keepdim=True).clamp_min(1e-6)
    return graph / scale[..., None]


def solve_innovation(residual: Tensor, graph: Tensor, valid: Tensor,
                     strength: float = 1.0, steps: int = 12) -> Tensor:
    """Solve (2 I + strength * L_W) delta = residual by contractive Jacobi.

    This minimizes .5||delta||^2 + .5||delta-residual||^2
    + strength/4 sum_ij W_ij ||delta_i-delta_j||^2.
    With degree <= 1 and strength=1, the contraction factor is <= 1/3.
    """
    residual = residual * valid[..., None]
    degree = graph.sum(-1, keepdim=True)
    denominator = 2.0 + strength * degree
    delta = residual / denominator
    for _ in range(steps):
        delta = (residual + strength * (graph @ delta)) / denominator
    return delta * valid[..., None]


class GroundedContextSegmenter(ContextualPhraseSegmenter):
    """GAR plus GCR: visual alias admission and bounded innovation reconstruction."""

    def __init__(self, backbone, config, tcpr_config, grounded_config=GroundedContextConfig()):
        super().__init__(backbone, config, tcpr_config)
        grounded_config.validate()
        self.grounded_config = grounded_config

    @torch.inference_mode()
    def read_grounded(self, prepared, whole_map: Tensor, bounded_map: Tensor,
                      bank, *, height: int, width: int, top: int, left: int,
                      box: tuple[int, int, int], valid_mask: Tensor):
        valid = valid_mask.reshape(1, -1).bool()
        text = F.normalize(bank.features.float(), dim=-1)
        local = F.normalize(prepared.geometry_projected.float(), dim=-1) @ text.T
        kwargs = dict(grid_height=prepared.grid_height, grid_width=prepared.grid_width,
                      patch_size=self.patch_size)
        whole = sample_overview_aliases(
            whole_map, image_height=height, image_width=width,
            tile_top=top, tile_left=left, **kwargs,
        )
        by, bx, side = box
        bounded = sample_overview_aliases(
            bounded_map, image_height=side, image_width=side,
            tile_top=top - by, tile_left=left - bx, **kwargs,
        )
        geometry = masked_geometry(prepared.geometry_patch_conditional.float(), valid, 1e-6)
        subset = torch.ones(text.shape[0], dtype=torch.bool, device=text.device)
        tau = self.config.alias_temperature

        def uniform_aggregate(aliases):
            return uniform_subset_scores(aliases, bank.parent_indices, bank.class_count, subset, tau)

        base = uniform_aggregate(local)
        reliability, _ = geometry_alias_reliability(local, geometry, bank, valid, tau)
        admitted, _ = geometry_alias_selection(
            reliability, bank, valid, topk=self.grounded_config.alias_topk,
            temperature=self.grounded_config.alias_selection_temperature,
            canonical_prior=self.grounded_config.canonical_alias_prior,
        )
        local_weights = geometry_alias_weights(
            reliability, bank, valid,
            temperature=self.grounded_config.local_alias_temperature,
            uniform_prior=self.grounded_config.local_uniform_prior,
            canonical_prior=self.grounded_config.canonical_alias_prior,
        )

        def local_aggregate(aliases):
            return weighted_alias_scores(aliases, bank, local_weights, tau)

        gar_local = local_aggregate(local)
        # GAR has no authority to erase an appearance mode from the local
        # classifier.  It only admits aliases as context evidence.  The graph
        # is conditioned on the coverage-preserving, uniform local readout.
        graph = evidence_graph(geometry, base, valid, tau)
        residual = (bounded - local) * admitted[:, None].to(local.dtype)
        centered = center_innovation(residual, valid)
        delta = solve_innovation(centered, graph, valid, self.grounded_config.graph_strength,
                                 self.grounded_config.solver_steps)
        fields = {
            "G_all20_uniform": base,
            "StructuredAliasUnion": uniform_aggregate(normalized_logmeanexp(
                torch.stack((local, geometry @ whole), dim=-1), tau)),
            "GAR_WeightedLocal": gar_local,
            "GAR_BoundedUnion": uniform_aggregate(normalized_logmeanexp(
                torch.stack((local, geometry @ bounded), dim=-1), tau)),
            # The direct-weighted local readout above is an ablation.  In the
            # final GAR+GCR path the entire 20-alias bank remains the local
            # coverage anchor; GAR only gates which aliases can create a
            # context innovation and a graph correction.
            "GAR_CenteredInnovation": uniform_aggregate(local + 0.5 * centered),
            "GAR_GCR": uniform_aggregate(local + delta),
        }
        shape = (prepared.grid_height, prepared.grid_width)
        output = {name: _tokens_to_map(torch.where(valid[..., None], value, base), *shape)
                  for name, value in fields.items()}
        selected = valid[..., None].expand_as(delta)
        equation_error = 2 * delta + self.grounded_config.graph_strength * (
            graph.sum(-1, keepdim=True) * delta - graph @ delta) - centered
        diagnostics = {
            "mean_admitted_aliases_per_class": float(
                admitted.sum().item() / max(bank.class_count, 1)
            ),
            "mean_admitted_text_margin": float(
                alias_text_margin(bank, tau)[admitted[0]].mean().item()
            ),
            "mean_alias_reliability": float(reliability[selected].mean().item()),
            "mean_context_innovation": float(residual[selected].abs().mean()),
            "mean_centered_innovation": float(centered[selected].abs().mean()),
            "mean_correction": float(delta[selected].abs().mean()),
            "solver_error": float(equation_error[selected].abs().mean()),
            "correction_spatial_mean": float((delta.sum(1) / valid.sum().clamp_min(1)).abs().mean()),
            "changed_gar_from_uniform": float(
                ((gar_local.argmax(-1) != base.argmax(-1)) & valid).sum().item()
                / max(int(valid.sum().item()), 1)
            ),
        }
        return output, diagnostics
