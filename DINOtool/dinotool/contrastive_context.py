"""Training-free pairwise alias evidence and spatial reconstruction."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .contextual_phrase_readout import (
    ContextualPhraseSegmenter, masked_geometry, normalized_logmeanexp,
    sample_overview_aliases,
)
from .hypothesis_readout import uniform_subset_scores
from .tcpr import _tokens_to_map


METHODS = (
    "G_all20_uniform", "BoundedUnion", "Native_all20_uniform",
    "ContrastiveLocal", "ContrastiveNoGraph", "ContrastiveGCR",
)


@dataclass(frozen=True)
class ContrastiveContextConfig:
    context_extent: int = 1024
    visual_radius: int = 2
    graph_strength: float = 1.0
    anchor_strength: float = 1.0
    huber_width: float = 0.07
    edge_temperature: float = 0.07
    solver_steps: int = 32

    def validate(self) -> None:
        if self.context_extent < 512 or self.context_extent % 16:
            raise ValueError("Context extent must be patch aligned and at least 512.")
        if self.visual_radius < 1 or self.solver_steps < 1:
            raise ValueError("Visual radius and solver steps must be positive.")
        if self.graph_strength < 0 or self.anchor_strength <= 0:
            raise ValueError("Graph and anchor strengths are invalid.")
        if self.huber_width <= 0 or self.edge_temperature <= 0:
            raise ValueError("Evidence and edge scales must be positive.")


def alias_matches(bank) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Find the nearest competing description for each class pair."""
    bank.validate()
    text = F.normalize(bank.features.float(), dim=-1)
    left_classes, right_classes, left_indices, right_indices = [], [], [], []
    for left in range(bank.class_count):
        for right in range(left + 1, bank.class_count):
            a = (bank.parent_indices == left).nonzero(as_tuple=True)[0]
            b = (bank.parent_indices == right).nonzero(as_tuple=True)[0]
            similarity = text[a] @ text[b].T
            left_classes.append(left)
            right_classes.append(right)
            left_indices.append(torch.stack((a, b[similarity.argmax(-1)]), dim=-1))
            right_indices.append(torch.stack((b, a[similarity.argmax(0)]), dim=-1))
    return (
        torch.tensor(left_classes, device=text.device),
        torch.tensor(right_classes, device=text.device),
        left_indices,
        right_indices,
    )


def pairwise_evidence(aliases: Tensor, matches, temperature: float) -> Tensor:
    """The antisymmetric A-vs-B margin from matched alias responses."""
    if aliases.ndim != 3 or temperature <= 0:
        raise ValueError("Expected [B,N,M] alias scores and positive temperature.")
    _, _, left_matches, right_matches = matches
    if not left_matches:
        return aliases.new_zeros((*aliases.shape[:2], 0))
    margins = []
    for a, b in zip(left_matches, right_matches):
        positive = aliases[..., a[:, 0]] - aliases[..., a[:, 1]]
        negative = aliases[..., b[:, 0]] - aliases[..., b[:, 1]]
        pos = temperature * (
            torch.logsumexp(positive / temperature, dim=-1)
            - positive.new_tensor(positive.shape[-1]).log()
        )
        neg = temperature * (
            torch.logsumexp(negative / temperature, dim=-1)
            - negative.new_tensor(negative.shape[-1]).log()
        )
        margins.append(0.5 * (pos - neg))
    return torch.stack(margins, dim=-1)


def local_pair_graph(geometry: Tensor, evidence: Tensor, valid: Tensor,
                     height: int, width: int, radius: int, temperature: float
                     ) -> tuple[Tensor, Tensor]:
    """Sparse symmetric visual graph, attenuated by pairwise disagreement."""
    batch, patches, pairs = evidence.shape
    if geometry.shape != (batch, patches, patches) or valid.shape != (batch, patches):
        raise ValueError("Geometry, evidence and valid mask have incompatible shapes.")
    if height * width != patches or radius < 1 or temperature <= 0:
        raise ValueError("Invalid patch grid or graph parameters.")
    index = torch.arange(patches, device=evidence.device)
    row, col = index // width, index % width
    neighbors = []
    allowed = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            y, x = row + dy, col + dx
            inside = (y >= 0) & (y < height) & (x >= 0) & (x < width)
            neighbors.append((y.clamp(0, height - 1) * width + x.clamp(0, width - 1)))
            allowed.append(inside)
    neighbor_ids = torch.stack(neighbors, dim=-1)
    edge_valid = torch.stack(allowed, dim=-1)[None]
    edge_valid = edge_valid & valid[..., None] & valid[:, neighbor_ids]
    visual = 0.5 * (
        geometry[:, index[:, None], neighbor_ids]
        + geometry[:, neighbor_ids, index[:, None]]
    )
    contrast = (evidence[:, :, None] - evidence[:, neighbor_ids]).abs()
    weights = visual[..., None].clamp_min(0) * torch.exp(-contrast / temperature)
    weights = weights * edge_valid[..., None]
    maximum_degree = weights.sum(2).amax(1, keepdim=True).clamp_min(1e-8)
    return neighbor_ids, weights / maximum_degree[:, :, None]


def reconstruct_pairs(base: Tensor, sources: tuple[Tensor, ...], matches,
                      graph: tuple[Tensor, Tensor] | None, valid: Tensor,
                      config: ContrastiveContextConfig) -> Tensor:
    """Projected gradient on a convex pair-fit and correction-graph energy."""
    left, right, _, _ = matches
    if base.ndim != 3 or valid.shape != base.shape[:2]:
        raise ValueError("Expected class scores [B,N,C] and valid [B,N].")
    if not sources or any(source.shape != (*base.shape[:2], left.numel()) for source in sources):
        raise ValueError("Every evidence source must have shape [B,N,P].")
    if left.numel() == 0:
        return base
    correction = torch.zeros_like(base)
    active = valid[..., None].to(base.dtype)
    # Complete pair incidence has spectral norm squared C. Normalized graph
    # Laplacian has norm at most two, giving a fixed safe gradient step.
    smooth = config.graph_strength if graph is not None else 0.0
    step = 1.0 / (config.anchor_strength + base.shape[-1] * (len(sources) + 2 * smooth))
    for _ in range(config.solver_steps):
        scores = base + correction
        pair_scores = scores[..., left] - scores[..., right]
        pair_gradient = sum(
            (pair_scores - source).clamp(-config.huber_width, config.huber_width)
            for source in sources
        )
        if graph is not None:
            neighbors, weights = graph
            pair_correction = correction[..., left] - correction[..., right]
            pair_gradient = pair_gradient + smooth * (
                weights * (pair_correction[:, :, None] - pair_correction[:, neighbors])
            ).sum(2)
        gradient = config.anchor_strength * correction
        gradient = gradient.scatter_add(-1, left[None, None].expand_as(pair_gradient), pair_gradient)
        gradient = gradient.scatter_add(-1, right[None, None].expand_as(pair_gradient), -pair_gradient)
        correction = (correction - step * gradient) * active
    return base + correction


class ContrastiveContextSegmenter(ContextualPhraseSegmenter):
    """Couple text-matched class evidence to bounded visual reconstruction."""

    needs_whole = False

    def __init__(self, backbone, config, tcpr_config,
                 contrastive_config: ContrastiveContextConfig = ContrastiveContextConfig()):
        super().__init__(backbone, config, tcpr_config)
        contrastive_config.validate()
        self.contrastive_config = contrastive_config
        self._match_cache = {}

    @torch.inference_mode()
    def read_grounded(self, prepared, whole_map: Tensor | None, bounded_map: Tensor,
                      bank, *, height: int, width: int, top: int, left: int,
                      box: tuple[int, int, int], valid_mask: Tensor):
        del whole_map
        valid = valid_mask.reshape(1, -1).bool()
        text = F.normalize(bank.features.float(), dim=-1)
        local = F.normalize(prepared.geometry_projected.float(), dim=-1) @ text.T
        native = F.normalize(prepared.native_projected.float(), dim=-1) @ text.T
        by, bx, side = box
        context = sample_overview_aliases(
            bounded_map, image_height=side, image_width=side,
            tile_top=top - by, tile_left=left - bx,
            grid_height=prepared.grid_height, grid_width=prepared.grid_width,
            patch_size=self.patch_size,
        )
        geometry = masked_geometry(prepared.geometry_patch_conditional.float(), valid, 1e-6)
        tau = self.config.alias_temperature
        subset = torch.ones(text.shape[0], device=text.device, dtype=torch.bool)

        def aggregate(scores: Tensor) -> Tensor:
            return uniform_subset_scores(scores, bank.parent_indices,
                                         bank.class_count, subset, tau)

        base = aggregate(local)
        bounded_aliases = normalized_logmeanexp(
            torch.stack((local, geometry @ context), dim=-1), tau
        )
        cache_key = id(bank)
        if cache_key not in self._match_cache:
            self._match_cache[cache_key] = alias_matches(bank)
        matches = self._match_cache[cache_key]
        local_pairs = pairwise_evidence(local, matches, tau)
        context_pairs = pairwise_evidence(bounded_aliases, matches, tau)
        native_pairs = pairwise_evidence(native, matches, tau)
        sources = (local_pairs, context_pairs, native_pairs)
        config = self.contrastive_config
        no_graph = reconstruct_pairs(base, sources, matches, None, valid, config)
        graph = local_pair_graph(geometry, local_pairs, valid,
                                 prepared.grid_height, prepared.grid_width,
                                 config.visual_radius, config.edge_temperature)
        with_graph = reconstruct_pairs(base, sources, matches, graph, valid, config)
        local_only = reconstruct_pairs(base, (local_pairs,), matches, None, valid, config)
        fields = {
            "G_all20_uniform": base,
            "BoundedUnion": aggregate(bounded_aliases),
            "Native_all20_uniform": aggregate(native),
            "ContrastiveLocal": local_only,
            "ContrastiveNoGraph": no_graph,
            "ContrastiveGCR": with_graph,
        }
        shape = prepared.grid_height, prepared.grid_width
        output = {
            name: _tokens_to_map(torch.where(valid[..., None], value, base), *shape)
            for name, value in fields.items()
        }
        active = valid.sum().clamp_min(1)
        diagnostics = {
            "mean_local_pair_magnitude": float((local_pairs.abs() * valid[..., None]).sum()
                                               / (active * max(local_pairs.shape[-1], 1))),
            "mean_context_pair_change": float(((context_pairs - local_pairs).abs()
                                                * valid[..., None]).sum()
                                               / (active * max(local_pairs.shape[-1], 1))),
            "mean_native_pair_change": float(((native_pairs - local_pairs).abs()
                                               * valid[..., None]).sum()
                                              / (active * max(local_pairs.shape[-1], 1))),
            "mean_final_correction": float(((with_graph - base).abs() * valid[..., None]).sum()
                                           / (active * bank.class_count)),
            "changed_from_geometry": float(((with_graph.argmax(-1) != base.argmax(-1))
                                             & valid).sum() / active),
        }
        return output, diagnostics
