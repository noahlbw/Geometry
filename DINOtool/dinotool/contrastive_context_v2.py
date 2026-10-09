"""Bounded-context anchor with pairwise alias reconciliation."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .contextual_phrase_readout import masked_geometry, normalized_logmeanexp, sample_overview_aliases
from .contrastive_context import (
    ContrastiveContextConfig, ContrastiveContextSegmenter, alias_matches,
    local_pair_graph, pairwise_evidence, reconstruct_pairs,
)
from .hypothesis_readout import uniform_subset_scores
from .tcpr import _tokens_to_map


METHODS = (
    "G_all20_uniform", "BoundedUnion", "Native_all20_uniform",
    "ContrastiveLocal", "PairwiseReconcile", "PairwiseGCR",
)


class PairwiseReconciliationSegmenter(ContrastiveContextSegmenter):
    """Use bounded evidence as an anchor and reconstruct signed pair margins."""

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

        local_logits = aggregate(local)
        bounded_aliases = normalized_logmeanexp(
            torch.stack((local, geometry @ context), dim=-1), tau
        )
        bounded_logits = aggregate(bounded_aliases)
        cache_key = id(bank)
        if cache_key not in self._match_cache:
            self._match_cache[cache_key] = alias_matches(bank)
        matches = self._match_cache[cache_key]
        local_pairs = pairwise_evidence(local, matches, tau)
        context_pairs = pairwise_evidence(bounded_aliases, matches, tau)
        config: ContrastiveContextConfig = self.contrastive_config
        local_only = reconstruct_pairs(
            local_logits, (local_pairs,), matches, None, valid, config
        )
        sources = (local_pairs, context_pairs)
        reconciled = reconstruct_pairs(
            bounded_logits, sources, matches, None, valid, config
        )
        graph = local_pair_graph(
            geometry, local_pairs, valid, prepared.grid_height, prepared.grid_width,
            config.visual_radius, config.edge_temperature,
        )
        reconstructed = reconstruct_pairs(
            bounded_logits, sources, matches, graph, valid, config
        )
        fields = {
            "G_all20_uniform": local_logits,
            "BoundedUnion": bounded_logits,
            "Native_all20_uniform": aggregate(native),
            "ContrastiveLocal": local_only,
            "PairwiseReconcile": reconciled,
            "PairwiseGCR": reconstructed,
        }
        shape = prepared.grid_height, prepared.grid_width
        output = {
            name: _tokens_to_map(torch.where(valid[..., None], value, local_logits), *shape)
            for name, value in fields.items()
        }
        active = valid.sum().clamp_min(1)
        diagnostics = {
            "mean_local_pair_magnitude": float((local_pairs.abs() * valid[..., None]).sum()
                                               / (active * max(local_pairs.shape[-1], 1))),
            "mean_context_pair_change": float(((context_pairs - local_pairs).abs()
                                                * valid[..., None]).sum()
                                               / (active * max(local_pairs.shape[-1], 1))),
            "mean_final_correction": float(((reconstructed - bounded_logits).abs()
                                            * valid[..., None]).sum()
                                           / (active * bank.class_count)),
            "changed_from_geometry": float(((reconstructed.argmax(-1)
                                              != local_logits.argmax(-1)) & valid).sum() / active),
        }
        return output, diagnostics
