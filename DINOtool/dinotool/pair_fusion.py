"""Shared-support pair comparisons for parallel multi-channel cost volumes."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class PairComparisonFusion(nn.Module):
    """Replace EPL with symmetric reads and antisymmetric cost updates.

    Spatial and semantic inputs are [B,D,Q,H,W]. Visual tokens are [B,V,H,W]
    and text is [Q,T]. No class IDs, target labels, or predicted masks enter
    this operator. All unordered query pairs are evaluated in chunks.
    """

    def __init__(self, cost_dim: int, visual_dim: int = 1024, text_dim: int = 1024,
                 relation_dim: int = 32, kernel_size: int = 3, pair_chunk: int = 16,
                 recompute_pairs: bool = True) -> None:
        super().__init__()
        if min(cost_dim, visual_dim, text_dim, relation_dim, pair_chunk) < 1:
            raise ValueError("Fusion dimensions and pair_chunk must be positive")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("Use an odd neighborhood kernel of at least three")
        self.cost_dim, self.visual_dim, self.text_dim = cost_dim, visual_dim, text_dim
        self.relation_dim, self.kernel_size, self.pair_chunk = relation_dim, kernel_size, pair_chunk
        self.recompute_pairs = recompute_pairs
        self.spatial_projection = nn.Sequential(nn.LayerNorm(cost_dim), nn.Linear(cost_dim, relation_dim))
        self.semantic_projection = nn.Sequential(nn.LayerNorm(cost_dim), nn.Linear(cost_dim, relation_dim))
        self.visual_projection = nn.Conv2d(visual_dim, relation_dim, 1, bias=False)
        self.text_projection = nn.Linear(text_dim, relation_dim, bias=False)
        self.visual_relation = nn.Linear(2 * relation_dim + 2, relation_dim)
        self.spatial_relation = nn.Linear(2 * relation_dim, relation_dim)
        self.relation_output = nn.Linear(relation_dim, relation_dim)
        self.pair_condition = nn.Sequential(
            nn.Linear(4 * relation_dim, relation_dim), nn.GELU(), nn.Linear(relation_dim, relation_dim)
        )
        self.message_projection = nn.Linear(cost_dim, cost_dim, bias=False)
        # A small nonzero map allows both relation branches to learn immediately.
        nn.init.normal_(self.message_projection.weight, std=0.01 / math.sqrt(cost_dim))
        radius = kernel_size // 2
        axis = torch.arange(-radius, radius + 1, dtype=torch.float32) / radius
        dy, dx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("offsets", torch.stack((dx.flatten(), dy.flatten()), -1), persistent=False)

    def _neighbors(self, features: Tensor) -> Tensor:
        batch, channels, height, width = features.shape
        unfolded = F.unfold(features, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        return unfolded.reshape(batch, channels, self.kernel_size ** 2, height * width).permute(0, 3, 2, 1)

    def _pair_messages(self, left: Tensor, right: Tensor, spatial: Tensor, semantic: Tensor,
                       text: Tensor, visual_relation: Tensor, values: Tensor, valid: Tensor):
        batch, _, height, width, dim = spatial.shape
        pair_count, pixels = left.numel(), height * width
        s_left, s_right = spatial[:, left], spatial[:, right]
        symmetric = torch.cat((s_left + s_right, (s_left - s_right).abs()), dim=-1)
        center_relation = symmetric.reshape(batch, pair_count, pixels, 2 * dim)
        member_relation = self._neighbors(
            symmetric.permute(0, 1, 4, 2, 3).reshape(batch * pair_count, 2 * dim, height, width)
        ).reshape(batch, pair_count, pixels, -1, 2 * dim)
        edge = member_relation - center_relation.unsqueeze(-2)
        relation = self.relation_output(F.gelu(self.spatial_relation(edge) + visual_relation[:, None]))

        h_left = semantic[:, left].reshape(batch, pair_count, pixels, dim)
        h_right = semantic[:, right].reshape(batch, pair_count, pixels, dim)
        text_pair = torch.cat((text[left] + text[right], (text[left] - text[right]).abs()), dim=-1)
        text_pair = text_pair[None, :, None].expand(batch, -1, pixels, -1)
        condition = self.pair_condition(torch.cat((text_pair, h_left + h_right, (h_left - h_right).abs()), -1))
        scores = (condition.unsqueeze(-2) * relation).sum(-1).float() / math.sqrt(dim)
        weights = scores.masked_fill(~valid[:, None], float("-inf")).softmax(-1)

        margin = values[:, :, left] - values[:, :, right]
        margin = margin.permute(0, 2, 1, 3, 4).reshape(batch * pair_count, self.cost_dim, height, width)
        members = self._neighbors(margin).reshape(batch, pair_count, pixels, -1, self.cost_dim)
        center = margin.flatten(2).transpose(1, 2).reshape(batch, pair_count, pixels, self.cost_dim)
        # Geometry affects the weights only; values remain semantic cost margins.
        with torch.autocast(device_type=values.device.type, enabled=False):
            read = (weights.unsqueeze(-1) * (members.float() - center.float().unsqueeze(-2))).sum(-2)
            messages = F.linear(read, self.message_projection.weight.float())
        return messages, weights

    def forward(self, spatial: Tensor, semantic: Tensor, visual: Tensor, text: Tensor,
                *, return_trace: bool = False):
        if spatial.ndim != 5 or spatial.shape != semantic.shape:
            raise ValueError("Expected matching spatial/semantic [B,D,Q,H,W] costs")
        batch, dim, queries, height, width = spatial.shape
        if dim != self.cost_dim or min(batch, queries, height, width) < 1:
            raise ValueError("Cost dimensions do not match the fusion module")
        if visual.shape != (batch, self.visual_dim, height, width) or text.shape != (queries, self.text_dim):
            raise ValueError("Visual/text dimensions do not match the cost grid and vocabulary")
        pairs = torch.triu_indices(queries, queries, offset=1, device=spatial.device)
        if queries == 1:
            if return_trace:
                return semantic, dict(pairs=pairs, weights=semantic.new_empty(batch, 0, height * width,
                                      self.kernel_size ** 2), delta=torch.zeros_like(semantic))
            return semantic

        spatial_repr = self.spatial_projection(spatial.permute(0, 2, 3, 4, 1))
        semantic_repr = self.semantic_projection(semantic.permute(0, 2, 3, 4, 1))
        text_repr = F.normalize(self.text_projection(F.normalize(text, dim=-1)), dim=-1)
        visual_repr = F.normalize(self.visual_projection(F.normalize(visual, dim=1)), dim=1)
        visual_center = visual_repr.flatten(2).transpose(1, 2).unsqueeze(-2)
        visual_members = self._neighbors(visual_repr)
        offsets = self.offsets.to(visual_members.dtype)[None, None].expand(batch, height * width, -1, -1)
        visual_relation = self.visual_relation(torch.cat((
            visual_center * visual_members, (visual_center - visual_members).abs(), offsets
        ), dim=-1))
        # Padding must never become a candidate member, including at image corners.
        valid = self._neighbors(semantic.new_ones(batch, 1, height, width)).squeeze(-1).bool()
        delta = semantic.new_zeros(batch, queries, height * width, dim, dtype=torch.float32)
        traces = []
        for start in range(0, pairs.shape[1], self.pair_chunk):
            left, right = pairs[:, start:start + self.pair_chunk]
            inputs = (left, right, spatial_repr, semantic_repr, text_repr, visual_relation, semantic, valid)
            if self.training and torch.is_grad_enabled() and self.recompute_pairs:
                messages, weights = checkpoint(self._pair_messages, *inputs, use_reentrant=False)
            else:
                messages, weights = self._pair_messages(*inputs)
            delta = delta.index_add(1, left, messages).index_add(1, right, -messages)
            if return_trace:
                traces.append(weights)
        delta = (delta / queries).reshape(batch, queries, height, width, dim).permute(0, 4, 1, 2, 3)
        result = semantic + delta.to(semantic.dtype)
        if return_trace:
            return result, dict(pairs=pairs, weights=torch.cat(traces, dim=1), delta=delta)
        return result
