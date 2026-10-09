"""Matched flat mask decoders for the context/member separation experiment.

The plain control is a text-conditioned, two-slot masked-attention decoder,
not an implementation or reproduction of Mask2Former. All variants have the
same parameters; only the evidence admitted to their attention differs.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MaskQueryUpdate(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.heads = heads

    def forward(self, query: Tensor, pixels: Tensor, log_support: Tensor) -> Tensor:
        batch, classes, regions, dim = query.shape
        local = query.reshape(batch * classes, regions, dim)
        normalized = self.norm(local)
        local = local + self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        query = local.reshape(batch, classes * regions, dim)
        q = self.query(self.cross_norm(query)).reshape(batch, classes * regions, self.heads, -1).transpose(1, 2)
        k = self.key(pixels).reshape(batch, -1, self.heads, dim // self.heads).transpose(1, 2)
        v = self.value(pixels).reshape(batch, -1, self.heads, dim // self.heads).transpose(1, 2)
        bias = log_support.reshape(batch, 1, classes * regions, -1).to(q.dtype)
        read = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        read = read.transpose(1, 2).reshape(batch, classes * regions, dim)
        query = query + self.out(read)
        return (query + self.ffn(query)).reshape(batch, classes, regions, dim)


class MembershipRound(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.member = MaskQueryUpdate(dim, heads)
        self.context = MaskQueryUpdate(dim, heads)
        self.exchange = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, 2 * dim))

    def forward(self, member: Tensor, context: Tensor, pixels: Tensor, member_bias: Tensor, context_bias: Tensor) -> tuple[Tensor, Tensor]:
        member = self.member(member, pixels, member_bias)
        context = self.context(context, pixels, context_bias)
        dm, dc = self.exchange(torch.cat((member, context), dim=-1)).chunk(2, dim=-1)
        member, context = member + dm, context + dc
        return member, context


class FlatMembershipDecoder(nn.Module):
    def __init__(self, dim: int = 128, heads: int = 4, stride: int = 4, radius: int = 7, rounds: int = 2, variant: str = "mask_dual") -> None:
        super().__init__()
        if variant not in {"mask_plain", "mask_dual", "mask_fixed"}:
            raise ValueError("Unknown mask experiment variant")
        if dim < 8 or dim % 8 or dim % heads or stride < 1 or radius < stride or rounds < 1:
            raise ValueError("Invalid flat membership decoder dimensions/support")
        self.variant, self.stride, self.radius = variant, stride, radius
        self.visual = nn.Sequential(nn.Conv2d(2048, dim, 1), nn.GroupNorm(8, dim), nn.GELU())
        self.pixel_norm = nn.LayerNorm(dim)
        self.position = nn.Linear(2, dim, bias=False)
        self.text = nn.Linear(1024, dim)
        self.roles = nn.Parameter(torch.zeros(2, dim))
        self.rounds = nn.ModuleList([MembershipRound(dim, heads) for _ in range(rounds)])
        # Shared across rounds so the fixed-read-support control also trains
        # every mask-head parameter through its final prediction.
        self.mask_embedding = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        nn.init.normal_(self.mask_embedding[-1].weight, std=1e-3)
        nn.init.zeros_(self.mask_embedding[-1].bias)

    def geometry(self, height: int, width: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        ry, rx = torch.meshgrid(torch.arange(0, height, self.stride, device=device), torch.arange(0, width, self.stride, device=device), indexing="ij")
        support = ((yy.flatten()[None] - ry.flatten()[:, None]).abs() <= self.radius) & ((xx.flatten()[None] - rx.flatten()[:, None]).abs() <= self.radius)
        coordinates = torch.stack((yy.flatten() / max(height - 1, 1), xx.flatten() / max(width - 1, 1)), dim=-1).float() * 2 - 1
        centers = torch.stack((ry.flatten() / max(height - 1, 1), rx.flatten() / max(width - 1, 1)), dim=-1).float() * 2 - 1
        return support, coordinates, centers

    def forward(self, x: Tensor, y: Tensor, initial_logits: Tensor, text: Tensor, *, return_states: bool = False) -> dict[str, Tensor]:
        field = self.visual(torch.cat((F.normalize(x, dim=1), F.normalize(y, dim=1)), dim=1))
        batch, dim, height, width = field.shape
        support, coordinates, centers = self.geometry(height, width, field.device)
        pixels = self.pixel_norm(field.flatten(2).transpose(1, 2))
        spatial_pixels = pixels + self.position(coordinates.to(pixels.dtype))[None]
        row_weights = support.to(pixels.dtype) / support.sum(-1, keepdim=True)
        anchors = torch.einsum("kn,bnd->bkd", row_weights.to(pixels.dtype), pixels)
        anchors = anchors + self.position(centers.to(pixels.dtype))[None]
        initial_query = anchors[:, None] + self.text(text)[None, :, None]
        member = initial_query + self.roles[0]
        context = initial_query + self.roles[1]
        relative_logits = initial_logits.float()
        if len(text) > 1:
            # CE does not identify a common logit offset. Remove it before
            # using sigmoid as a soft read support, not calibrated confidence.
            relative_logits = relative_logits - relative_logits.mean(1, keepdim=True)
        base = relative_logits.flatten(2)[:, :, None, :]
        mask_logits = base.expand(-1, -1, len(centers), -1)
        fixed_logits = mask_logits
        broad_bias = torch.zeros_like(mask_logits).masked_fill(~support[None, None], -torch.inf)
        changes = []
        for block in self.rounds:
            evidence = fixed_logits if self.variant == "mask_fixed" else mask_logits
            member_bias = F.logsigmoid(evidence).masked_fill(~support[None, None], -torch.inf)
            context_bias = member_bias if self.variant == "mask_plain" else broad_bias
            member, context = block(member, context, spatial_pixels, member_bias, context_bias)
            embedding = self.mask_embedding(torch.cat((member, context), dim=-1))
            delta = torch.einsum("bckd,bnd->bckn", embedding, pixels) / math.sqrt(dim)
            next_logits = base + delta.float()
            changes.append(((next_logits.sigmoid() - mask_logits.sigmoid()).abs() * support[None, None]).sum() / (support.sum() * batch * len(text)))
            mask_logits = next_logits
        # Partition-of-unity fusion bounds probabilities but does not prevent
        # false positives. The experiment must establish a semantic benefit.
        column_weights = support.float() / support.sum(0, keepdim=True)
        probability = (mask_logits.sigmoid() * column_weights[None, None]).sum(2)
        logits = torch.logit(probability.clamp(1e-6, 1 - 1e-6)).reshape(batch, len(text), height, width)
        result = {"coarse_logits": logits, "mean_member_change": torch.stack(changes).detach()}
        if return_states:
            result.update(region_probabilities=mask_logits.sigmoid(), support=column_weights, member_state=member, context_state=context)
        return result
