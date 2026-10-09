"""Sparse, two-scale soft regions and bidirectional pixel/region operations."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class RegionLayout:
    indices: Tensor
    valid: Tensor
    pixels: Tensor
    centers: Tensor
    scales: Tensor
    shapes: list[tuple[int, int]]

    @property
    def count(self) -> int:
        return len(self.centers)


def make_layout(height: int, width: int, device: torch.device) -> RegionLayout:
    yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    pixels = torch.stack(((xx.flatten() + 0.5) / width, (yy.flatten() + 0.5) / height), -1)
    dy, dx = torch.meshgrid(torch.arange(-1, 2, device=device), torch.arange(-1, 2, device=device), indexing="ij")
    indices, masks, centers, scales, shapes = [], [], [], [], []
    offset = 0
    for stride in (2, 4):
        rh, rw = math.ceil(height / stride), math.ceil(width / stride)
        cy = yy.flatten()[:, None] // stride + dy.flatten()
        cx = xx.flatten()[:, None] // stride + dx.flatten()
        valid = (cy >= 0) & (cy < rh) & (cx >= 0) & (cx < rw)
        indices.append(cy.clamp(0, rh - 1) * rw + cx.clamp(0, rw - 1) + offset)
        masks.append(valid)
        ry, rx = torch.meshgrid(torch.arange(rh, device=device), torch.arange(rw, device=device), indexing="ij")
        centers.append(torch.stack(((rx.flatten() + 0.5) * stride / width, (ry.flatten() + 0.5) * stride / height), -1).clamp(0, 1))
        scales.append(pixels.new_tensor([stride / width, stride / height]).expand(rh * rw, -1))
        shapes.append((rh, rw))
        offset += rh * rw
    return RegionLayout(torch.cat(indices, -1), torch.cat(masks, -1), pixels, torch.cat(centers), torch.cat(scales), shapes)


def region_pool(values: Tensor, weights: Tensor, layout: RegionLayout) -> tuple[Tensor, Tensor]:
    """Weighted pixel-to-region mean; float32 accumulation also under BF16."""
    b, n, d = values.shape
    k = weights.shape[-1]
    index = layout.indices.reshape(1, n * k, 1)
    mass = weights.float().new_zeros(b, layout.count, 1).scatter_add(
        1, index.expand(b, -1, 1), weights.float().reshape(b, n * k, 1))
    chunks = []
    for part in values.split(256, dim=-1):
        weighted = (part.float()[:, :, None] * weights.float()[..., None]).reshape(b, n * k, -1)
        total = weighted.new_zeros(b, layout.count, part.shape[-1]).scatter_add(
            1, index.expand(b, -1, part.shape[-1]), weighted)
        chunks.append(total / mass.clamp_min(1e-6))
    return torch.cat(chunks, -1).to(values.dtype), mass


def region_write(values: Tensor, weights: Tensor, layout: RegionLayout) -> Tensor:
    return torch.cat([
        (part[:, layout.indices].float() * weights.float()[..., None]).sum(2).to(values.dtype)
        for part in values.split(256, dim=-1)
    ], -1)


class SoftSupports(nn.Module):
    def __init__(self, dim: int, cost_dim: int):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(dim + cost_dim + 2), nn.Linear(dim + cost_dim + 2, dim))
        self.key = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.affinity = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2))

    def forward(self, visual: Tensor, cost: Tensor, nodes: Tensor, centers: Tensor, layout: RegionLayout):
        affinity = self.affinity(visual)
        query = self.query(torch.cat((visual, cost.mean(2), affinity.sigmoid()), -1))
        keys = self.key(nodes)[:, layout.indices]
        logits = (query[:, :, None].float() * keys.float()).sum(-1) / math.sqrt(query.shape[-1])
        delta = (layout.pixels[None, :, None] - centers[:, layout.indices]) / layout.scales[layout.indices][None]
        logits = logits - 0.5 * delta.square().sum(-1)
        logits = logits.masked_fill(~layout.valid[None], -torch.inf)
        # Each scale carries half the support mass; padded candidates never compete.
        weights = torch.cat([part.softmax(-1) * 0.5 for part in logits.split(9, -1)], -1)
        new_centers, mass = region_pool(layout.pixels[None].expand(len(visual), -1, -1), weights, layout)
        new_centers = torch.where(mass > 1e-6, new_centers, centers)
        return weights, new_centers, affinity


class RegionalRelation(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.geometry = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, heads))
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, nodes: Tensor, centers: Tensor, scales: Tensor) -> Tensor:
        delta = centers[:, :, None] - centers[:, None, :]
        size = scales.prod(-1).sqrt().clamp_min(1e-6)
        ratio = (size[:, None] / size[None, :]).log()[None, :, :, None].expand(len(nodes), -1, -1, -1)
        geometry = torch.cat((delta, ratio), -1).to(nodes.dtype)
        bias = self.geometry(geometry).permute(0, 3, 1, 2).flatten(0, 1)
        normalized = self.norm1(nodes)
        nodes = nodes + self.attention(normalized, normalized, normalized, attn_mask=bias, need_weights=False)[0]
        return nodes + self.ffn(self.norm2(nodes))


class RegionReasoningStage(nn.Module):
    def __init__(self, dim: int, cost_dim: int, heads: int, modes: int, *, update_state: bool):
        super().__init__()
        self.dim, self.cost_dim, self.modes = dim, cost_dim, modes
        self.support = SoftSupports(dim, cost_dim)
        self.region_input = nn.Sequential(nn.LayerNorm(2 * dim + cost_dim), nn.Linear(2 * dim + cost_dim, dim))
        self.relations = nn.ModuleList([RegionalRelation(dim, heads) for _ in range(2)])
        self.text_modes = nn.Linear(1024, dim * modes)
        self.mode_decoder = nn.TransformerDecoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.region_query = nn.Linear(dim, dim)
        self.evidence = nn.Sequential(nn.LayerNorm(2 * dim + cost_dim), nn.Linear(2 * dim + cost_dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, cost_dim))
        self.region_classifier = nn.Linear(cost_dim, 1)
        self.pixel_evidence = nn.Sequential(nn.LayerNorm(2 * cost_dim + dim), nn.Linear(2 * cost_dim + dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, cost_dim))
        nn.init.zeros_(self.pixel_evidence[-1].weight)
        nn.init.zeros_(self.pixel_evidence[-1].bias)
        self.update_state = update_state
        if update_state:
            self.semantic_feedback = nn.Linear(cost_dim, dim)
            self.visual_update = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, cost: Tensor, visual: Tensor, nodes: Tensor, centers: Tensor, text: Tensor, layout: RegionLayout):
        # Internal cost order is BNQD, unlike CAFe's BDQHW.
        b, n, q, d = cost.shape
        weights, centers, affinity = self.support(visual, cost, nodes, centers, layout)
        pooled_visual, _ = region_pool(visual, weights, layout)
        pooled_cost, _ = region_pool(cost.flatten(2), weights, layout)
        pooled_cost = pooled_cost.reshape(b, layout.count, q, d)
        nodes = self.region_input(torch.cat((nodes, pooled_visual, pooled_cost.mean(2)), -1))
        for relation in self.relations:
            nodes = relation(nodes, centers, layout.scales)
        modes = self.text_modes(text).reshape(1, q * self.modes, self.dim).expand(b, -1, -1)
        modes = self.mode_decoder(modes, nodes).reshape(b, q, self.modes, self.dim)
        match = torch.einsum("brd,bqmd->brqm", self.region_query(nodes), modes) / math.sqrt(self.dim)
        composed = torch.einsum("brqm,bqmd->brqd", match.softmax(-1), modes)
        evidence = self.evidence(torch.cat((nodes[:, :, None].expand(-1, -1, q, -1), composed, pooled_cost), -1))
        retrieved = region_write(evidence.flatten(2), weights, layout).reshape(b, n, q, d)
        correction = self.pixel_evidence(torch.cat((cost, retrieved, visual[:, :, None].expand(-1, -1, q, -1)), -1))
        if self.update_state:
            nodes = nodes + self.semantic_feedback(evidence.mean(2))
            visual = visual + self.visual_update(torch.cat((visual, region_write(nodes, weights, layout)), -1))
        auxiliary = {"weights": weights, "region_logits": self.region_classifier(evidence).squeeze(-1), "affinity_logits": affinity}
        return cost + correction, visual, nodes, centers, auxiliary


def region_auxiliary_losses(stages: list[dict], target: Tensor, classes: int, shape: tuple[int, int], layout: RegionLayout) -> dict[str, Tensor]:
    """Only the loss receives GT; supports are detached when defining regional targets."""
    valid = target != 255
    onehot = F.one_hot(target.masked_fill(~valid, 0), classes).permute(0, 3, 1, 2).float() * valid[:, None]
    fractions = F.adaptive_avg_pool2d(onehot, shape).flatten(2).transpose(1, 2)
    valid_fraction = fractions.sum(-1)
    labels = fractions.argmax(-1).reshape(-1, *shape)
    reliable = (fractions.max(-1).values >= 0.85).reshape(-1, *shape)
    losses, affinities = [], []
    for stage in stages:
        distribution, _ = region_pool(fractions, stage["weights"].detach(), layout)
        mass = distribution.sum(-1)
        distribution = distribution / mass[..., None].clamp_min(1e-6)
        log_probs = stage["region_logits"].float().log_softmax(-1)
        losses.append((-(distribution * log_probs).sum(-1) * mass).sum() / mass.sum().clamp_min(1e-6))
        logits = stage["affinity_logits"].reshape(-1, *shape, 2).float()
        horizontal = F.binary_cross_entropy_with_logits(logits[:, :, :-1, 0], (labels[:, :, :-1] == labels[:, :, 1:]).float(), reduction="none")
        vertical = F.binary_cross_entropy_with_logits(logits[:, :-1, :, 1], (labels[:, :-1, :] == labels[:, 1:, :]).float(), reduction="none")
        hv = reliable[:, :, :-1] & reliable[:, :, 1:]
        vv = reliable[:, :-1, :] & reliable[:, 1:, :]
        affinities.append(((horizontal * hv).sum() + (vertical * vv).sum()) / (hv.sum() + vv.sum()).clamp_min(1))
    return {"region_loss": torch.stack(losses).mean(), "affinity_loss": torch.stack(affinities).mean()}
