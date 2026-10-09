"""Image-shared mask proposals and recurrent, query-conditioned region assembly.

The proposal decoder uses PyTorch masked Transformer decoding. It is a compact
Mask2Former-style component, not a reproduction of the complete Mask2Former or
FC-CLIP system. No labels enter any forward method.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RegionAssemblyConfig:
    arm: str = "query_assembly"
    dim: int = 256
    heads: int = 8
    regions: int = 128
    parents: int = 32
    proposal_layers: int = 3
    rounds: int = 2
    stages: int = 3
    query_chunk: int = 8

    def validate(self) -> None:
        if self.arm not in {"flat_regions", "image_assembly", "query_assembly"}:
            raise ValueError("Unknown region assembly arm")
        if min(self.heads, self.proposal_layers, self.stages, self.query_chunk) < 1:
            raise ValueError("Head, layer, stage and chunk counts must be positive")
        if self.dim < 16 or self.dim % self.heads or self.dim % 8:
            raise ValueError("Region width must be >=16 and divisible by heads and eight")
        if not 2 <= self.parents <= self.regions or self.rounds < 2:
            raise ValueError("Require 2 <= parents <= regions and at least two recurrent reads")


def coordinates(height: int, width: int, reference: Tensor) -> Tensor:
    y = (torch.arange(height, device=reference.device, dtype=torch.float32) + 0.5) / height
    x = (torch.arange(width, device=reference.device, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), -1).reshape(-1, 2).to(reference.dtype)


def region_pool(values: Tensor, membership: Tensor) -> Tensor:
    """[B,N,D], [B,N,K] -> [B,K,D]; empty regions read zero."""
    with torch.autocast(values.device.type, enabled=False):
        weights = membership.float()
        pooled = weights.transpose(1, 2) @ values.float()
        pooled = pooled / weights.sum(1).unsqueeze(-1).clamp_min(1e-6)
    return pooled.to(values.dtype)


def region_write(values: Tensor, membership: Tensor) -> Tensor:
    with torch.autocast(values.device.type, enabled=False):
        result = membership.float() @ values.float()
    return result.to(values.dtype)


def compose_membership(children: Tensor, assignment: Tensor) -> Tensor:
    with torch.autocast(children.device.type, enabled=False):
        return children.float() @ assignment.float()


class SharedMaskProposals(nn.Module):
    """Encode one image into masks without a fixed-vocabulary classifier."""

    def __init__(self, config: RegionAssemblyConfig) -> None:
        super().__init__()
        d = config.dim
        self.x_projection = nn.Conv2d(1024, d, 1)
        self.y_projection = nn.Conv2d(1024, d, 1)
        self.spatial = nn.Sequential(nn.GroupNorm(8, d), nn.GELU(),
                                     nn.Conv2d(d, d, 3, padding=1), nn.GroupNorm(8, d))
        self.semantic_norm = nn.LayerNorm(d)
        self.position = nn.Linear(2, d)
        self.levels = nn.Parameter(torch.randn(3, d) * 0.02)
        self.queries = nn.Parameter(torch.randn(config.regions, d) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d, config.heads, d * 4, dropout=0.0,
                                       batch_first=True, norm_first=True)
            for _ in range(config.proposal_layers)
        ])
        self.mask_embedding = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.heads = config.heads

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        field = self.spatial(self.x_projection(F.normalize(x, dim=1)))
        semantics = self.semantic_norm(self.y_projection(F.normalize(y, dim=1)).flatten(2).transpose(1, 2))
        b, d, h, w = field.shape
        xy = coordinates(h, w, field)
        pixels = field.flatten(2).transpose(1, 2)
        levels = [field] + [F.adaptive_avg_pool2d(field, (max(1, math.ceil(h / s)), max(1, math.ceil(w / s))))
                            for s in (2, 4)]
        query = self.queries[None].expand(b, -1, -1)
        masks = self.mask_embedding(query) @ pixels.transpose(1, 2) / math.sqrt(d)
        for index, layer in enumerate(self.layers):
            level = levels[index % 3]
            lh, lw = level.shape[-2:]
            memory = level.flatten(2).transpose(1, 2)
            memory = memory + self.position(coordinates(lh, lw, level))[None] + self.levels[index % 3]
            blocked = F.interpolate(masks.reshape(b, -1, h, w).float(), (lh, lw), mode="bilinear",
                                    align_corners=False).flatten(2) < 0
            # A proposal with no admitted pixels must still be able to recover.
            blocked = blocked & ~blocked.all(-1, keepdim=True)
            blocked = blocked[:, None].expand(-1, self.heads, -1, -1).reshape(b * self.heads, -1, lh * lw)
            query = layer(query, memory, memory_mask=blocked.detach())
            masks = self.mask_embedding(query) @ pixels.transpose(1, 2) / math.sqrt(d)
        return pixels, semantics, masks.transpose(1, 2).float(), xy


class AssemblyRead(nn.Module):
    """Compose children, recognize parents, and update pixel-to-child membership."""

    def __init__(self, dim: int, cost_dim: int, parents: int, *, regroup: bool) -> None:
        super().__init__()
        self.regroup = regroup
        self.semantic = nn.Sequential(nn.LayerNorm(3 * dim + cost_dim),
                                      nn.Linear(3 * dim + cost_dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.member_score = nn.Linear(dim, 1)
        if regroup:
            self.parent_queries = nn.Parameter(torch.randn(parents, dim) * 0.02)
            self.parent_norm = nn.LayerNorm(dim)
            self.parent_key = nn.Linear(dim, dim)
            self.child_key = nn.Linear(dim, dim)
            self.position = nn.Linear(2, dim, bias=False)
            self.image_cost = nn.Linear(dim, cost_dim)
            self.update_child = nn.Sequential(nn.LayerNorm(3 * dim), nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
            self.update_pixel = nn.Sequential(nn.LayerNorm(dim + cost_dim), nn.Linear(dim + cost_dim, dim), nn.GELU())

    def forward(self, pixels: Tensor, semantics: Tensor, logits: Tensor, cost: Tensor,
                initial: Tensor, text: Tensor, xy: Tensor, *, conditioned: bool, update: bool) -> dict[str, Tensor]:
        membership = logits.float().softmax(-1)
        child = region_pool(pixels, membership)
        assignment = None
        if self.regroup:
            centers = region_pool(xy[None].expand(len(pixels), -1, -1), membership)
            condition = text if conditioned else pixels.mean(1)
            # Nonlinear conditioning matters: adding the same text term to every
            # parent after a linear key projection would cancel in the softmax.
            parent_queries = self.parent_key(self.parent_norm(self.parent_queries[None] + condition[:, None]))
            child_keys = self.child_key(child + self.position(centers))
            affinity = child_keys @ parent_queries.transpose(1, 2) / math.sqrt(child.shape[-1])
            assignment = affinity.float().softmax(-1)
            parent_membership = compose_membership(membership, assignment)
        else:
            parent_membership = membership
        structure = region_pool(pixels, parent_membership)
        content = region_pool(semantics, parent_membership)
        pooled_cost = region_pool(cost, parent_membership)
        evidence = self.semantic(torch.cat((structure, content, text[:, None].expand_as(structure), pooled_cost), -1))
        region_logits = self.member_score(evidence).squeeze(-1)
        with torch.autocast(pixels.device.type, enabled=False):
            probability = (parent_membership * region_logits.float().sigmoid()[:, None]).sum(-1)
            member_logits = torch.logit(probability.clamp(1e-6, 1 - 1e-6))
        next_logits = logits
        if self.regroup and update:
            # Image-only control never uses text, cost, or semantic evidence to
            # construct its geometry. Recognition still uses text in both arms.
            parent_message = evidence if conditioned else structure
            message = region_write(parent_message, assignment)
            condition = text if conditioned else pixels.mean(1)
            child_update = self.update_child(torch.cat((child, message, condition[:, None].expand_as(child)), -1))
            local_cost = self.image_cost(pixels)
            if conditioned:
                local_cost = local_cost + initial
            pixel_update = self.update_pixel(torch.cat((pixels, local_cost), -1))
            delta = pixel_update @ child_update.transpose(1, 2) / math.sqrt(child.shape[-1])
            next_logits = logits.float() + delta.float()
        result = dict(logits=next_logits, child_membership=membership, parent_membership=parent_membership,
                      member_logits=member_logits, dense_evidence=region_write(evidence, parent_membership))
        if assignment is not None:
            result["assignment"] = assignment
        return result


class RegionAssemblyStage(nn.Module):
    def __init__(self, config: RegionAssemblyConfig, cost_dim: int) -> None:
        super().__init__()
        self.config = config
        self.text = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, config.dim), nn.GELU())
        self.read = AssemblyRead(config.dim, cost_dim, config.parents, regroup=config.arm != "flat_regions")
        self.write = nn.Sequential(nn.LayerNorm(2 * cost_dim + config.dim),
                                   nn.Linear(2 * cost_dim + config.dim, config.dim), nn.GELU(), nn.Linear(config.dim, cost_dim))
        nn.init.zeros_(self.write[-1].weight)
        nn.init.zeros_(self.write[-1].bias)

    def forward(self, pixels: Tensor, semantics: Tensor, masks: Tensor, cost: Tensor, initial: Tensor,
                text: Tensor, xy: Tensor, *, return_regions: bool = False) -> tuple[Tensor, Tensor, Tensor, list[dict]]:
        b, d, q, h, w = cost.shape
        state = cost.permute(0, 2, 3, 4, 1).reshape(b, q, h * w, d)
        original = initial.permute(0, 2, 3, 4, 1).reshape(b, q, h * w, d)
        corrections, next_masks, member_logits, traces = [], [], [], []
        for start in range(0, q, self.config.query_chunk):
            stop = min(q, start + self.config.query_chunk)
            chunk = stop - start
            def repeat_image(value):
                return value[:, None].expand(-1, chunk, *value.shape[1:]).reshape(b * chunk, *value.shape[1:])
            local = state[:, start:stop].reshape(b * chunk, h * w, d)
            local_initial = original[:, start:stop].reshape_as(local)
            query = self.text(text[start:stop])[None].expand(b, -1, -1).reshape(b * chunk, -1)
            logits = masks[:, start:stop].reshape(b * chunk, h * w, -1)
            fields, meanings = repeat_image(pixels), repeat_image(semantics)
            round_logits = []
            count = self.config.rounds if self.config.arm != "flat_regions" else 1
            for iteration in range(count):
                result = self.read(fields, meanings, logits, local, local_initial, query, xy,
                                   conditioned=self.config.arm == "query_assembly", update=iteration + 1 < count)
                round_logits.append(result["member_logits"].reshape(b, chunk, h, w))
                if return_regions:
                    traces.append({"query_start": start, "query_stop": stop, "round": iteration,
                                   **{key: result[key] for key in ("child_membership", "parent_membership", "assignment") if key in result}})
                logits = result["logits"]
            correction = self.write(torch.cat((local, local_initial, result["dense_evidence"]), -1))
            corrections.append(correction.reshape(b, chunk, h * w, d))
            next_masks.append(logits.reshape(b, chunk, h * w, -1))
            member_logits.append(torch.stack(round_logits, 1))
        correction = torch.cat(corrections, 1).reshape(b, q, h, w, d).permute(0, 4, 1, 2, 3)
        return cost + correction, torch.cat(next_masks, 1), torch.cat(member_logits, 2), traces


def membership_loss(predictions: list[Tensor], target: Tensor, classes: int) -> Tensor:
    """Soft source-mask occupancy at patch resolution; ignore is never a negative."""
    valid = target != 255
    labels = F.one_hot(target.masked_fill(~valid, 0), classes).permute(0, 3, 1, 2).float()
    labels = labels * valid[:, None]
    losses = []
    for logits in predictions:
        shape = logits.shape[-2:]
        occupied = F.adaptive_avg_pool2d(labels, shape)
        fraction = F.adaptive_avg_pool2d(valid[:, None].float(), shape)
        desired = occupied / fraction.clamp_min(1e-6)
        desired = desired[:, None].expand_as(logits)
        weights = fraction[:, None]
        loss = F.binary_cross_entropy_with_logits(logits.float(), desired, reduction="none")
        losses.append((loss * weights).sum() / (fraction.sum() * classes * logits.shape[1]).clamp_min(1))
    return torch.stack(losses).mean()
