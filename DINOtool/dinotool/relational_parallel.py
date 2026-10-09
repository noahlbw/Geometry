"""Task-relational parallel fusion for CAFe-DINO cost volumes.

The module consumes parallel spatial/class aggregations and reconstructs one
cost field from a spatial edge-observation field and a query-relative field.
It deliberately avoids expert routing and explicit O(Q^2) query-pair tensors.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SpatialStencil:
    """Four canonical edges whose undirected closure is an 8-neighborhood."""

    directions = ((0, 1), (1, 0), (1, 1), (1, -1))

    @staticmethod
    def neighbor(value: Tensor, direction: int) -> Tensor:
        if direction == 0:
            return F.pad(value[..., 1:] , (0, 1))
        if direction == 1:
            return F.pad(value[..., 1:, :], (0, 0, 0, 1))
        if direction == 2:
            return F.pad(value[..., 1:, 1:], (0, 1, 0, 1))
        if direction == 3:
            return F.pad(value[..., 1:, :-1], (1, 0, 0, 1))
        raise ValueError("Unknown spatial stencil direction")

    @classmethod
    def difference(cls, value: Tensor) -> Tensor:
        """Return canonical neighbor-minus-center differences as [B,4,D,Q,H,W]."""
        if value.ndim != 5:
            raise ValueError("Expected [B,D,Q,H,W] tensor")
        return torch.stack([cls.neighbor(value, direction) - value for direction in range(4)], dim=1)

    @staticmethod
    def edge_valid(valid: Tensor) -> Tensor:
        """Return source/destination-valid masks as [B,4,H,W]."""
        if valid.ndim != 3:
            raise ValueError("Expected [B,H,W] validity mask")
        batch, height, width = valid.shape
        masks = []
        right = torch.zeros_like(valid)
        right[:, :, :-1] = valid[:, :, :-1] & valid[:, :, 1:]
        masks.append(right)
        down = torch.zeros_like(valid)
        down[:, :-1, :] = valid[:, :-1, :] & valid[:, 1:, :]
        masks.append(down)
        down_right = torch.zeros_like(valid)
        down_right[:, :-1, :-1] = valid[:, :-1, :-1] & valid[:, 1:, 1:]
        masks.append(down_right)
        down_left = torch.zeros_like(valid)
        down_left[:, :-1, 1:] = valid[:, :-1, 1:] & valid[:, 1:, :-1]
        masks.append(down_left)
        return torch.stack(masks, dim=1).reshape(batch, 4, height, width)

    @staticmethod
    def adjoint(field: Tensor) -> Tensor:
        """Exact transpose of ``difference`` for canonical directed edges."""
        if field.ndim != 6 or field.shape[1] != 4:
            raise ValueError("Expected [B,4,D,Q,H,W] edge field")
        right, down, down_right, down_left = field.unbind(1)
        result = -F.pad(right[..., :-1], (0, 1)) + F.pad(right[..., :-1], (1, 0))
        result = result - F.pad(down[..., :-1, :], (0, 0, 0, 1)) + F.pad(down[..., :-1, :], (0, 0, 1, 0))
        result = result - F.pad(down_right[..., :-1, :-1], (0, 1, 0, 1))
        result = result + F.pad(down_right[..., :-1, :-1], (1, 0, 1, 0))
        result = result - F.pad(down_left[..., :-1, 1:], (1, 0, 0, 1))
        result = result + F.pad(down_left[..., :-1, 1:], (0, 1, 1, 0))
        return result


def _channel_tokens(value: Tensor) -> Tensor:
    """[B,D,Q,H,W] -> [B,Q,D,H,W] for query-folded pointwise layers."""
    return value.permute(0, 2, 1, 3, 4).contiguous()


def _cost_layer_norm(value: Tensor, norm: nn.LayerNorm, projection: nn.Linear) -> Tensor:
    batch, dim, queries, height, width = value.shape
    tokens = value.permute(0, 2, 3, 4, 1).reshape(batch * queries, height, width, dim)
    tokens = projection(norm(tokens))
    return tokens.reshape(batch, queries, height, width, -1).permute(0, 1, 4, 2, 3).contiguous()


class SpatialRelationHead(nn.Module):
    """Predict nonzero, class-conditioned spatial cost differences."""

    def __init__(self, cost_dim: int, visual_dim: int, relation_dim: int) -> None:
        super().__init__()
        self.cost_dim = cost_dim
        self.relation_dim = relation_dim
        self.cost_norm = nn.LayerNorm(cost_dim)
        self.cost_projection = nn.Linear(cost_dim, relation_dim)
        self.visual_projection = nn.Conv2d(visual_dim, relation_dim, 1, bias=False)
        self.symmetric = nn.Sequential(
            nn.Conv2d(4 * relation_dim + 2, relation_dim, 1),
            nn.GELU(),
            nn.Conv2d(relation_dim, relation_dim, 1),
        )
        self.antisymmetric = nn.Conv2d(2 * relation_dim + 2, relation_dim, 1, bias=False)
        self.gate = nn.Conv2d(relation_dim, relation_dim, 1)
        self.delta = nn.Conv2d(relation_dim, cost_dim, 1, bias=False)
        self.weight = nn.Conv2d(relation_dim, 1, 1)
        nn.init.normal_(self.delta.weight, std=0.01 / math.sqrt(relation_dim))
        nn.init.zeros_(self.weight.weight)
        nn.init.zeros_(self.weight.bias)
        offsets = torch.tensor(SpatialStencil.directions, dtype=torch.float32)
        self.register_buffer("offsets", offsets, persistent=False)

    def forward(self, spatial: Tensor, visual: Tensor) -> tuple[Tensor, Tensor]:
        if spatial.ndim != 5 or visual.ndim != 4:
            raise ValueError("Expected spatial [B,D,Q,H,W] and visual [B,V,H,W]")
        batch, _, queries, height, width = spatial.shape
        if visual.shape[0] != batch or visual.shape[-2:] != (height, width):
            raise ValueError("Visual guidance must share the cost grid")
        base = _cost_layer_norm(spatial, self.cost_norm, self.cost_projection)
        visual_features = F.normalize(self.visual_projection(F.normalize(visual, dim=1)), dim=1)
        visual_features = visual_features[:, None].expand(-1, queries, -1, -1, -1)
        base_difference = SpatialStencil.difference(spatial)
        edge_valid = SpatialStencil.edge_valid(torch.ones(batch, height, width, dtype=torch.bool, device=spatial.device))
        relations, weights = [], []
        for direction in range(4):
            base_neighbor = SpatialStencil.neighbor(base, direction)
            visual_neighbor = SpatialStencil.neighbor(visual_features, direction)
            offset = self.offsets[direction].to(dtype=base.dtype, device=base.device)
            offset = offset.view(1, 1, 2, 1, 1).expand(batch, queries, -1, height, width)
            symmetric = torch.cat((base + base_neighbor, (base_neighbor - base).abs(),
                                   visual_features * visual_neighbor, (visual_neighbor - visual_features).abs(),
                                   offset), dim=2)
            antisymmetric = torch.cat((base_neighbor - base, visual_neighbor - visual_features, offset), dim=2)
            symmetric = symmetric.reshape(batch * queries, symmetric.shape[2], height, width)
            antisymmetric = antisymmetric.reshape(batch * queries, antisymmetric.shape[2], height, width)
            context = self.symmetric(symmetric)
            residual = self.delta(torch.sigmoid(self.gate(context)) * torch.tanh(self.antisymmetric(antisymmetric)))
            residual = residual.reshape(batch, queries, self.cost_dim, height, width).permute(0, 2, 1, 3, 4)
            relation = base_difference[:, direction] + residual
            weight = 0.25 + 0.75 * torch.sigmoid(self.weight(context))
            weight = weight.reshape(batch, queries, height, width)
            # Relation has [B,D,Q,H,W], while edge weights have [B,Q,H,W].
            # Keep both singleton axes here: omitting the query axis happens
            # to broadcast for B=1 but wrongly aligns B with D for B>1.
            relation_mask = edge_valid[:, direction, None, None].to(dtype=relation.dtype)
            weight_mask = edge_valid[:, direction, None].to(dtype=weight.dtype)
            relations.append(relation * relation_mask)
            weights.append(weight * weight_mask)
        return torch.stack(relations, dim=1), torch.stack(weights, dim=1)


class ClassRelationHead(nn.Module):
    """Predict a centered, O(Q) class-relative evidence field."""

    def __init__(self, cost_dim: int, text_dim: int, relation_dim: int) -> None:
        super().__init__()
        self.cost_dim = cost_dim
        self.cost_norm = nn.LayerNorm(cost_dim)
        self.cost_projection = nn.Linear(cost_dim, relation_dim)
        self.text_projection = nn.Linear(text_dim, relation_dim, bias=False)
        self.delta = nn.Sequential(
            nn.Conv2d(2 * relation_dim, relation_dim, 1),
            nn.GELU(),
            nn.Conv2d(relation_dim, cost_dim, 1, bias=False),
        )
        nn.init.normal_(self.delta[-1].weight, std=0.01 / math.sqrt(relation_dim))

    def forward(self, semantic: Tensor, text: Tensor) -> Tensor:
        if semantic.ndim != 5 or text.ndim != 2 or semantic.shape[2] != len(text):
            raise ValueError("Class relation inputs must share the query axis")
        batch, _, queries, height, width = semantic.shape
        cost_features = _cost_layer_norm(semantic, self.cost_norm, self.cost_projection)
        text_features = self.text_projection(F.normalize(text, dim=-1))
        text_features = text_features[None, :, :, None, None].expand(batch, -1, -1, height, width)
        delta = self.delta(torch.cat((cost_features, text_features), dim=2).reshape(batch * queries, -1, height, width))
        delta = delta.reshape(batch, queries, self.cost_dim, height, width).permute(0, 2, 1, 3, 4)
        candidate = semantic + delta
        return candidate - candidate.mean(dim=2, keepdim=True)


class AnchoredCostReconstruction(nn.Module):
    """Four fixed FP32 Richardson steps for the SPD relational reconstruction."""

    def __init__(self, spatial_weight: float = 0.05, class_weight: float = 0.05,
                 iterations: int = 4, step_size: float = 2.0 / 2.85) -> None:
        super().__init__()
        if min(spatial_weight, class_weight, step_size) <= 0 or iterations < 1:
            raise ValueError("Reconstruction weights, step size, and iterations must be positive")
        self.spatial_weight = float(spatial_weight)
        self.class_weight = float(class_weight)
        self.iterations = int(iterations)
        self.step_size = float(step_size)

    @staticmethod
    def center_queries(value: Tensor) -> Tensor:
        return value - value.mean(dim=2, keepdim=True)

    def operator(self, value: Tensor, edge_weight: Tensor) -> Tensor:
        difference = SpatialStencil.difference(value)
        spatial = SpatialStencil.adjoint(edge_weight.unsqueeze(2) * difference)
        return value + self.spatial_weight * spatial + self.class_weight * self.center_queries(value)

    def forward(self, anchor: Tensor, spatial_relation: Tensor, edge_weight: Tensor, class_relation: Tensor) -> Tensor:
        if anchor.ndim != 5 or spatial_relation.ndim != 6 or edge_weight.ndim != 5 or class_relation.shape != anchor.shape:
            raise ValueError("Invalid anchored reconstruction tensor contract")
        if spatial_relation.shape[:2] != (anchor.shape[0], 4) or spatial_relation.shape[2:] != anchor.shape[1:]:
            raise ValueError("Spatial relation must be [B,4,D,Q,H,W]")
        if edge_weight.shape != (anchor.shape[0], 4, anchor.shape[2], anchor.shape[3], anchor.shape[4]):
            raise ValueError("Edge weights must be [B,4,Q,H,W]")
        with torch.autocast(device_type=anchor.device.type, enabled=False):
            anchor32 = anchor.float()
            relation32 = spatial_relation.float()
            weight32 = edge_weight.float()
            class32 = class_relation.float()
            rhs = anchor32 + self.spatial_weight * SpatialStencil.adjoint(weight32.unsqueeze(2) * relation32)
            rhs = rhs + self.class_weight * class32
            value = anchor32
            for _ in range(self.iterations):
                value = value + self.step_size * (rhs - self.operator(value, weight32))
        return value.to(dtype=anchor.dtype)


def patch_class_proportions(target: Tensor, classes: int, height: int, width: int) -> tuple[Tensor, Tensor]:
    """Pool dense source labels into valid soft patch class proportions."""
    if target.ndim != 3 or min(height, width, classes) < 1:
        raise ValueError("Invalid dense target or relation grid")
    batch, image_height, image_width = target.shape
    if image_height % height or image_width % width:
        raise ValueError("Target resolution must divide the cost grid exactly")
    valid = target != 255
    clamped = target.clamp(0, classes - 1)
    one_hot = F.one_hot(clamped, num_classes=classes).permute(0, 3, 1, 2).to(dtype=torch.float32)
    one_hot = one_hot * valid[:, None].to(dtype=one_hot.dtype)
    kernel = (image_height // height, image_width // width)
    proportions = F.avg_pool2d(one_hot, kernel_size=kernel, stride=kernel)
    patch_valid = F.avg_pool2d(valid[:, None].float(), kernel_size=kernel, stride=kernel).squeeze(1) >= 1.0
    return proportions.reshape(batch, classes, height, width), patch_valid


def _mean_or_zero(value: Tensor, mask: Tensor) -> Tensor:
    if not mask.any():
        return value.sum() * 0
    return value.masked_select(mask).mean()


def relation_supervision(spatial_relation: Tensor, class_relation: Tensor, target: Tensor,
                         relation_readout: Tensor, epsilon: float = 0.05,
                         delta: float = 1.0) -> tuple[Tensor, Tensor]:
    """Source-only scalar relation losses on the same g/h consumed by reconstruction."""
    if spatial_relation.ndim != 6 or class_relation.ndim != 5:
        raise ValueError("Invalid relation tensors")
    batch, edges, dim, classes, height, width = spatial_relation.shape
    if edges != 4 or class_relation.shape != (batch, dim, classes, height, width):
        raise ValueError("Spatial/class relation tensor shapes disagree")
    proportions, patch_valid = patch_class_proportions(target, classes, height, width)
    tau = torch.log(proportions.to(device=class_relation.device) + epsilon)
    class_target = tau - tau.mean(dim=1, keepdim=True)
    unit = F.normalize(relation_readout.float(), dim=0)
    class_prediction = torch.einsum("bdqhw,d->bqhw", class_relation.float(), unit)
    class_values = F.huber_loss(class_prediction, class_target, reduction="none", delta=delta)
    class_loss = _mean_or_zero(class_values, patch_valid[:, None].expand_as(class_values))

    target_edge = SpatialStencil.difference(tau[:, None])[:, :, 0]
    predicted_edge = torch.einsum("bkdqhw,d->bkqhw", spatial_relation.float(), unit)
    edge_values = F.huber_loss(predicted_edge, target_edge, reduction="none", delta=delta)
    edge_valid = SpatialStencil.edge_valid(patch_valid)
    groups = []
    for direction in range(4):
        valid = edge_valid[:, direction, None].expand_as(edge_values[:, direction])
        changed = target_edge[:, direction].abs() > 1e-6
        for group in (changed, ~changed):
            selected = valid & group
            if selected.any():
                groups.append(edge_values[:, direction].masked_select(selected).mean())
    spatial_loss = torch.stack(groups).mean() if groups else edge_values.sum() * 0
    return spatial_loss, class_loss
