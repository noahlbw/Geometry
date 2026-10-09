"""Parallel evidence modules for the CAFe-DINO cost decoder.

The modules in this file do not own a DINO encoder or an upsampler. They
operate on CAFe's multi-channel cost tensor ``[B, D, Q, H, W]`` and keep the
three evidence states synchronous within every recurrent stage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    return math.gcd(channels, 32)


@dataclass(frozen=True)
class ParallelEvidenceConfig:
    """Capacity and recurrence settings for the PED decoder."""

    arm: str = "paired_support"
    evidence_dim: int = 512
    memory_tokens: int = 128
    memory_heads: int = 8
    read_heads: int = 8
    interaction_dim: int = 128
    message_hidden: int = 1024
    visual_blocks: int = 2
    local_kernel: int = 3
    stages: int = 6
    pyramid_grids: tuple[int, ...] = (8, 4, 2)
    support_points: int = 8
    surround_points: int = 8

    def validate(self) -> None:
        if self.arm not in {"plain", "naive_parallel", "serial_content", "ped", "anchored_multiscale", "paired_support"}:
            raise ValueError("Unknown PED study arm")
        if self.evidence_dim < 64 or self.evidence_dim % self.memory_heads or self.evidence_dim % self.read_heads:
            raise ValueError("evidence_dim must be >=64 and divisible by both head counts")
        if self.memory_tokens < 8 or self.interaction_dim < 8 or self.message_hidden < 64:
            raise ValueError("memory and message capacity are too small")
        if self.visual_blocks < 1 or self.local_kernel < 3 or self.local_kernel % 2 == 0:
            raise ValueError("visual_blocks and an odd local kernel are required")
        if self.stages < 1:
            raise ValueError("at least one recurrent stage is required")
        if not self.pyramid_grids or any(grid < 2 for grid in self.pyramid_grids):
            raise ValueError("pyramid grids must contain sizes >=2")
        if min(self.support_points, self.surround_points) < 4:
            raise ValueError("support and surround need at least four samples")

    def architecture(self) -> dict:
        return asdict(self)


class VisualRefinementBlock(nn.Module):
    """High-capacity local visual field update shared across query classes."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(dim), dim)
        self.depthwise = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.expand = nn.Conv2d(dim, 4 * dim, 1)
        self.project = nn.Conv2d(4 * dim, dim, 1)
        self.scale = nn.Parameter(torch.full((dim,), 1e-4))

    def forward(self, field: Tensor) -> Tensor:
        update = self.depthwise(self.norm(field))
        update = self.project(F.gelu(self.expand(update)))
        return field + update * self.scale[None, :, None, None]


class VisualEvidenceField(nn.Module):
    """Separately projects DINO X/Y tokens into an image-shared evidence field."""

    def __init__(self, dim: int, blocks: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2048, dim, 1),
            nn.GroupNorm(_groups(dim), dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(VisualRefinementBlock(dim) for _ in range(blocks))
        self.out_norm = nn.GroupNorm(_groups(dim), dim)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        field = self.stem(torch.cat((F.normalize(x, dim=1), F.normalize(y, dim=1)), dim=1))
        for block in self.blocks:
            field = block(field)
        return self.out_norm(field)


class CompressedVisualMemory(nn.Module):
    """Latent visual memory built once per image and reused by all PED rounds."""

    def __init__(self, dim: int, tokens: int, heads: int) -> None:
        super().__init__()
        self.dim, self.tokens, self.heads = dim, tokens, heads
        self.latents = nn.Parameter(torch.empty(tokens, dim))
        self.memory_coords = nn.Parameter(torch.empty(tokens, 2))
        self.coord = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.latents, std=0.02)
        nn.init.uniform_(self.memory_coords, -1, 1)

    @staticmethod
    def _coordinates(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        rows = torch.linspace(-1, 1, height, device=device, dtype=dtype)
        cols = torch.linspace(-1, 1, width, device=device, dtype=dtype)
        return torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1).reshape(-1, 2)

    def forward(self, field: Tensor) -> Tensor:
        batch, dim, height, width = field.shape
        values = field.flatten(2).transpose(1, 2)
        coords = self._coordinates(height, width, field.device, field.dtype)
        keys_and_values = values + self.coord(coords)[None]
        queries = self.latents[None].expand(batch, -1, -1) + self.coord(self.memory_coords)[None]
        head_dim = dim // self.heads
        q = self.query(queries).reshape(batch, self.tokens, self.heads, head_dim).transpose(1, 2)
        k = self.key(keys_and_values).reshape(batch, height * width, self.heads, head_dim).transpose(1, 2)
        v = self.value(keys_and_values).reshape(batch, height * width, self.heads, head_dim).transpose(1, 2)
        read = F.scaled_dot_product_attention(q, k, v)
        read = read.transpose(1, 2).reshape(batch, self.tokens, dim)
        return self.norm(queries + self.out(read))


class QueryVisualRead(nn.Module):
    """Query-conditioned local and compressed-memory visual value reading."""

    def __init__(self, cost_dim: int, text_dim: int, config: ParallelEvidenceConfig) -> None:
        super().__init__()
        self.cost_dim = cost_dim
        self.dim = config.evidence_dim
        self.heads = config.read_heads
        self.kernel = config.local_kernel
        self.state_query = nn.Linear(cost_dim, self.dim)
        self.initial_query = nn.Linear(cost_dim, self.dim, bias=False)
        self.text_query = nn.Linear(text_dim, self.dim, bias=False)
        self.key = nn.Linear(self.dim, self.dim, bias=False)
        self.value = nn.Linear(self.dim, self.dim, bias=False)
        self.local_bias = nn.Parameter(torch.zeros(self.heads, self.kernel**2))
        self.memory_bias = nn.Parameter(torch.zeros(self.heads, 1))
        self.update = nn.Sequential(
            nn.LayerNorm(cost_dim * 2 + self.dim),
            nn.Linear(cost_dim * 2 + self.dim, 4 * cost_dim),
            nn.GELU(),
            nn.Linear(4 * cost_dim, cost_dim),
        )

    def _local_bank(self, field: Tensor) -> tuple[Tensor, Tensor]:
        batch, dim, height, width = field.shape
        count = height * width
        values = F.unfold(field, self.kernel, padding=self.kernel // 2)
        values = values.reshape(batch, dim, self.kernel**2, count).permute(0, 3, 2, 1)
        valid = F.unfold(field.new_ones((batch, 1, height, width)), self.kernel, padding=self.kernel // 2)
        valid = valid.reshape(batch, self.kernel**2, count).permute(0, 2, 1).bool()
        return values, valid

    def forward(self, state: Tensor, field: Tensor, memory: Tensor, text: Tensor, initial: Tensor) -> Tensor:
        batch, cost_dim, classes, height, width = state.shape
        count = height * width
        state_tokens = state.permute(0, 2, 3, 4, 1).reshape(batch, classes, count, cost_dim)
        initial_tokens = initial.permute(0, 2, 3, 4, 1).reshape(batch, classes, count, cost_dim)
        query = self.state_query(state_tokens) + self.initial_query(initial_tokens)
        query = query + self.text_query(text)[None, :, None]
        head_dim = self.dim // self.heads
        query = query.reshape(batch, classes, count, self.heads, head_dim)
        local, valid = self._local_bank(field)
        local_key = self.key(local).reshape(batch, count, self.kernel**2, self.heads, head_dim)
        local_value = self.value(local).reshape(batch, count, self.kernel**2, self.heads, head_dim)
        memory_key = self.key(memory).reshape(batch, memory.shape[1], self.heads, head_dim)
        memory_value = self.value(memory).reshape(batch, memory.shape[1], self.heads, head_dim)
        local_score = torch.einsum("bcnhd,bnkhd->bcnhk", query, local_key)
        local_score = local_score * (head_dim ** -0.5) + self.local_bias[None, None, None]
        local_score = local_score.masked_fill(~valid[:, None, :, None], torch.finfo(local_score.dtype).min)
        memory_score = torch.einsum("bcnhd,bmhd->bcnhm", query, memory_key)
        memory_score = memory_score * (head_dim ** -0.5) + self.memory_bias[None, None, None]
        weights = torch.softmax(torch.cat((local_score, memory_score), dim=-1), dim=-1)
        local_weight, memory_weight = weights[..., : self.kernel**2], weights[..., self.kernel**2 :]
        local_read = torch.einsum("bcnhk,bnkhd->bcnhd", local_weight, local_value)
        memory_read = torch.einsum("bcnhm,bmhd->bcnhd", memory_weight, memory_value)
        read = (local_read + memory_read).flatten(3)
        update = self.update(torch.cat((state_tokens, initial_tokens, read), dim=-1))
        return state + update.reshape(batch, classes, height, width, cost_dim).permute(0, 4, 1, 2, 3)


class SpatiallyAnchoredPyramid(nn.Module):
    """A deterministic visual pyramid whose coarse tokens have known support."""

    def __init__(self, grids: tuple[int, ...]) -> None:
        super().__init__()
        self.grids = tuple(grids)

    def forward(self, field: Tensor) -> tuple[Tensor, ...]:
        levels = [field]
        for grid in self.grids:
            levels.append(F.adaptive_avg_pool2d(field, (grid, grid)))
        return tuple(levels)


class QueryConditionedEvidenceRead(nn.Module):
    """Read spatially anchored support and surround evidence for each query.

    Both study modes share every parameter and every sampling position.  The
    ``paired`` switch only changes whether support/surround evidence is
    normalized separately or merged into conventional deformable attention.
    """

    def __init__(self, cost_dim: int, text_dim: int, config: ParallelEvidenceConfig) -> None:
        super().__init__()
        self.cost_dim = cost_dim
        self.dim = config.evidence_dim
        self.heads = config.read_heads
        self.support_points = config.support_points
        self.surround_points = config.surround_points
        self.state_query = nn.Linear(cost_dim, self.dim)
        self.relation_query = nn.Linear(cost_dim, self.dim, bias=False)
        self.text_query = nn.Linear(text_dim, self.dim, bias=False)
        self.geometry = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, 6),
        )
        self.key = nn.Linear(self.dim, self.dim, bias=False)
        self.value = nn.Linear(self.dim, self.dim, bias=False)
        feature_dim = cost_dim + 6 * self.dim + 6
        self.state_update = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, config.message_hidden),
            nn.GELU(),
            nn.Linear(config.message_hidden, cost_dim),
        )
        self.register_buffer("support_unit", self._disk_points(self.support_points), persistent=False)
        self.register_buffer("surround_unit", self._ring_points(self.surround_points), persistent=False)

    @staticmethod
    def _disk_points(count: int) -> Tensor:
        angles = torch.arange(max(count - 1, 1), dtype=torch.float32) * (2 * math.pi / max(count - 1, 1))
        ring = torch.stack((angles.cos(), angles.sin()), dim=-1) * 0.72
        return torch.cat((torch.zeros(1, 2), ring[: count - 1]), dim=0)

    @staticmethod
    def _ring_points(count: int) -> Tensor:
        angles = torch.arange(count, dtype=torch.float32) * (2 * math.pi / count)
        return torch.stack((angles.cos(), angles.sin()), dim=-1) * 1.6

    @staticmethod
    def _grid_coordinates(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        ys = torch.linspace(-1, 1, height, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(height * width, 2)

    @staticmethod
    def _tokens(cost: Tensor) -> Tensor:
        batch, dim, classes, height, width = cost.shape
        return cost.permute(0, 2, 3, 4, 1).reshape(batch, classes, height * width, dim)

    def _coordinates(self, query: Tensor, height: int, width: int) -> tuple[Tensor, Tensor, Tensor]:
        batch, classes, count, _ = query.shape
        raw = self.geometry(query)
        base = self._grid_coordinates(height, width, query.device, query.dtype)[None, None]
        center = base + 0.30 * raw[..., :2].tanh()
        scale = 0.04 + 0.34 * raw[..., 2:4].sigmoid()
        direction = F.normalize(raw[..., 4:6], dim=-1, eps=1e-6)
        cosine, sine = direction.unbind(-1)

        def transform(unit: Tensor) -> Tensor:
            unit = unit.to(device=query.device, dtype=query.dtype)
            dx, dy = unit[:, 0], unit[:, 1]
            rotated = torch.stack((
                cosine[..., None] * dx - sine[..., None] * dy,
                sine[..., None] * dx + cosine[..., None] * dy,
            ), dim=-1)
            return center[..., None, :] + rotated * scale[..., None, :]

        geometry = torch.cat((center - base, scale, direction), dim=-1)
        return transform(self.support_unit), transform(self.surround_unit), geometry

    def _sample(self, pyramid: tuple[Tensor, ...], coordinates: Tensor) -> tuple[Tensor, Tensor]:
        batch, classes, count, points, _ = coordinates.shape
        reads, validities = [], []
        grid = coordinates.reshape(batch * classes, count, points, 2)
        for level in pyramid:
            _, dim, _, _ = level.shape
            repeated = level[:, None].expand(-1, classes, -1, -1, -1).reshape(batch * classes, dim, *level.shape[-2:])
            read = F.grid_sample(repeated, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            read = read.reshape(batch, classes, dim, count, points).permute(0, 1, 3, 4, 2)
            reads.append(read)
            validities.append((coordinates.abs() <= 1).all(dim=-1))
        return torch.cat(reads, dim=3), torch.cat(validities, dim=3)

    @staticmethod
    def _attend(scores: Tensor, values: Tensor, valid: Tensor) -> Tensor:
        masked = scores.masked_fill(~valid[:, :, :, None], torch.finfo(scores.dtype).min)
        weights = torch.softmax(masked, dim=-1)
        has_valid = valid.any(dim=-1, keepdim=True)[:, :, :, None]
        weights = torch.where(has_valid, weights, torch.zeros_like(weights))
        return torch.einsum("bcnhk,bcnkhd->bcnhd", weights, values).flatten(3)

    def forward(self, cost: Tensor, relation: Tensor, pyramid: tuple[Tensor, ...], text: Tensor, *, paired: bool) -> Tensor:
        batch, _, classes, height, width = cost.shape
        count = height * width
        state = self._tokens(cost)
        relation_state = self._tokens(relation)
        query = self.state_query(state) + self.relation_query(relation_state) + self.text_query(text)[None, :, None]
        support_coordinates, surround_coordinates, geometry = self._coordinates(query, height, width)
        support, support_valid = self._sample(pyramid, support_coordinates)
        surround, surround_valid = self._sample(pyramid, surround_coordinates)
        head_dim = self.dim // self.heads
        q = query.reshape(batch, classes, count, self.heads, head_dim)

        def score_and_value(values: Tensor) -> tuple[Tensor, Tensor]:
            keys = self.key(values).reshape(batch, classes, count, values.shape[3], self.heads, head_dim)
            projected = self.value(values).reshape(batch, classes, count, values.shape[3], self.heads, head_dim)
            scores = torch.einsum("bcnhd,bcnkhd->bcnhk", q, keys) * (head_dim ** -0.5)
            return scores, projected

        support_scores, support_values = score_and_value(support)
        surround_scores, surround_values = score_and_value(surround)
        if paired:
            support_read = self._attend(support_scores, support_values, support_valid)
            surround_read = self._attend(surround_scores, surround_values, surround_valid)
        else:
            merged_scores = torch.cat((support_scores, surround_scores), dim=-1)
            merged_values = torch.cat((support_values, surround_values), dim=3)
            merged_valid = torch.cat((support_valid, surround_valid), dim=-1)
            merged_read = self._attend(merged_scores, merged_values, merged_valid)
            support_read = surround_read = merged_read
        center = pyramid[0].flatten(2).transpose(1, 2)[:, None].expand(-1, classes, -1, -1)
        features = torch.cat((state, center, support_read, surround_read, support_read - center,
                              surround_read - center, query, geometry), dim=-1)
        update = self.state_update(features)
        return update.reshape(batch, classes, height, width, self.cost_dim).permute(0, 4, 1, 2, 3)


class ZeroCostFeedback(nn.Module):
    """A learned state-to-cost projection initialized as the exact zero map."""

    def __init__(self, cost_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(cost_dim)
        self.project = nn.Linear(cost_dim, cost_dim)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, state: Tensor) -> Tensor:
        batch, dim, classes, height, width = state.shape
        tokens = state.permute(0, 2, 3, 4, 1).reshape(batch, classes, height * width, dim)
        values = self.project(self.norm(tokens))
        return values.reshape(batch, classes, height, width, dim).permute(0, 4, 1, 2, 3)


class SynchronousEvidenceExchange(nn.Module):
    """Three-way vector exchange evaluated only from the old-round branch outputs."""

    def __init__(self, cost_dim: int, config: ParallelEvidenceConfig) -> None:
        super().__init__()
        rank = config.interaction_dim
        self.cost_dim = cost_dim
        self.as_left, self.as_right = nn.Linear(cost_dim, rank, bias=False), nn.Linear(cost_dim, rank, bias=False)
        self.ar_left, self.ar_right = nn.Linear(cost_dim, rank, bias=False), nn.Linear(cost_dim, rank, bias=False)
        self.sr_left, self.sr_right = nn.Linear(cost_dim, rank, bias=False), nn.Linear(cost_dim, rank, bias=False)
        input_dim = 4 * cost_dim + 3 * rank
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.message_hidden),
            nn.GELU(),
            nn.Linear(config.message_hidden, 3 * cost_dim),
        )
        self.gamma = 1 / math.sqrt(config.stages)
        nn.init.normal_(self.network[-1].weight, std=1e-4)
        nn.init.zeros_(self.network[-1].bias)

    @staticmethod
    def _tokens(cost: Tensor) -> Tensor:
        batch, dim, classes, height, width = cost.shape
        return cost.permute(0, 2, 3, 4, 1).reshape(batch, classes, height * width, dim)

    def forward(self, a: Tensor, s: Tensor, r: Tensor, initial: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        a_t, s_t, r_t, e_t = (F.layer_norm(self._tokens(value), (self.cost_dim,)) for value in (a, s, r, initial))
        pairs = (
            self.as_left(a_t) * self.as_right(s_t),
            self.ar_left(a_t) * self.ar_right(r_t),
            self.sr_left(s_t) * self.sr_right(r_t),
        )
        messages = self.network(torch.cat((a_t, s_t, r_t, e_t, *pairs), dim=-1))
        batch, classes, count, _ = messages.shape
        height, width = a.shape[-2:]
        messages = messages.reshape(batch, classes, count, 3, self.cost_dim).permute(3, 0, 4, 1, 2)
        messages = messages.reshape(3, batch, self.cost_dim, classes, height, width)
        return tuple(branch + self.gamma * message for branch, message in zip((a, s, r), messages))


class ParallelEvidenceStage(nn.Module):
    """One PED round: independent branch updates followed by synchronous exchange."""

    def __init__(self, spatial: nn.Module, channel: nn.Module, reader: QueryVisualRead, exchange: SynchronousEvidenceExchange) -> None:
        super().__init__()
        self.spatial = spatial
        self.channel = channel
        self.reader = reader
        self.exchange = exchange

    def forward(self, a: Tensor, s: Tensor, r: Tensor, initial: Tensor, text_guidance: Tensor,
                visual_guidance: Tensor, field: Tensor, memory: Tensor, text: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Each branch consumes an old-round state. Do not make this serial.
        a_hat = self.channel(a, text_guidance)
        s_hat = self.spatial(s, visual_guidance)
        r_hat = self.reader(r, field, memory, text, initial)
        return self.exchange(a_hat, s_hat, r_hat, initial)


class EvidenceReadout(nn.Module):
    """Maps persistent branch deltas back to CAFe's original cost width."""

    def __init__(self, cost_dim: int, hidden: int) -> None:
        super().__init__()
        self.readout = nn.Sequential(
            nn.LayerNorm(3 * cost_dim),
            nn.Linear(3 * cost_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, cost_dim),
        )
        nn.init.normal_(self.readout[-1].weight, std=1e-4)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, initial: Tensor, a: Tensor, s: Tensor, r: Tensor) -> Tensor:
        batch, dim, classes, height, width = initial.shape
        deltas = torch.cat((a - initial, s - initial, r - initial), dim=1)
        tokens = deltas.permute(0, 2, 3, 4, 1).reshape(batch, classes, height * width, 3 * dim)
        update = self.readout(tokens).reshape(batch, classes, height, width, dim).permute(0, 4, 1, 2, 3)
        return initial + update
