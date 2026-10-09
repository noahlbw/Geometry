"""Controlled pixel-query evidence addressing for PSDR mechanism tests."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PixelQueryEvidenceConfig:
    address_mode: str = "pixel_query"
    content_mode: str = "member_geometry"
    dim: int = 256
    heads: int = 8
    grids: tuple[int, ...] = (8, 4)
    regions_per_read: int = 4
    samples_per_region: int = 4
    rounds: int = 2
    query_chunk: int = 8
    visual_blocks: int = 2

    def validate(self) -> None:
        if self.address_mode not in {"image_only", "query_only", "pixel_only", "pixel_query"}:
            raise ValueError("Unknown evidence address mode")
        if self.content_mode not in {"region_mean", "member_geometry", "flat_attention"}:
            raise ValueError("Unknown evidence content mode")
        if self.dim < 32 or self.dim % self.heads or self.dim % 8:
            raise ValueError("Reader dim must be >=32 and divisible by heads and eight")
        if not self.grids or any(grid < 2 for grid in self.grids):
            raise ValueError("Evidence grids must contain values >=2")
        if min(self.regions_per_read, self.samples_per_region, self.rounds, self.query_chunk, self.visual_blocks) < 1:
            raise ValueError("Read budgets and recurrence must be positive")


def _coordinates(height: int, width: int, reference: Tensor) -> Tensor:
    ys = torch.linspace(-1, 1, height, dtype=reference.dtype, device=reference.device)
    xs = torch.linspace(-1, 1, width, dtype=reference.dtype, device=reference.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), -1).reshape(-1, 2)


class SharedGridEvidenceBank(nn.Module):
    """Class-agnostic region means plus fixed member-token controls."""

    def __init__(self, grids: tuple[int, ...], member_count: int) -> None:
        super().__init__()
        self.grids, self.member_count = tuple(grids), member_count
        if member_count == 1:
            unit = torch.zeros(1, 2)
        else:
            angle = torch.arange(member_count, dtype=torch.float32) * (2 * math.pi / member_count)
            unit = 0.62 * torch.stack((angle.cos(), angle.sin()), -1)
        self.register_buffer("member_unit", unit, persistent=False)

    def _layout(self, field: Tensor) -> tuple[Tensor, Tensor]:
        centers, scales = [], []
        for grid in self.grids:
            rows = (torch.arange(grid, dtype=field.dtype, device=field.device) + 0.5) * (2 / grid) - 1
            yy, xx = torch.meshgrid(rows, rows, indexing="ij")
            centers.append(torch.stack((xx, yy), -1).reshape(-1, 2))
            scales.append(field.new_full((grid * grid, 2), 1 / grid))
        return torch.cat(centers), torch.cat(scales)

    def forward(self, field: Tensor) -> dict[str, Tensor]:
        batch = len(field)
        means = [F.adaptive_avg_pool2d(field, (grid, grid)).flatten(2).transpose(1, 2) for grid in self.grids]
        mean = torch.cat(means, 1)
        centers, scales = self._layout(field)
        member_xy = centers[:, None] + scales[:, None] * self.member_unit.to(field.dtype)[None]
        members = F.grid_sample(field, member_xy.reshape(1, -1, 1, 2).expand(batch, -1, -1, -1), mode="bilinear", padding_mode="zeros", align_corners=True)
        members = members.squeeze(-1).transpose(1, 2).reshape(batch, len(centers), self.member_count, field.shape[1])
        return {"mean": mean, "members": members, "centers": centers, "scales": scales}


class PixelQueryEvidenceReader(nn.Module):
    """Execute a bounded image/query address and update only recipient costs."""

    def __init__(self, cost_dim: int, config: PixelQueryEvidenceConfig) -> None:
        super().__init__()
        self.config = config
        self.cost_dim, self.dim = cost_dim, config.dim
        self.text_input = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, config.dim), nn.GELU())
        # All M1 arms execute this exact encoder. The ablation changes only
        # which visual/text condition is presented; it does not swap readers.
        self.request_encoder = nn.Sequential(nn.LayerNorm(3 * config.dim), nn.Linear(3 * config.dim, 2 * config.dim), nn.GELU(), nn.Linear(2 * config.dim, config.dim))
        self.region_key = nn.Linear(config.dim, config.dim, bias=False)
        self.member_key = nn.Linear(config.dim, config.dim, bias=False)
        self.value = nn.Linear(config.dim, config.dim, bias=False)
        self.offset = nn.Sequential(nn.LayerNorm(2 * config.dim), nn.Linear(2 * config.dim, config.dim), nn.GELU(), nn.Linear(config.dim, 2 * config.samples_per_region))
        self.geometry = nn.Sequential(nn.Linear(4, config.dim), nn.GELU(), nn.Linear(config.dim, config.dim))
        self.candidate_score = nn.Sequential(nn.LayerNorm(2 * config.dim), nn.Linear(2 * config.dim, config.dim), nn.GELU(), nn.Linear(config.dim, 1))
        self.update = nn.Sequential(nn.LayerNorm(2 * cost_dim + config.dim), nn.Linear(2 * cost_dim + config.dim, 4 * cost_dim), nn.GELU(), nn.Linear(4 * cost_dim, cost_dim))
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def _request(self, field: Tensor, text: Tensor) -> Tensor:
        batch, _, height, width = field.shape
        pixels = field.flatten(2).transpose(1, 2)
        image = pixels.mean(1)[:, None, None].expand(-1, len(text), height * width, -1)
        pixel = pixels[:, None].expand(-1, len(text), -1, -1)
        query = self.text_input(text)[None, :, None].expand(batch, -1, height * width, -1)
        zeros = torch.zeros_like(pixel)
        if self.config.address_mode == "image_only":
            packed = torch.cat((image, zeros, zeros), -1)
        elif self.config.address_mode == "query_only":
            packed = torch.cat((zeros, query, zeros), -1)
        elif self.config.address_mode == "pixel_only":
            packed = torch.cat((pixel, zeros, zeros), -1)
        else:
            packed = torch.cat((pixel, query, pixel * query), -1)
        return self.request_encoder(packed)

    @staticmethod
    def _gather(values: Tensor, indices: Tensor) -> Tensor:
        """Gather B,M,... by B,Q,N,L indices -> B,Q,N,L,..."""
        batch, queries, pixels, count = indices.shape
        tail = values.shape[2:]
        expanded = values[:, None, None].expand(-1, queries, pixels, *values.shape[1:])
        view = indices[(...,) + (None,) * len(tail)].expand(batch, queries, pixels, count, *tail)
        return torch.gather(expanded, 3, view)

    def _sample(self, field: Tensor, coordinates: Tensor) -> Tensor:
        batch, queries, pixels, regions, samples, _ = coordinates.shape
        grid = coordinates.reshape(batch * queries, pixels, regions * samples, 2)
        repeated = field[:, None].expand(-1, queries, -1, -1, -1).reshape(batch * queries, *field.shape[1:])
        values = F.grid_sample(repeated, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return values.reshape(batch, queries, self.dim, pixels, regions, samples).permute(0, 1, 3, 4, 5, 2)

    def _region_candidates(self, request: Tensor, bank: dict[str, Tensor], field: Tensor, receiver_xy: Tensor) -> tuple[Tensor, Tensor]:
        region_key = self.region_key(bank["mean"])
        score = torch.einsum("bqnd,bmd->bqnm", request, region_key) / math.sqrt(self.dim)
        scores, indices = score.topk(self.config.regions_per_read, -1)
        keys, means = self._gather(region_key, indices), self._gather(bank["mean"], indices)
        centers, scales = bank["centers"][indices], bank["scales"][indices]
        packed = torch.cat((request[:, :, :, None].expand_as(keys), keys), -1)
        offsets = self.offset(packed).reshape(*keys.shape[:-1], self.config.samples_per_region, 2).tanh()
        coordinates = centers[..., None, :] + scales[..., None, :] * offsets
        relative = coordinates - receiver_xy[None, None, :, None, None]
        relation = self.geometry(torch.cat((relative, scales[..., None, :].expand_as(relative)), -1))
        if self.config.content_mode == "region_mean":
            mean_key = self.member_key(means)
            mean_score = (request[:, :, :, None] * mean_key).sum(-1) / math.sqrt(self.dim)
            # A fixed-region embedding baseline must not receive member-token
            # coordinates or recipient-relative geometry.
            return self.value(means) * mean_score.sigmoid()[..., None], scores
        if self.config.content_mode == "member_geometry":
            members = self._sample(field, coordinates)
            member_score = (request[:, :, :, None, None] * self.member_key(members)).sum(-1) / math.sqrt(self.dim)
            return (member_score.softmax(-1)[..., None] * (self.value(members) + relation)).sum(-2), scores
        # Same L x R attention budget and the same address generator, but it
        # reads static bank members with ordinary conditional attention. It
        # deliberately receives neither sampled content nor relative geometry.
        members = self._gather(bank["members"], indices)
        member_score = (request[:, :, :, None, None] * self.member_key(members)).sum(-1) / math.sqrt(self.dim)
        return (member_score.softmax(-1)[..., None] * self.value(members)).sum(-2), scores

    def forward(self, state: Tensor, initial: Tensor, field: Tensor, bank: dict[str, Tensor], text: Tensor, *, return_candidates: bool = False) -> dict[str, Tensor]:
        batch, cost_dim, queries, height, width = state.shape
        if cost_dim != self.cost_dim or len(text) != queries:
            raise ValueError("Cost/text contract differs from evidence reader configuration")
        request_all, xy = self._request(field, text), _coordinates(height, width, field)
        state_tokens = state.permute(0, 2, 3, 4, 1).reshape(batch, queries, height * width, cost_dim)
        initial_tokens = initial.permute(0, 2, 3, 4, 1).reshape_as(state_tokens)
        updates, candidate_updates, scores, addresses = [], [], [], []
        for start in range(0, queries, self.config.query_chunk):
            stop = min(queries, start + self.config.query_chunk)
            request = request_all[:, start:stop]
            candidate, address = self._region_candidates(request, bank, field, xy)
            rank = self.candidate_score(torch.cat((request[..., None, :].expand_as(candidate), candidate), -1)).squeeze(-1)
            read = (rank.softmax(-1)[..., None] * candidate).sum(-2)
            local, initial_local = state_tokens[:, start:stop], initial_tokens[:, start:stop]
            updates.append(self.update(torch.cat((local, initial_local, read), -1)))
            if return_candidates:
                count = candidate.shape[-2]
                packed = torch.cat((local[..., None, :].expand(-1, -1, -1, count, -1), initial_local[..., None, :].expand(-1, -1, -1, count, -1), candidate), -1)
                candidate_updates.append(self.update(packed))
            scores.append(rank)
            addresses.append(address)
        update = torch.cat(updates, 1).reshape(batch, queries, height, width, cost_dim).permute(0, 4, 1, 2, 3)
        result = {"cost": state + update, "utility_scores": torch.cat(scores, 1), "address_scores": addresses}
        if return_candidates:
            candidate = torch.cat(candidate_updates, 1)
            result["candidate_costs"] = (state_tokens[..., None, :] + candidate).permute(3, 0, 4, 1, 2).reshape(self.config.regions_per_read, batch, cost_dim, queries, height, width)
        return result
