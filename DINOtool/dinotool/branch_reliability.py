"""Image-only feasibility test for choosing broad VIP versus local Geometry.

This is a branch-selection audit, not a newly validated segmentation model.
No masks, class-area priors, fitted calibrators or target-specific parameters enter
the selector. Both candidate phrase families are excluded in both text banks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .competitive_ownership import alias_families
from .tcpr import TCPRTextBank


@dataclass(frozen=True)
class BranchReliabilityConfig:
    alias_temperature: float = 0.07
    family_cosine: float = 0.97
    geometry_temperature: float = 0.10
    neighbor_radius: float = 4.0
    neighbors: int = 32
    chunk_size: int = 128

    def validate(self) -> None:
        if (min(self.alias_temperature, self.geometry_temperature, self.neighbor_radius) <= 0
                or not 0 <= self.family_cosine <= 1
                or min(self.neighbors, self.chunk_size) < 1):
            raise ValueError("Invalid branch reliability configuration.")


def pair_margin(scores: Tensor, own: Tensor, rival: Tensor, temperature: float) -> Tensor:
    """Normalized LME own-minus-rival; masks may broadcast over neighbor slots."""
    own_count = own.sum(-1).clamp_min(1)
    rival_count = rival.sum(-1).clamp_min(1)
    scaled = scores.float() / temperature
    first = torch.logsumexp(scaled.masked_fill(~own, -torch.inf), -1) - own_count.log()
    second = torch.logsumexp(scaled.masked_fill(~rival, -torch.inf), -1) - rival_count.log()
    # Empty references cannot authorize a correction. Their identifiability is
    # checked by the caller; keep intermediate arithmetic finite as well.
    return torch.nan_to_num(temperature * (first - second), nan=0.0, posinf=0.0, neginf=0.0)


class BranchReliabilitySelector:
    def __init__(self, bank: TCPRTextBank,
                 config: BranchReliabilityConfig = BranchReliabilityConfig()):
        config.validate()
        bank.validate()
        self.config = config
        self.parents = bank.parent_indices
        self.family = torch.empty_like(self.parents)
        self.groups = alias_families(bank, config.family_cosine)
        for index, members in enumerate(self.groups):
            self.family[members] = index

    def config_dict(self) -> dict:
        return {**asdict(self.config), "family_count": len(self.groups),
                "decision": "VIP LFO margin > 0 and VIP+Geometry signed LFO margins > 0",
                "grounding": "the same signed reference also positive over shared raw-DINO neighbors",
                "unknown": "retain Multiscale; no confidence-to-background threshold"}

    def _neighbors(self, raw: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        height, width, channels = raw.shape
        features = F.normalize(raw.reshape(-1, channels).float(), dim=-1)
        y, x = torch.meshgrid(torch.arange(height, device=raw.device),
                              torch.arange(width, device=raw.device), indexing="ij")
        positions = torch.stack((y.flatten(), x.flatten()), -1).float()
        distance = (positions[:, None] - positions[None]).square().sum(-1)
        allowed = (distance <= self.config.neighbor_radius ** 2) & (distance > 0)
        allowed &= valid.flatten()[:, None] & valid.flatten()[None]
        scores = (features @ features.T) / self.config.geometry_temperature
        scores -= distance / (2 * self.config.neighbor_radius ** 2)
        scores = scores.masked_fill(~allowed, -1e9)
        k = min(self.config.neighbors, features.shape[0])
        selected, indices = scores.topk(k, -1)
        usable = allowed.gather(1, indices)
        weights = selected.softmax(-1) * usable.float()
        weights /= weights.sum(-1, keepdim=True).clamp_min(1e-12)
        return indices, weights

    @torch.inference_mode()
    def select(self, geometry_alias: Tensor, vip_alias: Tensor, raw: Tensor,
               geometry_labels: Tensor, vip_labels: Tensor,
               valid: Tensor) -> tuple[dict[str, Tensor], dict[str, int]]:
        h, w, aliases = geometry_alias.shape
        if (vip_alias.shape != geometry_alias.shape or raw.shape[:2] != (h, w)
                or geometry_labels.shape != (h, w) or vip_labels.shape != (h, w)
                or valid.shape != (h, w) or aliases != self.parents.numel()):
            raise ValueError("Selector observations must share a patch grid and alias bank.")
        g = geometry_alias.reshape(-1, aliases).float()
        v = vip_alias.reshape(-1, aliases).float()
        gl, vl = geometry_labels.flatten(), vip_labels.flatten()
        gmembers = self.parents[None] == gl[:, None]
        vmembers = self.parents[None] == vl[:, None]
        ga = g.masked_fill(~gmembers, -torch.inf).argmax(-1)
        va = v.masked_fill(~vmembers, -torch.inf).argmax(-1)
        excluded = ((self.family[None] == self.family[ga, None])
                    | (self.family[None] == self.family[va, None]))
        gmask, vmask = gmembers & ~excluded, vmembers & ~excluded
        identifiable = (gmask.sum(-1) > 0) & (vmask.sum(-1) > 0)
        eligible = valid.flatten() & (gl != vl) & identifiable
        # Both margins use v-minus-g direction. A positive sum means the VIP
        # surviving reference exceeds the opposing Geometry reference margin.
        gm = pair_margin(g, vmask, gmask, self.config.alias_temperature)
        vm = pair_margin(v, vmask, gmask, self.config.alias_temperature)
        local = gm + vm
        leave_out = eligible & (vm > 0) & (local > 0)
        indices, weights = self._neighbors(raw, valid)
        support = torch.zeros_like(local)
        for start in range(0, g.shape[0], self.config.chunk_size):
            end = min(start + self.config.chunk_size, g.shape[0])
            keys = indices[start:end]
            own = vmask[start:end, None]
            rival = gmask[start:end, None]
            neighbor_g = pair_margin(g[keys], own, rival, self.config.alias_temperature)
            neighbor_v = pair_margin(v[keys], own, rival, self.config.alias_temperature)
            support[start:end] = ((neighbor_g + neighbor_v) * weights[start:end]).sum(-1)
        supported = weights.sum(-1) > 0
        grounded = leave_out & supported & (support > 0)
        decisions = {"LeaveFamilyOut": leave_out.reshape(h, w),
                     "GroundedLeaveFamilyOut": grounded.reshape(h, w)}
        diagnostics = {"valid_patches": int(valid.sum()),
                       "disagreement_patches": int(((gl != vl) & valid.flatten()).sum()),
                       "identifiable_disagreements": int(eligible.sum()),
                       "lfo_proposed": int(leave_out.sum()),
                       "grounded_proposed": int(grounded.sum()),
                       "neighbor_supported_disagreements": int((eligible & supported).sum())}
        return decisions, diagnostics
