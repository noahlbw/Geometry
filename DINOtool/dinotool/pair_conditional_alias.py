"""Rival-conditioned alias subsets without globally deleting any phrase."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F

from .geometry_readout_trace import alias_class_scores


IMPLEMENTATION = "geometry-pair-conditional-alias-v1-20261002"
PRIMARY = "PairAlias_Coupled"
METHODS = ("Geometry", "BroadVIP", "MeanProb_VIP", "MeanLogit_VIP", "Anchored_VIP",
           "PairAlias_Local", PRIMARY, "RandomPair_Local", "RandomPair_Coupled",
           "FixedCount_Local", "FixedCount_Coupled")
PAIRS = {"pair_local": ("Geometry", "PairAlias_Local"),
         "pair_coupled": ("Anchored_VIP", PRIMARY),
         "semantic_vs_random_local": ("RandomPair_Local", "PairAlias_Local"),
         "semantic_vs_random_coupled": ("RandomPair_Coupled", PRIMARY),
         "normalization_local": ("FixedCount_Local", "PairAlias_Local"),
         "normalization_coupled": ("FixedCount_Coupled", PRIMARY)}


@dataclass(frozen=True)
class PairAliasConfig:
    temperature: float = 0.07
    canonical_contrast_fraction: float = 0.5
    random_seed: int = 20261002
    rival_source: str = "original local Geometry top2, before selection"


class PairConditionalAliases:
    def __init__(self, bank, config=PairAliasConfig()):
        bank.validate()
        if (bank.class_count < 2 or not 0 <= config.canonical_contrast_fraction <= 1
                or not math.isfinite(config.temperature) or config.temperature <= 0):
            raise ValueError("At least two classes and valid fixed contrast/temperature required.")
        self.bank, self.config = bank, config
        text = F.normalize(bank.features.float(), dim=-1)
        if not bool(torch.isfinite(text).all()):
            raise ValueError("Finite frozen text features required.")
        prototypes = torch.stack([text[(bank.parent_indices == c) & bank.canonical_mask][0]
                                  for c in range(bank.class_count)])
        similarity = text @ prototypes.T
        canonical_similarity = prototypes @ prototypes.T
        self.members = [(bank.parent_indices == c).nonzero().flatten() for c in range(bank.class_count)]
        self.selected, self.random = [], []
        generator = torch.Generator().manual_seed(config.random_seed)
        for c, members in enumerate(self.members):
            contrast = similarity[members, c, None] - similarity[members]
            canonical_gap = canonical_similarity[c, c] - canonical_similarity[c]
            selected = (contrast >= config.canonical_contrast_fraction * canonical_gap[None]).T
            selected[canonical_gap <= 1e-6] = True
            canonical = bank.canonical_mask[members]
            selected[:, canonical] = True
            random = torch.zeros_like(selected)
            alternatives = (~canonical).nonzero().flatten().cpu()
            canonical_index = int(canonical.nonzero().item())
            for d in range(bank.class_count):
                count = int(selected[d].sum())
                random[d, canonical_index] = True
                permutation = alternatives[torch.randperm(len(alternatives), generator=generator)]
                random[d, permutation[:count - 1].to(random.device)] = True
            self.selected.append(selected)
            self.random.append(random)

    def report(self):
        return {"config": asdict(self.config), "criterion":
                "cos(alias,own)-cos(alias,rival) >= 0.5*(1-cos(own,rival)); canonical retained; "
                "indistinguishable canonical directions retain all; no labels or images in mask construction",
                "classes": {name: {rival: {"count": int(self.selected[c][d].sum()),
                    "selected": [self.bank.alias_names[int(i)] for i in self.members[c][self.selected[c][d]]],
                    "random": [self.bank.alias_names[int(i)] for i in self.members[c][self.random[c][d]]]}
                    for d, rival in enumerate(self.bank.class_names) if d != c}
                    for c, name in enumerate(self.bank.class_names)}}

    def _table(self, aliases, masks, original, fixed_count=False):
        rows = []
        for c, members in enumerate(self.members):
            values = aliases[..., members] / self.config.temperature
            mask = masks[c]
            count = mask.sum(-1)
            log_count = (math.log(len(members)) if fixed_count else count.to(values.dtype).log())
            expanded = values[..., None, :].masked_fill(~mask, -torch.inf)
            scores = self.config.temperature * (torch.logsumexp(expanded, -1) - log_count)
            # Preserve the exact original arithmetic on unchanged subsets.
            scores = torch.where(mask.all(-1), original[..., c, None], scores)
            rows.append(scores)
        return torch.stack(rows, -2)

    @torch.inference_mode()
    def read(self, aliases, valid=None):
        if aliases.shape[-1] != len(self.bank.alias_names) or not bool(torch.isfinite(aliases).all()):
            raise ValueError("Matching finite alias responses required.")
        original = alias_class_scores(aliases, self.bank.parent_indices, self.bank.class_count,
                                      self.config.temperature)
        if valid is None:
            valid = torch.ones(aliases.shape[:-1], device=aliases.device, dtype=torch.bool)
        if valid.shape != aliases.shape[:-1]:
            raise ValueError("Validity must match alias positions.")
        rivals = original.topk(2, dim=-1).indices
        flat_pairs = rivals.reshape(-1, 2)
        positions = torch.arange(len(flat_pairs), device=aliases.device)
        outputs = {}
        for name, masks, fixed in (("selected", self.selected, False),
                                   ("random", self.random, False),
                                   ("fixed_count", self.selected, True)):
            table = self._table(aliases, masks, original, fixed).reshape(-1, self.bank.class_count,
                                                                       self.bank.class_count)
            left = table[positions, flat_pairs[:, 0], flat_pairs[:, 1]]
            right = table[positions, flat_pairs[:, 1], flat_pairs[:, 0]]
            changed = original.clone().reshape(-1, self.bank.class_count)
            changed.scatter_(1, flat_pairs, torch.stack((left, right), -1))
            outputs[name] = torch.where(valid[..., None], changed.reshape_as(original), original)
        counts = torch.stack([mask.sum(-1) for mask in self.selected])
        count_left = counts[flat_pairs[:, 0], flat_pairs[:, 1]]
        count_right = counts[flat_pairs[:, 1], flat_pairs[:, 0]]
        kept = torch.stack((count_left, count_right), -1).reshape(*valid.shape, 2)
        mass, full = torch.zeros_like(valid, dtype=aliases.dtype), torch.zeros_like(valid, dtype=torch.bool)
        for c, members in enumerate(self.members):
            for side in (0, 1):
                active = (rivals[..., side] == c) & valid
                other = rivals[..., 1 - side]
                removed = ~self.selected[c][other]
                responsibilities = (aliases[..., members] / self.config.temperature).softmax(-1)
                mass += .5 * (responsibilities * removed).sum(-1) * active
                full |= active & (kept[..., side] < len(members))
        diagnostics = {"mean_alias_count_in_competing_classes": float(kept[valid].float().mean()) if bool(valid.any()) else 0.,
                       "pair_selection_active_fraction": float(full[valid].float().mean()) if bool(valid.any()) else 0.,
                       "removed_responsibility_mass": float(mass[valid].mean()) if bool(valid.any()) else 0.,
                       "mean_absolute_local_score_change": float((outputs["selected"] - original)[valid].abs().mean())
                                                                if bool(valid.any()) else 0.,
                       "local_changed_patch_fraction": float((outputs["selected"].argmax(-1) != original.argmax(-1))[valid].float().mean())
                                                        if bool(valid.any()) else 0.}
        return outputs, diagnostics
