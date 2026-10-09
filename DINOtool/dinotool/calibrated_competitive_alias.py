"""Calibrated alias mixtures with leave-group-out competitive references."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F

from .soft_competitive_alias import apply_pair_margin
from .stratified_soft_alias import (ImageReferences, StratifiedAliasConfig,
    StratifiedSoftAliases, crop_stencil, reference_cells, weighted_wide_pairs)


IMPLEMENTATION = "geometry-calibrated-group-action-alias-v3-20261002"
PRIMARY = "GroupAction_Coupled"
SHUFFLED = tuple("ShuffledGroupAction" + str(i) + "_Coupled" for i in range(3))
METHODS = ("Geometry", "BroadVIP", "Anchored_VIP", "StratifiedAlias_Coupled",
           "CalibratedReplay_Coupled", PRIMARY, "TextPrior_Coupled", *SHUFFLED)
REPLAY = METHODS[:4]
PAIRS = {"primary_vs_all20": ("Anchored_VIP", PRIMARY),
         "primary_vs_v2": ("StratifiedAlias_Coupled", PRIMARY),
         "writer_vs_v2": ("StratifiedAlias_Coupled", "CalibratedReplay_Coupled"),
         "primary_vs_writer": ("CalibratedReplay_Coupled", PRIMARY),
         "primary_vs_text": ("TextPrior_Coupled", PRIMARY),
         **{"primary_vs_shuffle" + str(i): (method, PRIMARY) for i, method in enumerate(SHUFFLED)}}


@dataclass(frozen=True)
class CalibratedAliasConfig:
    text_holdout_count: int = 3
    action_retention: float = 0.5

    def validate(self, aliases):
        if not 1 <= self.text_holdout_count < aliases or not 0 < self.action_retention < 1:
            raise ValueError("Nonempty partial holdout and partial attenuation required.")


def text_holdouts(features, count=3):
    """Remove the tested alias and its closest text neighbors across classes."""
    if not 1 <= count < len(features):
        raise ValueError("Invalid text holdout size.")
    values = F.normalize(features.float(), dim=-1)
    similarity = values @ values.T
    similarity.fill_diagonal_(torch.inf)
    return torch.zeros_like(similarity, dtype=torch.bool).scatter_(1, similarity.topk(count, -1).indices, True)


def profiled_logits(crop, members):
    salience = crop.salience[members].softmax(-1)
    return crop.alias_logits[:, members] * (members.shape[-1] * salience)


@torch.inference_mode()
def calibrated_wide_pairs(crops, count, coordinates, image_size, pairs, weights, members, tau=1.):
    if weights.shape != (*pairs.shape, members.shape[-1]) or not tau > 0:
        raise ValueError("Matching pair weights and positive temperature required.")
    if not bool((weights > 0).all()) or not torch.allclose(weights.sum(-1), torch.ones_like(pairs, dtype=weights.dtype), atol=1e-6):
        raise ValueError("Positive normalized alias weights required.")
    output = coordinates.new_zeros(pairs.shape)
    for crop in crops:
        indices, coefficients = crop_stencil(crop, count, coordinates, image_size)
        evidence = profiled_logits(crop, members)[indices[:, :, None], pairs[:, None]]
        scores = torch.logsumexp(tau * evidence + (members.shape[-1] * weights[:, None]).log(), -1) / tau
        output += (scores * coefficients[..., None]).sum(1)
    return output


@torch.inference_mode()
def action_and_group_scores(crops, count, coordinates, image_size, members, holdouts, retention=.5, tau=1., chunk=128):
    """Exact single-alias attenuation marginals and group-excluded class witnesses."""
    aliases, classes, k = holdouts.shape[0], members.shape[0], members.shape[1]
    excluded = holdouts[:, members]
    remaining = k - excluded.sum(-1)
    if bool((remaining <= 0).any()):
        raise ValueError("Holdout removes an entire class.")
    crop_fields = []
    for crop in crops:
        evidence = profiled_logits(crop, members)
        total = (tau * evidence).logsumexp(-1)
        responsibility = torch.exp(tau * evidence - total[..., None])
        delta = -(torch.log1p(-(1 - retention) * responsibility)
                  + math.log(k / (k - 1 + retention))) / tau
        marginal = torch.empty((len(evidence), aliases), device=evidence.device)
        marginal[:, members.flatten()] = delta.flatten(-2)
        witness_salience = crop.salience[members][None].expand(aliases, -1, -1).masked_fill(excluded, -torch.inf).softmax(-1)
        witness_evidence = crop.alias_logits[:, members][:, None] * (remaining[None, ..., None] * witness_salience[None])
        group = (tau * witness_evidence).masked_fill(excluded[None], -torch.inf).logsumexp(-1) / tau
        crop_fields.append((crop, marginal, group))
    all_marginals, all_groups = [], []
    for start in range(0, len(coordinates), chunk):
        coords = coordinates[start:start + chunk]
        marginal = coords.new_zeros((len(coords), aliases))
        group = coords.new_zeros((len(coords), aliases, classes))
        for crop, values, witnesses in crop_fields:
            indices, coefficients = crop_stencil(crop, count, coords, image_size)
            marginal += (values[indices] * coefficients[..., None]).sum(1)
            group += (witnesses[indices] * coefficients[..., None, None]).sum(1)
        all_marginals.append(marginal)
        all_groups.append(group)
    return torch.cat(all_marginals), torch.cat(all_groups)


@torch.inference_mode()
def group_references(features, coordinates, local_aliases, broad_group, marginal, members, holdouts,
                     image_size, config=StratifiedAliasConfig()):
    aliases, classes, k = len(holdouts), len(members), members.shape[-1]
    excluded = holdouts[:, members]
    remaining = k - excluded.sum(-1)
    winner_rows, quality_rows = [], []
    for start in range(0, len(features), config.query_chunk):
        local = local_aliases[start:start + config.query_chunk, members]
        local = (local[:, None] / config.temperature).expand(-1, aliases, -1, -1).masked_fill(excluded[None], -torch.inf)
        scores = local.logsumexp(-1) - remaining.log()[None]
        lv, li = scores.topk(2, -1)
        bv, bi = broad_group[start:start + config.query_chunk].topk(2, -1)
        quality = (torch.tanh((lv[..., 0] - lv[..., 1]).clamp_min(0) / 2)
                   * torch.tanh((bv[..., 0] - bv[..., 1]).clamp_min(0) / 2)).sqrt()
        winner_rows.append(li[..., 0])
        quality_rows.append(quality * (li[..., 0] == bi[..., 0]))
    winners, quality = torch.cat(winner_rows), torch.cat(quality_rows)
    cells = reference_cells(coordinates, image_size, config.reference_cell_pixels)
    _, inverse = cells.unique(sorted=True, return_inverse=True)
    regions, capacity = int(inverse.max()) + 1, config.reference_pool_size
    indices = torch.zeros((aliases, classes, capacity), device=features.device, dtype=torch.long)
    available = torch.zeros_like(indices, dtype=torch.bool)
    expanded = inverse[:, None].expand(-1, aliases)
    token_ids = torch.arange(len(features), device=features.device)[:, None].expand(-1, aliases)
    for c in range(classes):
        values = quality.masked_fill(winners != c, 0.)
        best = torch.zeros((regions, aliases), device=features.device)
        best.scatter_reduce_(0, expanded, values, reduce="amax", include_self=True)
        matches = (values == best[inverse]) & (values > 0)
        chosen_tokens = torch.full((regions, aliases), len(features), device=features.device, dtype=torch.long)
        chosen_tokens.scatter_reduce_(0, expanded, token_ids.masked_fill(~matches, len(features)), reduce="amin", include_self=True)
        take = min(capacity, regions)
        scores, chosen = best.topk(take, 0)
        indices[:, c, :take] = chosen_tokens.gather(0, chosen).T.clamp_max(len(features) - 1)
        available[:, c, :take] = scores.T > 0
    used = indices[available].unique(sorted=True)
    if not len(used):
        used = torch.zeros(1, device=features.device, dtype=torch.long)
    remapped = torch.searchsorted(used, indices).clamp_max(len(used) - 1).masked_fill(~available, 0)
    return ImageReferences(F.normalize(features[used].float(), dim=-1), coordinates[used], cells[used],
                           quality[used], marginal[used], remapped, available, image_size, len(features))


class CalibratedCompetitiveAliases:
    def __init__(self, bank, config=StratifiedAliasConfig(), action_config=CalibratedAliasConfig()):
        self.base = StratifiedSoftAliases(bank, config)
        action_config.validate(len(bank.alias_names))
        self.bank, self.config, self.action_config = bank, config, action_config
        self.members, self.count = self.base.members, self.base.count
        self.holdouts = text_holdouts(bank.features, action_config.text_holdout_count)
        self.canonical = torch.stack([(bank.canonical_mask & (bank.parent_indices == c)).nonzero().flatten()[0]
                                     for c in range(bank.class_count)])
        self.permutations = []
        for seed in range(config.random_seed, config.random_seed + 3):
            generator = torch.Generator().manual_seed(seed)
            self.permutations.append(torch.stack([torch.randperm(self.count, generator=generator)
                for _ in range(bank.class_count)]).to(bank.features.device))

    def report(self):
        return {"implementation": IMPLEMENTATION, "config": asdict(self.config),
                "action_config": asdict(self.action_config), "text_holdouts": self.holdouts.nonzero().tolist(),
                "reference": "local/wide class agreement with tested alias and its two nearest text neighbors absent; query cell excluded",
                "utility": "contrast of exact broad half-attenuation marginals on Geometry-retrieved A/B witnesses",
                "writer": "fixed original salience-profiled logits; normalized reliability outside exponent",
                "coupling": "original Geometry anchor; same bounded pair-margin update and reconstruction",
                "shuffled_permutations": [value.tolist() for value in self.permutations], "target_labels_used": False}

    def text_weights(self, pairs):
        text = F.normalize(self.bank.features.float(), dim=-1)
        a, b = text[self.canonical[pairs[:, 0]]], text[self.canonical[pairs[:, 1]]]
        values = torch.einsum("qskd,qd->qsk", text[self.members[pairs]], a - b)
        values[:, 1] *= -1
        values = (values - values.mean(-1, keepdim=True)) / values.std(-1, keepdim=True).clamp_min(1e-6)
        values = values.clamp(-self.config.standardized_utility_limit, self.config.standardized_utility_limit)
        return self.config.uniform_prior / self.count + (1 - self.config.uniform_prior) * values.softmax(-1)

    @torch.inference_mode()
    def read(self, features, coordinates, local, broad, old_references, new_references, crops, crop_count, valid, tau=1.):
        names = ("StratifiedAlias_Coupled", "CalibratedReplay_Coupled", PRIMARY, "TextPrior_Coupled", *SHUFFLED)
        variants, totals = {name: [] for name in names}, {}
        for start in range(0, len(features), self.config.query_chunk):
            sl = slice(start, start + self.config.query_chunk)
            pairs, old_weights, _ = self.base.allocate(features[sl], coordinates[sl], local[sl], old_references, valid[sl])
            _, new_weights, diagnostics = self.base.allocate(features[sl], coordinates[sl], local[sl], new_references, valid[sl])
            allocations = {names[0]: old_weights["soft"], names[1]: old_weights["soft"], PRIMARY: new_weights["soft"],
                           "TextPrior_Coupled": self.text_weights(pairs)}
            for name, permutation in zip(SHUFFLED, self.permutations):
                allocations[name] = allocations[PRIMARY].gather(-1, permutation[pairs])
                error = (allocations[name].sort(-1).values - allocations[PRIMARY].sort(-1).values).abs().max()
                diagnostics[name + "_spectrum_error"] = float(error)
            before = broad[sl].gather(-1, pairs)
            margin = before[:, 0] - before[:, 1]
            for name, weights in allocations.items():
                writer = weighted_wide_pairs if name == names[0] else calibrated_wide_pairs
                wide = writer(crops, crop_count, coordinates[sl], new_references.image_size, pairs, weights, self.members, tau)
                correction = wide[:, 0] - wide[:, 1] - margin
                uniform = (weights == 1 / self.count).all(-1).all(-1)
                correction = torch.where(uniform | ~valid[sl], 0., correction)
                bound = self.config.maximum_margin_correction
                correction = bound * torch.tanh(correction / bound)
                result = apply_pair_margin(broad[sl], pairs, correction, valid[sl])
                variants[name].append(result)
                partition = (result.gather(-1, pairs).logsumexp(-1) - before.logsumexp(-1)).abs().max()
                diagnostics[name + "_partition_error"] = float(partition)
                diagnostics[name + "_mean_abs_correction"] = float(correction[valid[sl]].abs().mean()) if bool(valid[sl].any()) else 0.
                diagnostics[name + "_active_fraction"] = float((correction[valid[sl]].abs() > 1e-6).float().mean()) if bool(valid[sl].any()) else 0.
            n = int(valid[sl].sum())
            for field, value in diagnostics.items():
                if field.endswith("_error") or field.startswith("maximum_"):
                    totals[field] = max(totals.get(field, 0.), value)
                elif field.startswith("minimum_"):
                    totals[field] = min(totals.get(field, value), value)
                else:
                    totals[field] = totals.get(field, 0.) + value * n
        denominator = max(int(valid.sum()), 1)
        diagnostics = {field: value if field.endswith("_error") or field.startswith(("minimum_", "maximum_")) else value / denominator
                       for field, value in totals.items()}
        diagnostics["group_reference_pool_fraction"] = float(new_references.pool_valid.any(-1).float().mean())
        return {name: torch.cat(rows) for name, rows in variants.items()}, diagnostics
