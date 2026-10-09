"""Geometry-supported, leave-one-alias-out competitive evidence allocation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F

from .geometry_readout_trace import alias_class_scores
from .pair_conditional_alias import PairConditionalAliases


IMPLEMENTATION = "geometry-soft-counterfactual-alias-v1-20261002"
PRIMARY = "SoftCounterfactual_Coupled"
METHODS = ("Geometry", "BroadVIP", "MeanProb_VIP", "MeanLogit_VIP", "Anchored_VIP",
           "PairAlias_Local", "PairAlias_Coupled", "SoftText_Local", "SoftText_Coupled",
           "SoftCounterfactual_Local", PRIMARY, "ShuffledCounterfactual_Local",
           "ShuffledCounterfactual_Coupled")
PAIRS = {"soft_vs_hard_local": ("PairAlias_Local", "SoftCounterfactual_Local"),
         "soft_vs_hard_coupled": ("PairAlias_Coupled", PRIMARY),
         "soft_vs_all_local": ("Geometry", "SoftCounterfactual_Local"),
         "soft_vs_all_coupled": ("Anchored_VIP", PRIMARY),
         "utility_vs_simple_soft": ("SoftText_Coupled", PRIMARY),
         "utility_vs_shuffled": ("ShuffledCounterfactual_Coupled", PRIMARY)}


@dataclass(frozen=True)
class SoftAliasConfig:
    temperature: float = 0.07
    uniform_prior: float = 0.5
    relation_neighbors: int = 32
    minimum_effective_reference: float = 4.0
    standardized_utility_limit: float = 2.0
    maximum_margin_correction: float = 0.5
    random_seed: int = 20261002
    numerical_epsilon: float = 1e-6

    def validate(self):
        if (not 0 <= self.uniform_prior <= 1 or self.relation_neighbors < 1
                or self.minimum_effective_reference < 1
                or not all(math.isfinite(x) and x > 0 for x in
                    (self.temperature, self.standardized_utility_limit,
                     self.maximum_margin_correction, self.numerical_epsilon))):
            raise ValueError("Invalid frozen soft-alias configuration.")


def leave_one_out_scores(aliases, parents, classes, temperature=0.07):
    """Exact normalized LME after each alias deletion, in original alias order."""
    out = torch.empty_like(aliases, dtype=torch.float32)
    for c in range(classes):
        members = (parents == c).nonzero().flatten()
        k = len(members)
        if k < 2:
            raise ValueError("Each class needs at least two aliases for counterfactuals.")
        values = aliases[..., members].float() / temperature
        excluded = torch.eye(k, device=aliases.device, dtype=torch.bool)
        counterfactual = values[..., None, :].masked_fill(excluded, -torch.inf)
        out[..., members] = temperature * (torch.logsumexp(counterfactual, -1) - math.log(k - 1))
    return out


def leave_out_witnesses(scores, without_alias, parents):
    """Class winner and continuous winner margin with the tested alias absent."""
    changed = scores[..., None, :].expand(*without_alias.shape, scores.shape[-1]).clone()
    indices = parents.reshape(*([1] * (without_alias.ndim - 1)), -1, 1)
    changed.scatter_(-1, indices.expand(*without_alias.shape, 1), without_alias[..., None])
    values, winners = changed.topk(2, -1)
    quality = torch.tanh((values[..., 0] - values[..., 1]).clamp_min(0) / 2)
    return winners[..., 0], quality


def apply_pair_margin(scores, pairs, correction, valid):
    """Preserve the A/B partition sum and every other class probability."""
    before = scores.gather(-1, pairs)
    total = torch.logsumexp(before, -1)
    margin = before[..., 0] - before[..., 1] + correction
    after = torch.stack((total + F.logsigmoid(margin), total + F.logsigmoid(-margin)), -1)
    active = valid & (correction != 0)
    after = torch.where(active[..., None], after, before)
    return scores.clone().scatter(-1, pairs, after)


class SoftCompetitiveAliases:
    def __init__(self, bank, config=SoftAliasConfig()):
        bank.validate()
        config.validate()
        self.bank, self.config = bank, config
        self.members = torch.stack([(bank.parent_indices == c).nonzero().flatten()
                                    for c in range(bank.class_count)])
        self.count = self.members.shape[1]
        if bank.class_count < 2 or self.count < 2:
            raise ValueError("Two classes with equal alias counts >=2 required.")
        self.hard = PairConditionalAliases(bank)
        generator = torch.Generator().manual_seed(config.random_seed)
        self.permutations = torch.stack([torch.randperm(self.count, generator=generator)
                                        for _ in range(bank.class_count)]).to(bank.features.device)

    def report(self):
        return {"implementation": IMPLEMENTATION, "config": asdict(self.config),
                "reference": "local and broad leave-one-alias-out winners must agree; Geometry top32 donors",
                "utility": "A-minus-B difference of the signed exact local deletion marginal, standardized by reference variance",
                "fallback": "no reliable effective A/B references => uniform weights; no global deletion",
                "coupling": "original Geometry anchor; bounded pair correction adjusts broad observation; pair partition preserved",
                "shuffled_permutations": self.permutations.tolist(),
                "hard_control": self.hard.report(), "target_labels_used": False}

    @torch.inference_mode()
    def read(self, aliases, relation, broad, broad_without_alias, valid):
        if (aliases.ndim != 3 or aliases.shape[-1] != len(self.bank.alias_names)
                or relation.shape != (*aliases.shape[:2], aliases.shape[1])
                or broad.shape != (*aliases.shape[:2], self.bank.class_count)
                or broad_without_alias.shape != aliases.shape or valid.shape != aliases.shape[:2]
                or not all(bool(torch.isfinite(value).all()) for value in
                           (aliases, relation, broad, broad_without_alias)) or bool((relation < 0).any())):
            raise ValueError("Matching finite alias, relation, broad and validity tensors required.")
        cfg, k = self.config, self.count
        raw = alias_class_scores(aliases, self.bank.parent_indices, self.bank.class_count, cfg.temperature)
        original = raw / cfg.temperature
        pairs = original.topk(2, -1).indices
        alias_ids = self.members[pairs].flatten(-2)
        without = leave_one_out_scores(aliases, self.bank.parent_indices, self.bank.class_count, cfg.temperature)
        local_winner, local_quality = leave_out_witnesses(original, without / cfg.temperature, self.bank.parent_indices)
        wide_winner, wide_quality = leave_out_witnesses(broad, broad_without_alias, self.bank.parent_indices)
        quality = (local_quality * wide_quality).clamp_min(0).sqrt()
        agreement = (local_winner == wide_winner) & valid[..., None]
        quality = quality * agreement
        marginal = (raw[..., self.bank.parent_indices] - without) / cfg.temperature
        donor_weights, donors = relation.masked_fill(~valid[:, None], 0).topk(min(cfg.relation_neighbors, aliases.shape[1]), -1)
        donor_weights = donor_weights / donor_weights.sum(-1, keepdim=True).clamp_min(cfg.numerical_epsilon)
        batches = torch.arange(aliases.shape[0], device=aliases.device)[:, None, None, None]
        donor_indices, tested_ids = donors[..., None], alias_ids[..., None, :]
        winner = local_winner[batches, donor_indices, tested_ids]
        q = quality[batches, donor_indices, tested_ids]
        delta = marginal[batches, donor_indices, tested_ids]
        sign = torch.cat((torch.ones(k, device=aliases.device), -torch.ones(k, device=aliases.device)))
        delta = delta * sign
        means, variances, effective, certainties, fractions = [], [], [], [], []
        for side in (0, 1):
            membership = winner == pairs[..., side, None, None]
            unqualified = donor_weights[..., None] * membership
            weights = unqualified * q
            mass = weights.sum(-2)
            mean = (weights * delta).sum(-2) / mass.clamp_min(cfg.numerical_epsilon)
            variance = (weights * (delta - mean[..., None, :]).square()).sum(-2) / mass.clamp_min(cfg.numerical_epsilon)
            neff = mass.square() / weights.square().sum(-2).clamp_min(cfg.numerical_epsilon ** 2)
            means.append(mean)
            variances.append(variance)
            effective.append(neff)
            certainties.append(mass / unqualified.sum(-2).clamp_min(cfg.numerical_epsilon))
            fractions.append(unqualified.sum(-2))
        support = (effective[0] >= cfg.minimum_effective_reference) & (effective[1] >= cfg.minimum_effective_reference)
        error = (variances[0] / (effective[0] - 1).clamp_min(1)
                 + variances[1] / (effective[1] - 1).clamp_min(1) + cfg.numerical_epsilon ** 2).sqrt()
        utility = ((means[0] - means[1]) / error).clamp(-cfg.standardized_utility_limit, cfg.standardized_utility_limit)
        balance = 2 * torch.minimum(fractions[0], fractions[1]) / (fractions[0] + fractions[1]).clamp_min(cfg.numerical_epsilon)
        trust = (certainties[0] * certainties[1]).clamp_min(0).sqrt() * balance * support
        trust = trust * valid[..., None]
        utility = (utility * trust).reshape(*aliases.shape[:2], 2, k)
        proposed = utility.softmax(-1)
        weights = cfg.uniform_prior / k + (1 - cfg.uniform_prior) * proposed
        perm = self.permutations[pairs]
        shuffled = weights.gather(-1, perm)
        mask = torch.stack(self.hard.selected)[pairs[..., 0], pairs[..., 1]]
        reverse = torch.stack(self.hard.selected)[pairs[..., 1], pairs[..., 0]]
        text_masks = torch.stack((mask, reverse), -2)
        text_weights = cfg.uniform_prior / k + (1 - cfg.uniform_prior) * text_masks / text_masks.sum(-1, keepdim=True)
        competing = aliases.gather(-1, alias_ids).reshape(*aliases.shape[:2], 2, k)
        original_margin = original.gather(-1, pairs)
        original_margin = original_margin[..., 0] - original_margin[..., 1]
        variants, corrections = {}, {}
        for name, allocation in (("soft", weights), ("shuffled", shuffled), ("text", text_weights)):
            scores = torch.logsumexp(competing / cfg.temperature + allocation.log(), -1)
            correction = scores[..., 0] - scores[..., 1] - original_margin
            unchanged = (allocation == 1 / k).all(-1).all(-1)
            correction = torch.where(unchanged | ~valid, 0., correction)
            correction = cfg.maximum_margin_correction * torch.tanh(correction / cfg.maximum_margin_correction)
            variants[name] = {"local": apply_pair_margin(original, pairs, correction, valid),
                              "observation": apply_pair_margin(broad, pairs, correction, valid),
                              "weights": allocation, "pairs": pairs, "correction": correction}
            corrections[name] = correction
        active = valid[..., None].expand_as(trust)
        entropy = -(weights * weights.log()).sum(-1)
        diagnostics = {"reference_supported_alias_fraction": float(support[active].float().mean()) if bool(active.any()) else 0.,
                       "mean_reference_trust": float(trust[active].mean()) if bool(active.any()) else 0.,
                       "effective_alias_count": float(entropy[valid].exp().mean()) if bool(valid.any()) else 0.,
                       "minimum_alias_weight": float(weights.min()), "maximum_alias_weight": float(weights.max()),
                       "mean_weight_kl_from_uniform": float((weights * (weights.log() + math.log(k))).sum(-1)[valid].mean()) if bool(valid.any()) else 0.,
                       "soft_active_patch_fraction": float((corrections["soft"][valid].abs() > cfg.numerical_epsilon).float().mean()) if bool(valid.any()) else 0.,
                       "mean_absolute_soft_margin_correction": float(corrections["soft"][valid].abs().mean()) if bool(valid.any()) else 0.,
                       "maximum_absolute_soft_margin_correction": float(corrections["soft"].abs().max()),
                       "local_changed_patch_fraction": float((variants["soft"]["local"].argmax(-1)[valid] != original.argmax(-1)[valid]).float().mean()) if bool(valid.any()) else 0.,
                       "shuffled_weight_spectrum_error": float((weights.sort(-1).values - shuffled.sort(-1).values).abs().max()),
                       "pair_partition_error": float((variants["soft"]["local"].gather(-1, pairs).logsumexp(-1)
                                                       - original.gather(-1, pairs).logsumexp(-1)).abs().max())}
        return variants, diagnostics
