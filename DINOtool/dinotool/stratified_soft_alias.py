"""Class-stratified image references and branch-aligned soft alias readout."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F

from .soft_competitive_alias import apply_pair_margin, leave_one_out_scores, leave_out_witnesses


IMPLEMENTATION = "geometry-stratified-wide-alias-v2-20261002"
PRIMARY = "StratifiedAlias_Coupled"
METHODS = ("Geometry", "BroadVIP", "Anchored_VIP", "PairAlias_Coupled",
           "SoftCounterfactual_Coupled", PRIMARY, "ShuffledStratified_Coupled")
REPLAY = METHODS[:5]
PAIRS = {"new_vs_all20": ("Anchored_VIP", PRIMARY),
         "new_vs_old_soft": ("SoftCounterfactual_Coupled", PRIMARY),
         "new_vs_hard": ("PairAlias_Coupled", PRIMARY),
         "new_vs_shuffled": ("ShuffledStratified_Coupled", PRIMARY)}


@dataclass(frozen=True)
class StratifiedAliasConfig:
    temperature: float = 0.07
    uniform_prior: float = 0.5
    reference_pool_size: int = 64
    relation_neighbors: int = 32
    reference_cell_pixels: int = 64
    minimum_effective_reference: float = 4.0
    standardized_utility_limit: float = 2.0
    maximum_margin_correction: float = 0.5
    geometry_temperature: float = 0.10
    spatial_sigma: float = 0.25
    query_chunk: int = 128
    random_seed: int = 20261002
    numerical_epsilon: float = 1e-6

    def validate(self):
        positive = (self.temperature, self.minimum_effective_reference,
                    self.standardized_utility_limit, self.maximum_margin_correction,
                    self.geometry_temperature, self.spatial_sigma, self.numerical_epsilon)
        if (not 0 <= self.uniform_prior <= 1
                or not all(math.isfinite(x) and x > 0 for x in positive)
                or min(self.reference_pool_size, self.relation_neighbors,
                       self.reference_cell_pixels, self.query_chunk) < 1
                or self.relation_neighbors > self.reference_pool_size):
            raise ValueError("Invalid stratified alias configuration.")


@dataclass(frozen=True)
class ImageReferences:
    features: torch.Tensor
    coordinates: torch.Tensor
    cells: torch.Tensor
    quality: torch.Tensor
    marginal: torch.Tensor
    pool_indices: torch.Tensor
    pool_valid: torch.Tensor
    image_size: tuple[int, int]
    candidate_count: int


@dataclass(frozen=True)
class WideCrop:
    alias_logits: torch.Tensor
    salience: torch.Tensor
    top: int
    left: int
    actual_height: int
    actual_width: int
    grid_side: int = 21
    crop_side: int = 336


def reference_cells(coordinates, image_size, cell_pixels):
    height, width = image_size
    cells = (coordinates / cell_pixels).floor().long()
    return cells[..., 0] * math.ceil(width / cell_pixels) + cells[..., 1]


@torch.inference_mode()
def build_image_references(features, coordinates, local_aliases, broad, broad_without_alias,
                           parents, classes, image_size, config=StratifiedAliasConfig()):
    """One leave-out witness per spatial cell, class, and tested alias."""
    config.validate()
    if (features.ndim != 2 or coordinates.shape != (len(features), 2)
            or local_aliases.ndim != 2 or len(local_aliases) != len(features)
            or broad.shape != (len(features), classes)
            or broad_without_alias.shape != local_aliases.shape or len(features) < 1):
        raise ValueError("Matching nonempty image reference tensors required.")
    local = torch.stack([config.temperature * (
        torch.logsumexp(local_aliases[:, parents == c] / config.temperature, -1)
        - math.log(int((parents == c).sum()))) for c in range(classes)], -1)
    without = leave_one_out_scores(local_aliases[None], parents, classes, config.temperature)[0]
    winner, local_quality = leave_out_witnesses(local / config.temperature, without / config.temperature, parents)
    wide_winner, wide_quality = leave_out_witnesses(broad, broad_without_alias, parents)
    quality = (local_quality * wide_quality).clamp_min(0).sqrt() * (winner == wide_winner)
    marginal = broad[:, parents] - broad_without_alias
    cells = reference_cells(coordinates, image_size, config.reference_cell_pixels)
    _, inverse = cells.unique(sorted=True, return_inverse=True)
    regions = int(inverse.max()) + 1
    aliases, capacity = local_aliases.shape[-1], config.reference_pool_size
    indices = torch.zeros((aliases, classes, capacity), device=features.device, dtype=torch.long)
    available = torch.zeros_like(indices, dtype=torch.bool)
    token_indices = torch.arange(len(features), device=features.device)
    expanded_cells = inverse[:, None].expand(-1, aliases)
    for c in range(classes):
        values = quality.masked_fill(winner != c, 0.)
        best = torch.zeros((regions, aliases), device=features.device)
        best.scatter_reduce_(0, expanded_cells, values, reduce="amax", include_self=True)
        matches = (values == best[inverse]) & (values > 0)
        winners = torch.full((regions, aliases), len(features), device=features.device, dtype=torch.long)
        proposed = token_indices[:, None].expand_as(values).masked_fill(~matches, len(features))
        winners.scatter_reduce_(0, expanded_cells, proposed, reduce="amin", include_self=True)
        count = min(capacity, regions)
        scores, chosen = best.topk(count, dim=0)
        chosen_tokens = winners.gather(0, chosen).T
        indices[:, c, :count] = chosen_tokens.clamp_max(len(features) - 1)
        available[:, c, :count] = scores.T > 0
    used = indices[available].unique(sorted=True)
    if not len(used):
        used = torch.zeros(1, device=features.device, dtype=torch.long)
    remapped = torch.searchsorted(used, indices).clamp_max(len(used) - 1)
    remapped = remapped.masked_fill(~available, 0)
    return ImageReferences(F.normalize(features[used].float(), dim=-1), coordinates[used], cells[used],
                           quality[used], marginal[used], remapped, available, image_size, len(features))


def crop_stencil(crop, count, coordinates, image_size):
    """Compose historical grid_sample, crop overlap mean, and 21-to-336 interpolation."""
    resized_height, resized_width = count.shape
    scale = coordinates.new_tensor((resized_height / image_size[0], resized_width / image_size[1]))
    position = coordinates * scale - .5
    position[:, 0].clamp_(0, resized_height - 1)
    position[:, 1].clamp_(0, resized_width - 1)
    lower = position.floor().long()
    fraction = position - lower
    indices, coefficients = [], []
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        y = (lower[:, 0] + dy).clamp_max(resized_height - 1)
        x = (lower[:, 1] + dx).clamp_max(resized_width - 1)
        weight = (fraction[:, 0] if dy else 1 - fraction[:, 0]) * (fraction[:, 1] if dx else 1 - fraction[:, 1])
        inside = ((y >= crop.top) & (y < crop.top + crop.actual_height)
                  & (x >= crop.left) & (x < crop.left + crop.actual_width))
        weight = weight * inside / count[y, x]
        local = torch.stack((y - crop.top, x - crop.left), -1).float()
        local = ((local + .5) * crop.grid_side / crop.crop_side - .5).clamp(0, crop.grid_side - 1)
        base = local.floor().long()
        offset = local - base
        for sy, sx in ((0, 0), (0, 1), (1, 0), (1, 1)):
            gy = (base[:, 0] + sy).clamp_max(crop.grid_side - 1)
            gx = (base[:, 1] + sx).clamp_max(crop.grid_side - 1)
            coefficient = weight * (offset[:, 0] if sy else 1 - offset[:, 0]) * (offset[:, 1] if sx else 1 - offset[:, 1])
            indices.append(gy * crop.grid_side + gx)
            coefficients.append(coefficient)
    return torch.stack(indices, -1), torch.stack(coefficients, -1)


@torch.inference_mode()
def weighted_wide_pairs(crops, count, coordinates, image_size, pairs, weights, members, tau=1.):
    """Reliability reweights salience before the original nonlinear VIP aggregation."""
    if weights.shape != (*pairs.shape, members.shape[-1]) or not tau > 0:
        raise ValueError("Normalized per-pair alias weights required.")
    aliases = members[pairs]
    output = coordinates.new_zeros(pairs.shape)
    for crop in crops:
        indices, coefficients = crop_stencil(crop, count, coordinates, image_size)
        salience = crop.salience[aliases].softmax(-1)
        reliability = salience * weights
        reliability = reliability / reliability.sum(-1, keepdim=True)
        logits = crop.alias_logits[indices[:, :, None, None], aliases[:, None]]
        scaled = logits * (members.shape[-1] * reliability[:, None])
        scores = torch.logsumexp(tau * scaled, -1) / tau
        output += (scores * coefficients[..., None]).sum(1)
    return output


class StratifiedSoftAliases:
    def __init__(self, bank, config=StratifiedAliasConfig()):
        bank.validate()
        config.validate()
        self.bank, self.config = bank, config
        groups = [(bank.parent_indices == c).nonzero().flatten() for c in range(bank.class_count)]
        if bank.class_count < 2 or min(map(len, groups)) < 2 or len(set(map(len, groups))) != 1:
            raise ValueError("Equal nontrivial alias groups in two or more classes required.")
        self.members = torch.stack(groups)
        self.count = self.members.shape[-1]
        generator = torch.Generator().manual_seed(config.random_seed)
        self.permutations = torch.stack([torch.randperm(self.count, generator=generator)
            for _ in groups]).to(bank.features.device)

    def report(self):
        return {"implementation": IMPLEMENTATION, "config": asdict(self.config),
                "reference": "nonoverlapping image-wide Geometry views; leave-one-alias-out local/broad agreement; one witness per64px cell",
                "retrieval": "per tested alias and rival class: independently retrieve Geometry top32 from64 class-specific candidates; exclude query cell",
                "utility": "signed broad deletion marginal A-minus-B mean, standardized by weighted variance/effective count",
                "weighting": "half uniform plus half softmax trusted utility; reliability multiplies broad salience then renormalizes",
                "coupling": "branch-aligned weighted broad margin; original Geometry anchor; bounded pair update",
                "shuffled_permutations": self.permutations.tolist(), "target_labels_used": False}

    @torch.inference_mode()
    def allocate(self, features, coordinates, local, references, valid):
        cfg, count = self.config, self.count
        pairs = local.topk(2, -1).indices
        tested = self.members[pairs].flatten(-2)
        cells = reference_cells(coordinates, references.image_size, cfg.reference_cell_pixels)
        similarity = F.normalize(features.float(), dim=-1) @ references.features.T
        scale = coordinates.new_tensor(references.image_size)
        distance = ((coordinates[:, None] - references.coordinates[None]) / scale).square().sum(-1)
        geometry = similarity / cfg.geometry_temperature - distance / (2 * cfg.spatial_sigma ** 2)
        means, variances, effective, certainty = [], [], [], []
        alias_sign = coordinates.new_tensor([1.] * count + [-1.] * count)
        for side in (0, 1):
            candidate = references.pool_indices[tested, pairs[:, side, None]]
            present = references.pool_valid[tested, pairs[:, side, None]]
            present = present & (references.cells[candidate] != cells[:, None, None]) & valid[:, None, None]
            scores = geometry.gather(1, candidate.flatten(1)).reshape_as(candidate).masked_fill(~present, -torch.inf)
            scores, selected = scores.topk(min(cfg.relation_neighbors, candidate.shape[-1]), -1)
            candidate = candidate.gather(-1, selected)
            available = torch.isfinite(scores)
            maximum = scores.amax(-1, keepdim=True).masked_fill(~available.any(-1, keepdim=True), 0.)
            structural = torch.exp(scores - maximum).masked_fill(~available, 0.)
            structural = structural / structural.sum(-1, keepdim=True).clamp_min(cfg.numerical_epsilon)
            quality = references.quality[candidate, tested[..., None]]
            marginal = references.marginal[candidate, tested[..., None]] * alias_sign[None, :, None]
            weights = structural * quality
            mass = weights.sum(-1)
            mean = (weights * marginal).sum(-1) / mass.clamp_min(cfg.numerical_epsilon)
            variance = (weights * (marginal - mean[..., None]).square()).sum(-1) / mass.clamp_min(cfg.numerical_epsilon)
            neff = mass.square() / weights.square().sum(-1).clamp_min(cfg.numerical_epsilon ** 2)
            means.append(mean)
            variances.append(variance)
            effective.append(neff)
            certainty.append(mass)
        support = (effective[0] >= cfg.minimum_effective_reference) & (effective[1] >= cfg.minimum_effective_reference)
        error = (variances[0] / (effective[0] - 1).clamp_min(1)
                 + variances[1] / (effective[1] - 1).clamp_min(1) + cfg.numerical_epsilon ** 2).sqrt()
        utility = ((means[0] - means[1]) / error).clamp(-cfg.standardized_utility_limit, cfg.standardized_utility_limit)
        trust = (certainty[0] * certainty[1]).clamp_min(0).sqrt() * support * valid[:, None]
        utility = (utility * trust).reshape(-1, 2, count)
        weights = cfg.uniform_prior / count + (1 - cfg.uniform_prior) * utility.softmax(-1)
        shuffled = weights.gather(-1, self.permutations[pairs])
        active = valid[:, None].expand_as(support)
        diagnostics = {"reference_supported_alias_fraction": float(support[active].float().mean()) if bool(active.any()) else 0.,
                       "mean_reference_trust": float(trust[active].mean()) if bool(active.any()) else 0.,
                       "effective_alias_count": float((-(weights * weights.log()).sum(-1))[valid].exp().mean()) if bool(valid.any()) else 0.,
                       "mean_weight_kl_from_uniform": float((weights * (weights.log() + math.log(count))).sum(-1)[valid].mean()) if bool(valid.any()) else 0.,
                       "minimum_alias_weight": float(weights.min()), "maximum_alias_weight": float(weights.max()),
                       "shuffled_weight_spectrum_error": float((weights.sort(-1).values - shuffled.sort(-1).values).abs().max())}
        return pairs, {"soft": weights, "shuffled": shuffled}, diagnostics

    @torch.inference_mode()
    def read(self, features, coordinates, local, broad, references, crops, crop_count, valid, tau=1.):
        if (features.ndim != 2 or coordinates.shape != (len(features), 2)
                or local.shape != (len(features), self.bank.class_count)
                or broad.shape != local.shape or valid.shape != (len(features),)
                or not all(bool(torch.isfinite(x).all()) for x in (features, coordinates, local, broad))):
            raise ValueError("Matching finite query tensors required.")
        variants, totals = {"soft": [], "shuffled": []}, {}
        original = broad.gather(-1, local.topk(2, -1).indices)
        original_margin = original[:, 0] - original[:, 1]
        for start in range(0, len(features), self.config.query_chunk):
            stop = min(start + self.config.query_chunk, len(features))
            selection = slice(start, stop)
            pairs, allocations, stats = self.allocate(features[selection], coordinates[selection], local[selection], references, valid[selection])
            for name, weights in allocations.items():
                wide = weighted_wide_pairs(crops, crop_count, coordinates[selection], references.image_size,
                                           pairs, weights, self.members, tau)
                correction = wide[:, 0] - wide[:, 1] - original_margin[selection]
                uniform = (weights == 1 / self.count).all(-1).all(-1)
                correction = torch.where(uniform | ~valid[selection], 0., correction)
                bound = self.config.maximum_margin_correction
                correction = bound * torch.tanh(correction / bound)
                variants[name].append(apply_pair_margin(broad[selection], pairs, correction, valid[selection]))
                if name == "soft":
                    stats.update({"active_patch_fraction": float((correction[valid[selection]].abs() > self.config.numerical_epsilon).float().mean()) if bool(valid[selection].any()) else 0.,
                                  "mean_absolute_margin_correction": float(correction[valid[selection]].abs().mean()) if bool(valid[selection].any()) else 0.,
                                  "maximum_absolute_margin_correction": float(correction.abs().max()),
                                  "pair_partition_error": float((variants[name][-1].gather(-1, pairs).logsumexp(-1)
                                      - broad[selection].gather(-1, pairs).logsumexp(-1)).abs().max())})
            size = int(valid[selection].sum())
            for field, value in stats.items():
                if field.startswith("minimum_"):
                    totals[field] = min(totals.get(field, value), value)
                elif field.startswith("maximum_") or field.endswith("_error"):
                    totals[field] = max(totals.get(field, 0.), value)
                else:
                    totals[field] = totals.get(field, 0.) + value * size
        denominator = int(valid.sum())
        diagnostics = {field: value if field.startswith(("minimum_", "maximum_")) or field.endswith("_error") else value / max(denominator, 1)
                       for field, value in totals.items()}
        diagnostics.update({"image_reference_candidates": references.candidate_count,
                            "image_reference_selected_tokens": len(references.features),
                            "nonempty_reference_pool_fraction": float(references.pool_valid.any(-1).float().mean())})
        return {name: torch.cat(rows, 0) for name, rows in variants.items()}, diagnostics
