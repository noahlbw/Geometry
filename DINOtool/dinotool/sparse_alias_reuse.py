"""Budgeted alias correction using two contenders and already-computed features."""
from dataclasses import dataclass

import torch

from .fine_alias_view import CONFIG
from .stratified_soft_alias import crop_stencil


IMPLEMENTATION = 'geometry-native-reuse-top2-alias-v1-20261005'
PRIMARY = 'SparseNativeSoft'
METHODS = ('Geometry', 'NoAdmission_Exact', PRIMARY, 'SparseNativeHard', 'SparseGeometrySoft')


@dataclass(frozen=True)
class AliasLayout:
    members: torch.Tensor
    valid: torch.Tensor
    canonical: torch.Tensor

    @property
    def counts(self):
        return self.valid.sum(-1)


@dataclass(frozen=True)
class SparseCrop:
    evidence: torch.Tensor
    indices: torch.Tensor
    coefficients: torch.Tensor


def alias_layout(parents, canonical, classes):
    if (parents.ndim != 1 or parents.dtype != torch.long or classes < 2
            or canonical.shape != (classes,) or canonical.dtype != torch.long
            or bool(((parents < 0) | (parents >= classes)).any())
            or bool(((canonical < 0) | (canonical >= len(parents))).any())
            or not torch.equal(parents[canonical], torch.arange(classes, device=parents.device))):
        raise ValueError('Actual alias parents and one canonical alias per class required.')
    groups = [(parents == c).nonzero().flatten() for c in range(classes)]
    width = max(map(len, groups))
    members = canonical[:, None].expand(-1, width).clone()
    valid = torch.zeros_like(members, dtype=torch.bool)
    for c, group in enumerate(groups):
        members[c, :len(group)] = group
        valid[c, :len(group)] = True
    return AliasLayout(members, valid, valid & (members == canonical[:, None]))


def profile_aliases(alias_logits, salience, layout):
    if alias_logits.ndim != 2 or salience.shape != alias_logits.shape[1:]:
        raise ValueError('Actual token/alias responses and matching salience required.')
    scale = salience[layout.members].masked_fill(~layout.valid, -torch.inf).softmax(-1)
    result = alias_logits[:, layout.members] * (layout.counts[:, None] * scale)[None]
    return result.masked_fill(~layout.valid[None], -torch.inf)


def cache_sparse(crops, count, coordinates, image_size, layout):
    output = []
    for crop in crops:
        indices, coefficients = crop_stencil(crop, count, coordinates, image_size)
        output.append(SparseCrop(profile_aliases(crop.alias_logits, crop.salience, layout),
                                 indices, coefficients))
    return output


def contender_fields(observation, pairs):
    # Gather only the two selected classes, never a [query,alias,all-rivals] tensor.
    return observation.evidence[observation.indices[:, :, None], pairs[:, None]]


def survivor_allocation(source, keep, canonical, soft):
    if source.shape != keep.shape or canonical.shape != keep.shape or keep.dtype != torch.bool:
        raise ValueError('Matching sparse alias allocation fields required.')
    noncanonical = keep & ~canonical
    if soft:
        source = source.double().clamp_min(CONFIG.epsilon).masked_fill(~noncanonical, 0.)
        count = noncanonical.sum(-1, keepdim=True)
        allocated = source * count / source.sum(-1, keepdim=True).clamp_min(CONFIG.epsilon)
        return torch.where(noncanonical, allocated, keep.double())
    return keep.double()


@torch.inference_mode()
def sparse_scores(local, operator, broad, observations, witness, valid, layout, *, soft=True,
                  diagnostics=True, beta=CONFIG.beta):
    n, classes = local.shape
    if (classes != len(layout.members) or broad.shape != local.shape
            or operator.shape != (n, n) or valid.shape != (n,) or valid.dtype != torch.bool
            or witness.shape != (n, *layout.members.shape) or not observations or beta <= 0):
        raise ValueError('Matching local, wide, reused evidence and relation fields required.')
    base = local.double() + operator.double() @ (broad.double() - local.double())
    pairs = base.argsort(dim=-1, descending=True, stable=True)[:, :2]
    rows = torch.arange(n, device=local.device)
    selected = witness[rows[:, None], pairs].double()
    alias_valid = layout.valid[pairs]
    canonical = layout.canonical[pairs]
    counts = layout.counts[pairs].double()
    local_total = (beta * selected).logsumexp(-1) / beta
    local_mean = local_total - counts.log() / beta
    local_margin = selected - local_mean.flip(-1)[..., None]
    wide_margin = torch.zeros_like(selected)
    cached = []
    for crop in observations:
        evidence = contender_fields(crop, pairs).double()
        total = (beta * evidence).logsumexp(-1) / beta
        mean = total - counts[:, None].log() / beta
        margin = evidence - mean.flip(-1)[..., None]
        coefficients = crop.coefficients.double()
        wide_margin += (margin.masked_fill(~alias_valid[:, None], 0.)
                        * coefficients[..., None, None]).sum(1)
        cached.append((evidence, total, coefficients))
    reject = (wide_margin > 0) & (local_margin < 0) & ~canonical
    reject &= alias_valid & valid[:, None, None]
    keep = alias_valid & ~reject
    responsibility = (beta * selected).softmax(-1)
    allocation = survivor_allocation(responsibility, keep, canonical, soft)
    remaining = keep.sum(-1).double()
    normalizer = (counts / remaining).log()
    untouched = ((allocation == 1) | ~alias_valid).all(-1)
    directed = torch.zeros((n, 2), device=local.device, dtype=torch.float64)
    for evidence, total, coefficients in cached:
        changed = (beta * evidence + allocation.log()[:, None]).logsumexp(-1) / beta
        delta = (changed - total + normalizer[:, None] / beta).masked_fill(untouched[:, None], 0.)
        directed += (delta * coefficients[..., None]).sum(1)
    requested = directed[:, 0] - directed[:, 1]
    difference = local_total - broad.gather(1, pairs).double()
    target = difference[:, 0] - difference[:, 1]
    action = target.sign() * torch.minimum(requested.abs(), target.abs()) * (requested * target > 0)
    action = action.masked_fill(~valid, 0.)
    posterior = base.softmax(-1).gather(1, pairs)
    potential = torch.zeros_like(base)
    potential.scatter_(1, pairs, torch.stack((posterior[:, 1] * action, -posterior[:, 0] * action), -1))
    potential -= potential.mean(-1, keepdim=True)
    potential.masked_fill_(~valid[:, None], 0.)
    result = base + operator.double() @ potential
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError('Nonfinite sparse reuse correction.')
    stats = {}
    if diagnostics:
        eligible = alias_valid & ~canonical & valid[:, None, None]
        stats = {'fine_forwards': 0., 'additional_visual_forwards': 0.,
            'contenders_per_query': 2., 'evaluated_directed_edges_per_valid_query': 2.,
            'dense_directed_edges_per_query': float(classes * (classes - 1)),
            'rejected_noncanonical_fraction': float(reject[eligible].double().mean()) if bool(eligible.any()) else 0.,
            'survivor_mass_max_error': float((allocation.sum(-1) - remaining).abs().max()),
            'canonical_weight_max_error': float((allocation[canonical] - 1).abs().max()),
            'mean_absolute_action': float(action[valid].abs().mean()) if bool(valid.any()) else 0.,
            'active_action_fraction': float((action[valid] != 0).double().mean()) if bool(valid.any()) else 0.,
            'maximum_sparse_pair_alias_elements': max(c[0].numel() for c in cached)}
    return result, stats, {'pairs': pairs, 'weights': allocation, 'keep': keep,
                            'action': action, 'potential': potential, 'base': base}
