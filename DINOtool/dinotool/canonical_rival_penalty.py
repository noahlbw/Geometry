"""Suppress visually active foreign-canonical components of local aliases.

The foreign direction is orthogonal to the parent canonical. This algebraic
constraint is not semantic ground truth; four paired arms test its utility.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .fixed20_task_readout import POLICIES


IMPLEMENTATION = 'geometry-original20-canonical-rival-component-penalty-v1-20261008'
BASE = 'Fixed20_TaskCoupled'
PRIMARY = 'CanonicalRival_Penalty'
POOLED = 'CanonicalRival_Pooled'
SHUFFLED = 'CanonicalRival_AliasShuffle'
METHODS = (BASE, PRIMARY, POOLED, SHUFFLED)
SEED = 20261008


@dataclass(frozen=True)
class RivalPlan:
    members: torch.Tensor
    canonical: torch.Tensor
    protected: torch.Tensor
    rival: torch.Tensor
    own_rival_cosine: torch.Tensor
    norm: torch.Tensor
    alpha: torch.Tensor
    permutation: torch.Tensor


@torch.inference_mode()
def prepare_plan(bank):
    text = F.normalize(bank.features.float(), dim=-1)
    members = torch.stack([(bank.parent_indices == c).nonzero().flatten()
                          for c in range(bank.class_count)])
    if members.shape[1] != 20 or bank.class_count < 2:
        raise ValueError('Fixed twenty aliases and at least two classes required.')
    protected = bank.canonical_mask[members]
    if not bool((protected.sum(-1) == 1).all()):
        raise ValueError('Exactly one protected canonical per class required.')
    canonical = members[protected].reshape(-1)
    anchors = text[canonical]
    own = anchors[:, None].expand(-1, 20, -1)
    # Setup-only [alias,class] matrix. No pixel/alias/class tensor is built.
    similarity = text @ anchors.T
    similarity.scatter_(1, bank.parent_indices[:, None], -torch.inf)
    rival = similarity.argmax(-1)[members]
    foreign = anchors[rival]
    dot = (own*foreign).sum(-1)
    direction = foreign-dot[..., None]*own
    norm = direction.norm(dim=-1)
    unit = direction/norm.clamp_min(1e-6)[..., None]
    alpha = (text[members]*unit).sum(-1).clamp_min(0.)
    alpha.masked_fill_(protected | (norm <= 1e-6), 0.)
    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.arange(20, device=text.device).expand_as(members).clone()
    for c in range(bank.class_count):
        ids = (~protected[c]).nonzero().flatten()
        permutation[c, ids] = ids[torch.randperm(len(ids), generator=generator).to(ids.device)]
    return RivalPlan(members, canonical, protected, rival, dot,
                     norm.clamp_min(1e-6), alpha, permutation)


def penalties(alias_cosine, plan):
    canonical = alias_cosine[:, plan.canonical]
    own = canonical[..., None]
    foreign = canonical[:, plan.rival]
    active = ((foreign-plan.own_rival_cosine*own)/plan.norm).clamp_min(0.)
    return active*plan.alpha


def class_logits(cosine, plan, penalty=None):
    slots = cosine[:, plan.members]
    if penalty is not None:
        slots = slots-penalty
    # Retain the historical alias_class_scores multiply/divide rounding.
    return (.07*((slots/.07).logsumexp(-1)-math.log(20)))/.07


def tile_scores(cosine, broad, operator, plan, gain, methods=METHODS):
    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared paired arms required.')
    local = class_logits(cosine, plan)
    baseline = local.double()+gain*(operator.double()@(broad.double()-local.double()))
    result = {BASE: baseline} if BASE in methods else {}
    stats = dict(mean_penalty_cosine=0., active_alias_fraction=0.,
                 canonical_penalty_max=0., maximum_penalty_elements=0.)
    if set(methods) == {BASE}:
        return result, stats
    q = penalties(cosine, plan)
    stats.update(mean_penalty_cosine=float(q.mean()), active_alias_fraction=float((q > 0).float().mean()),
                 canonical_penalty_max=float(q[:, plan.protected].abs().max()),
                 maximum_penalty_elements=q.numel())
    for method in methods:
        if method == BASE:
            continue
        use = q
        if method == POOLED:
            use = q.sum(-1, keepdim=True).expand_as(q)/19
            use = use.masked_fill(plan.protected[None], 0.)
        elif method == SHUFFLED:
            use = q.gather(-1, plan.permutation[None].expand_as(q))
        changed = class_logits(cosine, plan, use)
        delta = changed.double()-local.double()
        # Exact L' + g H(W-L'); the local correction passes through I-gH.
        result[method] = baseline+delta-gain*(operator.double()@delta)
    return result, stats


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, plans, dataset, *, methods=METHODS):
    from eval_development_readout import observations, wide_scores
    from eval_geometry_vip_reliability import sample_broad
    from .device_probability_accumulator import DeviceProbabilityAccumulator
    from .inference import hann_blend_window

    policy = POLICIES[dataset]
    source = observations(image, geometry, vip, (policy.strength,), wide_policy=policy.wide_policy)
    height, width = source['size']
    blend = torch.from_numpy(hann_blend_window(512)).to(geometry.device)
    predictions, diagnostics = {}, {}
    for name, bank in banks.items():
        broad = wide_scores(source, queries[name], policy.tau, policy.tem)
        text = F.normalize(bank.features.float(), dim=-1)
        fields = []
        totals = dict(mean_penalty_cosine=0., active_alias_fraction=0.,
                      canonical_penalty_max=0., maximum_penalty_elements=0.)
        for tile in source['local']:
            cosine = tile['features'][policy.strength].float()@text.T
            wide = sample_broad(broad, tile['top'], tile['left'], height, width).reshape(len(cosine), bank.class_count)
            values, stats = tile_scores(cosine, wide, tile['operator'], plans[name], policy.coupling, methods)
            if not all(bool(torch.isfinite(value).all()) for value in values.values()):
                raise RuntimeError('Nonfinite component-penalty prediction.')
            if stats['canonical_penalty_max'] != 0.:
                raise RuntimeError('Canonical penalty changed.')
            fields.append((tile, values))
            for key, value in stats.items():
                totals[key] = max(totals[key], value) if key.endswith('_max') or key.startswith('maximum_') else totals[key]+value
        predictions[name] = {}
        # Comparison arms restore sequentially; deploy only the requested arm.
        for method in methods:
            with DeviceProbabilityAccumulator(bank.class_count, height, width, geometry.device) as accumulator:
                for tile, values in fields:
                    dense = F.interpolate(values[method].T.reshape(1, bank.class_count, 32, 32), (512, 512),
                                          mode='bilinear', align_corners=False)[0]
                    top, left = tile['top'], tile['left']
                    ah, aw = min(512, height-top), min(512, width-left)
                    accumulator.add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], left, top)
                predictions[name][method] = accumulator.finalize_resized(source['output_size'])
        for key in ('mean_penalty_cosine', 'active_alias_fraction'):
            totals[key] /= len(fields)
        diagnostics[name] = dict(totals, geometry_encodings=len(source['local']),
            wide_encodings=len(source['wide']), fine_forwards=0, additional_visual_forwards=0,
            additional_semantic_heads=0, fixed_alias_slots=20, wide_aliases_unchanged=True)
    return predictions, diagnostics
