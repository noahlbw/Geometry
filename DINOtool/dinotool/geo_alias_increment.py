"""Fixed-anchor rival discrimination and cardinality-free alias increments."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .native_alias_noise import signed_potential
from .stratified_soft_alias import crop_stencil
from .target_context_alias import alias_permutations


IMPLEMENTATION = 'geometry-canonical-witness-alias-increment-v1-20261003'
PRIMARY = 'GeoAliasIncrement_Exact'
COUNTS = (20, 30, 40)
REPLAY = ('Geometry', 'AllCount_Exact', 'RivalFineHard_Exact', 'Fixed20_Exact', 'Fixed20Hard_Exact')
NEW = ('IncrementAll_Exact', 'IncrementFine_Exact', 'IncrementText_Exact',
       'IncrementVisual_Exact', PRIMARY, 'IncrementShuffle_Exact')
METHODS = (*REPLAY, *NEW)


@dataclass(frozen=True)
class IncrementConfig:
    base_count: int = 20
    canonical_temperature: float = .07
    epsilon: float = 1e-6
    beta: float = 1.
    random_seed: int = 20261003
    query_chunk: int = 128


CONFIG = IncrementConfig()
GATE = {'minimum_mean_gain_vs_fixed20hard_pp': .1, 'minimum_domain_wins_vs_fixed20hard': 5,
        'maximum_protocol_loss_vs_fixed20hard_pp': 1., 'mean_above_all_and_shuffle_required': True,
        'no_automatic_full_rollout': True}


def canonical_positions(members, canonical):
    match = members == canonical[:, None]
    if not bool((match.sum(-1) == 1).all()):
        raise ValueError('Canonical aliases must belong to their own class.')
    return match.long().argmax(-1)


def sample_raw(crops, count, coordinates, image_size):
    result = coordinates.new_zeros(len(coordinates), crops[0].alias_logits.shape[-1])
    for crop in crops:
        ids, coeff = crop_stencil(crop, count, coordinates, image_size)
        result += (crop.alias_logits[ids] * coeff[..., None]).sum(1)
    return result


def fixed_profile(crop, members, config=CONFIG):
    if members.shape[-1] < config.base_count:
        raise ValueError('The historical20 prefix is required.')
    salience = crop.salience[members].float()
    base = salience[:, :config.base_count]
    # Added candidates never change the historical denominator or old factors.
    factors = config.base_count * (salience-base.logsumexp(-1, keepdim=True)).clamp(-80, 0).exp()
    factors[:, :config.base_count] = config.base_count * base.softmax(-1)
    return crop.alias_logits[:, members].float() * factors


def text_discrimination(features, members, canonical, config=CONFIG):
    text = F.normalize(features.float(), dim=-1)
    anchors = text[canonical]
    own = (text[members] * anchors[:, None]).sum(-1)
    rival = text[members] @ anchors.T
    separation = (1-anchors @ anchors.T).clamp_min(config.epsilon)
    return ((own[..., None]-rival)/separation[:, None]).clamp(0, 1)


def witness_fields(relation, local_canonical, fine_raw, canonical, valid, config=CONFIG):
    fine_canonical = fine_raw[:, canonical]
    lp = (local_canonical/config.canonical_temperature).softmax(-1)
    fp = (fine_canonical/config.canonical_temperature).softmax(-1)
    lv, li = lp.topk(2, -1)
    fv, fi = fp.topk(2, -1)
    confidence = ((lv[:, 0]-lv[:, 1]) * (fv[:, 0]-fv[:, 1])).clamp_min(0).sqrt()
    confidence *= (li[:, 0] == fi[:, 0]) & valid
    quality = fine_raw.new_zeros(len(valid), len(canonical)).scatter_(1, li[:, :1], confidence[:, None])
    retrieval = relation.float().masked_fill(~valid[None], 0.).clone()
    retrieval.fill_diagonal_(0.)
    retrieval /= retrieval.sum(-1, keepdim=True).clamp_min(config.epsilon)
    means = fine_raw.new_zeros(len(valid), len(canonical), fine_raw.shape[-1])
    mass = fine_raw.new_zeros(len(valid), len(canonical))
    for c in range(len(canonical)):
        ids = (quality[:, c] > 0).nonzero().flatten()
        if len(ids):
            weights = retrieval[:, ids] * quality[ids, c]
            mass[:, c] = weights.sum(-1)
            means[:, c] = (weights @ fine_raw[ids])/mass[:, c, None].clamp_min(config.epsilon)
    return means, mass, quality


def visual_discrimination(means, mass, members, canonical, valid, config=CONFIG):
    classes = len(members)
    own = means[:, torch.arange(classes, device=means.device)[:, None], members]
    rival = means[:, :, members].permute(0, 2, 3, 1)
    anchor_own = means[:, torch.arange(classes, device=means.device), canonical]
    anchor_rival = means[:, :, canonical].transpose(1, 2)
    separation = anchor_own[..., None]-anchor_rival
    known = (mass[:, :, None] > config.epsilon) & (mass[:, None, :] > config.epsilon)
    known &= separation > config.epsilon
    known &= valid[:, None, None]
    known &= ~torch.eye(classes, device=means.device, dtype=torch.bool)[None]
    scores = ((own[..., None]-rival)/separation[:, :, None].clamp_min(config.epsilon)).clamp(0, 1)
    return scores.masked_fill(~known[:, :, None], 0.), known


def anchored_veto(wide, wide_count, fine, fine_count, coordinates, image_size,
                  members, canonical, valid, config=CONFIG):
    positions = canonical_positions(members, canonical)
    observations = []
    for crops, count, size in ((wide, wide_count, image_size), (fine, fine_count, (512, 512))):
        margins = coordinates.new_zeros(len(valid), len(members), members.shape[-1], len(members))
        for crop in crops:
            evidence = fixed_profile(crop, members, config)
            anchors = evidence[:, torch.arange(len(members), device=members.device), positions]
            values = evidence[..., None]-anchors[:, None, None]
            ids, coeff = crop_stencil(crop, count, coordinates, size)
            margins += (values[ids] * coeff[..., None, None, None]).sum(1)
        observations.append(margins)
    veto = (observations[0] > 0) & (observations[1] < 0)
    veto[:, torch.arange(len(members), device=members.device), positions] = False
    return (~veto).float().masked_fill(~valid[:, None, None, None], 0.)


def max_excess_delta(evidence, weights, canonical_pos, config=CONFIG):
    """A fixed20 budget; zero additions and duplicate excess cannot gain mass."""
    classes = evidence.shape[-2]
    anchor = evidence[..., torch.arange(classes, device=evidence.device), canonical_pos]
    excess = (evidence-anchor[..., None]).clamp_min(0)
    old = excess[..., :config.base_count].amax(-1)
    capacity = (evidence[..., :config.base_count].amax(-1)
                -evidence[..., :config.base_count].amin(-1)).clamp_min(1/config.beta)
    admitted = (excess[..., None] * weights).amax(-2)
    change = admitted-old[..., None]
    return capacity[..., None] * torch.tanh(change/capacity[..., None])


def increment_observation(crops, count, coordinates, image_size, members, canonical, weights,
                          known, config=CONFIG):
    output = torch.zeros(len(coordinates), len(members), len(members), device=coordinates.device, dtype=torch.float64)
    positions = canonical_positions(members, canonical)
    for start in range(0, len(coordinates), config.query_chunk):
        sl = slice(start, start+config.query_chunk)
        for crop in crops:
            ids, coeff = crop_stencil(crop, count, coordinates[sl], image_size)
            evidence = fixed_profile(crop, members, config)[ids].double()
            change = max_excess_delta(evidence, weights[sl, None].double(), positions, config)
            output[sl] += (change * coeff.double()[..., None, None]).sum(1)
    return output.masked_fill(~known, 0.)


def increment_predictions(baseline, operator, wide, count, coordinates, image_size, members,
                          canonical, text_gate, visual_gate, veto, known, valid, config=CONFIG):
    shape = visual_gate.shape
    joint = text_gate[None] * visual_gate * veto
    allocations = {'IncrementAll_Exact': torch.ones(shape, device=visual_gate.device),
                   'IncrementFine_Exact': veto,
                   'IncrementText_Exact': text_gate[None].expand(shape) * veto,
                   'IncrementVisual_Exact': visual_gate * veto, PRIMARY: joint}
    permutation = alias_permutations(members, canonical, config.random_seed)[0]
    allocations['IncrementShuffle_Exact'] = joint.gather(2, permutation[None, :, :, None].expand_as(joint))
    error = float((allocations['IncrementShuffle_Exact'].sort(2).values-joint.sort(2).values).abs().max())
    if error:
        raise RuntimeError('Shuffled weights changed their spectrum.')
    values, frozen, diagnostics = {}, {}, {'weight_shuffle_spectrum_error': error, 'sources': {}}
    for method, weights in allocations.items():
        directed = increment_observation(wide, count, coordinates, image_size, members, canonical, weights, known, config)
        potential, checks = signed_potential(directed, valid)
        values[method] = baseline.double()+operator.double() @ potential
        frozen[method+'__directed'] = directed.cpu().numpy()
        diagnostics['sources'][method] = {**checks, 'mean_abs_directed': float(directed[valid].abs().mean()),
            'positive_directed_fraction': float((directed[valid] > 0).double().mean()),
            'negative_directed_fraction': float((directed[valid] < 0).double().mean())}
    positions = canonical_positions(members, canonical)
    noncanonical = torch.ones_like(members, dtype=torch.bool).scatter_(1, positions[:, None], False)
    eligible = known[:, :, None].expand_as(joint) & noncanonical[None, :, :, None]
    diagnostics.update(reference_pair_available_fraction=float(known[valid].double().mean()),
        mean_joint_weight=float(joint[eligible].mean()) if bool(eligible.any()) else 0.,
        zero_joint_weight_fraction=float((joint[eligible] == 0).double().mean()) if bool(eligible.any()) else 1.)
    frozen['joint_weights'] = joint.cpu().numpy()
    frozen['known_pairs'] = known.cpu().numpy()
    return values, frozen, diagnostics
