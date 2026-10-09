"""Conditional soft admission against the nearest semantic rival alias."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .fine_alias_view import CONFIG
from .native_query_alias import native_risk
from .rival_alias_fast import cache_observations, directed_cached, sampled_cached
from .rival_competition_admission import posterior_potential, project_actions, sampled_classes
from .rival_fine_support import PRIMARY as OLD_SUPPORT, support_control, support_scores


IMPLEMENTATION = 'frozen-semantic-matched-rival-alias-support-v1-20261005'
PRIMARY = 'RivalMatchedSupport_Exact'
CLASS_MEAN = 'RivalMatchedClassMean_Exact'
TEXT_ONLY = 'RivalMatchedTextOnly_Exact'
SHUFFLES = tuple('RivalMatchedShuffle'+str(i)+'_Exact' for i in range(3))
REPLAY = ('NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact', OLD_SUPPORT)
NEW_METHODS = (PRIMARY, CLASS_MEAN, TEXT_ONLY, *SHUFFLES)
METHODS = (*REPLAY, *NEW_METHODS)


@dataclass(frozen=True)
class RivalMatches:
    indices: torch.Tensor
    affinity: torch.Tensor


def semantic_rivals(template_features, members):
    if (template_features.ndim != 3 or members.ndim != 2
            or members.numel() != len(template_features)
            or not bool(torch.isfinite(template_features).all())):
        raise ValueError('Finite original alias/template features and complete class groups required.')
    text = F.normalize(template_features.float().mean(1), dim=-1)
    similarities = text @ text.T
    grouped = similarities[:, members]
    affinity, positions = grouped.max(-1)
    indices = members.gather(1, positions.T).T
    return RivalMatches(indices, affinity.clamp(0, 1).double())


def sampled_aliases(observations, coordinates):
    result = torch.zeros((len(coordinates), observations[0].evidence.shape[1]*observations[0].evidence.shape[2]),
                         device=coordinates.device, dtype=torch.float64)
    for crop in observations:
        scores = crop.evidence.flatten(1)[crop.indices]
        result += (scores.double()*crop.coefficients.double()[..., None]).sum(1)
    return result


def matched_weights(native, matches, old_risk, parents, canonical, valid, text_only=False):
    if (native.shape != old_risk.shape[:2] or matches.indices.shape != old_risk.shape[1:]
            or matches.affinity.shape != matches.indices.shape
            or parents.shape != native.shape[1:] or canonical.shape != old_risk.shape[2:]
            or valid.shape != native.shape[:1]
            or not bool(torch.isfinite(native).all() and torch.isfinite(matches.affinity).all())
            or bool(((matches.affinity < 0) | (matches.affinity > 1)).any())):
        raise ValueError('Matching finite native alias scores, semantic rivals and hard support required.')
    if text_only:
        competition = torch.full_like(old_risk, .5, dtype=torch.float64)
    else:
        # Two-phrase response allocation, not calibrated semantic correctness.
        margins = native.double()[..., None]-native.double()[:, matches.indices]
        competition = (-CONFIG.beta*margins).sigmoid()
    weights = (1-matches.affinity[None]*competition).clamp_min(CONFIG.epsilon)
    weights.masked_fill_(old_risk > 0, 0.)
    weights[:, canonical] = 1.
    weights.scatter_(-1, parents[None, :, None].expand(len(native), -1, 1), 1.)
    weights.masked_fill_(~valid[:, None, None], 1.)
    return weights


@torch.inference_mode()
def matched_scores(local, operator, broad, wide, wide_count, fine, fine_count,
                   coordinates, fine_coordinates, valid, members, canonical, parents,
                   image_size, *, matches, methods=METHODS):
    if not set(methods).issubset(METHODS):
        raise ValueError('Declared matched-rival endpoint required.')
    replay, _ = support_scores(local, operator, broad, wide, wide_count, fine, fine_count,
        coordinates, fine_coordinates, valid, members, canonical, parents, image_size,
        methods=REPLAY)
    cached_wide = cache_observations(wide, wide_count, coordinates, image_size, members)
    cached_fine = cache_observations(fine, fine_count, fine_coordinates, (512, 512), members)
    native_margin = sampled_cached(cached_fine, fine_coordinates).masked_fill(~valid[:, None, None], 0.)
    wide_margin = sampled_cached(cached_wide, coordinates)
    old_risk = native_risk(wide_margin, native_margin, native_margin,
                           parents, canonical, valid, CONFIG)
    native = sampled_aliases(cached_fine, fine_coordinates)
    weights = matched_weights(native, matches, old_risk, parents, canonical, valid)
    allocations = {PRIMARY: weights}
    if CLASS_MEAN in methods:
        allocations[CLASS_MEAN] = support_control(weights, old_risk, members, canonical)
    if TEXT_ONLY in methods:
        allocations[TEXT_ONLY] = matched_weights(native, matches, old_risk, parents, canonical, valid, text_only=True)
    for i, name in enumerate(SHUFFLES):
        if name in methods:
            allocations[name] = support_control(weights, old_risk, members, canonical, CONFIG.random_seed+i)
    base = replay['NoAdmission_Exact']
    posterior = base.softmax(-1)
    innovation = sampled_classes(cached_fine, fine_coordinates).double()-broad.double()
    target = (innovation[:, :, None]-innovation[:, None]).masked_fill(~valid[:, None, None], 0.)
    values = dict(replay)
    mask = valid[:, None, None] & (old_risk == 0)
    mask &= ~torch.isin(torch.arange(len(parents), device=parents.device), canonical)[None, :, None]
    mask &= parents[None, :, None] != torch.arange(len(members), device=parents.device)[None, None]
    joint = mask & (native_margin >= 0) & (wide_margin > 0)
    diagnostics = {'additional_visual_forwards': 0, 'canonical_weight_min': float(weights[:, canonical].min()),
        'old_rejected_weights_max': float(weights[old_risk > 0].abs().max()) if bool((old_risk > 0).any()) else 0.,
        'surviving_comparisons': int(mask.sum()),
        'surviving_noncanonical_weight_mean': float(weights[mask].mean()) if bool(mask.any()) else 1.,
        'joint_positive_comparisons': int(joint.sum()),
        'joint_positive_weight_mean': float(weights[joint].mean()) if bool(joint.any()) else 1.,
        'control_spectrum_max_errors': {}}
    for name, allocation in allocations.items():
        if name not in methods:
            continue
        directed = directed_cached(cached_wide, members, 1-allocation, valid, 'weighted')
        requested = directed-directed.transpose(-1, -2)
        bounded = project_actions(requested, target)
        potential, stats = posterior_potential(bounded, posterior, valid)
        values[name] = base+operator.double()@potential
        if name in SHUFFLES:
            error = float((allocation[:, members].sort(2).values-weights[:, members].sort(2).values).abs().max())
            diagnostics['control_spectrum_max_errors'][name] = error
            if error != 0:
                raise RuntimeError('Matched-rival alias-null spectrum changed.')
        if name == CLASS_MEAN:
            error = float((allocation[:, members].sum(2)-weights[:, members].sum(2)).abs().max())
            diagnostics['class_mean_weight_mass_max_error'] = error
            if error > 1e-10:
                raise RuntimeError('Matched-rival class control mass changed.')
        diagnostics[name] = stats
    values = {name: values[name] for name in methods}
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite semantic matched-rival scores.')
    return values, diagnostics
