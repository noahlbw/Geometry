"""Positive conditional alias response beyond a regularized rival text span."""
from dataclasses import dataclass

import torch

from .fine_alias_view import CONFIG
from .geo_alias_increment import sample_raw
from .native_query_alias import native_risk
from .rival_alias_fast import cache_observations, sampled_cached
from .rival_competition_admission import posterior_potential, project_actions, sampled_classes
from .rival_fine_support import support_control
from .rival_matched_support import semantic_rivals
from .rival_survivor_redistribution import (PRIMARY as SURVIVOR, allocation_directed,
    redistribute, redistribution_scores)


IMPLEMENTATION = 'frozen-rival-text-span-positive-residual-support-v1-20261005'
PRIMARY = 'RivalSubspaceSupport_Exact'
TEXT_ONLY = 'RivalSubspaceTextOnly_Exact'
CLASS_MEAN = 'RivalSubspaceClassMean_Exact'
SHUFFLES = tuple('RivalSubspaceShuffle'+str(i)+'_Exact' for i in range(3))
REPLAY = ('NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact', SURVIVOR)
METHODS = (*REPLAY, PRIMARY, TEXT_ONLY, CLASS_MEAN, *SHUFFLES)


@dataclass(frozen=True)
class RivalSubspace:
    coefficients: torch.Tensor
    template_mean_norms: torch.Tensor
    text_weights: torch.Tensor
    matches: object


def rival_subspaces(template_features, members):
    if (template_features.ndim != 3 or members.ndim != 2
            or members.numel() != len(template_features)
            or not torch.equal(members.flatten().sort().values,
                torch.arange(len(template_features), device=members.device))
            or not bool(torch.isfinite(template_features).all())):
        raise ValueError('Finite original template features and complete unique alias groups required.')
    means = template_features.double().mean(1)
    norms = means.norm(dim=-1)
    if bool((norms <= CONFIG.epsilon).any()):
        raise ValueError('Nonzero original-template alias means required.')
    text = means/norms[:, None]
    coefficients, text_weights = [], []
    for group in members:
        rival = text[group]
        gram = rival @ rival.T
        # Unit-norm queries and an identity prior give a stable, fixed ridge1 fit.
        coefficient = torch.linalg.solve(gram+torch.eye(len(group), device=gram.device,
            dtype=gram.dtype), rival @ text.T).T
        shared = coefficient @ rival
        residual = text-shared
        unique_energy, shared_energy = residual.square().sum(-1), shared.square().sum(-1)
        coefficients.append(coefficient)
        text_weights.append(unique_energy/(unique_energy+shared_energy).clamp_min(CONFIG.epsilon))
    return RivalSubspace(torch.stack(coefficients, 1), norms,
        torch.stack(text_weights, 1), semantic_rivals(template_features, members))


def subspace_weights(raw, source, old_risk, members, parents, canonical, valid, text_only=False):
    if (raw.shape != old_risk.shape[:2] or source.coefficients.shape != (*old_risk.shape[1:], members.shape[1])
            or source.template_mean_norms.shape != raw.shape[1:] or source.text_weights.shape != old_risk.shape[1:]
            or parents.shape != raw.shape[1:] or canonical.shape != old_risk.shape[2:]
            or valid.shape != raw.shape[:1] or not bool(torch.isfinite(raw).all())):
        raise ValueError('Matching raw original-template responses, rival spans and hard support required.')
    if text_only:
        weights = source.text_weights[None].expand_as(old_risk).clone()
    else:
        normalized = raw.double()/source.template_mean_norms
        shared = torch.einsum('nck,ack->nac', normalized[:, members], source.coefficients)
        unique = (normalized[..., None]-shared).clamp_min(0)
        positive_shared = shared.clamp_min(0)
        weights = unique/(unique+positive_shared).clamp_min(CONFIG.epsilon)
    weights = weights.clamp(CONFIG.epsilon, 1)
    weights.masked_fill_(old_risk > 0, 0.)
    weights[:, canonical] = 1.
    weights.scatter_(-1, parents[None, :, None].expand(len(raw), -1, 1), 1.)
    weights.masked_fill_(~valid[:, None, None], 1.)
    return weights


@torch.inference_mode()
def subspace_scores(local, operator, broad, wide, wide_count, fine, fine_count,
                    coordinates, fine_coordinates, valid, members, canonical, parents,
                    image_size, *, source, methods=METHODS):
    if not set(methods).issubset(METHODS):
        raise ValueError('Declared frozen rival-span endpoint required.')
    replay, _ = redistribution_scores(local, operator, broad, wide, wide_count, fine, fine_count,
        coordinates, fine_coordinates, valid, members, canonical, parents, image_size,
        matches=source.matches, methods=REPLAY)
    cached_wide = cache_observations(wide, wide_count, coordinates, image_size, members)
    cached_fine = cache_observations(fine, fine_count, fine_coordinates, (512, 512), members)
    native = sampled_cached(cached_fine, fine_coordinates).masked_fill(~valid[:, None, None], 0.)
    old_risk = native_risk(sampled_cached(cached_wide, coordinates), native, native,
        parents, canonical, valid, CONFIG)
    keep = old_risk == 0
    raw = sample_raw(fine, fine_count, fine_coordinates, (512, 512))
    source_weights = subspace_weights(raw, source, old_risk, members, parents, canonical, valid)
    weights = redistribute(source_weights, old_risk, members, canonical)
    allocations = {PRIMARY: weights}
    if TEXT_ONLY in methods:
        text = subspace_weights(raw, source, old_risk, members, parents, canonical, valid, text_only=True)
        allocations[TEXT_ONLY] = redistribute(text, old_risk, members, canonical)
    if CLASS_MEAN in methods:
        allocations[CLASS_MEAN] = keep.double()
    for i, name in enumerate(SHUFFLES):
        if name in methods:
            allocations[name] = support_control(weights, old_risk, members, canonical, CONFIG.random_seed+i)
    base = replay['NoAdmission_Exact']
    posterior = base.softmax(-1)
    innovation = sampled_classes(cached_fine, fine_coordinates).double()-broad.double()
    target = (innovation[:, :, None]-innovation[:, None]).masked_fill(~valid[:, None, None], 0.)
    eligible = valid[:, None, None] & keep
    eligible &= ~torch.isin(torch.arange(len(parents), device=parents.device), canonical)[None, :, None]
    eligible &= parents[None, :, None] != torch.arange(len(members), device=parents.device)[None, None]
    joint = eligible & (native >= 0) & (sampled_cached(cached_wide, coordinates) > 0)
    diagnostics = {'additional_visual_forwards': 0, 'canonical_weight_min': float(weights[:, canonical].min()),
        'old_rejected_weights_max': float(weights[~keep].abs().max()) if bool((~keep).any()) else 0.,
        'source_survivor_comparisons': int(eligible.sum()),
        'source_weight_sum': float(source_weights[eligible].sum()),
        'source_near_zero_comparisons': int((eligible & (source_weights <= CONFIG.epsilon)).sum()),
        'joint_positive_comparisons': int(joint.sum()), 'joint_positive_source_weight_sum': float(source_weights[joint].sum()),
        'all_mass_max_errors': {}, 'control_spectrum_max_errors': {}, 'ridge': 1.}
    original_mass = keep[:, members].sum(2).double()
    values = dict(replay)
    for name, allocation in allocations.items():
        if name not in methods:
            continue
        mass_error = float((allocation[:, members].sum(2)-original_mass).abs().max())
        diagnostics['all_mass_max_errors'][name] = mass_error
        if mass_error > 1e-10 or bool((allocation[:, canonical] != 1).any()):
            raise RuntimeError('Rival-span weighting changed survivor/canonical mass.')
        directed = allocation_directed(cached_wide, members, allocation, keep, valid)
        bounded = project_actions(directed-directed.transpose(-1, -2), target)
        potential, details = posterior_potential(bounded, posterior, valid)
        values[name] = base+operator.double()@potential
        diagnostics[name] = details
        if name in SHUFFLES:
            error = float((allocation[:, members].sort(2).values-weights[:, members].sort(2).values).abs().max())
            diagnostics['control_spectrum_max_errors'][name] = error
            if error != 0:
                raise RuntimeError('Rival-span weight-null spectrum changed.')
        if name == CLASS_MEAN:
            error = float((values[name]-replay['FineRivalProjected_Exact']).abs().max())
            diagnostics['class_mean_score_max_error'] = error
            if error != 0:
                raise RuntimeError('Rival-span class control must exactly restore projected hard.')
    values = {name: values[name] for name in methods}
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite rival-span scores.')
    return values, diagnostics
