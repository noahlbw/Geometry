"""Fixed-source counterfactuals for unscreened versus screened fine targets."""
import torch

from .fine_alias_view import CONFIG
from .native_query_alias import native_risk
from .rival_alias_fast import cache_observations, sampled_cached
from .rival_competition_admission import posterior_potential, project_actions, sampled_classes
from .rival_fine_support import support_control
from .rival_matched_support import matched_weights, sampled_aliases
from .rival_survivor_redistribution import (PRIMARY as SURVIVOR, redistribute,
    allocation_directed, redistribution_scores)


IMPLEMENTATION = 'frozen-survivor-projection-target-counterfactual-v1-20261005'
PRIMARY = 'SurvivorTargetMatched_Exact'
UNPROJECTED = 'SurvivorUnprojected_Exact'
CLASS_TARGET = 'SurvivorTargetClassMean_Exact'
CLASS_UNPROJECTED = 'SurvivorUnprojectedClassMean_Exact'
SHUFFLES = tuple('SurvivorTargetShuffle'+str(i)+'_Exact' for i in range(3))
REPLAY = ('NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact', SURVIVOR)
METHODS = (*REPLAY, PRIMARY, UNPROJECTED, CLASS_TARGET, CLASS_UNPROJECTED, *SHUFFLES)


def screened_target(target, fine_directed):
    if (target.ndim != 3 or target.shape[1] != target.shape[2] or target.shape != fine_directed.shape
            or not bool(torch.isfinite(target).all() and torch.isfinite(fine_directed).all())
            or not torch.allclose(target, -target.transpose(-1, -2), atol=1e-12, rtol=0)):
        raise ValueError('Matching finite original antisymmetric target and fine alias changes required.')
    return target+fine_directed-fine_directed.transpose(-1, -2)


def projection_statistics(requested, original, screened, posterior, valid):
    upper = torch.ones(requested.shape[1:], device=requested.device, dtype=torch.bool).triu(1)
    mask = valid[:, None, None] & upper[None]
    active = mask & (requested != 0)
    weights = posterior[:, :, None]*posterior[:, None]
    raw = requested.abs().masked_fill(~mask, 0.)
    old = original.abs().masked_fill(~mask, 0.)
    new = screened.abs().masked_fill(~mask, 0.)
    return {'active_pairs': int(active.sum()),
        'old_zeroed_pairs': int((active & (original == 0)).sum()),
        'screened_zeroed_pairs': int((active & (screened == 0)).sum()),
        'old_only_zeroed_pairs': int((active & (original == 0) & (screened != 0)).sum()),
        'screened_only_zeroed_pairs': int((active & (original != 0) & (screened == 0)).sum()),
        'requested_absolute_mass': float(raw.sum()), 'old_absolute_mass': float(old.sum()),
        'screened_absolute_mass': float(new.sum()),
        'requested_posterior_weighted_mass': float((raw*weights).sum()),
        'old_posterior_weighted_mass': float((old*weights).sum()),
        'screened_posterior_weighted_mass': float((new*weights).sum())}


@torch.inference_mode()
def target_audit_scores(local, operator, broad, wide, wide_count, fine, fine_count,
                        coordinates, fine_coordinates, valid, members, canonical, parents,
                        image_size, *, matches, methods=METHODS):
    if not set(methods).issubset(METHODS):
        raise ValueError('Declared fixed-source projection audit endpoint required.')
    replay, _ = redistribution_scores(local, operator, broad, wide, wide_count, fine, fine_count,
        coordinates, fine_coordinates, valid, members, canonical, parents, image_size,
        matches=matches, methods=REPLAY)
    cached_wide = cache_observations(wide, wide_count, coordinates, image_size, members)
    cached_fine = cache_observations(fine, fine_count, fine_coordinates, (512, 512), members)
    fine_margin = sampled_cached(cached_fine, fine_coordinates).masked_fill(~valid[:, None, None], 0.)
    old_risk = native_risk(sampled_cached(cached_wide, coordinates), fine_margin, fine_margin,
                           parents, canonical, valid, CONFIG)
    raw = matched_weights(sampled_aliases(cached_fine, fine_coordinates), matches, old_risk, parents, canonical, valid)
    primary_weights = redistribute(raw, old_risk, members, canonical)
    keep = old_risk == 0
    allocations = {PRIMARY: primary_weights}
    if CLASS_TARGET in methods or CLASS_UNPROJECTED in methods:
        allocations[CLASS_TARGET] = keep.double()
    for i, name in enumerate(SHUFFLES):
        if name in methods:
            allocations[name] = support_control(primary_weights, old_risk, members, canonical, CONFIG.random_seed+i)
    base = replay['NoAdmission_Exact']
    posterior = base.softmax(-1)
    innovation = sampled_classes(cached_fine, fine_coordinates).double()-broad.double()
    target = (innovation[:, :, None]-innovation[:, None]).masked_fill(~valid[:, None, None], 0.)
    original_mass = keep[:, members].sum(2).double()
    values = dict(replay)
    diagnostics = {'additional_visual_forwards': 0, 'canonical_weight_min': float(primary_weights[:, canonical].min()),
        'old_rejected_weights_max': float(primary_weights[~keep].abs().max()) if bool((~keep).any()) else 0.,
        'all_mass_max_errors': {}, 'control_spectrum_max_errors': {}, 'projection': {}}
    for name, allocation in allocations.items():
        error = float((allocation[:, members].sum(2)-original_mass).abs().max())
        diagnostics['all_mass_max_errors'][name] = error
        if error > 1e-10 or bool((allocation[:, canonical] != 1).any()):
            raise RuntimeError('Projection audit changed original survivor/canonical allocation mass.')
        wide_change = allocation_directed(cached_wide, members, allocation, keep, valid)
        fine_change = allocation_directed(cached_fine, members, allocation, keep, valid)
        requested = wide_change-wide_change.transpose(-1, -2)
        adjusted_target = screened_target(target, fine_change)
        old_projected = project_actions(requested, target)
        new_projected = project_actions(requested, adjusted_target)
        projected_potential, stats = posterior_potential(new_projected, posterior, valid)
        if name in methods:
            values[name] = base+operator.double()@projected_potential
        if name == PRIMARY:
            original_potential, _ = posterior_potential(old_projected, posterior, valid)
            old_scores = base+operator.double()@original_potential
            if not torch.equal(old_scores, replay[SURVIVOR]):
                raise RuntimeError('Original survivor score replay changed.')
            diagnostics['original_survivor_score_max_error'] = 0.
            if UNPROJECTED in methods:
                potential, _ = posterior_potential(requested, posterior, valid)
                values[UNPROJECTED] = base+operator.double()@potential
        if name == CLASS_TARGET and CLASS_UNPROJECTED in methods:
            potential, _ = posterior_potential(requested, posterior, valid)
            values[CLASS_UNPROJECTED] = base+operator.double()@potential
        if name in SHUFFLES:
            error = float((allocation[:, members].sort(2).values-primary_weights[:, members].sort(2).values).abs().max())
            diagnostics['control_spectrum_max_errors'][name] = error
            if error != 0:
                raise RuntimeError('Projection audit alias-null spectrum changed.')
        diagnostics['projection'][name] = projection_statistics(requested, old_projected, new_projected, posterior, valid)
        diagnostics[name] = stats
    values = {name: values[name] for name in methods}
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite projection counterfactual scores.')
    return values, diagnostics
