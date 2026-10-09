"""Redistribute survivor weights without changing hard/canonical prior mass."""
import torch

from .fine_alias_view import CONFIG
from .native_query_alias import native_risk
from .rival_alias_fast import cache_observations, sampled_cached
from .rival_competition_admission import posterior_potential, project_actions, sampled_classes
from .rival_fine_support import support_weights, support_control
from .rival_matched_support import (PRIMARY as MATCHED, REPLAY as MATCHED_REPLAY,
    matched_scores, matched_weights, sampled_aliases)


IMPLEMENTATION = 'frozen-hard-survivor-canonical-mass-redistribution-v1-20261005'
PRIMARY = 'SurvivorMatched_Exact'
CLASS_MEAN = 'SurvivorClassMean_Exact'
ABSOLUTE = 'SurvivorAbsolute_Exact'
TEXT_ONLY = 'SurvivorTextOnly_Exact'
SHUFFLES = tuple('SurvivorShuffle'+str(i)+'_Exact' for i in range(3))
REPLAY = (*MATCHED_REPLAY, MATCHED)
NEW_METHODS = (PRIMARY, CLASS_MEAN, ABSOLUTE, TEXT_ONLY, *SHUFFLES)
METHODS = (*REPLAY, *NEW_METHODS)


def redistribute(weights, old_risk, members, canonical):
    if (weights.shape != old_risk.shape or weights.ndim != 3
            or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any())):
        raise ValueError('Matching bounded-support finite nonnegative allocations required.')
    grouped = weights[:, members].double()
    survivors = old_risk[:, members] == 0
    noncanonical = survivors & ~torch.isin(members, canonical)[None, :, :, None]
    if bool((grouped[noncanonical] <= 0).any()):
        raise ValueError('Every surviving noncanonical alias needs positive source weight.')
    count = noncanonical.sum(2, keepdim=True)
    source = grouped.masked_fill(~noncanonical, 0.)
    mass = source.sum(2, keepdim=True)
    allocation = source*count/mass.clamp_min(CONFIG.epsilon)
    low = grouped.masked_fill(~noncanonical, torch.inf).amin(2, keepdim=True)
    high = grouped.masked_fill(~noncanonical, -torch.inf).amax(2, keepdim=True)
    # A constant source supplies no word preference; preserve exact hard identity.
    uniform = (count == 0) | (low == high)
    allocation = torch.where(uniform, noncanonical.double(), allocation)
    output = survivors.double()
    output = torch.where(noncanonical, allocation, output)
    result = weights.new_zeros(weights.shape, dtype=torch.float64)
    result[:, members] = output
    return result


def allocation_directed(observations, members, weights, keep, valid,
                        beta=CONFIG.beta, chunk=CONFIG.query_chunk):
    if (weights.shape != (len(valid), members.numel(), len(members))
            or keep.shape != weights.shape or keep.dtype != torch.bool
            or beta <= 0 or chunk < 1 or not bool(torch.isfinite(weights).all())
            or bool((weights < 0).any()) or bool((weights[~keep] != 0).any())
            or bool((weights[keep] <= 0).any())):
        raise ValueError('Finite positive survivors, zero rejected weights and original hard support required.')
    grouped, original = weights[:, members].double(), keep[:, members]
    remaining = original.sum(2).double()
    if bool((remaining == 0).any()) or float((grouped.sum(2)-remaining).abs().max()) > 1e-10:
        raise ValueError('Original survivor mass must remain fixed.')
    result = torch.zeros((len(valid), len(members), len(members)), dtype=torch.float64, device=weights.device)
    k = members.shape[1]
    for start in range(0, len(valid), chunk):
        sl = slice(start, start+chunk)
        log_weights = grouped[sl].log()[:, None]
        normalizer = (k/remaining[sl]).log()[:, None]
        untouched = (grouped[sl] == 1).all(2)[:, None]
        for crop in observations:
            # Keep the historical beta-before-double and original crop interpolation.
            evidence = (beta*crop.evidence[crop.indices[sl]]).double()
            full = evidence.logsumexp(-1)
            changed = (evidence[..., None]+log_weights).logsumexp(3)
            delta = ((changed-full[..., None]+normalizer)/beta).masked_fill(untouched, 0.)
            result[sl] += (delta*crop.coefficients[sl].double()[..., None, None]).sum(1)
    return result.masked_fill(~valid[:, None, None], 0.)


@torch.inference_mode()
def redistribution_scores(local, operator, broad, wide, wide_count, fine, fine_count,
                          coordinates, fine_coordinates, valid, members, canonical,
                          parents, image_size, *, matches, methods=METHODS):
    if not set(methods).issubset(METHODS):
        raise ValueError('Declared survivor-redistribution endpoint required.')
    replay, _ = matched_scores(local, operator, broad, wide, wide_count, fine, fine_count,
        coordinates, fine_coordinates, valid, members, canonical, parents, image_size,
        matches=matches, methods=REPLAY)
    cached_wide = cache_observations(wide, wide_count, coordinates, image_size, members)
    cached_fine = cache_observations(fine, fine_count, fine_coordinates, (512, 512), members)
    fine_margin = sampled_cached(cached_fine, fine_coordinates).masked_fill(~valid[:, None, None], 0.)
    old_risk = native_risk(sampled_cached(cached_wide, coordinates), fine_margin, fine_margin,
                           parents, canonical, valid, CONFIG)
    raw = matched_weights(sampled_aliases(cached_fine, fine_coordinates), matches, old_risk, parents, canonical, valid)
    weights = redistribute(raw, old_risk, members, canonical)
    allocations = {PRIMARY: weights}
    if CLASS_MEAN in methods:
        allocations[CLASS_MEAN] = redistribute(support_control(raw, old_risk, members, canonical), old_risk, members, canonical)
    if ABSOLUTE in methods:
        allocations[ABSOLUTE] = redistribute(support_weights(fine_margin, old_risk, parents, canonical, valid),
                                             old_risk, members, canonical)
    if TEXT_ONLY in methods:
        source = matched_weights(sampled_aliases(cached_fine, fine_coordinates), matches, old_risk,
                                 parents, canonical, valid, text_only=True)
        allocations[TEXT_ONLY] = redistribute(source, old_risk, members, canonical)
    for i, name in enumerate(SHUFFLES):
        if name in methods:
            allocations[name] = support_control(weights, old_risk, members, canonical, CONFIG.random_seed+i)
    base = replay['NoAdmission_Exact']
    posterior = base.softmax(-1)
    innovation = sampled_classes(cached_fine, fine_coordinates).double()-broad.double()
    target = (innovation[:, :, None]-innovation[:, None]).masked_fill(~valid[:, None, None], 0.)
    keep = old_risk == 0
    original_mass = keep[:, members].sum(2).double()
    values = dict(replay)
    mask = valid[:, None, None] & keep
    mask &= ~torch.isin(torch.arange(len(parents), device=parents.device), canonical)[None, :, None]
    mask &= parents[None, :, None] != torch.arange(len(members), device=parents.device)[None, None]
    diagnostics = {'additional_visual_forwards': 0, 'canonical_weight_min': float(weights[:, canonical].min()),
        'old_rejected_weights_max': float(weights[~keep].abs().max()) if bool((~keep).any()) else 0.,
        'surviving_comparisons': int(mask.sum()),
        'surviving_weight_mean': float(weights[mask].mean()) if bool(mask.any()) else 1.,
        'boosted_fraction': float((weights[mask] > 1).double().mean()) if bool(mask.any()) else 0.,
        'attenuated_fraction': float((weights[mask] < 1).double().mean()) if bool(mask.any()) else 0.,
        'control_spectrum_max_errors': {}, 'all_mass_max_errors': {}, 'canonical_prior_max_errors': {}}
    for name, allocation in allocations.items():
        if name not in methods:
            continue
        mass = allocation[:, members].sum(2)
        mass_error = float((mass-original_mass).abs().max())
        canonical_error = float((mass.reciprocal()-original_mass.reciprocal()).abs().max())
        diagnostics['all_mass_max_errors'][name] = mass_error
        diagnostics['canonical_prior_max_errors'][name] = canonical_error
        if mass_error > 1e-10 or canonical_error > 1e-12 or bool((allocation[:, canonical] != 1).any()):
            raise RuntimeError('Survivor/canonical prior mass changed.')
        directed = allocation_directed(cached_wide, members, allocation, keep, valid)
        bounded = project_actions(directed-directed.transpose(-1, -2), target)
        potential, stats = posterior_potential(bounded, posterior, valid)
        values[name] = base+operator.double()@potential
        if name in SHUFFLES:
            error = float((allocation[:, members].sort(2).values-weights[:, members].sort(2).values).abs().max())
            diagnostics['control_spectrum_max_errors'][name] = error
            if error != 0:
                raise RuntimeError('Survivor alias-null spectrum changed.')
        if name == CLASS_MEAN:
            error = float((values[name]-replay['FineRivalProjected_Exact']).abs().max())
            diagnostics['class_mean_score_max_error'] = error
            if error != 0:
                raise RuntimeError('Mass-preserving class mean must exactly recover projected hard.')
        diagnostics[name] = stats
    values = {name: values[name] for name in methods}
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite survivor-redistribution scores.')
    return values, diagnostics
