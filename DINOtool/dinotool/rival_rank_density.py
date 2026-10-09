"""Scale-free local rank support for the frozen alias-survivor allocation."""
from math import isqrt

import torch

from .fine_alias_view import CONFIG
from .fine_responsibility_audit import position_control
from .fine_responsibility_reader import responsibility_scores
from .rival_alias_fast import cache_observations
from .rival_competition_admission import posterior_potential, project_actions, sampled_classes
from .rival_conditional_density import (PRIMARY as OLD_DENSITY, density_responsibility,
    incremental_allocation, response_densities)
from .rival_fine_support import support_control
from .rival_matched_support import sampled_aliases
from .rival_responsibility_intersection import (ABSOLUTE as FINE_ONLY,
    protected_weights, sampled_log_responsibilities)
from .rival_survivor_redistribution import allocation_directed, redistribute


IMPLEMENTATION = 'geometry-referenced-tie-expanded-rank-neighborhood-alias-allocation-v1-20261005'
REPLAY = ('NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact')
PRIMARY = 'RivalRankDensity_Exact'
ONLY = 'RivalRankOnly_Exact'
OWN_ONLY = 'RivalRankOwnOnly_Exact'
REFERENCE_NULL = 'RivalRankReferenceShuffle_Exact'
SPATIAL = 'RivalRankPositionShuffle_Exact'
NO_HOLDOUT = 'RivalRankNoHoldout_Exact'
UNBALANCED = 'RivalRankUnbalanced_Exact'
UNIFORM = 'RivalRankUniformIncrement_Exact'
SHUFFLES = tuple('RivalRankFactorShuffle'+str(i)+'_Exact' for i in range(3))
METHODS = (*REPLAY, FINE_ONLY, OLD_DENSITY, PRIMARY, ONLY, OWN_ONLY,
    REFERENCE_NULL, SPATIAL, NO_HOLDOUT, UNBALANCED, UNIFORM, *SHUFFLES)


def rank_densities(alias, reference, valid, *, exclude_self=True, balance=True,
                   chunk=CONFIG.query_chunk):
    if (alias.ndim != 2 or reference.ndim != 2 or alias.shape[0] != len(valid)
            or reference.shape[0] != len(valid) or valid.dtype != torch.bool
            or not bool(torch.isfinite(alias).all() and torch.isfinite(reference).all())
            or bool((reference < 0).any()) or chunk < 1):
        raise ValueError('Finite alias responses, nonnegative references and Boolean validity required.')
    density = torch.zeros((*alias.shape, reference.shape[-1]), dtype=torch.float64, device=alias.device)
    available = torch.zeros_like(reference, dtype=torch.bool)
    ids = valid.nonzero().flatten()
    n = len(ids)
    if n < (2 if exclude_self else 1):
        return density, available
    neighbors = min(n, max(2, isqrt(n-1)+1))
    q = reference[ids].double()
    mass = q.sum(0, keepdim=True)-q if exclude_self else q.sum(0, keepdim=True).expand_as(q)
    known = mass > CONFIG.epsilon
    available[ids] = known
    for start in range(0, alias.shape[1], chunk):
        sl = slice(start, start+chunk)
        values, order = alias[ids, sl].double().T.contiguous().sort(dim=1, stable=True)
        first = torch.searchsorted(values, values, right=False)
        last = torch.searchsorted(values, values, right=True)
        middle = (first+last-1)//2
        low = (middle-neighbors//2).clamp(0, n-neighbors)
        high = low+neighbors-1
        low = torch.searchsorted(values, values.gather(1, low), right=False)
        high = torch.searchsorted(values, values.gather(1, high), right=True)
        ordered_q = q[order]
        prefix = torch.cat((torch.zeros_like(ordered_q[:, :1]), ordered_q.cumsum(1)), 1)
        # Include all boundary ties; exact ties must not depend on token order.
        total = prefix.gather(1, high[..., None].expand_as(ordered_q))
        total -= prefix.gather(1, low[..., None].expand_as(ordered_q))
        if exclude_self:
            total -= ordered_q
        total = total.clamp_min(0.)
        unsorted = torch.zeros_like(total).scatter_(1, order[..., None].expand_as(total), total).transpose(0, 1)
        complete = ((low == 0) & (high == n)).T
        complete = torch.zeros_like(complete).scatter_(0, order.T, complete)
        if balance:
            unsorted /= mass[:, None].clamp_min(CONFIG.epsilon)
            unsorted = torch.where(complete[..., None], torch.ones_like(unsorted), unsorted)
        else:
            unsorted = torch.where(complete[..., None], mass[:, None], unsorted)
        density[ids, sl] = unsorted.masked_fill(~known[:, None], 0.)
    if not bool(torch.isfinite(density).all()):
        raise RuntimeError('Nonfinite rank-neighborhood source.')
    return density, available


def rank_allocation(base, source, risk, parents, canonical, valid, members):
    result = incremental_allocation(base, source, risk, parents, canonical, valid, members)
    grouped = source[:, members]
    eligible = (risk[:, members] == 0) & ~torch.isin(members, canonical)[None, :, :, None]
    low = grouped.masked_fill(~eligible, torch.inf).amin(2)
    high = grouped.masked_fill(~eligible, -torch.inf).amax(2)
    neutral = (low == high) | (eligible.sum(2) == 0)
    result[:, members] = torch.where(neutral[:, :, None], base[:, members], result[:, members])
    return result


@torch.inference_mode()
def rank_scores(local, operator, broad, wide, wide_count, fine, fine_count,
                coordinates, fine_coordinates, valid, members, canonical, parents,
                image_size, *, methods=METHODS):
    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Nonempty declared rank-source endpoints required.')
    inputs = (local, operator, broad, wide, wide_count, fine, fine_count, coordinates,
        fine_coordinates, valid, members, canonical, parents, image_size)
    values, risk, _ = responsibility_scores(*inputs, methods=(*REPLAY, FINE_ONLY))
    observed_wide = cache_observations(wide, wide_count, coordinates, image_size, members)
    observed_fine = cache_observations(fine, fine_count, fine_coordinates, (512, 512), members)
    fine_pair, _ = sampled_log_responsibilities(observed_fine, valid)
    base_weights = redistribute(protected_weights(fine_pair, risk, parents, canonical, valid), risk, members, canonical)
    reference = local.double().softmax(-1)
    alias = sampled_aliases(observed_fine, fine_coordinates)
    density, available = rank_densities(alias, reference, valid)
    source, source_known = density_responsibility(density, available, parents)
    weights = rank_allocation(base_weights, source, risk, parents, canonical, valid, members)
    allocations = {PRIMARY: weights, UNIFORM: base_weights}
    if OLD_DENSITY in methods:
        old, known = response_densities(alias, reference, valid)
        old_source, _ = density_responsibility(old, known, parents)
        allocations[OLD_DENSITY] = incremental_allocation(base_weights, old_source, risk, parents, canonical, valid, members)
    if ONLY in methods:
        allocations[ONLY] = rank_allocation((risk == 0).double(), source, risk, parents, canonical, valid, members)
    if OWN_ONLY in methods:
        own = density.gather(-1, parents[None, :, None].expand(len(valid), -1, 1))
        own = torch.where(available[:, parents, None], own, .5).clamp_min(CONFIG.epsilon)
        allocations[OWN_ONLY] = rank_allocation(base_weights, own.expand_as(source), risk,
            parents, canonical, valid, members)
    for name, options, teacher in (
            (REFERENCE_NULL, {}, position_control(reference, valid)),
            (NO_HOLDOUT, {'exclude_self': False}, reference),
            (UNBALANCED, {'balance': False}, reference)):
        if name in methods:
            field, known = rank_densities(alias, teacher, valid, **options)
            changed, _ = density_responsibility(field, known, parents)
            allocations[name] = rank_allocation(base_weights, changed, risk, parents, canonical, valid, members)
    if SPATIAL in methods:
        allocations[SPATIAL] = rank_allocation(base_weights, position_control(source, valid), risk,
            parents, canonical, valid, members)
    protected = protected_weights(source.log(), risk, parents, canonical, valid)
    spectrum_errors = {}
    for i, name in enumerate(SHUFFLES):
        if name in methods:
            factor = support_control(protected, risk, members, canonical, CONFIG.random_seed+i)
            spectrum_errors[name] = float((factor[:, members].sort(2).values-protected[:, members].sort(2).values).abs().max())
            if spectrum_errors[name] != 0:
                raise RuntimeError('Incremental factor identity-null spectrum changed.')
            allocations[name] = rank_allocation(base_weights, factor, risk, parents, canonical, valid, members)
    keep = risk == 0
    eligible = valid[:, None, None] & keep
    eligible &= ~torch.isin(torch.arange(len(parents), device=parents.device), canonical)[None, :, None]
    eligible &= parents[None, :, None] != torch.arange(len(members), device=parents.device)[None, None]
    base = values[REPLAY[0]]
    posterior = base.softmax(-1)
    innovation = sampled_classes(observed_fine, fine_coordinates).double()-broad.double()
    target = (innovation[:, :, None]-innovation[:, None]).masked_fill(~valid[:, None, None], 0.)
    diagnostics = {'additional_visual_forwards': 0, 'canonical_weight_min': float(weights[:, canonical].min()),
        'old_rejected_weights_max': float(weights[~keep].abs().max()) if bool((~keep).any()) else 0.,
        'all_mass_max_errors': {}, 'control_spectrum_max_errors': spectrum_errors,
        'identity_null_preserves_fine_only_source': True,
        'eligible_source_comparisons': int(eligible.sum()),
        'available_source_comparisons': int((eligible & source_known).sum()),
        'mean_source_sum': float(source[eligible].sum()),
        'source_floor_comparisons': int((eligible & (source <= CONFIG.epsilon)).sum()),
        'neighbor_count_before_tie_expansion': min(int(valid.sum()), max(2, isqrt(max(int(valid.sum())-1, 0))+1)),
        'affine_invariant_source': True, 'class_reference_is_not_correctness': True}
    original_mass = keep[:, members].sum(2).double()
    for name, allocation in allocations.items():
        if name not in methods:
            continue
        error = float((allocation[:, members].sum(2)-original_mass).abs().max())
        diagnostics['all_mass_max_errors'][name] = error
        if error > 1e-10 or bool((allocation[:, canonical] != 1).any()) or bool((allocation[~keep] != 0).any()):
            raise RuntimeError('Rank source changed frozen hard/canonical/survivor mass.')
        directed = allocation_directed(observed_wide, members, allocation, keep, valid)
        action = project_actions(directed-directed.transpose(-1, -2), target)
        potential, _ = posterior_potential(action, posterior, valid)
        values[name] = base+operator.double()@potential
        if name == UNIFORM:
            diagnostics['uniform_fine_only_max_error'] = float((values[name]-values[FINE_ONLY]).abs().max())
            if diagnostics['uniform_fine_only_max_error'] != 0:
                raise RuntimeError('Neutral rank increment must restore fine-only exactly.')
    values = {name: values[name] for name in methods}
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite rank-source scores.')
    return values, diagnostics
