"""Soft rival allocation from held-out scalar alias-response densities."""
import torch

from .fine_alias_view import CONFIG
from .fine_responsibility_audit import position_control
from .native_query_alias import native_risk
from .rival_alias_fast import cache_observations, sampled_cached
from .rival_competition_admission import (posterior_potential, project_actions,
    retained_competition_scores, sampled_classes)
from .rival_fine_support import support_control
from .rival_matched_support import sampled_aliases
from .rival_responsibility_intersection import (ABSOLUTE as FINE_ONLY,
    protected_weights, sampled_log_responsibilities)
from .rival_survivor_redistribution import allocation_directed, redistribute


IMPLEMENTATION = 'geometry-referenced-leave-query-out-alias-response-density-v1-20261005'
REPLAY = ('NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact')
PRIMARY = 'RivalConditionalDensity_Exact'
ONLY = 'RivalDensityOnly_Exact'
REFERENCE_NULL = 'RivalDensityReferenceShuffle_Exact'
NO_HOLDOUT = 'RivalDensityNoHoldout_Exact'
UNBALANCED = 'RivalDensityUnbalanced_Exact'
SPATIAL = 'RivalDensityPositionShuffle_Exact'
UNIFORM = 'RivalDensityUniformIncrement_Exact'
SHUFFLES = tuple('RivalDensityAliasShuffle'+str(i)+'_Exact' for i in range(3))
METHODS = (*REPLAY, FINE_ONLY, PRIMARY, ONLY, REFERENCE_NULL, NO_HOLDOUT,
    UNBALANCED, SPATIAL, UNIFORM, *SHUFFLES)


def response_densities(alias, reference, valid, *, exclude_self=True, balance=True,
                       beta=CONFIG.beta, chunk=CONFIG.query_chunk):
    if (alias.ndim != 2 or reference.ndim != 2 or alias.shape[0] != len(valid)
            or reference.shape[0] != len(valid) or valid.dtype != torch.bool
            or not bool(torch.isfinite(alias).all() and torch.isfinite(reference).all())
            or bool((reference < 0).any()) or beta <= 0 or chunk < 1):
        raise ValueError('Finite scalar alias responses, nonnegative class references and validity required.')
    density = torch.zeros((*alias.shape, reference.shape[-1]), dtype=torch.float64, device=alias.device)
    available = torch.zeros_like(reference, dtype=torch.bool)
    ids = valid.nonzero().flatten()
    if len(ids) < (2 if exclude_self else 1):
        return density, available
    q = reference[ids].double()
    mass = q.sum(0, keepdim=True)-q if exclude_self else q.sum(0, keepdim=True).expand_as(q)
    known = mass > CONFIG.epsilon
    available[ids] = known
    for start in range(0, alias.shape[1], chunk):
        sl = slice(start, start+chunk)
        value, order = (beta*alias[ids, sl].double()).sort(dim=0, stable=True)
        t, positions = value.T, order.T
        log_q = q[positions].log()
        left = (log_q+t[..., None]).logcumsumexp(1)
        right = (log_q-t[..., None]).flip(1).logcumsumexp(1).flip(1)
        # Exclude self before summing, avoiding cancellation of dominant rows.
        if exclude_self:
            empty = torch.full_like(left[:, :1], -torch.inf)
            left = torch.cat((empty, left[:, :-1]), 1)
            right = torch.cat((right[:, 1:], empty), 1)
            total = (left-t[..., None]).exp()+(right+t[..., None]).exp()
        else:
            total = ((left-t[..., None]).exp()+(right+t[..., None]).exp()-q[positions]).clamp_min(0.)
        unsorted = torch.zeros_like(total).scatter_(1, positions[..., None].expand_as(total), total).transpose(0, 1)
        if balance:
            unsorted = unsorted/mass[:, None].clamp_min(CONFIG.epsilon)
        density[ids, sl] = unsorted.masked_fill(~known[:, None], 0.)
    if not bool(torch.isfinite(density).all()):
        raise RuntimeError('Nonfinite sorted response-density source.')
    return density, available


def density_responsibility(density, available, parents):
    if (density.ndim != 3 or available.shape != (density.shape[0], density.shape[2])
            or parents.shape != density.shape[1:2] or available.dtype != torch.bool
            or not bool(torch.isfinite(density).all()) or bool((density < 0).any())):
        raise ValueError('Matching finite nonnegative densities, parent classes and reference availability required.')
    own = density.gather(-1, parents[None, :, None].expand(len(density), -1, 1))
    total = own+density
    known = available[:, parents, None] & available[:, None] & (total > CONFIG.epsilon)
    source = torch.where(known, own/total.clamp_min(CONFIG.epsilon), .5).clamp_min(CONFIG.epsilon)
    return source, known


def incremental_allocation(base, source, risk, parents, canonical, valid, members):
    if base.shape != source.shape or base.shape != risk.shape:
        raise ValueError('Matching frozen allocation and incremental source required.')
    weights = base*source
    weights[:, canonical] = 1.
    weights.scatter_(-1, parents[None, :, None].expand(len(valid), -1, 1), 1.)
    weights.masked_fill_(~valid[:, None, None], 1.)
    return redistribute(weights, risk, members, canonical)


@torch.inference_mode()
def density_scores(local, operator, broad, wide, wide_count, fine, fine_count,
                   coordinates, fine_coordinates, valid, members, canonical, parents,
                   image_size, *, methods=METHODS):
    if not set(methods).issubset(METHODS):
        raise ValueError('Declared frozen density endpoint required.')
    replay, risk, _ = retained_competition_scores(local, operator, broad, wide, wide_count,
        fine, fine_count, coordinates, fine_coordinates, valid, members, canonical,
        parents, image_size, methods=REPLAY)
    observed_wide = cache_observations(wide, wide_count, coordinates, image_size, members)
    observed_fine = cache_observations(fine, fine_count, fine_coordinates, (512, 512), members)
    native = sampled_cached(observed_fine, fine_coordinates).masked_fill(~valid[:, None, None], 0.)
    expected = native_risk(sampled_cached(observed_wide, coordinates), native, native,
        parents, canonical, valid, CONFIG)
    if not torch.equal(risk, expected):
        raise RuntimeError('Original hard support changed.')
    fine_pair, _ = sampled_log_responsibilities(observed_fine, valid)
    base_weights = redistribute(protected_weights(fine_pair, risk, parents, canonical, valid),
        risk, members, canonical)
    reference = local.double().softmax(-1)
    alias = sampled_aliases(observed_fine, fine_coordinates)
    density, available = response_densities(alias, reference, valid)
    source, source_known = density_responsibility(density, available, parents)
    weights = incremental_allocation(base_weights, source, risk, parents, canonical, valid, members)
    allocations = {FINE_ONLY: base_weights, PRIMARY: weights, UNIFORM: base_weights}
    sources = {PRIMARY: source}
    if ONLY in methods:
        allocations[ONLY] = incremental_allocation((risk == 0).double(), source, risk, parents, canonical, valid, members)
    for name, options, teacher in (
            (REFERENCE_NULL, {}, position_control(reference, valid)),
            (NO_HOLDOUT, {'exclude_self': False}, reference),
            (UNBALANCED, {'balance': False}, reference)):
        if name in methods:
            field, known = response_densities(alias, teacher, valid, **options)
            changed, _ = density_responsibility(field, known, parents)
            sources[name] = changed
            allocations[name] = incremental_allocation(base_weights, changed, risk, parents, canonical, valid, members)
    if SPATIAL in methods:
        sources[SPATIAL] = position_control(source, valid)
        allocations[SPATIAL] = incremental_allocation(base_weights, sources[SPATIAL], risk,
            parents, canonical, valid, members)
    for i, name in enumerate(SHUFFLES):
        if name in methods:
            allocations[name] = support_control(weights, risk, members, canonical, CONFIG.random_seed+i)
    keep = risk == 0
    eligible = valid[:, None, None] & keep
    eligible &= ~torch.isin(torch.arange(len(parents), device=parents.device), canonical)[None, :, None]
    eligible &= parents[None, :, None] != torch.arange(len(members), device=parents.device)[None, None]
    base = replay[REPLAY[0]]
    posterior = base.softmax(-1)
    innovation = sampled_classes(observed_fine, fine_coordinates).double()-broad.double()
    target = (innovation[:, :, None]-innovation[:, None]).masked_fill(~valid[:, None, None], 0.)
    grouped = source[:, members]
    off_diagonal = ~torch.eye(len(members), dtype=torch.bool, device=local.device)[None, :, None]
    raw_range = grouped.masked_fill(~off_diagonal, -torch.inf).amax(-1)-grouped.masked_fill(
        ~off_diagonal, torch.inf).amin(-1)
    diagnostics = {'additional_visual_forwards': 0, 'canonical_weight_min': float(weights[:, canonical].min()),
        'old_rejected_weights_max': float(weights[~keep].abs().max()) if bool((~keep).any()) else 0.,
        'all_mass_max_errors': {}, 'control_spectrum_max_errors': {},
        'eligible_source_comparisons': int(eligible.sum()),
        'available_source_comparisons': int((eligible & source_known).sum()),
        'mean_source_sum': float(source[eligible].sum()),
        'source_floor_comparisons': int((eligible & (source <= CONFIG.epsilon)).sum()),
        'raw_rival_range_max_before_protection': float(raw_range[valid].max()) if bool(valid.any()) else 0.,
        'soft_reference': 'original Geometry20 local posterior; not a correctness teacher',
        'query_self_exclusion': True, 'query_exclusion_is_not_alias_or_scene_holdout': True}
    original_mass = keep[:, members].sum(2).double()
    values = dict(replay)
    for name, allocation in allocations.items():
        if name not in methods and name != FINE_ONLY:
            continue
        error = float((allocation[:, members].sum(2)-original_mass).abs().max())
        diagnostics['all_mass_max_errors'][name] = error
        if error > 1e-10 or bool((allocation[:, canonical] != 1).any()) or bool((allocation[~keep] != 0).any()):
            raise RuntimeError('Density source changed original canonical/support/mass.')
        directed = allocation_directed(observed_wide, members, allocation, keep, valid)
        action = project_actions(directed-directed.transpose(-1, -2), target)
        potential, stats = posterior_potential(action, posterior, valid)
        values[name] = base+operator.double()@potential
        diagnostics[name] = stats
        if name == UNIFORM:
            diagnostics['uniform_fine_only_max_error'] = float((values[name]-values[FINE_ONLY]).abs().max())
            if diagnostics['uniform_fine_only_max_error'] != 0:
                raise RuntimeError('Uniform incremental source must restore fine-only exactly.')
        if name in SHUFFLES:
            error = float((allocation[:, members].sort(2).values-weights[:, members].sort(2).values).abs().max())
            diagnostics['control_spectrum_max_errors'][name] = error
            if error != 0:
                raise RuntimeError('Density own identity-null spectrum changed.')
    values = {name: values[name] for name in methods}
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite density scores.')
    return values, diagnostics
