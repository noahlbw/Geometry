"""Conditional wide/fine alias responsibility intersection on frozen observers."""
import torch

from .fine_alias_view import CONFIG
from .native_query_alias import native_risk
from .rival_alias_fast import cache_observations, sampled_cached
from .rival_competition_admission import posterior_potential, project_actions, sampled_classes
from .rival_fine_support import support_control
from .rival_survivor_redistribution import (PRIMARY as SURVIVOR, allocation_directed,
    redistribute, redistribution_scores)


IMPLEMENTATION = 'frozen-pair-conditional-cross-view-responsibility-intersection-v1-20261005'
PRIMARY = 'PairResponsibilityIntersection_Exact'
WITHIN = 'WithinResponsibilityIntersection_Exact'
ABSOLUTE = 'FineResponsibilityOnly_Exact'
CLASS_MEAN = 'ResponsibilityClassMean_Exact'
SHUFFLES = tuple('ResponsibilityIntersectionShuffle'+str(i)+'_Exact' for i in range(3))
REPLAY = ('NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact', SURVIVOR)
METHODS = (*REPLAY, PRIMARY, WITHIN, ABSOLUTE, CLASS_MEAN, *SHUFFLES)


def sampled_log_responsibilities(observations, valid, beta=CONFIG.beta, chunk=CONFIG.query_chunk):
    if not observations or beta <= 0 or chunk < 1 or valid.dtype != torch.bool:
        raise ValueError('Cached observations, Boolean validity and positive source settings required.')
    classes, k = observations[0].evidence.shape[1:]
    device = valid.device
    pair = torch.full((len(valid), classes, k, classes), -torch.inf, device=device, dtype=torch.float64)
    within = torch.full((len(valid), classes, k), -torch.inf, device=device, dtype=torch.float64)
    coverage = torch.zeros(len(valid), device=device, dtype=torch.float64)
    for crop in observations:
        if crop.evidence.shape[1:] != (classes, k) or len(crop.indices) != len(valid):
            raise ValueError('All source crops must share actual query/class/alias identities.')
        evidence = (beta*crop.evidence).double()
        log_z = evidence.logsumexp(-1)
        log_q = evidence-log_z[..., None]
        log_class = log_z[:, :, None]-torch.logaddexp(log_z[:, :, None], log_z[:, None])
        coefficients = crop.coefficients.double()
        coverage += coefficients.sum(1)
        for start in range(0, len(valid), chunk):
            sl = slice(start, start+chunk)
            ids, log_alpha = crop.indices[sl], coefficients[sl].log()
            # Mix probabilities through the unchanged crop stencil in log space.
            w = (log_q[ids]+log_alpha[..., None, None]).logsumexp(1)
            p = (log_q[ids][..., None]+log_class[ids][:, :, :, None]
                 +log_alpha[..., None, None, None]).logsumexp(1)
            within[sl] = torch.logaddexp(within[sl], w)
            pair[sl] = torch.logaddexp(pair[sl], p)
    if bool((valid & (coverage <= 0)).any()):
        raise RuntimeError('Every valid source query needs observed crop coverage.')
    log_coverage = coverage.masked_fill(coverage <= 0, 1.).log()
    within -= log_coverage[:, None, None]
    pair -= log_coverage[:, None, None, None]
    within[~valid] = -torch.tensor(float(k), device=device, dtype=torch.float64).log()
    pair[~valid] = -torch.tensor(float(2*k), device=device, dtype=torch.float64).log()
    if not bool(torch.isfinite(within).all() and torch.isfinite(pair).all()):
        raise RuntimeError('Nonfinite sampled responsibility logs.')
    return pair.flatten(1, 2), within.flatten(1)


def intersection_log_weights(wide, fine):
    if wide.shape != fine.shape or not bool(torch.isfinite(wide).all() and torch.isfinite(fine).all()):
        raise ValueError('Matching finite wide/fine responsibility logs required.')
    return (fine-wide).clamp_max(0.)


def protected_weights(log_weights, old_risk, parents, canonical, valid):
    if log_weights.shape != old_risk.shape or not bool(torch.isfinite(log_weights).all()) or bool((log_weights > 0).any()):
        raise ValueError('Matching finite nonpositive responsibility log weights required.')
    weights = log_weights.exp().clamp_min(CONFIG.epsilon)
    weights.masked_fill_(old_risk > 0, 0.)
    weights[:, canonical] = 1.
    weights.scatter_(-1, parents[None, :, None].expand(len(valid), -1, 1), 1.)
    weights.masked_fill_(~valid[:, None, None], 1.)
    return weights


@torch.inference_mode()
def intersection_scores(local, operator, broad, wide, wide_count, fine, fine_count,
                        coordinates, fine_coordinates, valid, members, canonical, parents,
                        image_size, *, matches, methods=METHODS):
    if not set(methods).issubset(METHODS):
        raise ValueError('Declared frozen responsibility-intersection endpoint required.')
    replay, _ = redistribution_scores(local, operator, broad, wide, wide_count, fine, fine_count,
        coordinates, fine_coordinates, valid, members, canonical, parents, image_size,
        matches=matches, methods=REPLAY)
    observed_wide = cache_observations(wide, wide_count, coordinates, image_size, members)
    observed_fine = cache_observations(fine, fine_count, fine_coordinates, (512, 512), members)
    wide_margin = sampled_cached(observed_wide, coordinates)
    native = sampled_cached(observed_fine, fine_coordinates).masked_fill(~valid[:, None, None], 0.)
    old_risk = native_risk(wide_margin, native, native, parents, canonical, valid, CONFIG)
    keep = old_risk == 0
    wide_pair, wide_within = sampled_log_responsibilities(observed_wide, valid)
    fine_pair, fine_within = sampled_log_responsibilities(observed_fine, valid)
    logs = intersection_log_weights(wide_pair, fine_pair)
    identity_error = float((wide_pair+logs-torch.minimum(wide_pair, fine_pair)).abs().max())
    if identity_error > 1e-10:
        raise RuntimeError('Pre-protection responsibility-intersection identity failed.')
    source = protected_weights(logs, old_risk, parents, canonical, valid)
    weights = redistribute(source, old_risk, members, canonical)
    allocations = {PRIMARY: weights}
    if WITHIN in methods:
        log_weights = intersection_log_weights(wide_within, fine_within)[..., None].expand_as(old_risk)
        allocations[WITHIN] = redistribute(protected_weights(log_weights, old_risk, parents, canonical, valid),
            old_risk, members, canonical)
    if ABSOLUTE in methods:
        allocations[ABSOLUTE] = redistribute(protected_weights(fine_pair, old_risk, parents, canonical, valid),
            old_risk, members, canonical)
    if CLASS_MEAN in methods:
        allocations[CLASS_MEAN] = keep.double()
    for i, name in enumerate(SHUFFLES):
        if name in methods:
            allocations[name] = support_control(weights, old_risk, members, canonical, CONFIG.random_seed+i)
    base = replay['NoAdmission_Exact']
    posterior = base.softmax(-1)
    innovation = sampled_classes(observed_fine, fine_coordinates).double()-broad.double()
    target = (innovation[:, :, None]-innovation[:, None]).masked_fill(~valid[:, None, None], 0.)
    eligible = valid[:, None, None] & keep
    eligible &= ~torch.isin(torch.arange(len(parents), device=parents.device), canonical)[None, :, None]
    eligible &= parents[None, :, None] != torch.arange(len(members), device=parents.device)[None, None]
    joint = eligible & (wide_margin > 0) & (native >= 0)
    source_pairs = logs[:, members].exp().clamp_min(CONFIG.epsilon)
    rival_mask = ~torch.eye(len(members), device=source.device, dtype=torch.bool)
    differences = source_pairs.masked_fill(~rival_mask[None, :, None], -torch.inf).amax(-1)-source_pairs.masked_fill(
        ~rival_mask[None, :, None], torch.inf).amin(-1)
    diagnostics = {'additional_visual_forwards': 0, 'canonical_weight_min': float(weights[:, canonical].min()),
        'old_rejected_weights_max': float(weights[~keep].abs().max()) if bool((~keep).any()) else 0.,
        'intersection_log_identity_max_error_before_floor_and_protection': identity_error,
        'source_comparisons': int(eligible.sum()), 'source_weight_sum': float(source[eligible].sum()),
        'source_floor_comparisons': int((eligible & (source <= CONFIG.epsilon)).sum()),
        'joint_positive_comparisons': int(joint.sum()), 'joint_positive_source_weight_sum': float(source[joint].sum()),
        'raw_source_rival_range_max_before_protection': float(differences[valid].max()) if bool(valid.any()) else 0.,
        'all_mass_max_errors': {}, 'control_spectrum_max_errors': {}}
    original_mass = keep[:, members].sum(2).double()
    values = dict(replay)
    for name, allocation in allocations.items():
        if name not in methods:
            continue
        error = float((allocation[:, members].sum(2)-original_mass).abs().max())
        diagnostics['all_mass_max_errors'][name] = error
        if error > 1e-10 or bool((allocation[:, canonical] != 1).any()):
            raise RuntimeError('Intersection changed original survivor/canonical mass.')
        directed = allocation_directed(observed_wide, members, allocation, keep, valid)
        action = project_actions(directed-directed.transpose(-1, -2), target)
        potential, stats = posterior_potential(action, posterior, valid)
        values[name] = base+operator.double()@potential
        diagnostics[name] = stats
        if name in SHUFFLES:
            error = float((allocation[:, members].sort(2).values-weights[:, members].sort(2).values).abs().max())
            diagnostics['control_spectrum_max_errors'][name] = error
            if error != 0:
                raise RuntimeError('Intersection identity-null spectrum changed.')
        if name == CLASS_MEAN:
            error = float((values[name]-replay['FineRivalProjected_Exact']).abs().max())
            diagnostics['class_mean_score_max_error'] = error
            if error != 0:
                raise RuntimeError('Intersection class-mean control must restore projected hard.')
    values = {name: values[name] for name in methods}
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite responsibility-intersection scores.')
    return values, diagnostics
