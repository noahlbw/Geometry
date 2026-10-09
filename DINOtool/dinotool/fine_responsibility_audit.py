"""Own mechanistic controls for the frozen fine-only responsibility source."""
import torch

from .fine_alias_view import CONFIG
from .native_query_alias import native_risk
from .rival_alias_fast import cache_observations, sampled_cached
from .rival_competition_admission import (posterior_potential, project_actions,
    retained_competition_scores, sampled_classes)
from .rival_fine_support import support_control
from .rival_responsibility_intersection import (ABSOLUTE as PRIMARY,
    protected_weights, sampled_log_responsibilities)
from .rival_survivor_redistribution import allocation_directed, redistribute


IMPLEMENTATION = 'frozen-fine-responsibility-own-mechanism-audit-v1-20261005'
REPLAY = ('NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact')
WIDE = 'WideResponsibilityOnly_Exact'
WITHIN = 'FineWithinResponsibilityOnly_Exact'
FACTORIZED = 'FineFactorizedResponsibility_Exact'
SPATIAL = 'FineResponsibilityPositionShuffle_Exact'
CLASS_MEAN = 'FineResponsibilityUniform_Exact'
SHUFFLES = tuple('FineResponsibilityShuffle'+str(i)+'_Exact' for i in range(3))
METHODS = (*REPLAY, PRIMARY, WIDE, WITHIN, FACTORIZED, SPATIAL, CLASS_MEAN, *SHUFFLES)


def factorized_logs(pair, within, members):
    if pair.shape != (*within.shape, len(members)) or members.numel() != within.shape[1]:
        raise ValueError('Matching pair/within responsibilities and class members required.')
    class_pair = pair[:, members].logsumexp(2)
    grouped = within[:, members, None]+class_pair[:, :, None]
    result = torch.empty_like(pair)
    result[:, members] = grouped
    return result.clamp_max(0.)


def position_control(logs, valid, seed=CONFIG.random_seed):
    if logs.shape[0] != len(valid) or valid.dtype != torch.bool:
        raise ValueError('Matching Boolean query validity required.')
    ids = valid.nonzero().flatten()
    generator = torch.Generator(device=logs.device).manual_seed(seed)
    permutation = torch.randperm(len(ids), generator=generator, device=logs.device)
    result = logs.clone()
    result[ids] = logs[ids[permutation]]
    return result


@torch.inference_mode()
def audit_scores(local, operator, broad, wide, wide_count, fine, fine_count,
                 coordinates, fine_coordinates, valid, members, canonical, parents,
                 image_size, *, methods=METHODS):
    if not set(methods).issubset(METHODS):
        raise ValueError('Declared frozen fine-responsibility audit endpoint required.')
    replay, original_risk, _ = retained_competition_scores(local, operator, broad,
        wide, wide_count, fine, fine_count, coordinates, fine_coordinates, valid,
        members, canonical, parents, image_size, methods=REPLAY)
    observed_wide = cache_observations(wide, wide_count, coordinates, image_size, members)
    observed_fine = cache_observations(fine, fine_count, fine_coordinates, (512, 512), members)
    native = sampled_cached(observed_fine, fine_coordinates).masked_fill(~valid[:, None, None], 0.)
    risk = native_risk(sampled_cached(observed_wide, coordinates), native, native,
        parents, canonical, valid, CONFIG)
    if not torch.equal(risk, original_risk):
        raise RuntimeError('Original hard support changed.')
    fine_pair, fine_within = sampled_log_responsibilities(observed_fine, valid)
    wide_pair, _ = sampled_log_responsibilities(observed_wide, valid)
    factorized = factorized_logs(fine_pair, fine_within, members)
    sources = {PRIMARY: fine_pair, WIDE: wide_pair,
        WITHIN: fine_within[..., None].expand_as(risk), FACTORIZED: factorized,
        SPATIAL: position_control(fine_pair, valid)}
    allocations = {name: redistribute(protected_weights(logs, risk, parents, canonical, valid),
        risk, members, canonical) for name, logs in sources.items()}
    keep = risk == 0
    allocations[CLASS_MEAN] = keep.double()
    weights = allocations[PRIMARY]
    for i, name in enumerate(SHUFFLES):
        if name in methods:
            allocations[name] = support_control(weights, risk, members, canonical, CONFIG.random_seed+i)
    base = replay['NoAdmission_Exact']
    posterior = base.softmax(-1)
    innovation = sampled_classes(observed_fine, fine_coordinates).double()-broad.double()
    target = (innovation[:, :, None]-innovation[:, None]).masked_fill(~valid[:, None, None], 0.)
    eligible = valid[:, None, None] & keep
    eligible &= ~torch.isin(torch.arange(len(parents), device=parents.device), canonical)[None, :, None]
    eligible &= parents[None, :, None] != torch.arange(len(members), device=parents.device)[None, None]
    diagnostics = {'additional_visual_forwards': 0,
        'canonical_weight_min': float(weights[:, canonical].min()),
        'old_rejected_weights_max': float(weights[~keep].abs().max()) if bool((~keep).any()) else 0.,
        'all_mass_max_errors': {}, 'control_spectrum_max_errors': {},
        'eligible_source_comparisons': int(eligible.sum()), 'source_floor_counts': {
            name: int((eligible & (logs.exp() <= CONFIG.epsilon)).sum()) for name, logs in sources.items()},
        'primary_vs_within_allocation_max_difference': float((weights-allocations[WITHIN]).abs().max()),
        'factorized_vs_within_allocation_max_difference': float((allocations[FACTORIZED]-allocations[WITHIN]).abs().max()),
        'primary_vs_factorized_source_max_difference': float((fine_pair.exp()-factorized.exp()).abs().max())}
    original_mass = keep[:, members].sum(2).double()
    values = dict(replay)
    for name, allocation in allocations.items():
        if name not in methods:
            continue
        error = float((allocation[:, members].sum(2)-original_mass).abs().max())
        diagnostics['all_mass_max_errors'][name] = error
        if error > 1e-10 or bool((allocation[:, canonical] != 1).any()) or bool((allocation[~keep] != 0).any()):
            raise RuntimeError('Frozen survivor/canonical support or mass changed.')
        directed = allocation_directed(observed_wide, members, allocation, keep, valid)
        action = project_actions(directed-directed.transpose(-1, -2), target)
        potential, stats = posterior_potential(action, posterior, valid)
        values[name] = base+operator.double()@potential
        diagnostics[name] = stats
        if name in SHUFFLES:
            error = float((allocation[:, members].sort(2).values-weights[:, members].sort(2).values).abs().max())
            diagnostics['control_spectrum_max_errors'][name] = error
            if error != 0:
                raise RuntimeError('Own fine-source identity-null spectrum changed.')
        if name == CLASS_MEAN:
            error = float((values[name]-replay[REPLAY[-1]]).abs().max())
            diagnostics['class_mean_score_max_error'] = error
            if error != 0:
                raise RuntimeError('Uniform source must exactly restore projected hard.')
    values = {name: values[name] for name in methods}
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite fine-responsibility audit scores.')
    return values, diagnostics
