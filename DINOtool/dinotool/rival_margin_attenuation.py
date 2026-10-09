"""Competition-conditioned soft attenuation in the actual view-margin units."""
import math

import torch

from .fine_alias_view import CONFIG
from .native_query_alias import native_risk
from .rival_alias_fast import cache_observations, sampled_cached
from .rival_alias_attribution import alias_null
from .rival_competition_admission import (retained_competition_scores, sampled_classes,
    posterior_potential, project_actions, settings as competition_settings)


IMPLEMENTATION = 'geometry-frozen-rival-margin-soft-attenuation-v1-20261005'
BASE = ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact', 'FineBudgetOnly_Exact')
PRIMARY = 'MarginTransfer_Projected'
CANDIDATES = (PRIMARY, 'MarginNeutral_Projected', 'MarginDeficit_Projected')
CONTROLS = ('RatioSoft_Projected', 'TransferClassMean_Projected', 'ProjectedHardStrengthMatched')
SHUFFLES = tuple('TransferAliasShuffle'+str(i) for i in range(3))
METHODS = (*BASE, *CANDIDATES, *CONTROLS, *SHUFFLES)


def settings():
    return {**competition_settings(), 'frozen_risk_support': 'b>0 and f<0; canonical/self/invalid protected',
        'primary_log_weight': 'beta*(f-b) on frozen eligible alias/rival slots, zero elsewhere',
        'neutral_log_weight': '-beta*b on eligible slots', 'deficit_log_weight': 'beta*f on eligible slots',
        'writer': 'original normalized weighted logsumexp in original salience-profiled units',
        'action': 'same fine-target direction/segment projection and posterior pair solver',
        'new_parameters': 0, 'no_global_word_blacklist': True,
        'claim': 'matched view-margin attenuation, not calibrated correctness probabilities'}


def margin_log_weights(broad, fine, risk, beta=CONFIG.beta, kind='transfer'):
    if (broad.shape != fine.shape or risk.shape != broad.shape or kind not in ('transfer', 'neutral', 'deficit')
            or beta <= 0 or not bool(torch.isfinite(broad).all() and torch.isfinite(fine).all()
                                     and torch.isfinite(risk).all())
            or bool(((risk < 0) | (risk > 1)).any())):
        raise ValueError('Matching finite view margins, bounded frozen risk and positive beta required.')
    eligible = risk > 0
    if bool((eligible & ~((broad > 0) & (fine < 0))).any()):
        raise ValueError('Risk support must be the frozen cross-view sign reversal.')
    b, f = broad.double(), fine.double()
    value = {'transfer': f-b, 'neutral': -b, 'deficit': f}[kind]*beta
    return value.masked_fill(~eligible, 0.)


def discount_directed(observations, members, log_weights, valid, beta=CONFIG.beta, chunk=CONFIG.query_chunk):
    if (log_weights.shape != (len(valid), members.numel(), len(members)) or beta <= 0 or chunk < 1
            or bool(torch.isnan(log_weights).any()) or bool((log_weights > 0).any())):
        raise ValueError('Matching nonpositive log weights and positive reader settings required.')
    output = torch.zeros(len(valid), len(members), len(members), device=log_weights.device, dtype=torch.float64)
    k = members.shape[-1]
    for start in range(0, len(valid), chunk):
        sl = slice(start, start+chunk)
        grouped = log_weights[sl, members].double()
        log_mass = grouped.logsumexp(2)
        if not bool(torch.isfinite(log_mass).all()):
            raise ValueError('Every class/rival needs a surviving alias weight.')
        normalizer = (math.log(k)-log_mass)[:, None]
        untouched = (grouped == 0).all(2)[:, None]
        for crop in observations:
            evidence = (beta*crop.evidence[crop.indices[sl]]).double()
            full = evidence.logsumexp(-1)
            changed = (evidence[..., None]+grouped[:, None]).logsumexp(3)
            delta = ((changed-full[..., None]+normalizer)/beta).masked_fill(untouched, 0.)
            output[sl] += (delta*crop.coefficients[sl].double()[..., None, None]).sum(1)
    return output.masked_fill(~valid[:, None, None], 0.)


def class_mean_discount(log_weights, members, canonical):
    result = log_weights.clone()
    for group in members:
        aliases = group[~torch.isin(group, canonical)]
        result[:, aliases] = log_weights[:, aliases].mean(1, keepdim=True)
    result[:, canonical] = 0.
    return result


@torch.inference_mode()
def margin_attenuation_scores(local, operator, broad, wide, wide_count, fine, fine_count,
                              coordinates, fine_coordinates, valid, members, canonical, parents,
                              image_size, methods=METHODS):
    if not set(methods).issubset(METHODS): raise ValueError('Undeclared margin attenuation endpoint.')
    old_methods = tuple(m for m in methods if m in BASE)
    values, stats = {}, {}
    if old_methods:
        values, old_risk, stats = retained_competition_scores(local, operator, broad, wide, wide_count,
            fine, fine_count, coordinates, fine_coordinates, valid, members, canonical, parents,
            image_size, methods=old_methods)
        if set(methods).issubset(BASE): return values, old_risk, stats
    observed = cache_observations(wide, wide_count, coordinates, image_size, members)
    fine_observed = cache_observations(fine, fine_count, fine_coordinates, (512, 512), members)
    b = sampled_cached(observed, coordinates)
    f = sampled_cached(fine_observed, fine_coordinates).masked_fill(~valid[:, None, None], 0.)
    risk = native_risk(b, f, f, parents, canonical, valid, CONFIG)
    if old_methods and not torch.equal(risk, old_risk): raise RuntimeError('Frozen margin-risk source changed.')
    baseline = local.double()+operator.double()@(broad.double()-local.double())
    posterior = baseline.softmax(-1)
    innovation = sampled_classes(fine_observed, fine_coordinates).double()-broad.double()
    target = (innovation[..., None]-innovation[:, None]).masked_fill(~valid[:, None, None], 0.)
    transfer = margin_log_weights(b, f, risk)
    eligible = risk > 0
    ratio = torch.log1p(-risk.double())
    stats['canonical_risk_max'] = float(risk[:, canonical].abs().max())
    stats['eligible_comparisons'] = int(eligible.sum())
    stats['ratio_residual_advantage_fraction'] = float(((b.double()+ratio/CONFIG.beta)[eligible] > 0).double().mean()) if bool(eligible.any()) else 0.
    stats['transfer_residual_margin_max_error'] = float((b.double()+transfer/CONFIG.beta-f.double())[eligible].abs().max()) if bool(eligible.any()) else 0.
    stats['transfer_active_weight_mean'] = float(transfer[eligible].exp().mean()) if bool(eligible.any()) else 1.
    stats['ratio_active_weight_mean'] = float((1-risk.double())[eligible].mean()) if bool(eligible.any()) else 1.

    def correction(log_weights):
        directed = discount_directed(observed, members, log_weights, valid)
        action = project_actions(directed-directed.transpose(-1, -2), target)
        potential, diagnostic = posterior_potential(action, posterior, valid)
        return operator.double()@potential, diagnostic

    primary_delta, diagnostic = correction(transfer)
    values[PRIMARY] = baseline+primary_delta
    stats.update({PRIMARY+'__'+k: v for k, v in diagnostic.items()})
    for name, kind in zip(CANDIDATES[1:], ('neutral', 'deficit')):
        if name in methods:
            delta, _ = correction(margin_log_weights(b, f, risk, kind=kind))
            values[name] = baseline+delta
    if CONTROLS[0] in methods:
        delta, _ = correction(ratio)
        values[CONTROLS[0]] = baseline+delta
    if CONTROLS[1] in methods:
        changed = class_mean_discount(transfer, members, canonical)
        error = float((changed[:, members].sum(2)-transfer[:, members].sum(2)).abs().max())
        stats['class_mean_log_budget_max_error'] = error
        delta, _ = correction(changed)
        values[CONTROLS[1]] = baseline+delta
    if CONTROLS[2] in methods:
        if BASE[3] not in values:
            fixed, _, _ = retained_competition_scores(local, operator, broad, wide, wide_count,
                fine, fine_count, coordinates, fine_coordinates, valid, members, canonical,
                parents, image_size, methods=(BASE[3],))
            hard_delta = fixed[BASE[3]]-baseline
        else: hard_delta = values[BASE[3]]-baseline
        scale = primary_delta.norm()/hard_delta.norm().clamp_min(CONFIG.epsilon)
        values[CONTROLS[2]] = baseline+scale*hard_delta
        stats['strength_norm_error'] = float((scale*hard_delta).norm()-primary_delta.norm())
        stats['strength_zero_direction'] = float(hard_delta.norm() == 0)
        stats['strength_scale'] = float(scale)
    for i, name in enumerate(SHUFFLES):
        if name not in methods: continue
        changed = alias_null(transfer, members, canonical, CONFIG.random_seed+i)
        error = float((changed[:, members].sort(2).values-transfer[:, members].sort(2).values).abs().max())
        if error: raise RuntimeError('Alias-null log-weight spectrum changed.')
        delta, _ = correction(changed)
        values[name] = baseline+delta
        stats[name+'__spectrum_max_error'] = error
    values = {m: values[m] for m in methods}
    if not all(bool(torch.isfinite(v).all()) for v in values.values()): raise RuntimeError('Nonfinite margin attenuation score.')
    return values, risk, stats
