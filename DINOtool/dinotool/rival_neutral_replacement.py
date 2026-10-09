"""Preserve alias slots while capping contradicted competitive advantages."""
import math

import torch

from .fine_alias_view import CONFIG
from .native_query_alias import native_risk
from .rival_alias_fast import cache_observations, sampled_cached, directed_cached
from .rival_alias_attribution import alias_null
from .rival_competition_admission import (retained_competition_scores, sampled_classes,
    posterior_potential, project_actions, settings as competition_settings)
from .rival_margin_attenuation import BASE


IMPLEMENTATION = 'geometry-frozen-rival-neutral-slot-replacement-v1-20261005'
PRIMARY = 'RivalNeutral_Projected'
CANDIDATES = (PRIMARY, 'RivalFineCap_Projected')
CONTROLS = ('HardFixedMass_Projected', 'NeutralUnprojected', 'ProjectedHardStrengthMatched')
SHUFFLES = tuple('NeutralAliasShuffle'+str(i) for i in range(3))
METHODS = (*BASE, *CANDIDATES, *CONTROLS, *SHUFFLES)
BENCHMARK = (*CANDIDATES, CONTROLS[0])


def settings():
    return {**competition_settings(), 'risk_support': 'unchanged original b>0,f<0; canonical/self/invalid protected',
        'primary_writer': 'cap each eligible wide alias at rival original log-mean-exp; retain original K slots',
        'fine_cap_writer': 'cap at original rival log-mean-exp plus the same sampled fine alias/rival margin',
        'fixed_mass_control': 'ordinary hard removal without surviving-count renormalization',
        'counterfactual_direction': 'class deltas nonpositive before antisymmetrization and reconstruction',
        'action': 'unchanged fine-target projection and posterior pair solver',
        'new_parameters': 0, 'additional_visual_forwards': 0, 'no_global_word_blacklist': True}


def replacement_directed(observations, members, risk, valid, fine_margin=None,
                         kind='neutral', beta=CONFIG.beta, chunk=CONFIG.query_chunk):
    if (risk.shape != (len(valid), members.numel(), len(members)) or kind not in ('neutral', 'fine', 'fixed_mass')
            or beta <= 0 or chunk < 1 or not bool(torch.isfinite(risk).all())
            or bool(((risk < 0) | (risk > 1)).any())
            or (kind == 'fine' and (fine_margin is None or fine_margin.shape != risk.shape
                                    or not bool(torch.isfinite(fine_margin).all())))):
        raise ValueError('Matching bounded frozen risk, declared writer and fine margins required.')
    output = torch.zeros(len(valid), len(members), len(members), device=risk.device, dtype=torch.float64)
    for start in range(0, len(valid), chunk):
        sl = slice(start, start+chunk)
        eligible = risk[sl, members] > 0
        untouched = ~eligible.any(2)[:, None]
        for crop in observations:
            evidence = (beta*crop.evidence[crop.indices[sl]]).double()
            full = evidence.logsumexp(-1)
            original = evidence[..., None]
            if kind == 'fixed_mass':
                changed = original.masked_fill(eligible[:, None], -torch.inf)
            else:
                cap = (full-math.log(members.shape[-1]))[:, :, None, None]
                if kind == 'fine': cap = cap+beta*fine_margin[sl, members].double()[:, None]
                changed = torch.where(eligible[:, None], torch.minimum(original, cap), original)
            delta = ((changed.logsumexp(3)-full[..., None])/beta).masked_fill(untouched, 0.)
            if not bool(torch.isfinite(delta).all()): raise ValueError('Every fixed-mass class needs surviving finite evidence.')
            output[sl] += (delta*crop.coefficients[sl].double()[..., None, None]).sum(1)
    output.masked_fill_(~valid[:, None, None], 0.)
    if float(output.max()) > 1e-12: raise RuntimeError('Replacement unexpectedly boosted source class evidence.')
    return output


@torch.inference_mode()
def neutral_replacement_scores(local, operator, broad, wide, wide_count, fine, fine_count,
                               coordinates, fine_coordinates, valid, members, canonical, parents,
                               image_size, methods=METHODS):
    if not set(methods).issubset(METHODS): raise ValueError('Undeclared neutral replacement endpoint.')
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
    if old_methods and not torch.equal(risk, old_risk): raise RuntimeError('Frozen replacement source changed.')
    baseline = local.double()+operator.double()@(broad.double()-local.double())
    posterior = baseline.softmax(-1)
    innovation = sampled_classes(fine_observed, fine_coordinates).double()-broad.double()
    target = (innovation[..., None]-innovation[:, None]).masked_fill(~valid[:, None, None], 0.)
    stats['canonical_risk_max'] = float(risk[:, canonical].abs().max())
    stats['eligible_comparisons'] = int((risk > 0).sum())

    def correction(changed_risk=risk, kind='neutral', projected=True):
        directed = replacement_directed(observed, members, changed_risk, valid, f, kind)
        action = directed-directed.transpose(-1, -2)
        if projected: action = project_actions(action, target)
        potential, _ = posterior_potential(action, posterior, valid)
        return operator.double()@potential, directed

    primary_delta = None
    if PRIMARY in methods or CONTROLS[2] in methods:
        primary_delta, directed = correction()
        values[PRIMARY] = baseline+primary_delta
        stats['neutral_positive_source_delta_max'] = float(directed.max())
        stats['neutral_source_abs_mean'] = float(directed[valid].abs().mean()) if bool(valid.any()) else 0.
    if CANDIDATES[1] in methods:
        delta, _ = correction(kind='fine')
        values[CANDIDATES[1]] = baseline+delta
    if CONTROLS[0] in methods:
        delta, fixed = correction(kind='fixed_mass')
        values[CONTROLS[0]] = baseline+delta
        hard = directed_cached(observed, members, risk, valid)
        remaining = (risk[:, members] == 0).sum(2).double()
        normalizer = (members.shape[1]/remaining).log()/CONFIG.beta
        coverage = sum(crop.coefficients.double().sum(1) for crop in observed)
        expected = fixed+normalizer*coverage[:, None, None]
        error = float((expected-hard)[valid].abs().max()) if bool(valid.any()) else 0.
        if error > 1e-10: raise RuntimeError('Fixed-mass hard/count-normalized identity failed.')
        rivals = ~torch.eye(len(members), device=risk.device, dtype=torch.bool)
        active = ((risk[:, members] > 0).any(2) & rivals[None])[valid]
        stats['fixed_mass_identity_max_error'] = error
        stats['hard_source_boost_fraction'] = float((hard[valid][active] > 1e-12).double().mean()) if bool(active.any()) else 0.
        stats['normalizer_active_mean'] = float(normalizer[valid][active].mean()) if bool(active.any()) else 0.
        stats['fixed_mass_positive_source_delta_max'] = float(fixed.max())
    if CONTROLS[1] in methods:
        delta, _ = correction(projected=False)
        values[CONTROLS[1]] = baseline+delta
    if CONTROLS[2] in methods:
        if BASE[3] not in values:
            fixed, _, _ = retained_competition_scores(local, operator, broad, wide, wide_count,
                fine, fine_count, coordinates, fine_coordinates, valid, members, canonical, parents,
                image_size, methods=(BASE[3],))
            hard_delta = fixed[BASE[3]]-baseline
        else: hard_delta = values[BASE[3]]-baseline
        scale = primary_delta.norm()/hard_delta.norm().clamp_min(CONFIG.epsilon)
        values[CONTROLS[2]] = baseline+scale*hard_delta
        stats['strength_norm_error'] = float((scale*hard_delta).norm()-primary_delta.norm())
        stats['strength_zero_direction'] = float(hard_delta.norm() == 0)
        stats['strength_scale'] = float(scale)
    for i, name in enumerate(SHUFFLES):
        if name not in methods: continue
        changed = alias_null(risk, members, canonical, CONFIG.random_seed+i)
        error = int(((changed[:, members] > 0).sum(2)-(risk[:, members] > 0).sum(2)).abs().max())
        if error: raise RuntimeError('Neutral alias-null changed query/class/rival counts.')
        delta, _ = correction(changed)
        values[name] = baseline+delta
        stats[name+'__count_max_error'] = error
    values = {m: values[m] for m in methods}
    if not all(bool(torch.isfinite(v).all()) for v in values.values()): raise RuntimeError('Nonfinite neutral replacement score.')
    return values, risk, stats
