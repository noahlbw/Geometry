"""Attribute the existing single-sided rule without changing its predictions."""
import torch

from .fine_alias_view import CONFIG
from .native_alias_noise import signed_potential
from .native_query_alias import native_risk
from .reciprocal_alias_admission import (CLASS_MEAN as PREVIOUS_CLASS_MEAN,
    SHUFFLES as PREVIOUS_SHUFFLES, continuous_controls, fixed_slot_action)
from .rival_alias_fast import directed_cached, risk_statistics, sampled_cached
from .supported_positive_alias import (PROTOCOL as SOURCE_PROTOCOL,
    permuted_operator, predict_image as source_predict)
from .witness_cap_alias import match_previous


IMPLEMENTATION = 'geometry-bounded896-existing-one-sided-alias-audit-v1-20261006'
PRIMARY = 'OneSide_Soft'
OBSERVATION_MEAN = 'OneSide_ObservationMean'
HARD = 'OneSide_Hard'
CLASS_MEAN = 'OneSide_ClassMean'
MATCHED_CLASS_MEAN = 'OneSide_MatchedClassMean'
SHUFFLES = tuple('OneSide_AliasShuffle'+str(i) for i in range(3))
MATCHED_SHUFFLES = tuple('OneSide_MatchedAliasShuffle'+str(i) for i in range(3))
RIVAL_COLLAPSED = 'OneSide_RivalCollapsed'
MATCHED_RIVAL_COLLAPSED = 'OneSide_MatchedRivalCollapsed'
SHUFFLED_WRITE = 'OneSide_ShuffledWrite'
MATCHED_SHUFFLED_WRITE = 'OneSide_MatchedShuffledWrite'
DIRECT_MATCHED = 'OneSide_DirectMatched'
EQUAL_MEAN = 'OneSide_EqualLogitMean'
EQUAL_NO_ALIAS = 'OneSide_EqualLogitMeanNoAlias'
ALL_SHUFFLED_WRITE = 'OneSide_AllCorrespondenceShuffle'
METHODS = ('Geometry', 'NoAdmission_Exact', 'Geometry_PatchOnly2Coupled', PRIMARY,
    OBSERVATION_MEAN, HARD, CLASS_MEAN, MATCHED_CLASS_MEAN, *SHUFFLES,
    *MATCHED_SHUFFLES, RIVAL_COLLAPSED, MATCHED_RIVAL_COLLAPSED, SHUFFLED_WRITE,
    MATCHED_SHUFFLED_WRITE, DIRECT_MATCHED, EQUAL_MEAN, EQUAL_NO_ALIAS, ALL_SHUFFLED_WRITE)
DIAGNOSTICS = ('class_budget_max_error', 'rival_budget_max_error',
    'matched_norm_relative_error', 'matched_unmatchable',
    'positive_directed_delta_max', 'mass_log_fallbacks')
PROTOCOL = {**SOURCE_PROTOCOL,
    'alias_admission': 'EXACT existing single-sided wide-positive/fine-negative1-risk rule; no G truth gate',
    'aggregation': 'fixed20 outside-exponent soft weights; no survivor compensation',
    'writer': 'unchanged local anchor, equal wide/fine positive observation and half wide potential through original H',
    'controls': METHODS[4:], 'new_model_equations': False, 'fitted_parameters': 0,
    'interpretation': 'prospective attribution of a previously developed control; not a new candidate invention',
    'matching': 'per-tile whole valid class-field Frobenius norm after H; NOT pixelwise margin matching'}


def collapse_rivals(risk, parents):
    if (risk.ndim != 3 or risk.shape[1] != len(parents) or risk.shape[2] < 2
            or parents.dtype != torch.long or bool((parents < 0).any())
            or bool((parents >= risk.shape[2]).any())):
        raise ValueError('Matching alias parents and at least two competitor classes required.')
    alternatives = torch.arange(risk.shape[2], device=risk.device)[None] != parents[:, None]
    result = risk.double().sum(-1, keepdim=True)/(risk.shape[2]-1)
    result = result.expand_as(risk).masked_fill(~alternatives[None], 0.)
    error = float((result.sum(-1)-risk.double().sum(-1)).abs().max())
    return result, error


@torch.inference_mode()
def scores(local, operator, broad, fine_field, wide, fine, coordinates,
           relation, valid, members, canonical, parents, *, methods=METHODS[3:], risk_builder=None):
    if not methods or not set(methods).issubset(METHODS[3:]):
        raise ValueError('Declared one-sided audit methods required.')
    baseline = local.double()+operator.double()@(broad.double()-local.double())
    innovation = .5*(fine_field.double()-broad.double())
    innovation.masked_fill_(~valid[:, None], 0.)
    positive = baseline+operator.double()@innovation
    values = {OBSERVATION_MEAN: positive} if OBSERVATION_MEAN in methods else {}
    stats = dict.fromkeys((*DIAGNOSTICS, 'canonical_risk_max', 'mean_absolute_admission_potential',
        'normal_equation_max_error', 'gauge_max_error', 'deleted_alias_rival_fraction'), 0.)
    stats['retained_count_mean'] = float(members.shape[1])
    if set(methods) == {OBSERVATION_MEAN}:
        return values, stats
    wm = sampled_cached(wide, coordinates)
    fm = sampled_cached(fine, coordinates).masked_fill(~valid[:, None, None], 0.)
    risk = (native_risk(wm, fm, fm, parents, canonical, valid, CONFIG)
            if risk_builder is None else risk_builder(wm, fm, parents, canonical, valid))
    directed, diagnostics = fixed_slot_action(wide, members, risk, valid)
    potential, consistency = signed_potential(directed, valid)
    written = .5*(operator.double()@potential)
    if PRIMARY in methods:
        values[PRIMARY] = positive+written
    if HARD in methods:
        hard, _ = signed_potential(directed_cached(wide, members, risk, valid), valid)
        values[HARD] = positive+.5*(operator.double()@hard)

    def matched(field):
        result, info = match_previous(field, written, valid)
        stats['matched_norm_relative_error'] = max(stats['matched_norm_relative_error'],
            info['matched_previous_norm_relative_error'])
        stats['matched_unmatchable'] += info['matched_previous_unmatchable']
        return result

    if set(methods) & {SHUFFLED_WRITE, MATCHED_SHUFFLED_WRITE, ALL_SHUFFLED_WRITE}:
        shuffled = permuted_operator(operator, valid).double()
        shuffled_action = .5*(shuffled@potential)
        if SHUFFLED_WRITE in methods:
            values[SHUFFLED_WRITE] = positive+shuffled_action
        if MATCHED_SHUFFLED_WRITE in methods:
            values[MATCHED_SHUFFLED_WRITE] = positive+matched(shuffled_action)
        if ALL_SHUFFLED_WRITE in methods:
            corrected = broad.double()+innovation+.5*potential
            values[ALL_SHUFFLED_WRITE] = local.double()+shuffled@(corrected-local.double())
    if DIRECT_MATCHED in methods:
        values[DIRECT_MATCHED] = positive+matched(.5*potential)
    corrected = broad.double()+innovation
    if EQUAL_MEAN in methods:
        values[EQUAL_MEAN] = .5*(local.double()+corrected+.5*potential)
    if EQUAL_NO_ALIAS in methods:
        values[EQUAL_NO_ALIAS] = .5*(local.double()+corrected)

    pairs = ((CLASS_MEAN, MATCHED_CLASS_MEAN, PREVIOUS_CLASS_MEAN),
             *zip(SHUFFLES, MATCHED_SHUFFLES, PREVIOUS_SHUFFLES))
    if any(name in methods for row in pairs for name in row[:2]):
        controls, error = continuous_controls(risk, members, canonical)
        stats['class_budget_max_error'] = error
        for raw_name, matched_name, key in pairs:
            if raw_name in methods or matched_name in methods:
                delta, _ = fixed_slot_action(wide, members, controls[key], valid)
                changed, _ = signed_potential(delta, valid)
                changed_write = .5*(operator.double()@changed)
                if raw_name in methods:
                    values[raw_name] = positive+changed_write
                if matched_name in methods:
                    values[matched_name] = positive+matched(changed_write)
    if set(methods) & {RIVAL_COLLAPSED, MATCHED_RIVAL_COLLAPSED}:
        changed_risk, error = collapse_rivals(risk, parents)
        stats['rival_budget_max_error'] = error
        delta, _ = fixed_slot_action(wide, members, changed_risk, valid)
        changed, _ = signed_potential(delta, valid)
        changed_write = .5*(operator.double()@changed)
        if RIVAL_COLLAPSED in methods:
            values[RIVAL_COLLAPSED] = positive+changed_write
        if MATCHED_RIVAL_COLLAPSED in methods:
            values[MATCHED_RIVAL_COLLAPSED] = positive+matched(changed_write)
    stats.update({**risk_statistics(risk, potential, valid, members, canonical),
                  **diagnostics, **consistency})
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite one-sided audit prediction.')
    return values, stats


def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS,
                  cache=None, execution=None):
    return source_predict(image, geometry, banks, vip, queries, work, methods=methods,
        cache=cache, execution=execution, tile_scorer=scores, allowed_methods=METHODS,
        observation_method=OBSERVATION_MEAN, diagnostic_fields=DIAGNOSTICS)
