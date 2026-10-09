"""Separate conditional alias witnesses from the scored pool's salience and words."""
from dataclasses import dataclass
import math

import torch

from .fine_alias_view import CONFIG
from .native_alias_noise import signed_potential
from .native_query_alias import native_risk
from .reciprocal_alias_admission import continuous_controls, fixed_slot_action
from .rival_alias_fast import cache_observations, directed_cached, risk_statistics, sampled_cached
from .supported_positive_alias import (PROTOCOL as SOURCE_PROTOCOL, permuted_operator,
    predict_image as source_predict)


IMPLEMENTATION = 'geometry-bounded896-pool-isolated-alias-witness-v1-20261006'
PRIMARY = 'PoolIsolated_Canonical'
OBSERVATION_MEAN = 'PoolIsolated_ObservationMean'
PROFILE_POOL = 'PoolIsolated_ProfilePool'
PROFILE_CANONICAL = 'PoolIsolated_ProfileCanonical'
RAW_POOL = 'PoolIsolated_RawPool'
FINE_ONLY = 'PoolIsolated_FineOnly'
WIDE_HARD = 'PoolIsolated_WideHard'
CLASS_MEAN = 'PoolIsolated_ClassMean'
SHUFFLES = tuple('PoolIsolated_AliasShuffle'+str(i) for i in range(3))
SHUFFLED_WRITE = 'PoolIsolated_ShuffledWrite'
METHODS = ('Geometry', 'NoAdmission_Exact', 'Geometry_PatchOnly2Coupled', PRIMARY,
    OBSERVATION_MEAN, PROFILE_POOL, PROFILE_CANONICAL, RAW_POOL, FINE_ONLY,
    WIDE_HARD, CLASS_MEAN, *SHUFFLES, SHUFFLED_WRITE)
DIAGNOSTICS = ('profile_risk_mean', 'isolated_risk_mean', 'risk_support_change_fraction',
    'class_budget_max_error', 'positive_directed_delta_max', 'mass_log_fallbacks')
PROTOCOL = {**SOURCE_PROTOCOL,
    'alias_admission': 'original one-sided cross-scale risk from raw alias vs rival canonical, on both scales',
    'reference': 'raw existing mean-template logits; no vocabulary salience or noncanonical rival pool',
    'prediction': 'unchanged profiled20-word local/wide/fine class observations and calibration',
    'aggregation': 'continuous1-risk outside exponent, fixed20 slots; no survivor compensation',
    'writer': 'positive equal wide/fine plus half wide intervention through unchanged original H',
    'controls': METHODS[4:], 'additional_visual_or_head_encodings': 0, 'fitted_parameters': 0,
    'limitation': 'canonical isolation removes pool dependence, not canonical ambiguity or visual errors'}


@dataclass(frozen=True)
class IsolatedCrop:
    evidence: torch.Tensor
    margins: torch.Tensor
    indices: torch.Tensor
    coefficients: torch.Tensor
    raw: torch.Tensor


def cache_raw_observations(crops, count, coordinates, image_size, members):
    cached = cache_observations(crops, count, coordinates, image_size, members)
    return tuple(IsolatedCrop(row.evidence, row.margins, row.indices, row.coefficients,
                              crop.alias_logits[:, members]) for row, crop in zip(cached, crops))


def reference_margins(evidence, members, canonical, reference):
    if (evidence.ndim != 3 or evidence.shape[1:] != members.shape or reference not in ('pool', 'canonical')
            or canonical.shape != members.shape[:1] or not bool(torch.isfinite(evidence).all())):
        raise ValueError('Matching finite alias evidence, canonical indices and declared reference required.')
    if reference == 'pool':
        rival = evidence.logsumexp(-1)-math.log(members.shape[-1])
    else:
        matching = members == canonical[:, None]
        if not bool((matching.sum(-1) == 1).all()):
            raise ValueError('Exactly one actual canonical per class required.')
        positions = matching.long().argmax(-1)
        rival = evidence[:, torch.arange(len(members), device=members.device), positions]
    margins = torch.empty((len(evidence), members.numel(), len(members)), dtype=evidence.dtype, device=evidence.device)
    margins[:, members.flatten()] = evidence.flatten(1)[..., None]-rival[:, None]
    return margins


def sampled_reference(observations, members, canonical, valid, *, raw, reference):
    if not raw and reference == 'pool':
        return sampled_cached(observations, torch.empty(len(valid), 2, device=valid.device)).masked_fill(
            ~valid[:, None, None], 0.)
    first = observations[0]
    result = torch.zeros((len(valid), members.numel(), len(members)), dtype=first.evidence.dtype, device=valid.device)
    for crop in observations:
        source = crop.raw if raw else crop.evidence
        margins = reference_margins(source, members, canonical, reference)
        result += (margins[crop.indices]*crop.coefficients[..., None, None]).sum(1)
    return result.masked_fill(~valid[:, None, None], 0.)


@torch.inference_mode()
def isolated_scores(local, operator, broad, fine_field, wide, fine, coordinates,
                    relation, valid, members, canonical, parents, *, methods=METHODS[3:]):
    if not methods or not set(methods).issubset(METHODS[3:]):
        raise ValueError('Declared pool-isolation methods required.')
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

    def risk_for(raw, reference):
        wm = sampled_reference(wide, members, canonical, valid, raw=raw, reference=reference)
        fm = sampled_reference(fine, members, canonical, valid, raw=raw, reference=reference)
        return native_risk(wm, fm, fm, parents, canonical, valid, CONFIG)

    needs_isolated = bool(set(methods).intersection((PRIMARY, CLASS_MEAN, *SHUFFLES, SHUFFLED_WRITE)))
    risks = {}
    if needs_isolated:
        risks[PRIMARY] = risk_for(True, 'canonical')
    if PROFILE_POOL in methods or WIDE_HARD in methods or PRIMARY in methods or FINE_ONLY in methods:
        risks[PROFILE_POOL] = risk_for(False, 'pool')
    if PROFILE_CANONICAL in methods:
        risks[PROFILE_CANONICAL] = risk_for(False, 'canonical')
    if RAW_POOL in methods:
        risks[RAW_POOL] = risk_for(True, 'pool')
    if FINE_ONLY in methods:
        wm = sampled_reference(wide, members, canonical, valid, raw=False, reference='pool')
        fm = sampled_reference(fine, members, canonical, valid, raw=True, reference='canonical')
        risks[FINE_ONLY] = native_risk(wm, fm, fm, parents, canonical, valid, CONFIG)
    if needs_isolated:
        from .reciprocal_alias_admission import CLASS_MEAN as OLD_CLASS_MEAN, SHUFFLES as OLD_SHUFFLES

        controls, error = continuous_controls(risks[PRIMARY], members, canonical)
        stats['class_budget_max_error'] = error
        risks[CLASS_MEAN] = controls[OLD_CLASS_MEAN]
        risks.update({new: controls[old] for new, old in zip(SHUFFLES, OLD_SHUFFLES)})
    for method, risk in risks.items():
        if method not in methods and not (method == PRIMARY and SHUFFLED_WRITE in methods):
            continue
        directed, diagnostics = fixed_slot_action(wide, members, risk, valid)
        potential, consistency = signed_potential(directed, valid)
        if method in methods:
            values[method] = positive+.5*(operator.double()@potential)
        if method == PRIMARY:
            if SHUFFLED_WRITE in methods:
                values[SHUFFLED_WRITE] = positive+.5*(permuted_operator(operator, valid).double()@potential)
            stats.update({**risk_statistics(risk, potential, valid, members, canonical),
                          **diagnostics, **consistency})
            if PROFILE_POOL in risks:
                before = risks[PROFILE_POOL][valid]
                after = risk[valid]
                stats.update(profile_risk_mean=float(before.double().mean()),
                    isolated_risk_mean=float(after.double().mean()),
                    risk_support_change_fraction=float(((before > 0) != (after > 0)).double().mean()))
    if WIDE_HARD in methods:
        directed = directed_cached(wide, members, risks[PROFILE_POOL], valid)
        potential, _ = signed_potential(directed, valid)
        values[WIDE_HARD] = positive+.5*(operator.double()@potential)
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite isolated witness scores.')
    return values, stats


def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS,
                  cache=None, execution=None):
    return source_predict(image, geometry, banks, vip, queries, work, methods=methods,
        cache=cache, execution=execution, tile_scorer=isolated_scores, allowed_methods=METHODS,
        observation_method=OBSERVATION_MEAN, diagnostic_fields=DIAGNOSTICS,
        observation_cacher=cache_raw_observations)
