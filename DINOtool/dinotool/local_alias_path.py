"""Condition local lexical contributions and retain the original fidelity solve."""
from types import FunctionType

import torch
import torch.nn.functional as F

from .bounded_patch_only import readout as original_readout
from .fine_alias_view import CONFIG
from .matched_contribution_alias import contribution_margins
from .native_alias_noise import signed_potential
from .native_query_alias import native_risk
from .reciprocal_alias_admission import (CLASS_MEAN as OLD_CLASS_MEAN,
    SHUFFLES as OLD_SHUFFLES, continuous_controls, fixed_slot_action)
from .rival_alias_fast import CachedCrop, directed_cached, risk_statistics, sampled_cached
from .supported_positive_alias import (PROTOCOL as SOURCE_PROTOCOL,
    permuted_operator, predict_image as source_predict)
from .witness_cap_alias import match_previous


IMPLEMENTATION = 'geometry-bounded896-local-alias-fidelity-path-v1-20261006'
PRIMARY = 'LocalPath_JointSoft'
OBSERVATION_MEAN = 'LocalPath_ObservationMean'
OLD_SOFT = 'LocalPath_PreviousSoft'
OLD_HARD = 'LocalPath_PreviousHard'
MATCHED_OLD = 'LocalPath_MatchedPrevious'
CLASS_MEAN = 'LocalPath_ClassMean'
SHUFFLES = tuple('LocalPath_AliasShuffle'+str(i) for i in range(3))
SHUFFLED_WRITE = 'LocalPath_ShuffledWrite'
LOCAL_ONLY = 'LocalPath_LocalOnly'
REFERENCE_SHUFFLE = 'LocalPath_ReferenceShuffle'
DIRECT_MATCHED = 'LocalPath_DirectMatched'
METHODS = ('Geometry', 'NoAdmission_Exact', 'Geometry_PatchOnly2Coupled', PRIMARY,
    OBSERVATION_MEAN, OLD_SOFT, OLD_HARD, MATCHED_OLD, CLASS_MEAN, *SHUFFLES,
    SHUFFLED_WRITE, LOCAL_ONLY, REFERENCE_SHUFFLE, DIRECT_MATCHED)
DIAGNOSTICS = ('local_risk_mean', 'local_canonical_risk_max', 'local_invalid_write_max',
    'class_budget_max_error', 'positive_directed_delta_max', 'mass_log_fallbacks',
    'matched_previous_norm_relative_error', 'matched_previous_unmatchable', 'matched_previous_scale',
    'direct_match_norm_relative_error', 'direct_match_unmatchable', 'local_write_mean_absolute')
PROTOCOL = {**SOURCE_PROTOCOL,
    'alias_admission': 'local alias advantage contradicted by existing fine feature with the SAME local text bank',
    'local_template': 'unchanged normalized Geometry text bank and inherited0.07 scale; no salience added',
    'wide_action': 'exact prior single-sided wide soft intervention, unchanged',
    'local_writer': 'original fidelity objective: local anchor L+u; correction (I-H)u, no new mixing coefficient',
    'writer': 'unchanged H for positive wide/fine observation and prior half-wide word action',
    'controls': METHODS[4:], 'additional_visual_or_head_encodings': 0, 'fitted_parameters': 0,
    'limitation': 'fine features can be wrong; fidelity is to the corrected local anchor, not a no-harm guarantee'}


class FeatureCapture:
    def __init__(self, reader, state):
        self.reader, self.state = reader, state

    @property
    def calls(self):
        return self.reader.calls

    def __call__(self, observer, crops):
        features = self.reader(observer, crops)
        self.state['fine'] = features
        return features


class CapturedExecution:
    def __init__(self, execution, state):
        self.execution, self.state = execution, state

    def reader(self, observer):
        return FeatureCapture(self.execution.reader(observer), self.state)


def local_observations(features, texts, members, fine):
    if len(features) != len(fine) or not len(fine):
        raise ValueError('One already-computed fine feature tensor per original crop required.')
    output = []
    for value, crop in zip(features, fine):
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] != len(crop.evidence):
            raise ValueError('Original batch-one fine features required.')
        evidence = ((value.float()@texts.T)[0]/.07)[:, members]
        output.append(CachedCrop(evidence, contribution_margins(evidence), crop.indices, crop.coefficients))
    return output


def local_action(logits, risk, valid, members):
    if logits.shape != (len(valid), *members.shape):
        raise ValueError('Original grouped local alias logits required.')
    ids = torch.arange(len(valid), device=valid.device)[:, None]
    crop = CachedCrop(logits, contribution_margins(logits), ids, logits.new_ones(len(valid), 1))
    directed, diagnostics = fixed_slot_action((crop,), members, risk, valid)
    potential, consistency = signed_potential(directed, valid)
    return potential, {**diagnostics, **consistency}


@torch.inference_mode()
def local_scores(local, operator, broad, fine_field, wide, fine, coordinates,
                 relation, valid, members, canonical, parents, local_logits, fine_local,
                 *, methods=METHODS[3:]):
    if not methods or not set(methods).issubset(METHODS[3:]):
        raise ValueError('Declared local-alias methods required.')
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
    wide_risk = native_risk(wm, fm, fm, parents, canonical, valid, CONFIG)
    directed, diagnostics = fixed_slot_action(wide, members, wide_risk, valid)
    potential, consistency = signed_potential(directed, valid)
    wide_write = .5*(operator.double()@potential)
    previous = positive+wide_write
    if OLD_SOFT in methods:
        values[OLD_SOFT] = previous
    if OLD_HARD in methods:
        potential, _ = signed_potential(directed_cached(wide, members, wide_risk, valid), valid)
        values[OLD_HARD] = positive+.5*(operator.double()@potential)
    stats.update({**diagnostics, **consistency})
    local_methods = set(methods)-{OBSERVATION_MEAN, OLD_SOFT, OLD_HARD}
    if not local_methods:
        return values, stats
    lm = contribution_margins(local_logits)
    reference = sampled_cached(fine_local, coordinates).masked_fill(~valid[:, None, None], 0.)
    local_risk = native_risk(lm, reference, reference, parents, canonical, valid, CONFIG)
    potential, diagnostics = local_action(local_logits, local_risk, valid, members)
    fidelity = torch.eye(len(valid), device=operator.device, dtype=torch.float64)-operator.double()
    written = fidelity@potential
    if PRIMARY in methods:
        values[PRIMARY] = previous+written
    if LOCAL_ONLY in methods:
        values[LOCAL_ONLY] = positive+written
    if MATCHED_OLD in methods:
        matched, extra = match_previous(wide_write, wide_write+written, valid)
        values[MATCHED_OLD] = positive+matched
        stats.update(extra)
    if DIRECT_MATCHED in methods:
        matched, extra = match_previous(potential, written, valid)
        values[DIRECT_MATCHED] = previous+matched
        stats.update(direct_match_norm_relative_error=extra['matched_previous_norm_relative_error'],
            direct_match_unmatchable=extra['matched_previous_unmatchable'])
    if SHUFFLED_WRITE in methods:
        values[SHUFFLED_WRITE] = previous+permuted_operator(fidelity, valid).double()@potential
    if REFERENCE_SHUFFLE in methods:
        ids = valid.nonzero().flatten()
        generator = torch.Generator().manual_seed(CONFIG.random_seed)
        permutation = ids[torch.randperm(len(ids), generator=generator).to(ids.device)]
        shuffled = reference.clone()
        shuffled[ids] = reference[permutation]
        risk = native_risk(lm, shuffled, shuffled, parents, canonical, valid, CONFIG)
        delta, _ = local_action(local_logits, risk, valid, members)
        values[REFERENCE_SHUFFLE] = previous+fidelity@delta
    if any(m in methods for m in (CLASS_MEAN, *SHUFFLES)):
        controls, error = continuous_controls(local_risk, members, canonical)
        stats['class_budget_max_error'] = error
        for new, old in ((CLASS_MEAN, OLD_CLASS_MEAN), *zip(SHUFFLES, OLD_SHUFFLES)):
            if new in methods:
                delta, _ = local_action(local_logits, controls[old], valid, members)
                values[new] = previous+fidelity@delta
    stats.update({**risk_statistics(local_risk, potential, valid, members, canonical), **diagnostics,
        'local_risk_mean': float(local_risk[valid].double().mean()) if bool(valid.any()) else 0.,
        'local_canonical_risk_max': float(local_risk[:, canonical].abs().max()),
        'local_invalid_write_max': float(written[~valid].abs().max()) if bool((~valid).any()) else 0.,
        'local_write_mean_absolute': float(written[valid].abs().mean()) if bool(valid.any()) else 0.})
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite local-fidelity correction.')
    return values, stats


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS,
                  cache=None, execution=None):
    from .bounded_fine_execution import FineCoverageExecution

    state = {}
    texts = {id(queries[p].parents): F.normalize(bank.features.float(), dim=-1) for p, bank in banks.items()}
    for p, bank in banks.items():
        if tuple(bank.alias_names) != tuple(queries[p].aliases) or not torch.equal(bank.parent_indices, queries[p].parents):
            raise ValueError('Unchanged local/wide alias slot identity required.')
    def capture(*args, **kwargs):
        value = original_readout(*args, **kwargs)
        state['local'] = value[0]
        return value
    def score(local, operator, broad, fine_field, wide, fine, coordinates,
              relation, valid, members, canonical, parents, *, methods):
        if set(methods)-{OBSERVATION_MEAN, OLD_SOFT, OLD_HARD}:
            bank = texts[id(parents)]
            logits = ((state['local'].float()@bank.T)[0]/.07)[:, members]
            observations = local_observations(state['fine'], bank, members, fine)
        else:
            logits, observations = None, ()
        return local_scores(local, operator, broad, fine_field, wide, fine, coordinates,
            relation, valid, members, canonical, parents, logits, observations, methods=methods)
    template = getattr(source_predict, '__wrapped__', source_predict)
    predictor = FunctionType(template.__code__, dict(template.__globals__, readout=capture),
        template.__name__, template.__defaults__)
    predictor.__kwdefaults__ = template.__kwdefaults__
    backend = CapturedExecution(FineCoverageExecution(cached=True, burst=True) if execution is None else execution, state)
    return predictor(image, geometry, banks, vip, queries, work, methods=methods,
        cache=cache, execution=backend, tile_scorer=score, allowed_methods=METHODS,
        observation_method=OBSERVATION_MEAN, diagnostic_fields=DIAGNOSTICS)
