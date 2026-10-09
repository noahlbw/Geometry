"""Held-out RGB response references for jointly supported competing aliases."""
from dataclasses import dataclass
from math import isqrt

import torch

from .fine_alias_view import CONFIG
from .native_alias_noise import signed_potential
from .native_query_alias import native_risk
from .reciprocal_alias_admission import (CLASS_MEAN as OLD_CLASS, SHUFFLES as OLD_SHUFFLES,
    continuous_controls, fixed_slot_action)
from .rival_alias_fast import cache_observations, directed_cached, risk_statistics, sampled_cached
from .rival_rank_density import rank_densities
from .supported_positive_alias import PROTOCOL as SOURCE_PROTOCOL, permuted_operator, predict_image as source_predict
from .witness_cap_alias import match_previous


IMPLEMENTATION = 'geometry-bounded896-heldout-rank-alias-reference-v1-20261006'
PRIMARY = 'CrossRef_Soft'
OBSERVATION_MEAN = 'CrossRef_ObservationMean'
OLD_SOFT = 'CrossRef_PreviousSoft'
OLD_HARD = 'CrossRef_PreviousHard'
MATCHED_OLD = 'CrossRef_MatchedPrevious'
CLASS_MEAN = 'CrossRef_MatchedClassMean'
SHUFFLES = tuple('CrossRef_MatchedAliasShuffle'+str(i) for i in range(3))
REFERENCE_NULL = 'CrossRef_MatchedReferenceShuffle'
IN_IMAGE = 'CrossRef_MatchedInImage'
RIVAL_NULL = 'CrossRef_MatchedRivalCollapsed'
WRITE_NULL = 'CrossRef_MatchedShuffledWrite'
DIRECT = 'CrossRef_DirectMatched'
METHODS = ('Geometry', 'NoAdmission_Exact', 'Geometry_PatchOnly2Coupled', PRIMARY,
    OBSERVATION_MEAN, OLD_SOFT, OLD_HARD, MATCHED_OLD, CLASS_MEAN, *SHUFFLES,
    REFERENCE_NULL, IN_IMAGE, RIVAL_NULL, WRITE_NULL, DIRECT)
DIAGNOSTICS = ('reference_known_fraction', 'joint_positive_fraction', 'additional_risk_mean',
    'additional_risk_fraction', 'previous_risk_mean', 'class_budget_max_error',
    'matched_norm_relative_error', 'matched_unmatchable', 'positive_directed_delta_max',
    'mass_log_fallbacks')
PROTOCOL = {**SOURCE_PROTOCOL,
    'alias_admission': 'old contradiction risk plus held-out class-balanced rank evidence on wide-positive/fine-nonnegative comparisons',
    'reference_source': '64 independent RGB images/domain; UDD5 train RGB, other domains transductive held-out evaluation RGB',
    'reference_labels': 'canonical-only raw fine-logit softmax times top2 probability gap; fallible pseudo-reference, no masks',
    'reference_sampling': 'fine32-grid offset4/stride8; remove padded centers; at most16 centers/crop',
    'reference_lookup': 'ceil(sqrt(N)) scalar-response rank neighbors with expanded boundary ties; no extrapolation',
    'reference_risk': 'positive part of (rival density-own density)/(rival density+own density)',
    'reference_availability': 'at least one weighted reference-token equivalent/class, otherwise UNKNOWN',
    'aggregation': 'unchanged fixed20-slot outside-exponent weights; no survivor compensation',
    'writer': 'unchanged positive wide/fine observation and half wide potential through original H',
    'additional_query_encodings': 0, 'calibration_images_per_domain': 64, 'fitted_parameters': 0,
    'controls': METHODS[4:],
    'precedents': 'empirical rank neighborhoods, pseudo-labels, balanced density contrasts and ridge writing are known operations',
    'limitation': 'canonical pseudo-categories can be wrong; corpus agreement does not certify semantics'}


def canonical_reference(raw, canonical):
    if raw.ndim != 2 or len(canonical) < 2 or not bool(torch.isfinite(raw).all()):
        raise ValueError('Finite raw responses and at least two canonical classes required.')
    probabilities = raw[:, canonical].double().softmax(-1)
    top = probabilities.topk(2, dim=-1).values
    return probabilities*(top[:, :1]-top[:, 1:2])


@dataclass(frozen=True)
class RankReference:
    values: torch.Tensor
    prefix: torch.Tensor
    masses: torch.Tensor

    @classmethod
    def build(cls, raw, reference):
        if (raw.ndim != 2 or reference.ndim != 2 or len(raw) != len(reference) or len(raw) < 2
                or not bool(torch.isfinite(raw).all() and torch.isfinite(reference).all())
                or bool((reference < 0).any())):
            raise ValueError('At least two finite calibration rows and nonnegative pseudo-references required.')
        values, order = raw.double().T.contiguous().sort(dim=1, stable=True)
        weights = reference.double()[order]
        prefix = torch.cat((torch.zeros_like(weights[:, :1]), weights.cumsum(1)), dim=1)
        return cls(values, prefix, reference.double().sum(0))

    def lookup(self, raw):
        if raw.ndim != 2 or raw.shape[1] != len(self.values) or not bool(torch.isfinite(raw).all()):
            raise ValueError('Finite query responses with the unchanged alias layout required.')
        n = self.values.shape[1]
        neighbors = min(n, max(2, isqrt(n-1)+1))
        density = torch.zeros((*raw.shape, len(self.masses)), dtype=torch.float64, device=raw.device)
        available = torch.zeros_like(density, dtype=torch.bool)
        for start in range(0, raw.shape[1], 32):
            sl = slice(start, start+32)
            values = self.values[sl].contiguous()
            queries = raw[:, sl].double().T.contiguous()
            first = torch.searchsorted(values, queries, right=False)
            last = torch.searchsorted(values, queries, right=True)
            middle = (first+last-1)//2
            low = (middle-neighbors//2).clamp(0, n-neighbors)
            high = low+neighbors-1
            low = torch.searchsorted(values, values.gather(1, low), right=False)
            high = torch.searchsorted(values, values.gather(1, high), right=True)
            prefix = self.prefix[sl]
            total = prefix.gather(1, high[..., None].expand(-1, -1, len(self.masses)))
            total -= prefix.gather(1, low[..., None].expand(-1, -1, len(self.masses)))
            total = total.clamp_min(0.)/self.masses.clamp_min(CONFIG.epsilon)
            known = ((queries >= values[:, :1]) & (queries <= values[:, -1:]))[..., None]
            known = known & (self.masses >= 1.)[None, None]
            density[:, sl] = total.transpose(0, 1).masked_fill(~known.transpose(0, 1), 0.)
            available[:, sl] = known.transpose(0, 1)
        return density, available


@dataclass(frozen=True)
class RawCachedCrop:
    evidence: torch.Tensor
    margins: torch.Tensor
    indices: torch.Tensor
    coefficients: torch.Tensor
    raw: torch.Tensor


def cache_raw(crops, count, coordinates, image_size, members):
    cached = cache_observations(crops, count, coordinates, image_size, members)
    return tuple(RawCachedCrop(c.evidence, c.margins, c.indices, c.coefficients, source.alias_logits)
                 for c, source in zip(cached, crops))


def sampled_raw(crops, coordinates):
    result = coordinates.new_zeros((len(coordinates), crops[0].raw.shape[1]))
    for crop in crops:
        result += (crop.raw[crop.indices]*crop.coefficients[..., None]).sum(1)
    return result


def joint_reference_risk(density, available, wide, fine, parents, canonical, valid):
    if density.shape != wide.shape or available.shape != density.shape or fine.shape != wide.shape:
        raise ValueError('Matching query/alias/rival density, availability and margins required.')
    own = density.gather(-1, parents[None, :, None].expand(len(valid), -1, 1))
    own_known = available.gather(-1, parents[None, :, None].expand(len(valid), -1, 1))
    total = own+density
    eligible = valid[:, None, None] & (wide > 0) & (fine >= 0)
    eligible &= own_known & available & (total > CONFIG.epsilon)
    risk = torch.where(eligible, (density-own).clamp_min(0.)/total.clamp_min(CONFIG.epsilon), 0.)
    risk[:, canonical] = 0.
    risk.scatter_(-1, parents[None, :, None].expand(len(valid), -1, 1), 0.)
    return risk


@torch.inference_mode()
def scores(local, operator, broad, fine_field, wide, fine, coordinates,
           relation, valid, members, canonical, parents, reference, shuffled_reference=None,
           *, methods=METHODS[3:]):
    if not methods or not set(methods).issubset(METHODS[3:]):
        raise ValueError('Declared held-out reference endpoints required.')
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
    old = native_risk(wm, fm, fm, parents, canonical, valid, CONFIG)

    def field(risk):
        directed, info = fixed_slot_action(wide, members, risk, valid)
        potential, consistency = signed_potential(directed, valid)
        return .5*(operator.double()@potential), potential, {**info, **consistency}

    old_write = None
    if set(methods) & {OLD_SOFT, MATCHED_OLD}:
        old_write, _, _ = field(old)
        if OLD_SOFT in methods:
            values[OLD_SOFT] = positive+old_write
    if OLD_HARD in methods:
        hard, _ = signed_potential(directed_cached(wide, members, old, valid), valid)
        values[OLD_HARD] = positive+.5*(operator.double()@hard)
    if not set(methods)-{OBSERVATION_MEAN, OLD_SOFT, OLD_HARD}:
        return values, stats
    raw = sampled_raw(fine, coordinates)
    density, available = reference.lookup(raw)
    extra = joint_reference_risk(density, available, wm, fm, parents, canonical, valid)
    risk = torch.maximum(old.double(), extra)
    written, potential, info = field(risk)
    if PRIMARY in methods:
        values[PRIMARY] = positive+written

    def matched(changed):
        result, diagnostics = match_previous(changed, written, valid)
        stats['matched_norm_relative_error'] = max(stats['matched_norm_relative_error'],
            diagnostics['matched_previous_norm_relative_error'])
        stats['matched_unmatchable'] += diagnostics['matched_previous_unmatchable']
        return positive+result

    if MATCHED_OLD in methods:
        values[MATCHED_OLD] = matched(old_write)
    if DIRECT in methods:
        values[DIRECT] = matched(.5*potential)
    if WRITE_NULL in methods:
        values[WRITE_NULL] = matched(.5*(permuted_operator(operator, valid).double()@potential))
    if set(methods) & {CLASS_MEAN, *SHUFFLES}:
        controls, error = continuous_controls(risk, members, canonical)
        stats['class_budget_max_error'] = error
        for name, key in ((CLASS_MEAN, OLD_CLASS), *zip(SHUFFLES, OLD_SHUFFLES)):
            if name in methods:
                values[name] = matched(field(controls[key])[0])
    if RIVAL_NULL in methods:
        from .one_sided_alias_audit import collapse_rivals
        changed, _ = collapse_rivals(risk, parents)
        values[RIVAL_NULL] = matched(field(changed)[0])
    if REFERENCE_NULL in methods:
        null, known = shuffled_reference.lookup(raw)
        changed = joint_reference_risk(null, known, wm, fm, parents, canonical, valid)
        values[REFERENCE_NULL] = matched(field(torch.maximum(old.double(), changed))[0])
    if IN_IMAGE in methods:
        source = canonical_reference(raw, canonical)
        source_density, known = rank_densities(raw, source, valid, exclude_self=True)
        masses = source[valid].sum(0)[None]-source
        known &= masses >= 1.
        known = known[:, None].expand_as(source_density)
        changed = joint_reference_risk(source_density, known, wm, fm, parents, canonical, valid)
        values[IN_IMAGE] = matched(field(torch.maximum(old.double(), changed))[0])
    stats.update({**risk_statistics(risk, potential, valid, members, canonical), **info,
        'reference_known_fraction': float(available[valid].double().mean()),
        'joint_positive_fraction': float(((wm[valid] > 0) & (fm[valid] >= 0)).double().mean()),
        'additional_risk_mean': float(extra[valid].mean()),
        'additional_risk_fraction': float((extra[valid] > 0).double().mean()),
        'previous_risk_mean': float(old[valid].double().mean())})
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite held-out reference scores.')
    return values, stats


def predict_image(image, geometry, banks, vip, queries, work=None, *, references,
                  shuffled_references=None, methods=METHODS, cache=None, execution=None):
    def score(*args, methods):
        parents = args[-1]
        key = next(p for p, q in queries.items() if q.parents is parents)
        return scores(*args, references[key], None if shuffled_references is None else shuffled_references[key], methods=methods)
    return source_predict(image, geometry, banks, vip, queries, work, methods=methods, cache=cache,
        execution=execution, tile_scorer=score, allowed_methods=METHODS,
        observation_method=OBSERVATION_MEAN, diagnostic_fields=DIAGNOSTICS, observation_cacher=cache_raw)
