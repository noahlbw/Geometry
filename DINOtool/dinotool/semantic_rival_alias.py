"""Judge each alias against a semantically matched rival response distribution."""
import torch
import torch.nn.functional as F

from .fine_alias_view import CONFIG
from .native_alias_noise import signed_potential
from .native_query_alias import native_risk
from .reciprocal_alias_admission import (CLASS_MEAN as PREVIOUS_CLASS_MEAN,
    SHUFFLES as PREVIOUS_SHUFFLES, continuous_controls, fixed_slot_action)
from .rival_alias_fast import CachedCrop, directed_cached, risk_statistics, sampled_cached
from .supported_positive_alias import (PROTOCOL as SOURCE_PROTOCOL,
    permuted_operator, predict_image as source_predict)
from .witness_cap_alias import match_previous


IMPLEMENTATION = 'geometry-bounded896-semantic-rival-reference-v1-20261006'
PRIMARY = 'SemanticRival_Soft'
OBSERVATION_MEAN = 'SemanticRival_ObservationMean'
OLD_SOFT = 'SemanticRival_PreviousSoft'
OLD_HARD = 'SemanticRival_PreviousHard'
MATCHED_OLD = 'SemanticRival_MatchedPrevious'
CLASS_MEAN = 'SemanticRival_ClassMean'
SHUFFLES = tuple('SemanticRival_AliasShuffle'+str(i) for i in range(3))
SHUFFLED_WRITE = 'SemanticRival_ShuffledWrite'
KEY_NULL = 'SemanticRival_KeyShuffle'
MATCHED_KEY_NULL = 'SemanticRival_MatchedKeyShuffle'
DIRECT_MATCHED = 'SemanticRival_DirectMatched'
METHODS = ('Geometry', 'NoAdmission_Exact', 'Geometry_PatchOnly2Coupled', PRIMARY,
    OBSERVATION_MEAN, OLD_SOFT, OLD_HARD, MATCHED_OLD, CLASS_MEAN, *SHUFFLES,
    SHUFFLED_WRITE, KEY_NULL, MATCHED_KEY_NULL, DIRECT_MATCHED)
TEXT_TEMPERATURE = .07
DIAGNOSTICS = ('semantic_risk_mean', 'previous_risk_mean', 'risk_support_change_fraction',
    'rival_affinity_max_mass_error', 'rival_affinity_mean_entropy', 'class_budget_max_error',
    'positive_directed_delta_max', 'mass_log_fallbacks', 'matched_previous_norm_relative_error',
    'matched_previous_unmatchable', 'matched_previous_scale', 'matched_key_norm_relative_error',
    'matched_key_unmatchable', 'direct_match_norm_relative_error', 'direct_match_unmatchable')
PROTOCOL = {**SOURCE_PROTOCOL,
    'alias_admission': 'unchanged one-sided wide/fine contradiction against alias-specific semantic rival references',
    'rival_reference': 'text-cosine softmax over each rival class original20 words, inherited0.07 scale',
    'text_source': 'normalized mean of existing VIP query template vectors; no new text encoding',
    'prediction': 'unchanged original20 local/wide/fine scores and salience; only judgment reference changes',
    'aggregation': 'fixed20-slot outside-exponent soft attenuation; no survivor redistribution',
    'writer': 'unchanged positive observation and half wide potential through original H',
    'controls': METHODS[4:], 'additional_visual_or_head_encodings': 0, 'fitted_parameters': 0,
    'limitation': 'text similarity is not semantic ownership or truth; shared observation errors can survive',
    'precedents': 'softmax text matching, weighted LME, contradiction risk and ridge writing are known operations'}


def rival_affinity(texts, members):
    if (texts.ndim != 2 or members.ndim != 2 or members.dtype != torch.long
            or members.numel() != len(texts) or len(members) < 2 or members.shape[1] < 2
            or not torch.equal(members.flatten().sort().values, torch.arange(len(texts), device=members.device))
            or not bool(torch.isfinite(texts).all()) or bool((texts.norm(dim=-1) == 0).any())):
        raise ValueError('Finite nonzero existing text vectors and a complete equal-count class layout required.')
    normalized = F.normalize(texts.double(), dim=-1)
    similarity = torch.einsum('ad,ckd->ack', normalized, normalized[members])
    return (similarity/TEXT_TEMPERATURE).softmax(-1)


def shuffled_affinity(affinity):
    generator = torch.Generator().manual_seed(CONFIG.random_seed)
    result = affinity.clone()
    for c in range(affinity.shape[1]):
        permutation = torch.randperm(affinity.shape[2], generator=generator).to(affinity.device)
        result[:, c] = affinity[:, c, permutation]
    return result


def semantic_margins(evidence, members, affinity):
    if (evidence.ndim != 3 or evidence.shape[1:] != members.shape
            or affinity.shape != (members.numel(), *members.shape)
            or not bool(torch.isfinite(evidence).all() and torch.isfinite(affinity).all())
            or bool((affinity < 0).any()) or float((affinity.sum(-1)-1).abs().max()) > 1e-10):
        raise ValueError('Finite grouped observations and row-stochastic semantic references required.')
    logits = evidence.double()
    shift = logits.amax(-1)
    mass = torch.einsum('nck,ack->nac', (logits-shift[..., None]).exp(), affinity.double())
    reference = mass.clamp_min(torch.finfo(torch.float64).tiny).log()+shift[:, None]
    # Sparse external test references can underflow; only those entries need log-domain repair.
    missing = (mass == 0).nonzero()
    if len(missing):
        n, a, c = missing.unbind(-1)
        reference[n, a, c] = (logits[n, c]+affinity[a, c].double().log()).logsumexp(-1)
    result = evidence.new_empty((len(evidence), members.numel(), len(members)))
    result[:, members.flatten()] = (logits.flatten(1)[..., None]-reference[:, members.flatten()]).to(evidence.dtype)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError('Nonfinite semantic rival margins.')
    return result


def semantic_observations(observations, members, affinity):
    return tuple(CachedCrop(crop.evidence, semantic_margins(crop.evidence, members, affinity),
                            crop.indices, crop.coefficients) for crop in observations)


@torch.inference_mode()
def semantic_scores(local, operator, broad, fine_field, wide, fine, coordinates,
                    relation, valid, members, canonical, parents, affinity, *, methods=METHODS[3:]):
    if not methods or not set(methods).issubset(METHODS[3:]):
        raise ValueError('Declared semantic-rival methods required.')
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
    old_risk = native_risk(wm, fm, fm, parents, canonical, valid, CONFIG)
    old_write = None
    if set(methods) & {OLD_SOFT, MATCHED_OLD}:
        delta, _ = fixed_slot_action(wide, members, old_risk, valid)
        potential, _ = signed_potential(delta, valid)
        old_write = .5*(operator.double()@potential)
        if OLD_SOFT in methods:
            values[OLD_SOFT] = positive+old_write
    if OLD_HARD in methods:
        potential, _ = signed_potential(directed_cached(wide, members, old_risk, valid), valid)
        values[OLD_HARD] = positive+.5*(operator.double()@potential)
    if not set(methods)-{OBSERVATION_MEAN, OLD_SOFT, OLD_HARD}:
        return values, stats

    def risk_for(weights):
        wide_margin = sampled_cached(semantic_observations(wide, members, weights), coordinates)
        fine_margin = sampled_cached(semantic_observations(fine, members, weights), coordinates)
        fine_margin.masked_fill_(~valid[:, None, None], 0.)
        return native_risk(wide_margin, fine_margin, fine_margin, parents, canonical, valid, CONFIG)

    risk = risk_for(affinity)
    delta, diagnostics = fixed_slot_action(wide, members, risk, valid)
    potential, consistency = signed_potential(delta, valid)
    written = .5*(operator.double()@potential)
    if PRIMARY in methods:
        values[PRIMARY] = positive+written
    if SHUFFLED_WRITE in methods:
        values[SHUFFLED_WRITE] = positive+.5*(permuted_operator(operator, valid).double()@potential)
    if MATCHED_OLD in methods:
        matched, extra = match_previous(old_write, written, valid)
        values[MATCHED_OLD] = positive+matched
        stats.update(extra)
    if DIRECT_MATCHED in methods:
        matched, extra = match_previous(.5*potential, written, valid)
        values[DIRECT_MATCHED] = positive+matched
        stats.update(direct_match_norm_relative_error=extra['matched_previous_norm_relative_error'],
                     direct_match_unmatchable=extra['matched_previous_unmatchable'])
    if set(methods) & {KEY_NULL, MATCHED_KEY_NULL}:
        changed_risk = risk_for(shuffled_affinity(affinity))
        directed, _ = fixed_slot_action(wide, members, changed_risk, valid)
        changed_potential, _ = signed_potential(directed, valid)
        changed_write = .5*(operator.double()@changed_potential)
        if KEY_NULL in methods:
            values[KEY_NULL] = positive+changed_write
        if MATCHED_KEY_NULL in methods:
            matched, extra = match_previous(changed_write, written, valid)
            values[MATCHED_KEY_NULL] = positive+matched
            stats.update(matched_key_norm_relative_error=extra['matched_previous_norm_relative_error'],
                         matched_key_unmatchable=extra['matched_previous_unmatchable'])
    if any(m in methods for m in (CLASS_MEAN, *SHUFFLES)):
        controls, error = continuous_controls(risk, members, canonical)
        stats['class_budget_max_error'] = error
        for new, old in ((CLASS_MEAN, PREVIOUS_CLASS_MEAN), *zip(SHUFFLES, PREVIOUS_SHUFFLES)):
            if new in methods:
                directed, _ = fixed_slot_action(wide, members, controls[old], valid)
                changed, _ = signed_potential(directed, valid)
                values[new] = positive+.5*(operator.double()@changed)
    stats.update({**risk_statistics(risk, potential, valid, members, canonical), **diagnostics, **consistency,
        'semantic_risk_mean': float(risk[valid].double().mean()) if bool(valid.any()) else 0.,
        'previous_risk_mean': float(old_risk[valid].double().mean()) if bool(valid.any()) else 0.,
        'risk_support_change_fraction': float(((risk[valid] > 0) != (old_risk[valid] > 0)).double().mean()) if bool(valid.any()) else 0.,
        'rival_affinity_max_mass_error': float((affinity.sum(-1)-1).abs().max()),
        'rival_affinity_mean_entropy': float(-(affinity*affinity.clamp_min(1e-300).log()).sum(-1).mean())})
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite semantic-rival scores.')
    return values, stats


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS,
                  cache=None, execution=None):
    affinities = {}
    if set(methods)-set(METHODS[:3])-{OBSERVATION_MEAN, OLD_SOFT, OLD_HARD}:
        for p, query in queries.items():
            if tuple(banks[p].alias_names) != tuple(query.aliases) or not torch.equal(banks[p].parent_indices, query.parents):
                raise ValueError('Unchanged original alias slot identity required.')
            members = torch.stack([(query.parents == c).nonzero().flatten() for c in range(len(query.class_names))])
            affinities[id(query.parents)] = rival_affinity(query.features.float().mean(1), members)
    def score(local, operator, broad, fine_field, wide, fine, coordinates,
              relation, valid, members, canonical, parents, *, methods):
        return semantic_scores(local, operator, broad, fine_field, wide, fine, coordinates,
            relation, valid, members, canonical, parents, affinities.get(id(parents)), methods=methods)
    return source_predict(image, geometry, banks, vip, queries, work, methods=methods,
        cache=cache, execution=execution, tile_scorer=score, allowed_methods=METHODS,
        observation_method=OBSERVATION_MEAN, diagnostic_fields=DIAGNOSTICS)
