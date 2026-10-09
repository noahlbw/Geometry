"""Keep conditional rival evidence until Geometry-constrained logistic reading."""
import torch
import torch.nn.functional as F

from .fine_alias_view import CONFIG
from .native_alias_noise import signed_potential
from .native_query_alias import native_risk
from .one_sided_alias_audit import (HARD as INCUMBENT_HARD,
    OBSERVATION_MEAN as INCUMBENT_MEAN, PRIMARY as INCUMBENT_SOFT,
    collapse_rivals, scores as incumbent_scores)
from .reciprocal_alias_admission import (CLASS_MEAN as OLD_CLASS, SHUFFLES as OLD_SHUFFLES,
    continuous_controls, fixed_slot_action)
from .rival_alias_fast import directed_cached, risk_statistics, sampled_cached
from .supported_positive_alias import PROTOCOL as SOURCE_PROTOCOL, predict_image as source_predict
from .witness_cap_alias import match_previous


IMPLEMENTATION = 'geometry-bounded896-pair-likelihood-coupling-v1-20261006'
PRIMARY = 'PairLikelihood_Soft'
NO_ALIAS = 'PairLikelihood_NoAlias'
OBSERVATION_MEAN = 'PairLikelihood_PreviousMean'
OLD_SOFT = 'PairLikelihood_PreviousSoft'
HARD = 'PairLikelihood_PreviousHard'
PROJECTED = 'PairLikelihood_ProjectFirst'
MATCHED_PROJECTED = 'PairLikelihood_MatchedProjectFirst'
MATCHED_OLD = 'PairLikelihood_MatchedPreviousSoft'
CLASS_MEAN = 'PairLikelihood_MatchedClassMean'
SHUFFLES = tuple('PairLikelihood_MatchedAliasShuffle'+str(i) for i in range(3))
RIVAL_NULL = 'PairLikelihood_MatchedRivalCollapsed'
WRITE_NULL = 'PairLikelihood_MatchedWordPosition'
DIRECT = 'PairLikelihood_DirectMatched'
METHODS = ('Geometry', 'NoAdmission_Exact', 'Geometry_PatchOnly2Coupled', PRIMARY,
    NO_ALIAS, OBSERVATION_MEAN, OLD_SOFT, HARD, PROJECTED, MATCHED_PROJECTED,
    MATCHED_OLD, CLASS_MEAN, *SHUFFLES,
    RIVAL_NULL, WRITE_NULL, DIRECT)
ITERATIONS = 24
DIAGNOSTICS = ('class_budget_max_error', 'rival_budget_max_error', 'matched_norm_relative_error',
    'matched_unmatchable', 'positive_directed_delta_max', 'mass_log_fallbacks',
    'solver_initial_objective', 'solver_final_objective', 'solver_objective_increase',
    'solver_max_gradient', 'solver_lipschitz_bound', 'solver_iterations',
    'pair_cycle_fraction', 'word_intervention_norm', 'solver_gauge_max_error')
PROTOCOL = {**SOURCE_PROTOCOL,
    'alias_admission': 'EXACT existing one-sided wide-positive/fine-negative risk; no new source or truth gate',
    'aggregation': 'unchanged fixed20-slot outside-exponent soft attenuation; no survivor redistribution',
    'writer': 'local L2 fidelity plus Geometry-pulled Bradley-Terry likelihood of pair-conditioned observations',
    'likelihood_coefficient': '4/C; matches quadratic Gram curvature at neutral pair probabilities',
    'solver': '24 fixed accelerated gradient steps; Lipschitz1+max column sum of masked row-stochastic W',
    'semantic_source': 'same positive half-wide/half-fine observations, same risk and directed actions',
    'additional_visual_or_head_encodings': 0, 'fitted_parameters': 0,
    'controls': METHODS[4:],
    'limitation': 'jointly wrong broad/fine positives still survive; no new correctness source',
    'precedents': 'Bradley-Terry logistic consistency, convex ridge fidelity and accelerated gradient are known operations'}


def geometry_weights(relation, valid):
    if (relation.shape != (len(valid), len(valid)) or valid.dtype != torch.bool
            or not bool(torch.isfinite(relation).all()) or bool((relation < 0).any())):
        raise ValueError('Finite nonnegative relation and Boolean valid tokens required.')
    weights = relation.double().masked_fill(~valid[None], 0.)
    weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-12)
    return weights.masked_fill(~valid[:, None], 0.)


def logistic_objective(local, value, weights, probabilities, valid):
    classes = local.shape[1]
    observed = weights@value
    margins = observed[:, :, None]-observed[:, None]
    off_diagonal = ~torch.eye(classes, device=local.device, dtype=torch.bool)
    loss = (F.softplus(margins)-probabilities*margins)[:, off_diagonal]
    return .5*(value-local).square().sum()+(2/classes)*loss[valid].sum()


def logistic_gradient(local, value, weights, probabilities):
    observed = weights@value
    margins = observed[:, :, None]-observed[:, None]
    residual = margins.sigmoid()-probabilities
    return value-local+(4/local.shape[1])*(weights.T@residual.sum(-1))


@torch.inference_mode()
def reconstruct(local, weights, target, valid, *, iterations=ITERATIONS):
    if (local.ndim != 2 or local.shape[1] < 2 or weights.shape != (len(local), len(local))
            or target.shape != (len(local), local.shape[1], local.shape[1]) or iterations < 1
            or not bool(torch.isfinite(local).all() and torch.isfinite(weights).all() and torch.isfinite(target).all())
            or bool((weights < 0).any())
            or not torch.allclose(target, -target.transpose(-1, -2), atol=1e-10, rtol=0)):
        raise ValueError('Finite local scores, nonnegative Geometry and antisymmetric pair targets required.')
    local = local.double()
    probabilities = target.double().sigmoid()
    bound = 1+weights.sum(0).max()
    acceleration = (bound.sqrt()-1)/(bound.sqrt()+1)
    value = lookahead = local.clone()
    initial = logistic_objective(local, value, weights, probabilities, valid)
    for _ in range(iterations):
        changed = lookahead-logistic_gradient(local, lookahead, weights, probabilities)/bound
        changed -= (changed-local).mean(-1, keepdim=True)
        lookahead = changed+acceleration*(changed-value)
        value = changed
    final = logistic_objective(local, value, weights, probabilities, valid)
    gradient = logistic_gradient(local, value, weights, probabilities)
    gauge = float((value-local).sum(-1).abs().max())
    if not bool(torch.isfinite(value).all()) or float(final-initial) > 1e-8 or gauge > 1e-10:
        raise RuntimeError('Pair reconstruction must be finite, gauge-preserving and improve its frozen objective.')
    return value, dict(solver_initial_objective=float(initial), solver_final_objective=float(final),
        solver_objective_increase=max(0., float(final-initial)), solver_max_gradient=float(gradient.abs().max()),
        solver_lipschitz_bound=float(bound), solver_iterations=float(iterations), solver_gauge_max_error=gauge)


@torch.inference_mode()
def scores(local, operator, broad, fine_field, wide, fine, coordinates,
           relation, valid, members, canonical, parents, *, methods=METHODS[3:]):
    if not methods or not set(methods).issubset(METHODS[3:]):
        raise ValueError('Declared pair-likelihood endpoints required.')
    old_names = {OBSERVATION_MEAN: INCUMBENT_MEAN, OLD_SOFT: INCUMBENT_SOFT, HARD: INCUMBENT_HARD}
    if set(methods).issubset(old_names):
        old, diagnostics = incumbent_scores(local, operator, broad, fine_field, wide, fine,
            coordinates, relation, valid, members, canonical, parents,
            methods=tuple(old_names[name] for name in methods))
        return {name: old[old_names[name]] for name in methods}, diagnostics
    innovation = .5*(fine_field.double()-broad.double())
    innovation.masked_fill_(~valid[:, None], 0.)
    positive = local.double()+operator.double()@(broad.double()-local.double())+operator.double()@innovation
    values = {OBSERVATION_MEAN: positive} if OBSERVATION_MEAN in methods else {}
    stats = dict.fromkeys((*DIAGNOSTICS, 'canonical_risk_max', 'mean_absolute_admission_potential',
        'normal_equation_max_error', 'gauge_max_error', 'deleted_alias_rival_fraction'), 0.)
    stats['retained_count_mean'] = float(members.shape[1])
    if set(methods) == {OBSERVATION_MEAN}:
        return values, stats
    weights = geometry_weights(relation, valid)
    observed = weights@(broad.double()+innovation)
    base_target = observed[:, :, None]-observed[:, None]
    no_alias, _ = reconstruct(local, weights, base_target, valid)
    if NO_ALIAS in methods:
        values[NO_ALIAS] = no_alias
    if not set(methods)-{OBSERVATION_MEAN, NO_ALIAS}:
        return values, stats
    wm = sampled_cached(wide, coordinates)
    fm = sampled_cached(fine, coordinates).masked_fill(~valid[:, None, None], 0.)
    risk = native_risk(wm, fm, fm, parents, canonical, valid, CONFIG)
    directed, info = fixed_slot_action(wide, members, risk, valid)
    potential, consistency = signed_potential(directed, valid)
    old_intervention = .5*(operator.double()@potential)
    if OLD_SOFT in methods:
        values[OLD_SOFT] = positive+old_intervention
    if HARD in methods:
        hard, _ = signed_potential(directed_cached(wide, members, risk, valid), valid)
        values[HARD] = positive+.5*(operator.double()@hard)

    def solve(action, geometry=weights):
        margins = action-action.transpose(-1, -2)
        target = base_target+.5*(weights@margins.flatten(1)).reshape_as(margins)
        return reconstruct(local, geometry, target, valid)

    primary, diagnostics = solve(directed)
    intervention = primary-no_alias
    if PRIMARY in methods:
        values[PRIMARY] = primary
    def matched(changed):
        result, errors = match_previous(changed, intervention, valid)
        stats['matched_norm_relative_error'] = max(stats['matched_norm_relative_error'], errors['matched_previous_norm_relative_error'])
        stats['matched_unmatchable'] += errors['matched_previous_unmatchable']
        return no_alias+result

    if set(methods) & {PROJECTED, MATCHED_PROJECTED}:
        projected = solve(potential[:, :, None].expand_as(directed))[0]
        if PROJECTED in methods:
            values[PROJECTED] = projected
        if MATCHED_PROJECTED in methods:
            values[MATCHED_PROJECTED] = matched(projected-no_alias)
    if MATCHED_OLD in methods:
        values[MATCHED_OLD] = matched(old_intervention)

    if set(methods) & {CLASS_MEAN, *SHUFFLES}:
        controls, error = continuous_controls(risk, members, canonical)
        stats['class_budget_max_error'] = error
        for name, key in ((CLASS_MEAN, OLD_CLASS), *zip(SHUFFLES, OLD_SHUFFLES)):
            if name in methods:
                changed, _ = fixed_slot_action(wide, members, controls[key], valid)
                values[name] = matched(solve(changed)[0]-no_alias)
    if RIVAL_NULL in methods:
        changed, error = collapse_rivals(risk, parents)
        stats['rival_budget_max_error'] = error
        action, _ = fixed_slot_action(wide, members, changed, valid)
        values[RIVAL_NULL] = matched(solve(action)[0]-no_alias)
    if WRITE_NULL in methods:
        generator = torch.Generator().manual_seed(CONFIG.random_seed)
        ids = valid.nonzero().flatten()
        shuffled = directed.clone()
        shuffled[ids] = directed[ids[torch.randperm(len(ids), generator=generator).to(ids.device)]]
        values[WRITE_NULL] = matched(solve(shuffled)[0]-no_alias)
    if DIRECT in methods:
        identity = torch.diag(valid.double())
        direct_base, _ = reconstruct(local, identity, base_target, valid)
        values[DIRECT] = matched(solve(directed, identity)[0]-direct_base)
    cycle = directed-directed.transpose(-1, -2)-(potential[:, :, None]-potential[:, None])
    margins = directed-directed.transpose(-1, -2)
    stats.update({**risk_statistics(risk, potential, valid, members, canonical), **info, **consistency, **diagnostics,
        'pair_cycle_fraction': float(cycle[valid].square().sum()/margins[valid].square().sum().clamp_min(CONFIG.epsilon)),
        'word_intervention_norm': float(intervention[valid].norm())})
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite coupled likelihood scores.')
    return values, stats


def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS,
                  cache=None, execution=None):
    return source_predict(image, geometry, banks, vip, queries, work, methods=methods, cache=cache,
        execution=execution, tile_scorer=scores, allowed_methods=METHODS,
        observation_method=OBSERVATION_MEAN, diagnostic_fields=DIAGNOSTICS)
