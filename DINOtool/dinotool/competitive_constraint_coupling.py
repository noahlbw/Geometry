"""Joint spatial/competitive constraints, rather than separated class projection."""
import torch

from .bounded_patch_only import PRIMARY as BASELINE
from .fine_alias_view import CONFIG
from .native_alias_noise import signed_potential
from .native_query_alias import native_risk
from .rival_alias_fast import directed_cached, risk_statistics, sampled_cached
from .rival_competition_admission import posterior_potential
from .supported_positive_alias import (PROTOCOL as SOURCE_PROTOCOL, class_only_risk,
    predict_image as source_predict)
from .tlp import _conjugate_gradient


IMPLEMENTATION = 'geometry-bounded896-joint-competitive-constraints-v1-20261006'
PRIMARY = 'CompetitiveConstraint_Hard'
NO_ALIAS = 'CompetitiveConstraint_NoAlias'
UNIFORM = 'CompetitiveConstraint_Uniform'
OBSERVATION_MEAN = 'CompetitiveConstraint_ObservationMean'
CLASS_MEAN = 'CompetitiveConstraint_ClassMean'
SHUFFLES = tuple('CompetitiveConstraint_AliasShuffle'+str(i) for i in range(3))
SEPARATED = 'CompetitiveConstraint_Separated'
PRIOR_SHUFFLE = 'CompetitiveConstraint_PriorShuffle'
NORM_MATCHED = 'CompetitiveConstraint_UniformNormMatched'
METHODS = ('Geometry', 'NoAdmission_Exact', BASELINE, PRIMARY, NO_ALIAS, UNIFORM,
    OBSERVATION_MEAN, CLASS_MEAN, *SHUFFLES, SEPARATED, PRIOR_SHUFFLE, NORM_MATCHED)
DIAGNOSTICS = ('cg_iterations', 'cg_relative_residual', 'constraint_trace_error',
               'class_mean_preservation_error', 'norm_matching_error', 'empty_graph_rows')
PROTOCOL = {**SOURCE_PROTOCOL,
    'alias_admission': 'unchanged query/class/rival wide-positive fine-negative hard contradiction, no G truth gate',
    'reconstruction': 'joint SPD competitive constraints on original valid-normalized Geometry W',
    'writer': 'unprojected pair intervention transported by W before joint class/spatial solve',
    'competition_prior': 'unchanged patch-only2 Geometry local logits softmax, temperature1',
    'constraint_budget': 'off-diagonal p(c)*p(r) graph Laplacian trace exactly C-1',
    'solver': 'existing matrix-free float64 conjugate-gradient; max64, relative tolerance1e-9',
    'controls': METHODS[4:], 'cg_max_iterations': 64, 'cg_tolerance': 1e-9}


def geometry_weights(relation, valid):
    if (relation.shape != (len(valid), len(valid)) or valid.dtype != torch.bool
            or not bool(torch.isfinite(relation).all()) or bool((relation < 0).any())):
        raise ValueError('Finite nonnegative Geometry relation and validity required.')
    weights = relation.double().masked_fill(~valid[None], 0.)
    weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-12)
    return weights.masked_fill(~valid[:, None], 0.)


def competitive_graph(posterior, valid):
    if (posterior.ndim != 2 or len(posterior) != len(valid) or posterior.shape[1] < 2 or valid.dtype != torch.bool
            or not bool(torch.isfinite(posterior).all()) or bool((posterior < 0).any())
            or not torch.allclose(posterior.double().sum(-1), torch.ones(len(valid), device=posterior.device,
                dtype=torch.float64), atol=1e-12, rtol=0)):
        raise ValueError('Normalized finite class prior required.')
    classes = posterior.shape[1]
    posterior = posterior.double()
    self_edge = torch.eye(classes, device=posterior.device, dtype=torch.bool)
    edges = (posterior[:, :, None]*posterior[:, None]).masked_fill(self_edge, 0.)
    trace = edges.sum((1, 2))
    empty = trace == 0
    if bool(empty.any()):
        edges[empty] = (~self_edge).double()/classes**2
        trace = edges.sum((1, 2))
    edges *= ((classes-1)/trace)[:, None, None]
    edges.masked_fill_(~valid[:, None, None], 0.)
    laplacian = torch.diag_embed(edges.sum(-1))-edges
    error = float((laplacian.diagonal(dim1=-2, dim2=-1).sum(-1)[valid]-(classes-1)).abs().max()) if bool(valid.any()) else 0.
    return edges, laplacian, dict(constraint_trace_error=error, empty_graph_rows=float((empty & valid).sum()))


@torch.inference_mode()
def solve_constraints(local, observation, margins, weights, posterior, valid):
    if (local.ndim != 2 or observation.shape != local.shape
            or margins.shape != (len(local), local.shape[1], local.shape[1])
            or weights.shape != (len(local), len(local)) or posterior.shape != local.shape
            or not bool(torch.isfinite(local).all() and torch.isfinite(observation).all()
                        and torch.isfinite(margins).all() and torch.isfinite(weights).all())
            or not torch.allclose(margins, -margins.transpose(-1, -2), atol=1e-10, rtol=0)):
        raise ValueError('Matching finite anchored fields, antisymmetric pair actions and Geometry weights required.')
    edges, laplacian, stats = competitive_graph(posterior, valid)
    read = weights@observation.double()
    transported = (weights@margins.double().flatten(1)).reshape_as(margins)
    target = read[:, :, None]-read[:, None]+.5*transported
    rhs = local.double()+weights.T@(edges*target).sum(-1)

    def system(value):
        reads = torch.matmul(weights, value)
        action = torch.einsum('ncd,...nd->...nc', laplacian, reads)
        return value+torch.matmul(weights.T, action)

    solution, iterations, _ = _conjugate_gradient(system, rhs[None, None],
        local.double()[None, None], PROTOCOL['cg_max_iterations'], PROTOCOL['cg_tolerance'])
    solution = solution[0, 0]
    solution -= (solution-local.double()).mean(-1, keepdim=True)
    residual = float((system(solution)-rhs).norm()/rhs.norm().clamp_min(1e-12))
    gauge = float((solution-local.double()).sum(-1).abs().max())
    if residual > 1e-7 or gauge > 1e-8 or not bool(torch.isfinite(solution).all()):
        raise RuntimeError('Joint constraint solve did not meet the frozen residual/gauge tolerance.')
    return solution, {**stats, 'cg_iterations': float(iterations), 'cg_relative_residual': residual,
                      'class_mean_preservation_error': gauge}


def shuffled_prior(posterior, valid):
    ids = valid.nonzero().flatten()
    generator = torch.Generator().manual_seed(CONFIG.random_seed)
    changed = posterior.clone()
    changed[ids] = posterior[ids[torch.randperm(len(ids), generator=generator).to(valid.device)]]
    return changed


@torch.inference_mode()
def constraint_scores(local, operator, broad, fine_field, wide, fine, coordinates,
                      relation, valid, members, canonical, parents, *, methods=METHODS[3:]):
    if not methods or not set(methods).issubset(METHODS[3:]):
        raise ValueError('Declared joint-constraint methods required.')
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
    weights = geometry_weights(relation, valid)
    posterior = local.double().softmax(-1)
    observation = (broad.double()+fine_field.double())*.5
    full = sampled_cached(wide, coordinates)
    native = sampled_cached(fine, coordinates).masked_fill(~valid[:, None, None], 0.)
    risk = native_risk(full, native, native, parents, canonical, valid, CONFIG)
    directed = directed_cached(wide, members, risk, valid)
    potential, consistency = signed_potential(directed, valid)
    margins = directed-directed.transpose(-1, -2)
    uniform = positive+.5*(operator.double()@potential)
    if UNIFORM in methods:
        values[UNIFORM] = uniform
    stats.update({**risk_statistics(risk, potential, valid, members, canonical), **consistency})
    needs_primary = PRIMARY in methods or NORM_MATCHED in methods
    if needs_primary:
        primary, diagnostics = solve_constraints(local, observation, margins, weights, posterior, valid)
        stats.update(diagnostics)
        if PRIMARY in methods:
            values[PRIMARY] = primary
    if NO_ALIAS in methods or NORM_MATCHED in methods:
        no_alias, _ = solve_constraints(local, observation, torch.zeros_like(margins), weights, posterior, valid)
        if NO_ALIAS in methods:
            values[NO_ALIAS] = no_alias
    if NORM_MATCHED in methods:
        reference = uniform-positive
        requested = primary-no_alias
        scale = requested.norm()/reference.norm().clamp_min(1e-12)
        matched = reference*scale
        error = float((matched.norm()-requested.norm()).abs())
        if error > 1e-7:
            raise RuntimeError('Same-base alias intervention norm could not be matched.')
        stats['norm_matching_error'] = error
        values[NORM_MATCHED] = no_alias+matched
    if SEPARATED in methods:
        delta, _ = posterior_potential(margins, posterior, valid)
        values[SEPARATED] = positive+.5*(operator.double()@delta)
    if PRIOR_SHUFFLE in methods:
        values[PRIOR_SHUFFLE], _ = solve_constraints(local, observation, margins, weights,
                                                    shuffled_prior(posterior, valid), valid)
    if CLASS_MEAN in methods:
        changed = class_only_risk(risk, members, canonical)
        action = directed_cached(wide, members, changed, valid, 'weighted')
        values[CLASS_MEAN], _ = solve_constraints(local, observation, action-action.transpose(-1, -2),
                                                 weights, posterior, valid)
    if any(name in methods for name in SHUFFLES):
        from .bounded_fine_coverage import SHUFFLES as OLD_SHUFFLES, alias_controls

        controls = alias_controls(risk, members, canonical)
        for name, old in zip(SHUFFLES, OLD_SHUFFLES):
            if name in methods:
                action = directed_cached(wide, members, controls[old], valid)
                values[name], _ = solve_constraints(local, observation, action-action.transpose(-1, -2),
                                                    weights, posterior, valid)
    return values, stats


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS,
                  cache=None, execution=None):
    return source_predict(image, geometry, banks, vip, queries, work, methods=methods,
        cache=cache, execution=execution, tile_scorer=constraint_scores, allowed_methods=METHODS,
        observation_method=OBSERVATION_MEAN, diagnostic_fields=DIAGNOSTICS)
