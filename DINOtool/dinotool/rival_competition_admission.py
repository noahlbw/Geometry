"""Fine alias admission with prediction-conditioned competitive action reconciliation."""
from dataclasses import asdict

import torch

from .fine_alias_view import CONFIG
from .native_alias_noise import signed_potential
from .native_query_alias import native_risk
from .rival_alias_fast import cache_observations, sampled_cached, directed_cached, risk_statistics


IMPLEMENTATION = 'geometry-frozen-fine-competition-action-v1-20261005'
PRIMARY = 'FineRivalPosterior_Exact'
ALTERNATIVES = ('FineActionProjected_Exact', 'FineRivalProjected_Exact')
CONTROLS = ('FineTopTwo_Exact', 'FinePosteriorStrengthMatched_Exact', 'FineBudgetOnly_Exact')
METHODS = ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact', PRIMARY, *ALTERNATIVES, *CONTROLS)


def settings():
    return {'source': asdict(CONFIG), 'rival_posterior': 'softmax of unchanged NoAdmission patch logits, temperature1',
            'projection': 'clip signed hard action onto segment0..(fine-minus-wide class pair margin)',
            'fine_class_reference': 'same profiled fine crop class logsumexp, same interpolation stencils',
            'strength_control': 'one window scalar matching reconstructed Frobenius norm to posterior action',
            'canonical_alias_protected': True, 'additional_visual_forwards': 0, 'domain_routing': False}


def manifest_samples(samples, manifest, dataset):
    keys = manifest['sample_keys']
    lookup = {sample.key: sample for sample in samples}
    if (manifest['dataset'] != dataset or not keys or len(keys) != len(set(keys))
            or len(lookup) != len(samples) or not set(keys).issubset(lookup)):
        raise ValueError('A nonempty unique manifest of actual dataset keys is required.')
    return [lookup[key] for key in keys]


def posterior_potential(margins, posterior, valid):
    if (margins.ndim != 3 or margins.shape[1] != margins.shape[2]
            or posterior.shape != margins.shape[:2] or valid.shape != margins.shape[:1]
            or not bool(torch.isfinite(margins).all() and torch.isfinite(posterior).all())
            or bool((posterior < 0).any())
            or not torch.allclose(posterior.sum(-1), torch.ones_like(posterior[:, 0]), atol=1e-12, rtol=0)
            or not torch.allclose(margins, -margins.transpose(-1, -2), atol=1e-12, rtol=0)):
        raise ValueError('Finite antisymmetric actions and normalized nonnegative class posterior required.')
    # Closed-form weighted pair least squares for edge weights p(c)*p(d).
    potential = (margins.double()*posterior.double()[:, None]).sum(-1)
    potential -= potential.mean(-1, keepdim=True)
    potential.masked_fill_(~valid[:, None], 0.)
    residual = potential[:, :, None]-potential[:, None]-margins
    weights = posterior[:, :, None]*posterior[:, None]
    error = float((residual*weights).sum(-1)[valid].abs().max()) if bool(valid.any()) else 0.
    gauge = float(potential.sum(-1).abs().max())
    if max(error, gauge) > 1e-10:
        raise RuntimeError('Weighted competitive normal equations failed.')
    return potential, {'weighted_normal_equation_max_error': error, 'gauge_max_error': gauge}


def project_actions(margins, target):
    if (margins.shape != target.shape or margins.ndim != 3
            or not bool(torch.isfinite(margins).all() and torch.isfinite(target).all())
            or not torch.allclose(margins, -margins.transpose(-1, -2), atol=1e-12, rtol=0)
            or not torch.allclose(target, -target.transpose(-1, -2), atol=1e-12, rtol=0)):
        raise ValueError('Finite matching antisymmetric requested and fine-target actions required.')
    return target.sign()*torch.minimum(margins.abs(), target.abs())*(margins*target > 0)


def sampled_classes(observations, coordinates, beta=CONFIG.beta):
    output = coordinates.new_zeros((len(coordinates), observations[0].evidence.shape[1]))
    for crop in observations:
        scores = (beta*crop.evidence).logsumexp(-1)/beta
        output += (scores[crop.indices]*crop.coefficients[..., None]).sum(1)
    return output


def top_two_potential(margins, baseline, valid):
    pairs = baseline.topk(2, -1).indices
    rows = torch.arange(len(baseline), device=baseline.device)
    values = margins[rows, pairs[:, 0], pairs[:, 1]]*.5
    output = torch.zeros_like(baseline, dtype=torch.float64)
    output.scatter_(1, pairs[:, :1], values[:, None])
    output.scatter_(1, pairs[:, 1:], -values[:, None])
    return output.masked_fill(~valid[:, None], 0.)


@torch.inference_mode()
def retained_competition_scores(local, operator, broad, wide, wide_count, fine, fine_count,
                                coordinates, fine_coordinates, valid, members, canonical, parents,
                                image_size, methods=METHODS):
    if not set(methods).issubset(METHODS):
        raise ValueError('Unknown declared screening endpoint.')
    wide = cache_observations(wide, wide_count, coordinates, image_size, members)
    fine = cache_observations(fine, fine_count, fine_coordinates, (512, 512), members)
    full = sampled_cached(wide, coordinates)
    native = sampled_cached(fine, fine_coordinates).masked_fill(~valid[:, None, None], 0.)
    risk = native_risk(full, native, native, parents, canonical, valid, CONFIG)
    directed = directed_cached(wide, members, risk, valid)
    original, consistency = signed_potential(directed, valid)
    margins = directed-directed.transpose(-1, -2)
    baseline = local.double()+operator.double()@(broad.double()-local.double())
    values = {'Geometry': local.double(), 'NoAdmission_Exact': baseline}
    diagnostics = {**risk_statistics(risk, original, valid, members, canonical), **consistency}
    original_delta = None
    if 'RivalFineHard_Exact' in methods or 'FinePosteriorStrengthMatched_Exact' in methods:
        original_delta = operator.double()@original
        values['RivalFineHard_Exact'] = baseline+original_delta
    if any(m in methods for m in (PRIMARY, 'FineRivalProjected_Exact', 'FinePosteriorStrengthMatched_Exact')):
        posterior = baseline.softmax(-1)
        weighted, stats = posterior_potential(margins, posterior, valid)
        weighted_delta = operator.double()@weighted
        values[PRIMARY] = baseline+weighted_delta
        diagnostics.update({PRIMARY+'__'+k: v for k, v in stats.items()})
        diagnostics['posterior_effective_rival_count'] = float((1/posterior[valid].square().sum(-1)).mean())
    if any(m in methods for m in (*ALTERNATIVES, 'FineBudgetOnly_Exact')):
        classes = sampled_classes(fine, fine_coordinates).double()
        innovation = classes-broad.double()
        target = (innovation[:, :, None]-innovation[:, None]).masked_fill(~valid[:, None, None], 0.)
        projected = project_actions(margins, target)
        active = (margins != 0) & valid[:, None, None]
        diagnostics['projection_nonzero_action_retention'] = float((projected[active] != 0).double().mean()) if bool(active.any()) else 1.
        if 'FineActionProjected_Exact' in methods:
            potential, stats = signed_potential(projected*.5, valid)
            values['FineActionProjected_Exact'] = baseline+operator.double()@potential
        if 'FineRivalProjected_Exact' in methods:
            potential, stats = posterior_potential(projected, posterior, valid)
            values['FineRivalProjected_Exact'] = baseline+operator.double()@potential
        if 'FineBudgetOnly_Exact' in methods:
            action = target.sign()*torch.minimum(margins.abs(), target.abs())
            potential, stats = signed_potential(action*.5, valid)
            values['FineBudgetOnly_Exact'] = baseline+operator.double()@potential
    if 'FineTopTwo_Exact' in methods:
        values['FineTopTwo_Exact'] = baseline+operator.double()@top_two_potential(margins, baseline, valid)
    if 'FinePosteriorStrengthMatched_Exact' in methods:
        norm = original_delta.norm()
        scale = weighted_delta.norm()/norm.clamp_min(CONFIG.epsilon)
        values['FinePosteriorStrengthMatched_Exact'] = baseline+original_delta*scale
        diagnostics['strength_matched_scale'] = float(scale)
        diagnostics['strength_matched_norm_error'] = float((original_delta*scale).norm()-weighted_delta.norm())
        diagnostics['strength_control_zero_original_action'] = float(norm == 0)
    values = {m: values[m] for m in methods}
    if not all(bool(torch.isfinite(v).all()) for v in values.values()) or bool((risk[:, canonical] != 0).any()):
        raise RuntimeError('Nonfinite competition action or unprotected canonical word.')
    return values, risk, diagnostics
