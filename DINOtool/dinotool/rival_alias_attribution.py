"""Matched decision interventions for the frozen competition-admission candidate."""
import torch

from .fine_alias_view import CONFIG
from .rival_alias_fast import cache_observations, directed_cached
from .rival_competition_admission import (retained_competition_scores, sampled_classes,
    posterior_potential, project_actions, settings as competition_settings)
from .target_context_alias import alias_permutations


IMPLEMENTATION = 'frozen-rival-conditional-alias-attribution-v1-20261005'
PRIMARY = 'FineRivalProjected_Exact'
BASE = ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact', PRIMARY, 'FineBudgetOnly_Exact')
ALIAS = tuple('ProjectedAliasShuffle'+str(i) for i in range(3))
SPATIAL = tuple('ProjectedSpatialShuffle'+str(i) for i in range(3))
RIVAL = tuple('ProjectedRivalDegree'+str(i) for i in range(3))
CLASS = ('FineClassCoupled_NoAdmission', 'FineClassStrengthMatched')
METHODS = (*BASE, *CLASS, *ALIAS, *SPATIAL, *RIVAL)
SWAP_ATTEMPTS = 512


def settings():
    return {**competition_settings(), 'frozen_candidate': PRIMARY,
        'alias_null': 'within-class noncanonical permutation; per-query/class/rival decision counts retained',
        'spatial_null': 'valid-query permutation; total class/alias/rival decisions retained',
        'rival_null': 'binary2x2 switches preserving both alias and rival deletion degrees at every query/class',
        'swap_attempts': SWAP_ATTEMPTS, 'random_seeds': [CONFIG.random_seed+i for i in range(3)],
        'rival_null_not_uniform_sampler': True,
        'fine_class_control': 'same profiled fine class scores, no alias-specific admission',
        'class_strength_control': 'one window scalar matching candidate reconstructed Frobenius norm'}


def alias_null(risk, members, canonical, seed):
    permutation = alias_permutations(members, canonical, seed)[0]
    grouped = risk[:, members]
    result = torch.empty_like(risk)
    result[:, members] = grouped.gather(2, permutation[None, :, :, None].expand_as(grouped))
    return result


def spatial_null(risk, valid, seed):
    ids = valid.nonzero().flatten()
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(ids), generator=generator).to(risk.device)
    result = risk.clone()
    result[ids] = risk[ids[order]]
    return result


def rival_degree_null(risk, members, canonical, seed, attempts=SWAP_ATTEMPTS):
    if attempts < 1: raise ValueError('Positive switch attempts required.')
    q, classes, k, _ = risk[:, members].shape
    grouped = risk[:, members] > 0
    aliases = torch.stack([(row != word).nonzero().flatten() for row, word in zip(members, canonical)])
    rivals = torch.stack([(torch.arange(classes, device=risk.device) != c).nonzero().flatten() for c in range(classes)])
    ci = torch.arange(classes, device=risk.device)
    decision = grouped[:, ci[:, None, None], aliases[:, :, None], rivals[:, None, :]].clone()
    original = decision.clone()
    nr, nc = decision.shape[-2:]
    switches = 0
    if min(nr, nc) > 1:
        flat = decision.reshape(q*classes, nr, nc)
        generator = torch.Generator().manual_seed(seed)
        rows = torch.randint(nr, (attempts, q*classes), generator=generator)
        other_rows = (rows+torch.randint(1, nr, rows.shape, generator=generator)) % nr
        cols = torch.randint(nc, rows.shape, generator=generator)
        other_cols = (cols+torch.randint(1, nc, rows.shape, generator=generator)) % nc
        rows, other_rows, cols, other_cols = (v.to(risk.device) for v in (rows, other_rows, cols, other_cols))
        nodes = torch.arange(q*classes, device=risk.device)
        accepted = torch.zeros((), dtype=torch.int64, device=risk.device)
        for i in range(attempts):
            a, b, c, d = rows[i], other_rows[i], cols[i], other_cols[i]
            x, y, z, w = flat[nodes, a, c], flat[nodes, a, d], flat[nodes, b, c], flat[nodes, b, d]
            use = (x == w) & (y == z) & (x != y)
            flat[nodes, a, c] = x ^ use
            flat[nodes, a, d] = y ^ use
            flat[nodes, b, c] = z ^ use
            flat[nodes, b, d] = w ^ use
            accepted += use.sum()
        switches = int(accepted)
    grouped[:, ci[:, None, None], aliases[:, :, None], rivals[:, None, :]] = decision
    result = torch.zeros_like(risk)
    result[:, members] = grouped.to(risk.dtype)
    stats = {'accepted_switches': switches,
        'decision_changed_fraction': float((decision != original).double().mean()),
        'query_class_changed_fraction': float((decision != original).flatten(-2).any(-1).double().mean()),
        'alias_degree_max_error': int((decision.sum(-1)-original.sum(-1)).abs().max()),
        'rival_degree_max_error': int((decision.sum(-2)-original.sum(-2)).abs().max())}
    if stats['alias_degree_max_error'] or stats['rival_degree_max_error']:
        raise RuntimeError('Degree-preserving rival intervention changed the budget.')
    return result, stats


@torch.inference_mode()
def attribution_scores(local, operator, broad, wide, wide_count, fine, fine_count,
                       coordinates, fine_coordinates, valid, members, canonical, parents,
                       image_size, methods=METHODS):
    if not set(methods).issubset(METHODS): raise ValueError('Undeclared attribution endpoint.')
    interventions = set(methods)-set(BASE)
    requested = BASE if interventions else methods
    values, risk, stats = retained_competition_scores(local, operator, broad, wide, wide_count,
        fine, fine_count, coordinates, fine_coordinates, valid, members, canonical, parents,
        image_size, methods=requested)
    if not interventions: return values, risk, stats
    observed_wide = cache_observations(wide, wide_count, coordinates, image_size, members)
    observed_fine = cache_observations(fine, fine_count, fine_coordinates, (512, 512), members)
    innovation = sampled_classes(observed_fine, fine_coordinates).double()-broad.double()
    target = (innovation[:, :, None]-innovation[:, None]).masked_fill(~valid[:, None, None], 0.)
    baseline = values['NoAdmission_Exact']
    posterior = baseline.softmax(-1)
    if any(m in methods for m in CLASS):
        potential = innovation-innovation.mean(-1, keepdim=True)
        delta = operator.double()@potential.masked_fill(~valid[:, None], 0.)
        if CLASS[0] in methods: values[CLASS[0]] = baseline+delta
        if CLASS[1] in methods:
            candidate_delta = values[PRIMARY]-baseline
            scale = candidate_delta.norm()/delta.norm().clamp_min(CONFIG.epsilon)
            values[CLASS[1]] = baseline+delta*scale
            stats['class_strength_norm_error'] = float((delta*scale).norm()-candidate_delta.norm())
            stats['class_strength_scale'] = float(scale)
            stats['class_strength_zero_direction'] = float(delta.norm() == 0)
    source_decision = risk[:, members] > 0
    for family, names in (('alias', ALIAS), ('spatial', SPATIAL), ('rival', RIVAL)):
        for i, name in enumerate(names):
            if name not in methods: continue
            seed = CONFIG.random_seed+i
            if family == 'alias': changed = alias_null(risk, members, canonical, seed)
            elif family == 'spatial': changed = spatial_null(risk, valid, seed)
            else:
                changed, report = rival_degree_null(risk, members, canonical, seed)
                stats.update({name+'__'+k: v for k, v in report.items()})
            decisions = changed[:, members] > 0
            if family == 'spatial':
                error = int((decisions.sum(0)-source_decision.sum(0)).abs().max())
            else:
                error = int((decisions.sum(2)-source_decision.sum(2)).abs().max())
            if (error or bool((changed[:, canonical] != 0).any()) or bool((changed[~valid] != 0).any())
                    or bool(changed.gather(-1, parents[None, :, None].expand(len(valid), -1, 1)).any())):
                raise RuntimeError('Null changed its declared decision budget or protection.')
            directed = directed_cached(observed_wide, members, changed, valid)
            margins = directed-directed.transpose(-1, -2)
            potential, _ = posterior_potential(project_actions(margins, target), posterior, valid)
            values[name] = baseline+operator.double()@potential
            stats[name+'__budget_max_error'] = error
            stats[name+'__decision_changed_fraction'] = float((decisions[valid] != source_decision[valid]).double().mean())
    values = {m: values[m] for m in methods}
    if not all(bool(torch.isfinite(v).all()) for v in values.values()): raise RuntimeError('Nonfinite attribution scores.')
    return values, risk, stats
