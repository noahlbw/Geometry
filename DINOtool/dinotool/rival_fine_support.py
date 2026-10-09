"""Fine alias/rival support weights on the unchanged hard admission support."""
import torch

from .fine_alias_view import CONFIG
from .native_query_alias import native_risk
from .rival_alias_fast import cache_observations, sampled_cached, directed_cached
from .rival_competition_admission import (posterior_potential, project_actions,
    retained_competition_scores, sampled_classes)


IMPLEMENTATION = 'frozen-fine-alias-rival-support-weighting-v1-20261005'
PRIMARY = 'FineRivalSupport_Exact'
SHUFFLES = tuple('FineSupportShuffle'+str(i)+'_Exact' for i in range(3))
NEW_METHODS = (PRIMARY, 'FineSupportClassMean_Exact', *SHUFFLES)
REPLAY = ('NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact')
METHODS = (*REPLAY, *NEW_METHODS)


def support_weights(native, old_risk, parents, canonical, valid):
    if (native.shape != old_risk.shape or native.ndim != 3
            or parents.shape != native.shape[1:2] or canonical.shape != native.shape[2:]
            or valid.shape != native.shape[:1]
            or not bool(torch.isfinite(native).all() and torch.isfinite(old_risk).all())):
        raise ValueError('Finite matching fine margins and original admission identities required.')
    # Two-option response responsibility, not a probability of semantic correctness.
    weights = (CONFIG.beta*native.double()).sigmoid().clamp_min(CONFIG.epsilon)
    weights.masked_fill_(old_risk > 0, 0.)
    weights[:, canonical] = 1.
    weights.scatter_(-1, parents[None, :, None].expand(len(native), -1, 1), 1.)
    weights.masked_fill_(~valid[:, None, None], 1.)
    return weights


def support_control(weights, old_risk, members, canonical, seed=None):
    grouped = weights[:, members].permute(0, 1, 3, 2).contiguous()
    eligible = (old_risk[:, members].permute(0, 1, 3, 2) == 0)
    eligible &= ~torch.isin(members, canonical)[None, :, None]
    rows = grouped.reshape(-1, members.shape[1])
    mask = eligible.reshape_as(rows)
    count = mask.sum(-1, keepdim=True)
    if seed is None:
        mean = rows.masked_fill(~mask, 0.).sum(-1, keepdim=True)/count.clamp_min(1)
        changed = torch.where(mask, mean, rows)
    else:
        generator = torch.Generator(device=weights.device).manual_seed(seed)
        positions = torch.arange(rows.shape[1], device=rows.device).expand_as(rows)
        source = positions.masked_fill(~mask, rows.shape[1]).argsort(-1)
        destination = torch.rand(rows.shape, generator=generator, device=rows.device).masked_fill(~mask, torch.inf).argsort(-1)
        replacement = torch.where(positions < count, rows.gather(1, source), rows.gather(1, destination))
        changed = rows.clone().scatter_(1, destination, replacement)
    output = weights.clone()
    output[:, members] = changed.reshape_as(grouped).permute(0, 1, 3, 2)
    return output


@torch.inference_mode()
def support_scores(local, operator, broad, wide, wide_count, fine, fine_count,
                   coordinates, fine_coordinates, valid, members, canonical, parents,
                   image_size, methods=METHODS):
    if not set(methods).issubset(METHODS):
        raise ValueError('Declared support endpoint required.')
    replay, old_risk, _ = retained_competition_scores(local, operator, broad, wide, wide_count,
        fine, fine_count, coordinates, fine_coordinates, valid, members, canonical, parents,
        image_size, methods=REPLAY)
    cached_wide = cache_observations(wide, wide_count, coordinates, image_size, members)
    cached_fine = cache_observations(fine, fine_count, fine_coordinates, (512, 512), members)
    native = sampled_cached(cached_fine, fine_coordinates).masked_fill(~valid[:, None, None], 0.)
    actual_risk = native_risk(sampled_cached(cached_wide, coordinates), native, native,
                              parents, canonical, valid, CONFIG)
    if not torch.equal(actual_risk, old_risk):
        raise RuntimeError('Original conditional hard support changed.')
    base = replay['NoAdmission_Exact']
    posterior = base.softmax(-1)
    innovation = sampled_classes(cached_fine, fine_coordinates).double()-broad.double()
    target = (innovation[:, :, None]-innovation[:, None]).masked_fill(~valid[:, None, None], 0.)
    weights = support_weights(native, old_risk, parents, canonical, valid)
    allocations = {PRIMARY: weights}
    if 'FineSupportClassMean_Exact' in methods:
        allocations['FineSupportClassMean_Exact'] = support_control(weights, old_risk, members, canonical)
    for i, name in enumerate(SHUFFLES):
        if name in methods:
            allocations[name] = support_control(weights, old_risk, members, canonical, CONFIG.random_seed+i)
    values = dict(replay)
    diagnostics = {'additional_visual_forwards': 0, 'canonical_weight_min': float(weights[:, canonical].min()),
        'old_rejected_weights_max': float(weights[old_risk > 0].abs().max()) if bool((old_risk > 0).any()) else 0.,
        'mean_noncanonical_weight': float(weights[:, ~torch.isin(torch.arange(len(parents), device=parents.device), canonical)].mean()),
        'control_spectrum_max_errors': {}}
    for name, allocation in allocations.items():
        if name not in methods:
            continue
        directed = directed_cached(cached_wide, members, 1-allocation, valid, 'weighted')
        requested = directed-directed.transpose(-1, -2)
        bounded = project_actions(requested, target)
        potential, stats = posterior_potential(bounded, posterior, valid)
        values[name] = base+operator.double()@potential
        if name in SHUFFLES:
            error = float((allocation[:, members].sort(2).values-weights[:, members].sort(2).values).abs().max())
            diagnostics['control_spectrum_max_errors'][name] = error
            if error != 0:
                raise RuntimeError('Alias-null weight spectrum changed.')
        if name == 'FineSupportClassMean_Exact':
            error = float((allocation[:, members].sum(2)-weights[:, members].sum(2)).abs().max())
            diagnostics['class_mean_weight_mass_max_error'] = error
            if error > 1e-10:
                raise RuntimeError('Class-mean control weight mass changed.')
        diagnostics[name] = stats
    values = {name: values[name] for name in methods}
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite support-weighted scores.')
    return values, diagnostics
