"""Equivalent requested-endpoint reader for the frozen fine-only source."""
import torch

from .fine_alias_view import CONFIG
from .native_alias_noise import signed_potential
from .native_query_alias import native_risk
from .rival_alias_fast import cache_observations, directed_cached, sampled_cached
from .rival_competition_admission import posterior_potential, project_actions, sampled_classes
from .rival_fine_support import support_control
from .rival_responsibility_intersection import ABSOLUTE as PRIMARY, protected_weights, sampled_log_responsibilities
from .rival_survivor_redistribution import allocation_directed, redistribute


IMPLEMENTATION = 'frozen-fine-responsibility-requested-endpoint-reader-v1-20261005'
PROJECTED = 'FineRivalProjected_Exact'
WIDE = 'WideResponsibilityOnly_Exact'
WITHIN = 'FineWithinResponsibilityOnly_Exact'
SHUFFLES = tuple('FineResponsibilityShuffle'+str(i)+'_Exact' for i in range(3))
METHODS = ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact', PROJECTED,
    PRIMARY, WIDE, WITHIN, *SHUFFLES)


@torch.inference_mode()
def responsibility_scores(local, operator, broad, wide, wide_count, fine, fine_count,
                          coordinates, fine_coordinates, valid, members, canonical,
                          parents, image_size, *, methods=METHODS):
    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Nonempty declared frozen responsibility endpoints required.')
    observed_wide = cache_observations(wide, wide_count, coordinates, image_size, members)
    observed_fine = cache_observations(fine, fine_count, fine_coordinates, (512, 512), members)
    native = sampled_cached(observed_fine, fine_coordinates).masked_fill(~valid[:, None, None], 0.)
    risk = native_risk(sampled_cached(observed_wide, coordinates), native, native, parents, canonical, valid, CONFIG)
    base = local.double()+operator.double()@(broad.double()-local.double())
    values = {'Geometry': local.double(), 'NoAdmission_Exact': base}
    diagnostics = {'additional_visual_forwards': 0., 'canonical_risk_max': float(risk[:, canonical].abs().max())}
    if 'RivalFineHard_Exact' in methods or PROJECTED in methods:
        directed = directed_cached(observed_wide, members, risk, valid)
        if 'RivalFineHard_Exact' in methods:
            potential, _ = signed_potential(directed, valid)
            values['RivalFineHard_Exact'] = base+operator.double()@potential
    needs_source = set(methods).intersection((PRIMARY, WIDE, WITHIN, *SHUFFLES))
    if PROJECTED in methods or needs_source:
        posterior = base.softmax(-1)
        innovation = sampled_classes(observed_fine, fine_coordinates).double()-broad.double()
        target = (innovation[:, :, None]-innovation[:, None]).masked_fill(~valid[:, None, None], 0.)
    if PROJECTED in methods:
        action = project_actions(directed-directed.transpose(-1, -2), target)
        potential, _ = posterior_potential(action, posterior, valid)
        values[PROJECTED] = base+operator.double()@potential
    allocations = {}
    if set(methods).intersection((PRIMARY, WITHIN, *SHUFFLES)):
        fine_pair, fine_within = sampled_log_responsibilities(observed_fine, valid)
        if PRIMARY in methods or set(methods).intersection(SHUFFLES):
            weights = redistribute(protected_weights(fine_pair, risk, parents, canonical, valid),
                risk, members, canonical)
            allocations[PRIMARY] = weights
            for i, name in enumerate(SHUFFLES):
                if name in methods:
                    allocations[name] = support_control(weights, risk, members, canonical, CONFIG.random_seed+i)
        if WITHIN in methods:
            allocations[WITHIN] = redistribute(protected_weights(fine_within[..., None].expand_as(risk),
                risk, parents, canonical, valid), risk, members, canonical)
    if WIDE in methods:
        wide_pair, _ = sampled_log_responsibilities(observed_wide, valid)
        allocations[WIDE] = redistribute(protected_weights(wide_pair, risk, parents, canonical, valid),
            risk, members, canonical)
    keep = risk == 0
    mass = keep[:, members].sum(2).double()
    for name, allocation in allocations.items():
        if name not in methods:
            continue
        error = float((allocation[:, members].sum(2)-mass).abs().max())
        if error > 1e-10 or bool((allocation[:, canonical] != 1).any()) or bool((allocation[~keep] != 0).any()):
            raise RuntimeError('Requested reader changed frozen survivor/canonical support or mass.')
        directed = allocation_directed(observed_wide, members, allocation, keep, valid)
        action = project_actions(directed-directed.transpose(-1, -2), target)
        potential, _ = posterior_potential(action, posterior, valid)
        values[name] = base+operator.double()@potential
        diagnostics[name+'__mass_max_error'] = error
        if name in SHUFFLES:
            error = float((allocation[:, members].sort(2).values-weights[:, members].sort(2).values).abs().max())
            diagnostics[name+'__spectrum_max_error'] = error
            if error != 0:
                raise RuntimeError('Requested reader changed own identity-null spectrum.')
    values = {name: values[name] for name in methods}
    if not all(bool(torch.isfinite(v).all()) for v in values.values()):
        raise RuntimeError('Nonfinite requested responsibility scores.')
    return values, risk, diagnostics
