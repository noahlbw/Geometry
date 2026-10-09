"""Preserve directed alias admission before consistent competitive writeback."""
from dataclasses import dataclass

import numpy as np
import torch

from .alias_spatial_allocation import ROUNDING_GUARD, SEEDS, capacity_balanced_mean, validated_magnitudes
from .contrastive_reversal_alias import METHODS as PREVIOUS_METHODS


IMPLEMENTATION = 'geometry-rival-preserving-alias-reader-v1-20261003'
PRIMARY = 'RivalPreserving_Exact'
ALIAS_SHUFFLES = tuple('RivalAliasShuffle'+str(i)+'_Exact' for i in range(3))
SPATIAL_SHUFFLES = tuple('DirectionalShuffle'+str(i)+'_Exact' for i in range(3))
METHODS = (*PREVIOUS_METHODS, PRIMARY, 'RivalMeanLogit', 'RivalShuffledSupport_Exact',
    'RivalTextOnly_Exact', *ALIAS_SHUFFLES, 'DirectionalMean_Exact', *SPATIAL_SHUFFLES)


@dataclass(frozen=True)
class RivalPreservingConfig:
    query_chunk: int = 128
    numerical_tolerance: float = 1e-10


CONFIG = RivalPreservingConfig()


def competitive_potential(directed, valid, config=CONFIG):
    if (directed.ndim != 3 or directed.shape[-1] != directed.shape[-2]
            or directed.shape[-1] < 2 or valid.shape != directed.shape[:1]
            or not bool(torch.isfinite(directed).all()) or bool((directed > 0).any())
            or bool((directed[~valid] != 0).any()) or bool((directed.diagonal(dim1=-2, dim2=-1) != 0).any())):
        raise ValueError('Finite directed suppression, zero self/invalid actions required.')
    actions = directed.double()
    margins = actions-actions.transpose(-1, -2)
    classes = directed.shape[-1]
    potential = margins.sum(-1)/classes
    realized = potential[:, :, None]-potential[:, None]
    cycle = margins-realized
    normal_error = float(cycle.sum(-1).abs().max())
    gauge_error = float(potential.sum(-1).abs().max())
    if max(normal_error, gauge_error) > config.numerical_tolerance:
        raise RuntimeError('Competitive consistency numerical identity failed.')
    stats = {'normal_equation_max_error': normal_error, 'gauge_max_error': gauge_error,
        'requested_margin_energy': float(margins.square().sum()/2),
        'discarded_cycle_energy': float(cycle.square().sum()/2),
        'realized_margin_energy': float(realized.square().sum()/2),
        'mean_directed_suppression': float(actions[valid].abs().mean()) if bool(valid.any()) else 0.,
        'mean_potential_magnitude': float(potential[valid].abs().mean()) if bool(valid.any()) else 0.,
        'positive_potential_fraction': float((potential[valid] > 0).double().mean()) if bool(valid.any()) else 0.}
    return potential, margins, stats


def directional_controls(directed, cap, valid):
    """Match each directed pair's budget and capacities, not final potential spectra."""
    directed, cap = np.asarray(directed, np.float64), np.asarray(cap, np.float64)
    n, classes, other = directed.shape
    if other != classes or cap.shape != (n, classes) or np.any(np.diagonal(directed, axis1=1, axis2=2) != 0):
        raise ValueError('Directed class-pair fields and class capacities required.')
    bounds = np.broadcast_to(cap[:, :, None], directed.shape).copy()
    ids = np.arange(classes)
    bounds[:, ids, ids] = 0
    flat, bounds = directed.reshape(n, -1), bounds.reshape(n, -1)
    magnitude, effective = validated_magnitudes(flat, bounds, valid)
    known = np.flatnonzero(valid)
    outputs = {'DirectionalMean_Exact': capacity_balanced_mean(flat, bounds, valid).reshape(directed.shape)}
    for name, seed in zip(SPATIAL_SHUFFLES, SEEDS):
        shuffled = np.zeros_like(magnitude)
        generator = np.random.default_rng(seed)
        for c in range(classes*classes):
            values = np.sort(magnitude[known, c])[::-1]
            available = np.ones(len(known), bool)
            # Zero placements need no writes; this keeps large pair-control audits tractable.
            for value in values[values > 0]:
                eligible = np.flatnonzero(available & (effective[known, c] >= value))
                if not len(eligible):
                    raise RuntimeError('Feasible directed-pair assignment failed.')
                chosen = generator.choice(eligible)
                shuffled[known[chosen], c] = value
                available[chosen] = False
        outputs[name] = -shuffled.reshape(directed.shape)
    for name, value in outputs.items():
        if (np.max(np.abs(value.sum(0)-directed.sum(0))) > 1e-10
                or np.max(-value-np.broadcast_to(cap[:, :, None], directed.shape)) > ROUNDING_GUARD
                or np.any(value[~valid] != 0) or np.any(np.diagonal(value, axis1=1, axis2=2) != 0)):
            raise RuntimeError('Directed-pair budget/capacity/validity changed: '+name)
    return outputs
