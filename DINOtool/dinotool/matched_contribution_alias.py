"""Measure contextual alias reversal in the actual profiled prediction units."""
from dataclasses import dataclass
import math

import torch

from .rival_preserving_alias import METHODS as LEGACY


IMPLEMENTATION = 'geometry-matched-contribution-alias-source-v1-20261003'
PRIMARY = 'MatchedContribution_Exact'
ALIAS_SHUFFLES = tuple('MatchedAliasShuffle'+str(i)+'_Exact' for i in range(3))
SPATIAL_SHUFFLES = tuple('MatchedDirectionalShuffle'+str(i)+'_Exact' for i in range(3))
NEW_METHODS = (PRIMARY, 'MatchedMeanLogit', 'MatchedShuffledSupport_Exact', 'MatchedTextOnly_Exact',
               *ALIAS_SHUFFLES, 'MatchedDirectionalMean_Exact', *SPATIAL_SHUFFLES)
METHODS = (*LEGACY, *NEW_METHODS)
GATE_CONTROLS = ('ContrastReversal_Exact', 'RivalPreserving_Exact', 'RivalShuffledSupport_Exact',
    *tuple('RivalAliasShuffle'+str(i)+'_Exact' for i in range(3)), 'DirectionalMean_Exact',
    *tuple('DirectionalShuffle'+str(i)+'_Exact' for i in range(3)), *NEW_METHODS[1:])
GATE = {'minimum_mean_gain_pp': .1, 'minimum_domain_wins': 5, 'maximum_protocol_loss_pp': 1.,
        'mean_above_controls': list(GATE_CONTROLS), 'no_automatic_full_rollout': True}


@dataclass(frozen=True)
class MatchedContributionConfig:
    beta: float = 1.
    epsilon: float = 1e-6
    random_seed: int = 20261003
    query_chunk: int = 128


CONFIG = MatchedContributionConfig()


def contribution_margins(profiled, beta=1.):
    """At each wide token, log-mean-exp of alias margins recovers class margins."""
    if profiled.ndim != 3 or profiled.shape[1] < 2 or profiled.shape[2] < 2 or beta <= 0:
        raise ValueError('Finite [token,class,alias] profiled evidence required.')
    if not bool(torch.isfinite(profiled).all()):
        raise ValueError('Nonfinite profiled evidence.')
    k = profiled.shape[-1]
    rival = (beta*profiled).logsumexp(-1)/beta-math.log(k)/beta
    return profiled.flatten(1)[..., None]-rival[:, None]


def matched_reversal(full, kept, removed, parents, canonical, supported, config=CONFIG):
    if (full.ndim != 3 or full.shape[-1] < 2 or kept.shape != (2, *full.shape)
            or removed.shape != kept.shape or parents.shape != full.shape[1:2]
            or canonical.shape != full.shape[2:] or supported.shape != full.shape[:1]
            or parents.dtype != torch.long or canonical.dtype != torch.long or supported.dtype != torch.bool):
        raise ValueError('Matched two-fill alias/rival margins and actual group identities required.')
    if (not bool(torch.isfinite(full).all() and torch.isfinite(kept).all() and torch.isfinite(removed).all())
            or bool((parents < 0).any()) or bool((parents >= full.shape[-1]).any())
            or len(canonical.unique()) != len(canonical)):
        raise ValueError('Finite source and valid class identities required.')
    eligible = (full[None] > 0) & (kept < 0) & (removed > 0)
    strength = (-kept).clamp_min(0)/(removed-kept).clamp_min(config.epsilon)
    risk = torch.where(eligible, strength.clamp(0, 1), 0.).amin(0)
    risk[:, canonical] = 0
    risk.masked_fill_(~supported[:, None, None], 0)
    risk.scatter_(-1, parents[None, :, None].expand(len(full), -1, 1), 0)
    return risk


def stencil_margins(profiled, indices, coefficients, beta=1.):
    if (indices.ndim != 2 or coefficients.shape != indices.shape or indices.dtype != torch.long
            or not bool(torch.isfinite(coefficients).all()) or bool((coefficients < 0).any())
            or bool((indices < 0).any()) or bool((indices >= len(profiled)).any())):
        raise ValueError('Valid nonnegative original crop stencil required.')
    margins = contribution_margins(profiled, beta)
    return (margins[indices]*coefficients[..., None, None]).sum(1)
