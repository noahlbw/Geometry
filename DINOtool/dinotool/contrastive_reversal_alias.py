"""Reject background-exclusive alias/rival advantages in matched interventions."""
import torch

from .target_context_alias import TargetContextConfig


IMPLEMENTATION = 'geometry-contrastive-reversal-alias-source-v1-20261003'
PRIMARY = 'ContrastReversal_Exact'
ALIAS_SHUFFLES = tuple('ShuffledAlias'+str(i)+'_Reversal' for i in range(3))
SPATIAL_SHUFFLES = tuple('SpatialShuffle'+str(i)+'_Exact' for i in range(3))
METHODS = ('Geometry', 'Anchored_Exact', 'TargetContext_Exact', 'MeanLogit_Original', PRIMARY,
    'Reversal_MeanLogit', 'RelativeDependence_Exact', 'ShuffledSupport_Reversal',
    'SingleFill0_Reversal', 'SingleFill1_Reversal', 'TextOnly_Exact', *ALIAS_SHUFFLES,
    'CapacityMean_Exact', *SPATIAL_SHUFFLES)
CONFIG = TargetContextConfig()


def reversal_risk(full, kept, removed, parents, canonical, supported, config=CONFIG, fill=None):
    if (full.ndim != 2 or kept.shape != (2, *full.shape) or removed.shape != kept.shape
            or parents.shape != full.shape[1:] or supported.shape != full.shape[:1]
            or fill not in (None, 0, 1)):
        raise ValueError('Two matched raw alias observations and query validity required.')
    if not bool(torch.isfinite(full).all() and torch.isfinite(kept).all() and torch.isfinite(removed).all()):
        raise ValueError('Finite observations required.')
    original_margin = full[..., None]-full[:, None, canonical]
    target_margin = kept[..., None]-kept[:, :, None, canonical]
    background_margin = removed[..., None]-removed[:, :, None, canonical]
    reversal = (original_margin[None] > 0) & (target_margin < 0) & (background_margin > 0)
    strength = (-target_margin).clamp_min(0)/(background_margin-target_margin).clamp_min(config.epsilon)
    gamma = torch.where(reversal, strength.clamp(0, 1), 0.)
    relative = (original_margin[None]-target_margin).clamp_min(0)/(
        (original_margin[None]-target_margin).abs()+(original_margin[None]-background_margin).abs()).clamp_min(config.epsilon)
    result = {}
    for name, values in (('reversal', gamma), ('relative_dependence', relative.clamp(0, 1))):
        risk = values.amin(0) if fill is None else values[fill]
        risk[:, canonical] = 0
        risk.masked_fill_(~supported[:, None, None], 0)
        risk.scatter_(-1, parents[None, :, None].expand(len(full), -1, 1), 0)
        result[name] = risk
    return result
