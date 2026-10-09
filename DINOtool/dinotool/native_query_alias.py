"""Unmasked native-resolution witnesses for bounded contextual alias admission."""
from dataclasses import dataclass

import torch


IMPLEMENTATION = 'geometry-native-query-alias-witness-v1-20261003'
PRIMARY = 'NativeQuery_Exact'
LEGACY = ('Geometry', 'Anchored_Exact', 'MeanLogit_Original', 'ContrastReversal_Exact',
          'RivalPreserving_Exact', 'MatchedContribution_Exact')
ALIAS_SHUFFLES = tuple('NativeAliasShuffle'+str(i)+'_Exact' for i in range(3))
SPATIAL_SHUFFLES = tuple('NativeDirectionalShuffle'+str(i)+'_Exact' for i in range(3))
NEW_METHODS = (PRIMARY, 'NativeQueryOnly_Exact', 'NativeSharedSupport_Exact',
    'NativeShuffledSupport_Exact', *ALIAS_SHUFFLES, 'NativeMeanLogit',
    'NativeObservationMeanLogit', 'NativeObservationAnchored',
    'NativeDirectionalMean_Exact', *SPATIAL_SHUFFLES)
METHODS = (*LEGACY, *NEW_METHODS)
GATE_CONTROLS = (*LEGACY[2:], *NEW_METHODS[1:])
GATE = {'minimum_mean_gain_pp': .1, 'minimum_domain_wins': 5,
    'maximum_protocol_loss_pp': 1., 'mean_above_controls': list(GATE_CONTROLS),
    'no_automatic_full_rollout': True}


@dataclass(frozen=True)
class NativeQueryConfig:
    beta: float = 1.
    epsilon: float = 1e-6
    native_window_side: int = 512
    native_crop_side: int = 336
    native_stride: int = 224
    shared_cell_pixels: int = 128
    random_seed: int = 20261003
    query_chunk: int = 128


CONFIG = NativeQueryConfig()


def native_crop_positions(height, width, config=CONFIG):
    from .inference import tile_starts

    if not (0 < height <= config.native_window_side and 0 < width <= config.native_window_side):
        raise ValueError('Actual dimensions within the fixed native window required.')
    overlap = config.native_crop_side-config.native_stride
    if not 0 <= overlap < config.native_crop_side:
        raise ValueError('Positive native stride no larger than crop side required.')
    positions = [(t, l, min(config.native_crop_side, height-t), min(config.native_crop_side, width-l))
        for t in tile_starts(height, config.native_crop_side, overlap)
        for l in tile_starts(width, config.native_crop_side, overlap)]
    if not 1 <= len(positions) <= 4:
        raise RuntimeError('Native crop grid violates the declared maximum of four forwards.')
    return positions


def query_relation(relation, valid):
    if (relation.shape != (len(valid), len(valid)) or valid.dtype != torch.bool
            or not bool(torch.isfinite(relation).all()) or bool((relation < 0).any())):
        raise ValueError('Finite nonnegative Geometry relation and validity required.')
    weights = relation.float().masked_fill(~valid[None], 0.)
    mass = weights.sum(-1)
    known = valid & (mass > 0)
    weights = weights / mass[:, None].clamp_min(CONFIG.epsilon)
    weights = weights.masked_fill(~known[:, None], 0.)
    return weights, known


def support_controls(weights, coordinates, valid, config=CONFIG):
    if (weights.shape != (len(valid), len(valid)) or coordinates.shape != (len(valid), 2)
            or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any())):
        raise ValueError('Matching query relation, coordinates and validity required.')
    cells = (coordinates/config.shared_cell_pixels).floor().long()
    _, assignments = cells.unique(dim=0, sorted=True, return_inverse=True)
    shared = torch.zeros_like(weights)
    for cell in assignments.unique().tolist():
        use = valid & (assignments == cell)
        if bool(use.any()):
            shared[use] = weights[use].mean(0)
    ids = valid.nonzero().flatten()
    shuffled = torch.zeros_like(weights)
    if len(ids):
        generator = torch.Generator().manual_seed(config.random_seed)
        permutation = ids[torch.randperm(len(ids), generator=generator).to(ids.device)]
        shuffled[:, ids] = weights[:, permutation]
    return {'shared': shared, 'shuffle': shuffled}


def support_margins(weights, margins):
    if (margins.ndim != 3 or weights.shape != (len(margins), len(margins))
            or not bool(torch.isfinite(weights).all() and torch.isfinite(margins).all())
            or bool((weights < 0).any())):
        raise ValueError('Finite query relation and alias/rival margins required.')
    return (weights @ margins.flatten(1)).reshape_as(margins)


def native_risk(broad, native, supported, parents, canonical, known, config=CONFIG):
    """Require both fine query and its visual support to contradict a broad advantage."""
    if (broad.ndim != 3 or native.shape != broad.shape or supported.shape != broad.shape
            or parents.shape != broad.shape[1:2] or canonical.shape != broad.shape[2:]
            or known.shape != broad.shape[:1] or known.dtype != torch.bool
            or parents.dtype != torch.long or canonical.dtype != torch.long
            or not bool(torch.isfinite(broad).all() and torch.isfinite(native).all()
                        and torch.isfinite(supported).all())
            or bool((parents < 0).any()) or bool((parents >= broad.shape[-1]).any())
            or bool((canonical < 0).any()) or bool((canonical >= broad.shape[1]).any())
            or len(canonical.unique()) != len(canonical)):
        raise ValueError('Matching finite observations and actual class identities required.')
    advantage = broad.clamp_min(0)
    negative = torch.minimum((-native).clamp_min(0), (-supported).clamp_min(0))
    eligible = (broad > 0) & (native < 0) & (supported < 0)
    risk = torch.where(eligible, negative/(advantage+negative).clamp_min(config.epsilon), 0.)
    risk[:, canonical] = 0.
    risk.masked_fill_(~known[:, None, None], 0.)
    risk.scatter_(-1, parents[None, :, None].expand(len(broad), -1, 1), 0.)
    return risk
