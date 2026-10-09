"""Fixed-action writeback diagnosis; no deployed selector or fitted coefficient."""
import torch

from .bounded_alias_path_audit import donor_targets, privileged_donor_changes


IMPLEMENTATION = 'geometry-bounded896-fixed-alias-norm-matched-write-audit-v1-20261006'
WRITERS = ('H', 'TwoH', 'DirectLegacy', 'DirectMatched', 'GeometryMatched', 'PositiveHMatched')
MATCHED = WRITERS[3:]
BASELINE = 'Geometry_PatchOnly2Coupled'


def field_power(fields, tiles):
    return torch.stack([value[tile.valid].double().square().sum(0)
                        for value, tile in zip(fields, tiles)]).sum(0)


def match_magnitude(reference, template, tiles):
    """Match every action column across all valid tile fields, before pixel assembly."""
    target = field_power(reference, tiles)
    current = field_power(template, tiles)
    fallback = (current == 0) & (target > 0)
    scale = (target/torch.where(current > 0, current, torch.ones_like(current))).sqrt()
    result = [torch.where(fallback[None], ref, value*scale[None])
              for ref, value in zip(reference, template)]
    actual = field_power(result, tiles)
    relative_error = (actual-target).abs()/target.clamp_min(1e-30)
    if not bool(torch.isfinite(scale).all()) or float(relative_error.max()) > 1e-10:
        raise RuntimeError('Magnitude matching failed.')
    return result, dict(norm_max_relative_error=float(relative_error.max()),
                        fallback_columns=float(fallback.sum()))


def writer_changes(source, donor_changes):
    """Reuse identical changes; matched routes exclude padded donors and queries."""
    tiles = source.tiles
    if len(donor_changes) != len(tiles) or not tiles:
        raise ValueError('Matching nonempty original tile/action fields required.')
    changes, reference, geometry, positive, direct = [], [], [], [], []
    for tile, change in zip(tiles, donor_changes):
        if (change.ndim != 2 or len(change) != len(tile.valid)
                or tile.operator.shape != (len(change), len(change))
                or tile.relation is None or tile.relation.shape != tile.operator.shape
                or not bool(torch.isfinite(change).all())):
            raise ValueError('Finite action, original H/G and matching validity required.')
        delta = change.double()
        valid_delta = delta.masked_fill(~tile.valid[:, None], 0.)
        g = tile.relation.double().masked_fill(~tile.valid[None], 0.)
        if not bool(torch.isfinite(g).all()) or bool((g < 0).any()):
            raise ValueError('Original finite nonnegative Geometry relation required.')
        g = (g/g.sum(-1, keepdim=True).clamp_min(1e-12)).masked_fill(~tile.valid[:, None], 0.)
        changes.append(delta)
        reference.append(tile.operator.double()@delta)
        direct.append(valid_delta)
        geometry.append(g@valid_delta)
        positive.append(tile.operator.double().clamp_min(0)@valid_delta)
    result = dict(H=reference, TwoH=[2*x for x in reference], DirectLegacy=changes)
    diagnostics = {}
    for name, template in zip(MATCHED, (direct, geometry, positive)):
        result[name], metrics = match_magnitude(reference, template, tiles)
        diagnostics.update({name+'_'+k: v for k, v in metrics.items()})
    valid_values = torch.cat([value[tile.valid].flatten() for value, tile in zip(reference, tiles)])
    diagnostics['H_positive_patch_field_fraction'] = float((valid_values > 1e-12).double().mean())
    return result, diagnostics


def privileged_changes(source, target):
    """Freeze ONE labeled donor policy before choosing any writer; audit only."""
    result = []
    for tile in source.tiles:
        _, wide = privileged_donor_changes(tile.local_changes[0], tile.wide_changes[0], source.canonical,
            donor_targets(target, tile, source), 'Wide')
        result.append(wide)
    return result
