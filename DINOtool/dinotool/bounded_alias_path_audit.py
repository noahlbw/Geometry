"""Fixed-source alias counterfactuals; labeled controls are audit-only."""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


IMPLEMENTATION = 'geometry-bounded896-alias-path-counterfactual-audit-v1-20261006'
PATHS = ('Local', 'Wide', 'Joint', 'DirectWide')
POOLS = ('FixedSlots', 'SurvivorNormalized')
CHUNK = 4


def removal_changes(evidence):
    """Exact log-mass change for each single removal, without cancellation."""
    if evidence.ndim != 3 or evidence.shape[-1] < 2 or not bool(torch.isfinite(evidence).all()):
        raise ValueError('Finite query/class/alias evidence with at least two aliases required.')
    values = evidence.double()
    prefix = values.logcumsumexp(-1)
    suffix = values.flip(-1).logcumsumexp(-1).flip(-1)
    empty = torch.full_like(values[..., :1], -torch.inf)
    remaining = torch.logaddexp(torch.cat((empty, prefix[..., :-1]), -1),
                               torch.cat((suffix[..., 1:], empty), -1))
    fixed = remaining - values.logsumexp(-1, keepdim=True)
    normalized = fixed + math.log(values.shape[-1] / (values.shape[-1] - 1))
    return fixed, normalized


def routed_change(local_change, wide_change, operator, path):
    if local_change.shape != wide_change.shape or operator.shape != (len(local_change), len(local_change)):
        raise ValueError('Matching donor changes and original reconstruction operator required.')
    local, wide = local_change.double(), wide_change.double()
    if path == 'Local':
        return local - operator.double() @ local
    if path == 'Wide':
        return operator.double() @ wide
    if path == 'Joint':
        return local + operator.double() @ (wide-local)
    if path == 'DirectWide':
        return wide
    raise ValueError('Undeclared audit path.')


def privileged_donor_changes(local_changes, wide_changes, canonical, target, path):
    """One noncanonical removal per wrong-class donor, not a deployable selector."""
    if local_changes.shape != wide_changes.shape or target.shape != local_changes.shape[:1]:
        raise ValueError('Matching source changes and donor labels required.')
    n, classes, count = local_changes.shape
    if canonical.shape != (classes,) or count < 2 or path not in PATHS:
        raise ValueError('Matching protected canonical positions and declared path required.')
    criterion = local_changes if path == 'Local' else wide_changes
    if path == 'Joint':
        criterion = local_changes + wide_changes
    protected = torch.arange(count, device=criterion.device)[None] == canonical[:, None]
    chosen = criterion.masked_fill(protected[None], torch.inf).argmin(-1)
    known = (target >= 0) & (target < classes)
    active = known[:, None] & (target[:, None] != torch.arange(classes, device=target.device)[None])
    ids = chosen[..., None]
    return (local_changes.gather(-1, ids)[..., 0].masked_fill(~active, 0.),
            wide_changes.gather(-1, ids)[..., 0].masked_fill(~active, 0.))


@dataclass
class AuditTile:
    top: int
    left: int
    scores: dict
    local_changes: torch.Tensor
    wide_changes: torch.Tensor
    operator: torch.Tensor
    valid: torch.Tensor
    relation: torch.Tensor | None = None
    local_aliases: torch.Tensor | None = None
    wide_aliases: torch.Tensor | None = None
    native_aliases: torch.Tensor | None = None


@dataclass
class AuditSource:
    size: tuple
    output_size: tuple
    tiles: list
    members: torch.Tensor
    canonical: torch.Tensor
    aliases: tuple
    diagnostics: dict


@torch.inference_mode()
def assemble(source, changes=None, method='Geometry_PatchOnly2Coupled'):
    """Reuse the original probability stitching and full-size restoration."""
    from .device_probability_accumulator import DeviceProbabilityAccumulator
    from .inference import hann_blend_window

    first = source.tiles[0].scores[method]
    classes, device = first.shape[-1], first.device
    blend = torch.from_numpy(hann_blend_window(512)).to(device)
    with DeviceProbabilityAccumulator(classes, *source.size, device) as accumulator:
        for i, tile in enumerate(source.tiles):
            value = tile.scores[method]
            if changes is not None:
                value = value.double() + changes[i].double()
            dense = F.interpolate(value.T.reshape(1, classes, 32, 32), (512, 512),
                                  mode='bilinear', align_corners=False)[0]
            if method == 'Geometry':
                dense = dense / .07
            ah, aw = min(512, source.size[0]-tile.top), min(512, source.size[1]-tile.left)
            accumulator.add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], tile.left, tile.top)
        return accumulator.finalize_resized(source.output_size)


@torch.inference_mode()
def capture(image, geometry, banks, vip, queries, cache=None, *, retain_alias_evidence=False):
    from benchmark_grouped_class_readout import CheckedGroupedCache
    from eval_geometry_vip_reliability import sample_broad
    from eval_rival_fine_full import prepare_wide
    from eval_sparse_alias_reuse import forbid_fine
    from .alias_action_capacity import reconstruction_operator
    from .bounded_patch_only import PRIMARY, readout
    from .bounded_physical_coupling import _BoundedBranch, geometry_windows, resize_geometry
    from .gear_ov import _crop_at
    from .rival_alias_count import canonical_indices

    cache = CheckedGroupedCache() if cache is None else cache
    cache.verify = False
    resized = resize_geometry(image)
    height, width = resized.shape[-2:]
    local_branch = _BoundedBranch(geometry, 512, 4, 'prepare_image')
    wide_branch = _BoundedBranch(vip, 336, 4, 'crop_patch_features')
    broad, crops, _, count = prepare_wide(image, wide_branch, {'clean': (banks, queries)})
    sources = {}
    wide_changes = {}
    wide_evidence = {}
    for p, bank in banks.items():
        query = queries[p]
        members = torch.stack([(bank.parent_indices == c).nonzero().flatten() for c in range(bank.class_count)])
        if members.shape[-1] != 20 or bank.alias_names != query.aliases:
            raise ValueError('Unchanged matched20 alias banks required.')
        canonical_global = canonical_indices(bank.class_names, bank.alias_names, bank.parent_indices)
        canonical = (members == canonical_global[:, None]).long().argmax(-1)
        source = AuditSource((height, width), tuple(image.shape[-2:]), [], members, canonical,
                             bank.alias_names, {})
        sources[p] = source
        fields = torch.zeros(2, members.numel(), *count.shape, dtype=torch.float64, device=geometry.device)
        alias_field = torch.zeros(members.numel(), *count.shape, device=geometry.device) if retain_alias_evidence else None
        for crop in crops['clean', p]:
            weights = crop.salience[members].softmax(-1)
            # Match the observer's frozen profile, including its mean divisor.
            evidence = crop.alias_logits[:, members] * (weights/weights.mean(-1, keepdim=True))
            if retain_alias_evidence:
                dense_alias = F.interpolate(evidence.permute(1, 2, 0).reshape(1, members.numel(), 21, 21),
                                            (336, 336), mode='bilinear', align_corners=False)[0]
                alias_field[:, crop.top:crop.top+crop.actual_height, crop.left:crop.left+crop.actual_width] += (
                    dense_alias[:, :crop.actual_height, :crop.actual_width])
            changes = torch.stack(removal_changes(evidence))
            dense = F.interpolate(changes.permute(0, 2, 3, 1).reshape(2, members.numel(), 21, 21),
                                  (336, 336), mode='bilinear', align_corners=False)
            fields[:, :, crop.top:crop.top+crop.actual_height, crop.left:crop.left+crop.actual_width] += (
                dense[:, :, :crop.actual_height, :crop.actual_width])
        wide_changes[p] = fields/count[None, None]
        if retain_alias_evidence:
            wide_evidence[p] = alias_field/count[None]
    with forbid_fine():
        for top, left in geometry_windows(height, width):
            prepared = local_branch.prepare_image(_crop_at(resized, top, left, 512).to(geometry.device))
            y, x = torch.meshgrid(top+(torch.arange(32, device=geometry.device)+.5)*16,
                                  left+(torch.arange(32, device=geometry.device)+.5)*16, indexing='ij')
            valid = (y.flatten() < height) & (x.flatten() < width)
            operator, residual = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
            with geometry.backbone._autocast():
                local_features = readout(geometry.backbone.model.visual_model.head, prepared, 2.)[0]
            for p, bank in banks.items():
                text = F.normalize(bank.features.float(), dim=-1)
                aliases = (local_features.float() @ text.T)[0]
                original_raw = cache.geometry((prepared.geometry_projected.float() @ text.T)[0],
                                               bank.parent_indices, bank.class_count)
                local_raw = cache.geometry(aliases, bank.parent_indices, bank.class_count)
                local = local_raw/.07
                wide = sample_broad(broad['clean', p], top, left, height, width).reshape_as(local)
                scores = {'Geometry': original_raw,
                    'NoAdmission_Exact': (original_raw/.07).double()+operator @ (wide.double()-(original_raw/.07).double()),
                    PRIMARY: local.double()+operator @ (wide.double()-local.double())}
                local_delta = torch.stack(removal_changes(aliases[:, sources[p].members]/.07))
                sampled = torch.stack([sample_broad(field.float(), top, left, height, width)
                                       for field in wide_changes[p]])
                wide_delta = sampled.reshape(2, 1024, bank.class_count, 20).double()
                tile = AuditTile(top, left, scores, local_delta, wide_delta, operator, valid,
                                 prepared.geometry_patch_conditional[0].clone())
                if retain_alias_evidence:
                    tile.local_aliases = aliases[:, sources[p].members]/.07
                    tile.wide_aliases = sample_broad(wide_evidence[p], top, left, height, width).reshape_as(tile.local_aliases)
                    tile.native_aliases = (prepared.native_projected.float() @ text.T)[0][:, sources[p].members]/.07
                sources[p].tiles.append(tile)
                sources[p].diagnostics['operator_residual_max'] = max(
                    residual, sources[p].diagnostics.get('operator_residual_max', 0.))
    if not 1 <= local_branch.calls <= 4 or not 1 <= wide_branch.calls <= 4:
        raise RuntimeError('Actual bounded branch call budget differs.')
    for source in sources.values():
        for tile in source.tiles:
            if not all(bool(torch.isfinite(value).all()) for value in (*tile.scores.values(),
                                                                      tile.local_changes, tile.wide_changes)):
                raise RuntimeError('Nonfinite frozen audit source.')
        operators = torch.stack([tile.operator for tile in source.tiles])
        source.diagnostics.update(geometry_encodings=local_branch.calls, wide_encodings=wide_branch.calls,
            additional_visual_forwards=0, fine_forwards=0, extra_semantic_heads=0,
            operator_negative_entry_fraction=float((operators < 0).double().mean()),
            local_normalized_positive_fraction=float(torch.stack([t.local_changes[1] for t in source.tiles]).gt(0).double().mean()),
            wide_normalized_positive_fraction=float(torch.stack([t.wide_changes[1] for t in source.tiles]).gt(0).double().mean()),
            source_frozen_before_target_mask=True)
    return sources


def donor_targets(target, tile, source):
    values = torch.as_tensor(target, device=tile.valid.device)
    ys = (tile.top+(torch.arange(32, device=values.device)+.5)*16)/source.size[0]*target.shape[0]
    xs = (tile.left+(torch.arange(32, device=values.device)+.5)*16)/source.size[1]*target.shape[1]
    sampled = values[ys.long().clamp(0, target.shape[0]-1)[:, None],
                     xs.long().clamp(0, target.shape[1]-1)[None]].flatten().long()
    return sampled.masked_fill(~tile.valid, -1)


@torch.inference_mode()
def candidate_predictions(source, path, pool):
    """Enumerate global single-word removals with a fixed restoration batch cap."""
    pool_id = POOLS.index(pool)
    tile_changes = [routed_change(t.local_changes[pool_id].flatten(1),
                                  t.wide_changes[pool_id].flatten(1), t.operator, path)
                    for t in source.tiles]
    yield from candidate_predictions_from_changes(source, tile_changes)


@torch.inference_mode()
def candidate_predictions_from_changes(source, tile_changes):
    """Stitch already-frozen single-alias fields using the original batch contract."""
    from .inference import hann_blend_window

    classes, count = source.members.shape
    device = source.tiles[0].operator.device
    if len(tile_changes) != len(source.tiles) or any(
            changes.shape != (1024, classes*count) for changes in tile_changes):
        raise ValueError('One query/global-alias change field per original tile required.')
    blend = torch.from_numpy(hann_blend_window(512)).to(device)
    for start in range(0, classes*count, CHUNK):
        ids = torch.arange(start, min(start+CHUNK, classes*count), device=device)
        owners = ids//count
        probabilities = torch.zeros(len(ids), classes, *source.size, dtype=torch.float32, device=device)
        normalizer = torch.zeros(*source.size, device=device)
        for tile, changes in zip(source.tiles, tile_changes):
            scores = tile.scores['Geometry_PatchOnly2Coupled'][None].expand(len(ids), -1, -1).clone()
            scores[torch.arange(len(ids), device=device)[:, None],
                   torch.arange(1024, device=device)[None], owners[:, None]] += changes[:, ids].T
            dense = F.interpolate(scores.permute(0, 2, 1).reshape(len(ids), classes, 32, 32),
                                  (512, 512), mode='bilinear', align_corners=False).softmax(1).float()
            ah, aw = min(512, source.size[0]-tile.top), min(512, source.size[1]-tile.left)
            probabilities[:, :, tile.top:tile.top+ah, tile.left:tile.left+aw] += dense[:, :, :ah, :aw]*blend[None, None, :ah, :aw]
            normalizer[tile.top:tile.top+ah, tile.left:tile.left+aw] += blend[:ah, :aw]
        if not bool((normalizer > 0).all()):
            raise RuntimeError('Uncovered counterfactual probability assembly.')
        restored = F.interpolate(probabilities/normalizer[None, None], source.output_size,
                                 mode='bilinear', align_corners=False)
        yield start, restored.argmax(1).to(torch.uint8)


@torch.inference_mode()
def privileged_prediction(source, target, path, pool):
    pool_id = POOLS.index(pool)
    changes = []
    for tile in source.tiles:
        local, wide = privileged_donor_changes(tile.local_changes[0], tile.wide_changes[0], source.canonical,
                                               donor_targets(target, tile, source), path)
        if pool_id:
            labels = donor_targets(target, tile, source)
            active = ((labels >= 0) & (labels < len(source.members)))[:, None] & (
                labels[:, None] != torch.arange(len(source.members), device=labels.device)[None])
            offset = math.log(20/19)*active
            local, wide = local+offset, wide+offset
        changes.append(routed_change(local, wide, tile.operator, path))
    return assemble(source, changes)
