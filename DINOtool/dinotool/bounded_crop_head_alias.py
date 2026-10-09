"""Rival-conditioned alias weights from cropped, already encoded patch tokens."""
from contextlib import ExitStack
import math

import torch
import torch.nn.functional as F

from .alias_action_capacity import reconstruction_operator
from .bounded_patch_only import PRIMARY as BASELINE, PROTOCOL as BASE_PROTOCOL, readout
from .bounded_physical_coupling import _BoundedBranch, geometry_windows, resize_geometry
from .calibrated_competitive_alias import profiled_logits
from .device_probability_accumulator import DeviceProbabilityAccumulator
from .gear_ov import _crop_at
from .inference import hann_blend_window
from .matched_contribution_alias import contribution_margins, stencil_margins
from .matched_readout_controls import qkv
from .native_alias_noise import hard_pair_observation, signed_potential
from .native_query_alias import native_risk, query_relation
from .rival_alias_count import canonical_indices
from .stratified_soft_alias import WideCrop, crop_stencil
from .tcpr import _finish_attention_block, _finish_head


IMPLEMENTATION = 'geometry-bounded896-crop-head-rival-soft-v1-20261005'
PRIMARY = 'CropHead_RivalSoft'
CONTROLS = ('CropHead_RivalHard', 'CropHead_ClassMean', 'CropHead_AliasShuffle',
            'CropHead_ObservationMean')
METHODS = ('Geometry', 'NoAdmission_Exact', BASELINE, PRIMARY, *CONTROLS)
PROTOCOL = {**BASE_PROTOCOL, 'alias_admission': 'rival-specific cropped-native-head soft weights',
    'witness': 'four disjoint16x16 token subsets of existing32x32 raw backbone grid; one batched frozen head call',
    'witness_attention': 'native QK, unchanged prefix inputs/residual/MLP; invalid patch keys masked; no VIP proxy',
    'risk': 'unchanged native_risk; wide advantage contradicted by crop-native query AND G-supported crop-native margin',
    'weight': 'max(1-risk,1e-6); canonical weight1; normalized weighted LME outside exponent',
    'writer': 'original pair least-squares potential; unchanged H; baseline + H @ potential',
    'additional_rgb_encodings': 0, 'maximum_cropped_head_tokens': 4 * (256 + 5),
    'limits': 'backbone tokens and prefix retain global context; not independent fine RGB observations or calibrated truth',
    'controls': CONTROLS, 'fitted_parameters': 0}


def quadrant_indices(height, width, device):
    if height != 32 or width != 32:
        raise ValueError('Require the frozen32x32 Geometry grid.')
    grid = torch.arange(height * width, device=device).reshape(height, width)
    return torch.stack([grid[t:t+16, l:l+16].flatten() for t in (0, 16) for l in (0, 16)])


@torch.inference_mode()
def crop_head_features(head, prepared, valid):
    ids = quadrant_indices(prepared.grid_height, prepared.grid_width, valid.device)
    tokens, prefix = prepared.backbone_tokens, prepared.prefix_tokens
    if tokens.shape[0] != 1 or valid.shape != (1024,) or valid.dtype != torch.bool:
        raise ValueError('One prepared grid and matching valid patches required.')
    patches = tokens[0, prefix:][ids]
    current = torch.cat((tokens[:, :prefix].expand(4, -1, -1), patches), 1)
    visible = torch.cat((torch.ones((4, prefix), dtype=torch.bool, device=valid.device), valid[ids]), 1)
    for block in head.blocks[:prepared.block_index+1]:
        query, key, value = qkv(block, current)
        logits = (query.float() @ key.float().transpose(-1, -2)) * block.attn.scale
        weights = logits.masked_fill(~visible[:, None, None], -torch.inf).softmax(-1).to(value.dtype)
        current = _finish_attention_block(block, current, weights @ value)
    projected = _finish_head(head, current, prepared.block_index)
    features = F.normalize(projected[:, prefix:].float(), dim=-1)
    return features.masked_fill(~valid[ids, None], 0.), ids


def supported_margins(profiled, relation, valid):
    weights, known = query_relation(relation, valid)
    aliases = profiled.flatten(1)
    classes = profiled.logsumexp(-1) - math.log(profiled.shape[-1])
    # Linearity avoids applying the dense relation to an alias-by-rival tensor.
    support = (weights @ aliases)[..., None] - (weights @ classes)[:, None]
    return contribution_margins(profiled), support, classes, known


def weight_controls(risk, canonical, classes, count, *, seed=20261005):
    grouped = risk.reshape(len(risk), classes, count, classes)
    soft = (1. - risk).clamp_min(1e-6)
    shared = grouped.clone()
    shuffled = grouped.clone()
    generator = torch.Generator().manual_seed(seed)
    for c in range(classes):
        ids = torch.arange(count, device=risk.device)
        ids = ids[~torch.isin(ids + c * count, canonical)]
        shared[:, c, ids] = grouped[:, c, ids].mean(1, keepdim=True)
        permutation = torch.randperm(len(ids), generator=generator).to(risk.device)
        shuffled[:, c, ids] = grouped[:, c, ids[permutation]]
    return {PRIMARY: soft,
        'CropHead_ClassMean': (1. - shared.flatten(1, 2)).clamp_min(1e-6),
        'CropHead_AliasShuffle': (1. - shuffled.flatten(1, 2)).clamp_min(1e-6)}


def soft_pair_delta(crops, count, coordinates, image_size, members, weights, valid, chunk=128):
    if (weights.shape != (len(valid), members.numel(), len(members))
            or not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any())
            or bool((weights > 1).any())):
        raise ValueError('Positive bounded query/alias/rival weights required.')
    output = torch.zeros((len(valid), len(members), len(members)), device=weights.device, dtype=torch.float64)
    sources = [(crop, profiled_logits(crop, members).double().softmax(-1)) for crop in crops]
    for start in range(0, len(valid), chunk):
        sl = slice(start, start+chunk)
        grouped = weights[sl].reshape(-1, len(members), members.shape[1], len(members)).double()
        mass = grouped.mean(2)
        unchanged = (grouped == 1).all(2)
        for crop, responsibilities in sources:
            indices, coefficients = crop_stencil(crop, count, coordinates[sl], image_size)
            retained = torch.einsum('qsck,qckd->qscd', responsibilities[indices], grouped)
            delta = (retained.log() - mass[:, None].log()).masked_fill(unchanged[:, None], 0.)
            output[sl] += (delta * coefficients.double()[..., None, None]).sum(1)
    output.masked_fill_(~valid[:, None, None], 0.)
    output.diagonal(dim1=-2, dim2=-1).zero_()
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError('Nonfinite weighted alias action.')
    return output


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS, cache=None):
    from benchmark_grouped_class_readout import CheckedGroupedCache
    from eval_geometry_vip_reliability import sample_broad
    from eval_matched_contribution_alias import crop_from_features
    from eval_rival_fine_full import prepare_wide, tile_coordinates
    from eval_sparse_alias_reuse import forbid_fine

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared crop-head alias methods required.')
    cache = CheckedGroupedCache() if cache is None else cache
    cache.verify = False
    resized = resize_geometry(image)
    height, width = resized.shape[-2:]
    windows = geometry_windows(height, width)
    local_branch = _BoundedBranch(geometry, 512, 4, 'prepare_image')
    wide_branch = _BoundedBranch(vip, 336, 4, 'crop_patch_features')
    wide, crops, _, count = prepare_wide(image, wide_branch, {'clean': (banks, queries)})
    texts = {p: F.normalize(bank.features.float(), dim=-1) for p, bank in banks.items()}
    extra = bool(set(methods) - {'Geometry', 'NoAdmission_Exact', BASELINE})
    totals = {p: dict(tiles=0, risk_mean=0., active_fraction=0., weight_mean=0., potential_mean=0.) for p in banks}
    blend = torch.from_numpy(hann_blend_window(512)).to(geometry.device)
    with forbid_fine(), ExitStack() as stack:
        accumulators = {(p, m): stack.enter_context(DeviceProbabilityAccumulator(
            bank.class_count, height, width, geometry.device)) for p, bank in banks.items() for m in methods}
        for top, left in windows:
            prepared = local_branch.prepare_image(_crop_at(resized, top, left, 512).to(geometry.device))
            coordinates = tile_coordinates(top, left, geometry.device)
            valid = (coordinates[:, 0] < height) & (coordinates[:, 1] < width)
            operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
            with geometry.backbone._autocast():
                features = readout(geometry.backbone.model.visual_model.head, prepared, 2.)[0]
                observed, ids = crop_head_features(geometry.backbone.model.visual_model.head, prepared, valid) if extra else (None, None)
            for p, bank in banks.items():
                raw = cache.geometry((prepared.geometry_projected.float() @ texts[p].T)[0], bank.parent_indices, bank.class_count)
                local = cache.geometry((features.float() @ texts[p].T)[0], bank.parent_indices, bank.class_count) / .07
                broad = sample_broad(wide['clean', p], top, left, height, width).reshape_as(local)
                baseline = local.double() + operator.double() @ (broad.double() - local.double())
                original = (raw / .07).double()
                values = {'Geometry': raw, 'NoAdmission_Exact': original + operator.double() @ (broad.double() - original), BASELINE: baseline}
                row = dict(risk_mean=0., active_fraction=0., weight_mean=1., potential_mean=0.)
                if extra:
                    query = queries[p]
                    members = torch.stack([(query.parents == c).nonzero().flatten() for c in range(bank.class_count)])
                    if not torch.equal(members.flatten(), torch.arange(members.numel(), device=members.device)):
                        raise ValueError('Require original class-contiguous alias order.')
                    canonical = canonical_indices(query.class_names, query.aliases, query.parents)
                    native = local.new_zeros((len(valid), *members.shape))
                    for quadrant in range(4):
                        use = valid[ids[quadrant]]
                        if bool(use.any()):
                            blank = WideCrop(features.new_empty(1, 1), features.new_empty(1), 0, 0, 256, 256, 16, 256)
                            source = crop_from_features(observed[quadrant, use][None], query, blank)
                            native[ids[quadrant, use]] = profiled_logits(source, members)
                    margin, supported, native_class, known = supported_margins(native, prepared.geometry_patch_conditional[0], valid)
                    full = torch.zeros_like(margin)
                    for crop in crops['clean', p]:
                        indices, coefficients = crop_stencil(crop, count, coordinates, (height, width))
                        full += stencil_margins(profiled_logits(crop, members), indices, coefficients)
                    risk = native_risk(full, margin, supported, query.parents, canonical, known)
                    weights = weight_controls(risk, canonical, *members.shape)
                    for method in methods:
                        if method in weights:
                            directed = soft_pair_delta(crops['clean', p], count, coordinates, (height, width), members, weights[method], valid)
                        elif method == 'CropHead_RivalHard':
                            directed = hard_pair_observation(crops['clean', p], count, coordinates, (height, width), members, risk, valid)
                        elif method == 'CropHead_ObservationMean':
                            observation = native_class + math.log(members.shape[1])
                            values[method] = local.double() + operator.double() @ ((broad.double() + observation.double()) * .5 - local.double())
                            continue
                        else:
                            continue
                        potential, _ = signed_potential(directed, valid)
                        values[method] = baseline + operator.double() @ potential
                        if method == PRIMARY:
                            row['potential_mean'] = float(potential[valid].abs().mean())
                    if bool((risk[:, canonical] != 0).any()):
                        raise RuntimeError('Canonical alias protection changed.')
                    row.update(risk_mean=float(risk[valid].mean()), active_fraction=float((risk[valid] > 0).float().mean()),
                               weight_mean=float(weights[PRIMARY][valid].mean()))
                for method in methods:
                    if not bool(torch.isfinite(values[method]).all()):
                        raise RuntimeError('Nonfinite crop-head coupled scores.')
                    dense = F.interpolate(values[method].T.reshape(1, bank.class_count, 32, 32), (512, 512),
                        mode='bilinear', align_corners=False)[0]
                    if method == 'Geometry':
                        dense = dense / .07
                    ah, aw = min(512, height-top), min(512, width-left)
                    accumulators[p, method].add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], left, top)
                totals[p]['tiles'] += 1
                for key, value in row.items():
                    totals[p][key] += value
        predictions = {p: {m: accumulators[p, m].finalize_resized(tuple(image.shape[-2:])) for m in methods} for p in banks}
    for row in totals.values():
        for key in ('risk_mean', 'active_fraction', 'weight_mean', 'potential_mean'):
            row[key] /= row['tiles']
        row.update(geometry_encodings=local_branch.calls, wide_encodings=wide_branch.calls,
            fine_forwards=0, additional_visual_forwards=0, native_resolution_encodings=0,
            cropped_head_groups=4 * row['tiles'] if extra else 0, cropped_head_batch_calls=row['tiles'] if extra else 0)
    return predictions, totals
