"""Separate spatial broad-view innovation from window-wide class offsets."""
from contextlib import ExitStack

import torch
import torch.nn.functional as F

from .alias_action_capacity import reconstruction_operator
from .bounded_physical_coupling import VIEW_PROTOCOL, _BoundedBranch, geometry_windows, resize_geometry
from .device_probability_accumulator import DeviceProbabilityAccumulator
from .gear_ov import _crop_at
from .inference import hann_blend_window
from .matched_readout_controls import run_head


IMPLEMENTATION = 'geometry-bounded896-zero-dc-coupling-v1-20261005'
PRIMARY = 'Geometry_ContrastCoupled'
METHODS = ('Geometry', 'NoAdmission_Exact', PRIMARY, 'SCLIP_NoAdmission', 'SCLIP_ContrastCoupled')
PROTOCOL = {**VIEW_PROTOCOL, 'reconstruction': 'L + P H P (B-L); P centers valid patches per class',
    'alias_admission': 'none; fixed20 equal-pool aggregation retained',
    'local_control': 'published SCLIP_Two head on same backbone tokens; original Geometry H remains',
    'extra_backbone_encodings': 0, 'parameters_fitted': 0}


def center_valid(values, valid):
    if values.ndim != 2 or valid.shape != values.shape[:1] or valid.dtype != torch.bool:
        raise ValueError('Token-class scores and boolean token validity required.')
    if not bool(valid.any()):
        return torch.zeros_like(values)
    return (values - values[valid].mean(0, keepdim=True)).masked_fill(~valid[:, None], 0.)


def contrast_coupling(local, broad, operator, valid):
    if local.shape != broad.shape or operator.shape != (len(local), len(local)):
        raise ValueError('Matching local/broad scores and Geometry reconstruction required.')
    innovation = center_valid(broad.double() - local.double(), valid)
    correction = center_valid(operator.double() @ innovation, valid)
    return local.double() + correction


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS, cache=None):
    from benchmark_grouped_class_readout import CheckedGroupedCache
    from eval_geometry_vip_reliability import sample_broad
    import eval_rival_fine_full as reference
    from eval_sparse_alias_reuse import forbid_fine

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared contrast-coupling methods required.')
    cache = CheckedGroupedCache() if cache is None else cache
    cache.verify = False
    resized = resize_geometry(image)
    height, width = resized.shape[-2:]
    windows = geometry_windows(height, width)
    local_branch = _BoundedBranch(geometry, 512, 4, 'prepare_image')
    wide_branch = _BoundedBranch(vip, 336, 4, 'crop_patch_features')
    texts = {p: F.normalize(bank.features.float(), dim=-1) for p, bank in banks.items()}
    wide, _, _, _ = reference.prepare_wide(image, wide_branch, {'clean': (banks, queries)})
    blend = torch.from_numpy(hann_blend_window(512)).to(geometry.device)
    totals = {p: {'tiles': 0, 'correction_mean_error': 0., 'removed_offset_rms': 0.} for p in banks}
    scl = any(name.startswith('SCLIP') for name in methods)
    with forbid_fine(), ExitStack() as stack:
        accumulators = {(p, m): stack.enter_context(DeviceProbabilityAccumulator(
            bank.class_count, height, width, geometry.device)) for p, bank in banks.items() for m in methods}
        for top, left in windows:
            prepared = local_branch.prepare_image(_crop_at(resized, top, left, 512).to(geometry.device))
            coordinates = reference.tile_coordinates(top, left, geometry.device)
            valid = (coordinates[:, 0] < height) & (coordinates[:, 1] < width)
            operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
            features = {'Geometry': prepared.geometry_projected}
            if scl:
                with geometry.backbone._autocast():
                    features['SCLIP'] = run_head(geometry.backbone.model.visual_model.head,
                        prepared.backbone_tokens, prepared.backbone_tokens[:, prepared.prefix_tokens:],
                        prepared.geometry_patch_conditional, prepared.prefix_tokens, 'SCLIP_Two', prepared.block_index)[0]
            ah, aw = min(512, height - top), min(512, width - left)
            for p, bank in banks.items():
                raw = {name: cache.geometry((value.float() @ texts[p].T)[0], bank.parent_indices, bank.class_count)
                       for name, value in features.items()}
                local = raw['Geometry'] / .07
                broad = sample_broad(wide['clean', p], top, left, height, width).reshape_as(local)
                values = {'Geometry': raw['Geometry'],
                    'NoAdmission_Exact': local.double() + operator.double() @ (broad.double() - local.double())}
                for name, suffix in (('Geometry', PRIMARY), ('SCLIP', 'SCLIP_ContrastCoupled')):
                    if name in raw:
                        scores = raw[name] / .07
                        if suffix in methods:
                            values[suffix] = contrast_coupling(scores, broad, operator, valid)
                        if name == 'SCLIP' and 'SCLIP_NoAdmission' in methods:
                            values['SCLIP_NoAdmission'] = scores.double() + operator.double() @ (broad.double() - scores.double())
                for method in methods:
                    dense = F.interpolate(values[method].T.reshape(1, bank.class_count, 32, 32), (512, 512),
                                          mode='bilinear', align_corners=False)[0]
                    if method == 'Geometry':
                        dense = dense / .07
                    accumulators[p, method].add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], left, top)
                totals[p]['tiles'] += 1
                totals[p]['removed_offset_rms'] += float((broad.double() - local.double())[valid].mean(0).square().mean().sqrt())
                if PRIMARY in methods:
                    error = float((values[PRIMARY] - local.double())[valid].mean(0).abs().max())
                    totals[p]['correction_mean_error'] = max(totals[p]['correction_mean_error'], error)
        predictions = {p: {m: accumulators[p, m].finalize_resized(tuple(image.shape[-2:])) for m in methods} for p in banks}
    if local_branch.calls != len(windows) or not 1 <= wide_branch.calls <= 4:
        raise RuntimeError('Actual encoding count differs from bounded view protocol.')
    for row in totals.values():
        row.update(geometry_encodings=local_branch.calls, wide_encodings=wide_branch.calls,
                   fine_forwards=0, additional_visual_forwards=0)
        row['removed_offset_rms'] /= row['tiles']
    return predictions, totals
