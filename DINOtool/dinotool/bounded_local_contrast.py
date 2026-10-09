"""Keep broad calibration; compensate contrast only inside the local head."""
from contextlib import ExitStack

import torch
import torch.nn.functional as F

from .alias_action_capacity import reconstruction_operator
from .bounded_physical_coupling import VIEW_PROTOCOL, _BoundedBranch, geometry_windows, resize_geometry
from .device_probability_accumulator import DeviceProbabilityAccumulator
from .gear_ov import _crop_at
from .geometry_conservative_contrast import read_head, settings
from .inference import hann_blend_window


IMPLEMENTATION = 'geometry-bounded896-conservative-local-coupled-v1-20261005'
PRIMARY = 'Geometry_ConservativeCoupled'
HEADS = {PRIMARY: 'Geometry_ConservativeContrast',
         'Geometry_DoubleRowCoupled': 'Geometry_DoubleRow',
         'Geometry_GainOnlyCoupled': 'Geometry_ContrastGainOnly',
         'SCLIP_Coupled': 'SCLIP_Two'}
METHODS = ('Geometry', 'NoAdmission_Exact', *HEADS)
PROTOCOL = {**VIEW_PROTOCOL, 'reconstruction': 'L_head + H(B-L_head); unchanged original G/H and all broad class offsets',
    'alias_admission': 'none; unchanged20 pools', 'local_readout': settings(),
    'extra_backbone_encodings': 0, 'fitted_parameters': 0,
    'controls': 'original Geometry, double whole read, input-derived whole-read gain, published SCLIP_Two'}


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS, cache=None,
                  head_names=HEADS, readout=read_head, primary=PRIMARY, score_controls=None):
    from benchmark_grouped_class_readout import CheckedGroupedCache
    from eval_geometry_vip_reliability import sample_broad
    from eval_sparse_alias_reuse import forbid_fine
    import eval_rival_fine_full as reference

    score_controls = {} if score_controls is None else score_controls
    allowed = ('Geometry', 'NoAdmission_Exact', *head_names, *score_controls)
    if not methods or not set(methods).issubset(allowed):
        raise ValueError('Declared local-contrast methods required.')
    head_methods = tuple(dict.fromkeys(m if m in head_names else score_controls[m][0]
        for m in methods if m in head_names or m in score_controls))
    if not set(head_methods).issubset(head_names):
        raise ValueError('Score controls require a declared local head.')
    cache = CheckedGroupedCache() if cache is None else cache
    cache.verify = False
    resized = resize_geometry(image)
    height, width = resized.shape[-2:]
    windows = geometry_windows(height, width)
    local_branch = _BoundedBranch(geometry, 512, 4, 'prepare_image')
    wide_branch = _BoundedBranch(vip, 336, 4, 'crop_patch_features')
    wide, _, _, _ = reference.prepare_wide(image, wide_branch, {'clean': (banks, queries)})
    texts = {p: F.normalize(bank.features.float(), dim=-1) for p, bank in banks.items()}
    totals = {p: dict(tiles=0, primary_descriptor_displacement=0.) for p in banks}
    blend = torch.from_numpy(hann_blend_window(512)).to(geometry.device)
    with forbid_fine(), ExitStack() as stack:
        accumulators = {(p, m): stack.enter_context(DeviceProbabilityAccumulator(
            bank.class_count, height, width, geometry.device)) for p, bank in banks.items() for m in methods}
        for top, left in windows:
            prepared = local_branch.prepare_image(_crop_at(resized, top, left, 512).to(geometry.device))
            coordinates = reference.tile_coordinates(top, left, geometry.device)
            valid = (coordinates[:, 0] < height) & (coordinates[:, 1] < width)
            operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
            features = {'Geometry': prepared.geometry_projected}
            with geometry.backbone._autocast():
                for method in head_methods:
                    features[method] = readout(geometry.backbone.model.visual_model.head,
                                              prepared, head_names[method])[0]
            if any(not bool(torch.isfinite(value).all()) for value in features.values()):
                raise RuntimeError('Nonfinite local descriptors.')
            displacement = float((features[primary] - prepared.geometry_projected).abs().mean()) if primary in features else 0.
            ah, aw = min(512, height - top), min(512, width - left)
            for p, bank in banks.items():
                raw = {m: cache.geometry((f.float() @ texts[p].T)[0], bank.parent_indices, bank.class_count)
                       for m, f in features.items()}
                broad = sample_broad(wide['clean', p], top, left, height, width).reshape_as(raw['Geometry'])
                values = {'Geometry': raw['Geometry']}
                for method in methods:
                    if method == 'Geometry':
                        continue
                    if method in score_controls:
                        source, rule = score_controls[method]
                        values[method] = rule((raw[source] / .07).double(), broad.double(), operator.double(), valid)
                        continue
                    local = raw['Geometry' if method == 'NoAdmission_Exact' else method] / .07
                    values[method] = local.double() + operator.double() @ (broad.double() - local.double())
                for method in methods:
                    dense = F.interpolate(values[method].T.reshape(1, bank.class_count, 32, 32), (512, 512),
                                          mode='bilinear', align_corners=False)[0]
                    if method == 'Geometry':
                        dense = dense / .07
                    accumulators[p, method].add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], left, top)
                totals[p]['tiles'] += 1
                totals[p]['primary_descriptor_displacement'] += displacement
        predictions = {p: {m: accumulators[p, m].finalize_resized(tuple(image.shape[-2:])) for m in methods} for p in banks}
    if local_branch.calls != len(windows) or not 1 <= wide_branch.calls <= 4:
        raise RuntimeError('Actual branch call budget differs.')
    for row in totals.values():
        row['primary_descriptor_displacement'] /= row['tiles']
        row.update(geometry_encodings=local_branch.calls, wide_encodings=wide_branch.calls,
                   fine_forwards=0, additional_visual_forwards=0, native_resolution_encodings=0)
    return predictions, totals
