"""Two image-wide detail observations and sparse rival-conditioned alias use."""
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
from .native_query_alias import query_relation
from .rival_alias_count import canonical_indices
from .stratified_soft_alias import WideCrop, crop_stencil


IMPLEMENTATION = 'geometry-bounded896-global-detail2-sparse-rival-soft-v1-20261005'
PRIMARY = 'Detail2_RivalSoft'
CLASS_MEAN = 'Detail2_ClassMean'
SHUFFLE = 'Detail2_AliasShuffle'
OBSERVATION_MEAN = 'Detail2_ObservationMean'
HARD = 'Detail2_RivalHard'
METHODS = ('Geometry', 'NoAdmission_Exact', BASELINE, PRIMARY, CLASS_MEAN,
           SHUFFLE, OBSERVATION_MEAN, HARD)
PROTOCOL = {**BASE_PROTOCOL, 'alias_admission': 'soft alias use for the frozen baseline top2 only',
    'detail_source': 'original RGB; two disjoint256-bounded-pixel regions sampled directly to512',
    'maximum_detail_encodings_per_image': 2, 'fine_forwards': 2,
    'detail_reader': 'frozen native DINO.text head, not VIP proxy',
    'acquisition': 'global local/wide JS disagreement times original H column energy; no labels',
    'weight': 'unchanged positive-wide/negative-detail/negative-G-support contradiction; canonical1',
    'aggregation': 'normalized weighted LME of stencil-sampled profiled wide alias evidence',
    'writer': 'two-class antisymmetric potential through original H; untouched donors outside detail',
    'extra_backbone_encodings': 'zero to two per whole image, never per local window',
    'fitted_parameters': 0, 'controls': (CLASS_MEAN, SHUFFLE, OBSERVATION_MEAN, HARD)}


def detail_positions(height, width):
    if min(height, width) < 1 or max(height, width) != 896:
        raise ValueError('Require the bounded896 image.')
    return tuple((t, l, min(256, height-t), min(256, width-l))
                 for t in range(0, height, 256) for l in range(0, width, 256))


def select_details(records, height, width):
    positions = detail_positions(height, width)
    scores = torch.zeros(len(positions), device=records[0]['valid'].device, dtype=torch.float64)
    for record in records:
        energy = record['operator'].square().sum(0)
        coordinates, valid = record['coordinates'], record['valid']
        for local, broad in record['fields'].values():
            p, q = local.double().softmax(-1), broad.double().softmax(-1)
            mean = (p+q)*.5
            js = .5*((p*(p.clamp_min(1e-300).log()-mean.clamp_min(1e-300).log())).sum(-1)
                     +(q*(q.clamp_min(1e-300).log()-mean.clamp_min(1e-300).log())).sum(-1))
            priority = js.clamp_min(0)*energy*(p.argmax(-1) != q.argmax(-1))*valid
            for i, (top, left, h, w) in enumerate(positions):
                inside = ((coordinates[:, 0] >= top) & (coordinates[:, 0] < top+h)
                          & (coordinates[:, 1] >= left) & (coordinates[:, 1] < left+w))
                scores[i] += priority[inside].sum()/len(record['fields'])
    order = scores.argsort(descending=True, stable=True).tolist()
    return tuple(positions[i] for i in order[:2] if float(scores[i]) > 0)


def detail_rgb(image, position, bounded_size):
    top, left, height, width = position
    bh, bw = bounded_size
    if image.ndim != 3 or image.shape[0] != 3 or not (0 < height <= 256 and 0 < width <= 256):
        raise ValueError('One RGB image and a bounded detail region required.')
    if not (0 <= top < bh and 0 <= left < bw and top+height <= bh and left+width <= bw):
        raise ValueError('Detail region outside the bounded image.')
    axis = (torch.arange(512, device=image.device, dtype=torch.float32)+.5)*.5
    yy, xx = torch.meshgrid(axis+top, axis+left, indexing='ij')
    grid = torch.stack((2*xx/bw-1, 2*yy/bh-1), -1)
    rgb = F.grid_sample(image[None], grid[None], mode='bilinear', padding_mode='border', align_corners=False)
    visible = (yy < top+height) & (xx < left+width)
    return rgb.masked_fill(~visible[None, None], 0.)


def sample_evidence(crops, count, coordinates, image_size, members):
    field = coordinates.new_zeros((len(coordinates), *members.shape))
    coverage = coordinates.new_zeros(len(coordinates))
    for crop in crops:
        ids, coefficients = crop_stencil(crop, count, coordinates, image_size)
        field += (profiled_logits(crop, members)[ids]*coefficients[..., None, None]).sum(1)
        coverage += coefficients.sum(-1)
    return field, coverage


def sparse_weights(wide, fine, supported, pairs, canonical, known):
    if wide.shape != fine.shape or fine.shape != supported.shape or wide.ndim != 3:
        raise ValueError('Matching query/class/alias evidence required.')
    rows = torch.arange(len(wide), device=wide.device)[:, None]
    def margins(values):
        rival = (values.logsumexp(-1)-math.log(values.shape[-1]))[rows, pairs.flip(-1)]
        return values[rows, pairs]-rival[..., None]
    broad_margin, fine_margin, support_margin = map(margins, (wide, fine, supported))
    negative = torch.minimum((-fine_margin).clamp_min(0), (-support_margin).clamp_min(0))
    eligible = (broad_margin > 0) & (fine_margin < 0) & (support_margin < 0) & known[:, None, None]
    risk = torch.where(eligible, negative/(broad_margin.clamp_min(0)+negative).clamp_min(1e-6), 0.)
    protected = canonical[pairs]
    risk.scatter_(-1, protected[..., None], 0.)
    weights = (1-risk).clamp_min(1e-6)
    shared, shuffled = weights.clone(), weights.clone()
    generator = torch.Generator().manual_seed(20261005)
    for c in range(wide.shape[1]):
        slots = torch.arange(wide.shape[-1], device=wide.device)
        slots = slots[slots != canonical[c]]
        permutation = slots[torch.randperm(len(slots), generator=generator).to(wide.device)]
        use = pairs == c
        selected = weights[use]
        class_weights, null_weights = selected.clone(), selected.clone()
        class_weights[:, slots] = selected[:, slots].mean(-1, keepdim=True)
        null_weights[:, slots] = selected[:, permutation]
        shared[use], shuffled[use] = class_weights, null_weights
    hard = torch.where(risk > 0, torch.zeros_like(weights), torch.ones_like(weights))
    return {PRIMARY: weights, CLASS_MEAN: shared, SHUFFLE: shuffled, HARD: hard}


def pair_potential(wide, pairs, weights, valid):
    rows = torch.arange(len(wide), device=wide.device)[:, None]
    selected = wide[rows, pairs].double()
    total = weights.double().sum(-1)
    delta = (selected+weights.double().log()).logsumexp(-1)-selected.logsumexp(-1)
    delta -= (total/weights.shape[-1]).log()
    delta.masked_fill_((weights == 1).all(-1) | ~valid[:, None], 0.)
    contrast = .5*(delta[:, 0]-delta[:, 1])
    output = torch.zeros(wide.shape[:2], device=wide.device, dtype=torch.float64)
    output.scatter_(1, pairs, torch.stack((contrast, -contrast), -1))
    return output


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS, cache=None):
    from benchmark_grouped_class_readout import CheckedGroupedCache
    from eval_geometry_vip_reliability import sample_broad
    from eval_matched_contribution_alias import crop_from_features
    from eval_rival_fine_full import prepare_wide, tile_coordinates
    from eval_sparse_alias_reuse import forbid_fine

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared global-detail methods required.')
    extra = bool(set(methods)-set(METHODS[:3]))
    cache = CheckedGroupedCache() if cache is None else cache
    cache.verify = False
    resized = resize_geometry(image)
    height, width = resized.shape[-2:]
    local_branch = _BoundedBranch(geometry, 512, 4, 'prepare_image')
    wide_branch = _BoundedBranch(vip, 336, 4, 'crop_patch_features')
    detail_branch = _BoundedBranch(geometry, 512, 2, 'prepare_image')
    wide, crops, _, count = prepare_wide(image, wide_branch, {'clean': (banks, queries)})
    texts = {p: F.normalize(bank.features.float(), dim=-1) for p, bank in banks.items()}
    records = []
    for top, left in geometry_windows(height, width):
        prepared = local_branch.prepare_image(_crop_at(resized, top, left, 512).to(geometry.device))
        coordinates = tile_coordinates(top, left, geometry.device)
        valid = (coordinates[:, 0] < height) & (coordinates[:, 1] < width)
        operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
        with geometry.backbone._autocast():
            features = readout(geometry.backbone.model.visual_model.head, prepared, 2.)[0]
        fields, values = {}, {}
        for p, bank in banks.items():
            raw = cache.geometry((prepared.geometry_projected.float() @ texts[p].T)[0], bank.parent_indices, bank.class_count)
            local = cache.geometry((features.float() @ texts[p].T)[0], bank.parent_indices, bank.class_count)/.07
            broad = sample_broad(wide['clean', p], top, left, height, width).reshape_as(local)
            fields[p] = (local, broad)
            original = (raw/.07).double()
            values[p] = {'Geometry': raw, 'NoAdmission_Exact': original+operator@(broad.double()-original),
                         BASELINE: local.double()+operator@(broad.double()-local.double())}
        records.append(dict(top=top, left=left, prepared=prepared, coordinates=coordinates,
                            valid=valid, operator=operator, fields=fields, values=values))
    positions = select_details(records, height, width) if extra else ()
    fine_count = torch.zeros(height, width, device=geometry.device)
    fine_crops = {p: [] for p in banks}
    with forbid_fine():
        for position in positions:
            top, left, h, w = position
            fine_count[top:top+h, left:left+w] += 1
            observed = detail_branch.prepare_image(detail_rgb(image, position, (height, width)).to(geometry.device))
            for p, query in queries.items():
                blank = WideCrop(image.new_empty(1, 1), image.new_empty(1), top, left, h, w, 32, 256)
                crop = crop_from_features(observed.native_projected, query, blank)
                fine_crops[p].append(WideCrop(crop.alias_logits, crop.salience, top, left, h, w, 32, 256))
    fine_count.clamp_min_(1)
    totals = {p: dict(tiles=0, active_fraction=0., observed_fraction=0.) for p in banks}
    blend = torch.from_numpy(hann_blend_window(512)).to(geometry.device)
    with ExitStack() as stack:
        accumulators = {(p, m): stack.enter_context(DeviceProbabilityAccumulator(
            bank.class_count, height, width, geometry.device)) for p, bank in banks.items() for m in methods}
        for record in records:
            top, left = record['top'], record['left']
            valid, coordinates, operator = record['valid'], record['coordinates'], record['operator']
            for p, bank in banks.items():
                values = record['values'][p]
                known = torch.zeros_like(valid)
                active = 0.
                if extra and positions:
                    query = queries[p]
                    members = torch.stack([(query.parents == c).nonzero().flatten() for c in range(bank.class_count)])
                    ids = canonical_indices(query.class_names, query.aliases, query.parents)
                    canonical = ids-members[:, 0]
                    broad_evidence, _ = sample_evidence(crops['clean', p], count, coordinates, (height, width), members)
                    fine_evidence, coverage = sample_evidence(fine_crops[p], fine_count, coordinates, (height, width), members)
                    known = valid & (coverage > 0)
                    fine_evidence /= coverage.clamp_min(1e-6)[:, None, None]
                    support, supported_known = query_relation(record['prepared'].geometry_patch_conditional[0], known)
                    supported = (support@fine_evidence.flatten(1)).reshape_as(fine_evidence)
                    pairs = values[BASELINE].topk(2, -1).indices
                    allocations = sparse_weights(broad_evidence, fine_evidence, supported, pairs, canonical, known & supported_known)
                    for method in methods:
                        if method in allocations:
                            potential = pair_potential(broad_evidence, pairs, allocations[method], valid)
                            values[method] = values[BASELINE]+operator@potential
                        elif method == OBSERVATION_MEAN:
                            fine_class = fine_evidence.logsumexp(-1)
                            innovation = .5*(fine_class-record['fields'][p][1]).double()
                            innovation.masked_fill_(~known[:, None], 0.)
                            values[method] = values[BASELINE]+operator@innovation
                    active = float((allocations[PRIMARY] != 1).float().mean())
                for method in methods:
                    value = values.get(method, values[BASELINE])
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeError('Nonfinite sparse detail readout.')
                    dense = F.interpolate(value.T.reshape(1, bank.class_count, 32, 32), (512, 512),
                                          mode='bilinear', align_corners=False)[0]
                    if method == 'Geometry':
                        dense = dense/.07
                    ah, aw = min(512, height-top), min(512, width-left)
                    accumulators[p, method].add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], left, top)
                totals[p]['tiles'] += 1
                totals[p]['active_fraction'] += active
                totals[p]['observed_fraction'] += float(known[valid].float().mean())
        predictions = {p: {m: accumulators[p, m].finalize_resized(tuple(image.shape[-2:])) for m in methods} for p in banks}
    for row in totals.values():
        row['active_fraction'] /= row['tiles']
        row['observed_fraction'] /= row['tiles']
        row.update(geometry_encodings=local_branch.calls, wide_encodings=wide_branch.calls,
                   fine_forwards=detail_branch.calls, additional_visual_forwards=detail_branch.calls,
                   selected_detail_regions=len(positions), native_resolution_encodings=0)
    if detail_branch.calls != len(positions) or detail_branch.calls > 2:
        raise RuntimeError('Whole-image detail budget differs.')
    return predictions, totals
