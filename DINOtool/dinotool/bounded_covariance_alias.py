"""Geometry-supported response contrast, without pseudo-labelled cohorts."""
from contextlib import ExitStack

import torch
import torch.nn.functional as F

from .alias_action_capacity import reconstruction_operator
from .bounded_canonical_pair_alias import sample_raw, weight_controls
from .bounded_patch_only import PRIMARY as BASELINE, PROTOCOL as BASE_PROTOCOL, readout
from .bounded_physical_coupling import _BoundedBranch, geometry_windows, resize_geometry
from .calibrated_competitive_alias import profiled_logits
from .device_probability_accumulator import DeviceProbabilityAccumulator
from .gear_ov import _crop_at
from .inference import hann_blend_window
from .rival_alias_count import canonical_indices
from .stratified_soft_alias import crop_stencil


IMPLEMENTATION = 'geometry-bounded896-conditional-covariance-fixedmass-v1-20261006'
PRIMARY = 'GeometryCov_RivalSoft'
CLASS_MEAN = 'GeometryCov_ClassMean'
SHUFFLE = 'GeometryCov_AliasShuffle'
GLOBAL = 'GeometryCov_GlobalSupport'
POSITION_NULL = 'GeometryCov_ResponseShuffle'
TEXT_ONLY = 'GeometryCov_TextOnly'
NORMALIZED = 'GeometryCov_NormalizedPool'
METHODS = ('Geometry', 'NoAdmission_Exact', BASELINE, PRIMARY, CLASS_MEAN, SHUFFLE,
           GLOBAL, POSITION_NULL, TEXT_ONLY, NORMALIZED)
CAPACITY = 32
PROTOCOL = {**BASE_PROTOCOL,
    'alias_admission': 'Geometry-supported conditional response correlation; fixed-slot soft attenuation',
    'source': 'wide raw alias variation versus local canonical pair-margin variation; no pseudo-labelled cohorts',
    'support_capacity': CAPACITY,
    'support': 'top32 positive original-G neighbours; self/padding excluded; normalized support only',
    'weight': '(1+signed weighted Pearson correlation)/2; canonical1; zero-variance fallback1; floor1e-6',
    'aggregation': 'outside-exponent weights; original K slots and salience kept; no survivor renormalization',
    'scope': 'unchanged baseline top2 competitors; no permanent word deletion',
    'writer': 'antisymmetric alias potential through original H',
    'fine_forwards': 0, 'additional_visual_forwards': 0, 'additional_semantic_heads': 0,
    'fitted_parameters': 0,
    'controls': (CLASS_MEAN, SHUFFLE, GLOBAL, POSITION_NULL, TEXT_ONLY, NORMALIZED),
    'limits': 'canonical variation and correlation are not class truth; constant coherent errors can survive',
    'precedents': 'weighted correlation and fixed-mass attenuation have precedents; claim no new statistical theorem'}


def neighbourhood(relation, valid, *, global_support=False):
    n = len(valid)
    if relation.shape != (n, n) or valid.dtype != torch.bool:
        raise ValueError('Matching Geometry matrix and valid queries required.')
    k = min(CAPACITY, n)
    if global_support:
        ids = valid.nonzero().flatten()
        if len(ids):
            chosen = ids[torch.linspace(0, len(ids)-1, min(k, len(ids)), device=valid.device).round().long()]
            indices = chosen[None].expand(n, -1)
            mass = torch.ones_like(indices, dtype=torch.float64)
        else:
            indices = torch.zeros((n, 1), device=valid.device, dtype=torch.long)
            mass = torch.zeros_like(indices, dtype=torch.float64)
    else:
        weights = relation.double().masked_fill(~valid[None], 0.).clone()
        weights.fill_diagonal_(0.)
        mass, indices = weights.topk(k, -1, sorted=False)
    mass = mass.masked_fill((indices == torch.arange(n, device=valid.device)[:, None])
                           | ~valid[indices] | ~valid[:, None], 0.)
    return indices, mass/mass.sum(-1, keepdim=True).clamp_min(1e-300)


def covariance_weights(evidence, canonical_scores, pairs, canonical, neighbours, valid):
    if (evidence.ndim != 3 or canonical_scores.shape != evidence.shape[:2]
            or pairs.shape != (len(valid), 2) or len(evidence) != len(valid)):
        raise ValueError('Matching raw alias fields, canonical fields and class pairs required.')
    ids, mass = neighbours
    own = evidence[ids[:, :, None], pairs[:, None]].double()
    reference = canonical_scores[ids[:, :, None], pairs[:, None]].double()
    direction = reference[..., 0]-reference[..., 1]
    direction = direction-(direction*mass).sum(-1, keepdim=True)
    centered = own-(own*mass[..., None, None]).sum(1, keepdim=True)
    covariance = (centered*direction[..., None, None]*mass[..., None, None]).sum(1)
    covariance[:, 1] *= -1
    variance = (centered.square()*mass[..., None, None]).sum(1)
    ref_variance = (direction.square()*mass).sum(-1)
    known = (variance > 1e-12) & (ref_variance[:, None, None] > 1e-12)
    known &= valid[:, None, None] & (pairs[:, 0] != pairs[:, 1])[:, None, None]
    correlation = covariance/(variance*ref_variance[:, None, None]).sqrt().clamp_min(1e-300)
    weights = torch.where(known, (1+correlation.clamp(-1, 1))*.5, 1.).clamp_min(1e-6)
    weights.scatter_(-1, canonical[pairs][..., None], 1.)
    return weights, known


def text_weights(alias_features, canonical_features, pairs, canonical, valid):
    values = F.normalize(alias_features.double(), dim=-1)
    anchors = F.normalize(canonical_features.double(), dim=-1)
    direction = anchors[pairs[:, 0]]-anchors[pairs[:, 1]]
    norm = direction.norm(dim=-1)
    correlation = (values[pairs]*direction[:, None, None]).sum(-1)/norm[:, None, None].clamp_min(1e-300)
    correlation[:, 1] *= -1
    active = valid & (pairs[:, 0] != pairs[:, 1]) & (norm > 1e-12)
    weights = torch.where(active[:, None, None], (1+correlation.clamp(-1, 1))*.5, 1.).clamp_min(1e-6)
    weights.scatter_(-1, canonical[pairs][..., None], 1.)
    return weights


def fixed_slot_potential(crops, count, coordinates, image_size, members, pairs, weights, valid, *, normalized=False):
    delta = torch.zeros((len(valid), 2), device=weights.device, dtype=torch.float64)
    rows = torch.arange(len(valid), device=weights.device)[:, None, None]
    unchanged = (weights == 1).all(-1)
    for crop in crops:
        ids, coefficients = crop_stencil(crop, count, coordinates, image_size)
        probabilities = profiled_logits(crop, members).double().softmax(-1)[ids]
        selected = probabilities[rows, torch.arange(ids.shape[1], device=ids.device)[None, :, None], pairs[:, None]]
        change = (selected*weights[:, None]).sum(-1).clamp_min(1e-300).log().clamp_max(0.)
        if normalized:
            change -= weights.mean(-1)[:, None].log()
        change.masked_fill_(unchanged[:, None], 0.)
        delta += (change*coefficients.double()[..., None]).sum(1)
    contrast = .5*(delta[:, 0]-delta[:, 1])
    contrast.masked_fill_(~valid | (pairs[:, 0] == pairs[:, 1]), 0.)
    result = torch.zeros((len(valid), len(members)), device=weights.device, dtype=torch.float64)
    result.scatter_add_(1, pairs, torch.stack((contrast, -contrast), -1))
    return result


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS, cache=None):
    from benchmark_grouped_class_readout import CheckedGroupedCache
    from eval_geometry_vip_reliability import sample_broad
    from eval_rival_fine_full import prepare_wide, tile_coordinates
    from eval_sparse_alias_reuse import forbid_fine

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared conditional covariance methods required.')
    cache = CheckedGroupedCache() if cache is None else cache
    cache.verify = False
    resized = resize_geometry(image)
    height, width = resized.shape[-2:]
    local_branch = _BoundedBranch(geometry, 512, 4, 'prepare_image')
    wide_branch = _BoundedBranch(vip, 336, 4, 'crop_patch_features')
    wide, crops, _, count = prepare_wide(image, wide_branch, {'clean': (banks, queries)})
    texts = {p: F.normalize(bank.features.float(), dim=-1) for p, bank in banks.items()}
    extra = bool(set(methods)-set(METHODS[:3]))
    totals = {p: dict(tiles=0, known_alias_fraction=0., active_weight_fraction=0., potential_mean=0.) for p in banks}
    blend = torch.from_numpy(hann_blend_window(512)).to(geometry.device)
    with forbid_fine(), ExitStack() as stack:
        accumulators = {(p, m): stack.enter_context(DeviceProbabilityAccumulator(
            bank.class_count, height, width, geometry.device)) for p, bank in banks.items() for m in methods}
        for top, left in geometry_windows(height, width):
            prepared = local_branch.prepare_image(_crop_at(resized, top, left, 512).to(geometry.device))
            coordinates = tile_coordinates(top, left, geometry.device)
            valid = (coordinates[:, 0] < height) & (coordinates[:, 1] < width)
            relation = prepared.geometry_patch_conditional[0]
            operator, _ = reconstruction_operator(relation, valid)
            with geometry.backbone._autocast():
                features = readout(geometry.backbone.model.visual_model.head, prepared, 2.)[0]
            neighbours = neighbourhood(relation, valid) if extra else None
            for p, bank in banks.items():
                raw = cache.geometry((prepared.geometry_projected.float()@texts[p].T)[0], bank.parent_indices, bank.class_count)
                local_aliases = (features.float()@texts[p].T)[0]
                local = cache.geometry(local_aliases, bank.parent_indices, bank.class_count)/.07
                broad = sample_broad(wide['clean', p], top, left, height, width).reshape_as(local)
                original = (raw/.07).double()
                baseline = local.double()+operator@(broad.double()-local.double())
                values = {'Geometry': raw, 'NoAdmission_Exact': original+operator@(broad.double()-original), BASELINE: baseline}
                if extra:
                    query = queries[p]
                    members = torch.stack([(query.parents == c).nonzero().flatten() for c in range(bank.class_count)])
                    canonical_ids = canonical_indices(query.class_names, query.aliases, query.parents)
                    canonical = canonical_ids-members[:, 0]
                    evidence = sample_raw(crops['clean', p], count, coordinates, (height, width), members)
                    canonical_scores = local_aliases[:, canonical_ids]
                    pairs = baseline.topk(2, -1).indices
                    weights, known = covariance_weights(evidence, canonical_scores, pairs, canonical, neighbours, valid)
                    allocations = {PRIMARY: weights, NORMALIZED: weights}
                    if set(methods) & {CLASS_MEAN, SHUFFLE}:
                        allocations[CLASS_MEAN], allocations[SHUFFLE] = weight_controls(weights, pairs, canonical)
                    if GLOBAL in methods:
                        allocations[GLOBAL] = covariance_weights(evidence, canonical_scores, pairs, canonical,
                            neighbourhood(relation, valid, global_support=True), valid)[0]
                    if POSITION_NULL in methods:
                        ids = valid.nonzero().flatten()
                        generator = torch.Generator().manual_seed(20261006)
                        permutation = ids[torch.randperm(len(ids), generator=generator).to(ids.device)]
                        permuted = canonical_scores.clone()
                        permuted[ids] = canonical_scores[permutation]
                        allocations[POSITION_NULL] = covariance_weights(evidence, permuted, pairs, canonical, neighbours, valid)[0]
                    if TEXT_ONLY in methods:
                        allocations[TEXT_ONLY] = text_weights(query.features.float().mean(1)[members],
                            texts[p][canonical_ids], pairs, canonical, valid)
                    for method in methods:
                        if method in allocations:
                            potential = fixed_slot_potential(crops['clean', p], count, coordinates, (height, width),
                                members, pairs, allocations[method], valid, normalized=method == NORMALIZED)
                            values[method] = baseline+operator@potential
                            if method == PRIMARY:
                                totals[p]['potential_mean'] += float(potential[valid].abs().mean())
                    totals[p]['known_alias_fraction'] += float(known[valid].float().mean())
                    totals[p]['active_weight_fraction'] += float((weights[valid] != 1).float().mean())
                for method in methods:
                    value = values[method]
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeError('Nonfinite conditional covariance readout.')
                    dense = F.interpolate(value.T.reshape(1, bank.class_count, 32, 32), (512, 512), mode='bilinear', align_corners=False)[0]
                    if method == 'Geometry':
                        dense = dense/.07
                    ah, aw = min(512, height-top), min(512, width-left)
                    accumulators[p, method].add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], left, top)
                totals[p]['tiles'] += 1
        predictions = {p: {m: accumulators[p, m].finalize_resized(tuple(image.shape[-2:])) for m in methods} for p in banks}
    for row in totals.values():
        for key in ('known_alias_fraction', 'active_weight_fraction', 'potential_mean'):
            row[key] /= row['tiles']
        row.update(geometry_encodings=local_branch.calls, wide_encodings=wide_branch.calls,
                   fine_forwards=0, additional_visual_forwards=0, additional_semantic_heads=0, maximum_support_neighbours=CAPACITY)
    return predictions, totals
