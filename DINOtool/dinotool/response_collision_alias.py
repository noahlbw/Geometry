"""Conditional alias attenuation from Geometry-supported response collisions."""
from contextlib import ExitStack

import torch
import torch.nn.functional as F

from .alias_action_capacity import reconstruction_operator
from .bounded_canonical_pair_alias import sample_raw, weight_controls
from .bounded_covariance_alias import CAPACITY, fixed_slot_potential, neighbourhood
from .bounded_patch_only import PRIMARY as BASELINE, PROTOCOL as BASE_PROTOCOL, readout
from .bounded_physical_coupling import _BoundedBranch, geometry_windows, resize_geometry
from .device_probability_accumulator import DeviceProbabilityAccumulator
from .gear_ov import _crop_at
from .inference import hann_blend_window
from .rival_alias_count import canonical_indices
from .supported_positive_alias import permuted_operator
from .witness_cap_alias import match_previous


IMPLEMENTATION = 'geometry-bounded896-response-collision-fixed-slot-v1-20261006'
PRIMARY = 'ResponseCollision_Soft'
POOLED = 'ResponseCollision_PooledPattern'
MATCHED_POOLED = 'ResponseCollision_MatchedPooledPattern'
CLASS_MEAN = 'ResponseCollision_ClassMean'
MATCHED_CLASS_MEAN = 'ResponseCollision_MatchedClassMean'
SHUFFLES = tuple('ResponseCollision_AliasShuffle'+str(i) for i in range(3))
MATCHED_SHUFFLES = tuple('ResponseCollision_MatchedAliasShuffle'+str(i) for i in range(3))
GLOBAL = 'ResponseCollision_GlobalSupport'
POSITION_NULL = 'ResponseCollision_PositionNull'
SHUFFLED_WRITE = 'ResponseCollision_MatchedShuffledWrite'
DIRECT = 'ResponseCollision_DirectMatched'
METHODS = ('Geometry', 'NoAdmission_Exact', BASELINE, PRIMARY, POOLED,
    MATCHED_POOLED, CLASS_MEAN, MATCHED_CLASS_MEAN, *SHUFFLES, *MATCHED_SHUFFLES,
    GLOBAL, POSITION_NULL, SHUFFLED_WRITE, DIRECT)
DIAGNOSTICS = ('known_alias_fraction', 'active_weight_fraction', 'potential_mean',
    'collision_risk_mean', 'canonical_weight_max_error', 'matched_norm_relative_error',
    'matched_unmatchable', 'gauge_max_error')
PROTOCOL = {**BASE_PROTOCOL,
    'alias_admission': 'positive squared cross-word response correlation in Geometry supports',
    'source': 'existing wide raw alias fields, not canonical margins or text-span projections',
    'support': 'top32 original-G neighbours; current query and padding excluded',
    'support_capacity': CAPACITY,
    'weight': '1 minus rival-responsibility-weighted positive squared correlation; floor1e-6',
    'rival_responsibility': 'softmax of original raw-logit neighbour means; no new temperature',
    'aggregation': 'original fixed20 slots and salience; no survivor renormalization',
    'scope': 'unchanged bounded baseline top2 competitors; canonical protected; unknown neutral',
    'writer': 'unchanged original H writes antisymmetric fixed-slot pair potential',
    'fine_forwards': 0, 'additional_visual_forwards': 0, 'additional_semantic_heads': 0,
    'fitted_parameters': 0, 'controls': METHODS[4:],
    'precedents': 'correlation, collision penalties and ridge are familiar operations; no priority claim',
    'limits': 'shared response can be valid context; unvarying or correlated wrong evidence can survive'}


def collision_weights(evidence, pairs, canonical, neighbours, valid, *, chunk=64,
                      rival_evidence=None, pooled=False):
    rival_evidence = evidence if rival_evidence is None else rival_evidence
    ids, mass = neighbours
    n, classes, slots = evidence.shape
    if (rival_evidence.shape != evidence.shape or pairs.shape != (n, 2)
            or canonical.shape != (classes,) or valid.shape != (n,) or valid.dtype != torch.bool
            or ids.shape != mass.shape or ids.shape[0] != n or chunk < 1
            or not bool(torch.isfinite(evidence).all() and torch.isfinite(rival_evidence).all())):
        raise ValueError('Matching finite response fields, pairs and Geometry supports required.')
    weights = torch.ones((n, 2, slots), dtype=torch.float64, device=evidence.device)
    known = torch.zeros_like(weights, dtype=torch.bool)
    for start in range(0, n, chunk):
        sl = slice(start, min(n, start+chunk))
        chosen, support = ids[sl], mass[sl].double()
        own = evidence[chosen[:, :, None], pairs[sl, None]].double()
        rival = rival_evidence[chosen[:, :, None], pairs[sl, None].flip(-1)].double()
        own_mean = (own*support[..., None, None]).sum(1)
        rival_mean = (rival*support[..., None, None]).sum(1)
        if pooled:
            own, rival = own.mean(-1, keepdim=True), rival.mean(-1, keepdim=True)
            own_mean, rival_mean = own_mean.mean(-1, keepdim=True), rival_mean.mean(-1, keepdim=True)
        x, y = own-own_mean[:, None], rival-rival_mean[:, None]
        own_var = (x.square()*support[..., None, None]).sum(1)
        rival_var = (y.square()*support[..., None, None]).sum(1)
        covariance = torch.einsum('nsdk,nsdl,ns->ndkl', x, y, support)
        available = (own_var[..., :, None] > 1e-12) & (rival_var[..., None, :] > 1e-12)
        available &= (valid[sl] & (pairs[sl, 0] != pairs[sl, 1]))[:, None, None, None]
        denominator = (own_var[..., :, None]*rival_var[..., None, :]).sqrt().clamp_min(1e-300)
        collision = (covariance/denominator).clamp(0, 1).square().masked_fill(~available, 0.)
        risk = (collision*rival_mean.softmax(-1)[..., None, :]).sum(-1)
        current_known = available.any(-1)
        if pooled:
            risk, current_known = risk.expand(-1, -1, slots), current_known.expand(-1, -1, slots)
        weights[sl] = (1-risk).clamp_min(1e-6)
        known[sl] = current_known
    weights.scatter_(-1, canonical[pairs][..., None], 1.)
    known.scatter_(-1, canonical[pairs][..., None], False)
    return weights, known


@torch.inference_mode()
def scores(baseline, operator, relation, evidence, crops, count, coordinates,
           image_size, members, pairs, canonical, valid, *, methods=METHODS[3:]):
    if not methods or not set(methods).issubset(METHODS[3:]):
        raise ValueError('Declared frozen collision endpoints required.')
    support = neighbourhood(relation, valid)
    weights, known = collision_weights(evidence, pairs, canonical, support, valid)
    allocations = {PRIMARY: weights}
    if set(methods) & {POOLED, MATCHED_POOLED}:
        allocations[POOLED] = collision_weights(evidence, pairs, canonical, support, valid, pooled=True)[0]
    if set(methods) & {CLASS_MEAN, MATCHED_CLASS_MEAN, *SHUFFLES, *MATCHED_SHUFFLES}:
        for i, name in enumerate(SHUFFLES):
            shared, shuffled = weight_controls(weights, pairs, canonical, seed=20261006+i)
            if i == 0:
                allocations[CLASS_MEAN] = shared
            allocations[name] = shuffled
    if GLOBAL in methods:
        allocations[GLOBAL] = collision_weights(evidence, pairs, canonical,
            neighbourhood(relation, valid, global_support=True), valid)[0]
    if POSITION_NULL in methods:
        ids = valid.nonzero().flatten()
        generator = torch.Generator().manual_seed(20261006)
        permutation = ids[torch.randperm(len(ids), generator=generator).to(ids.device)]
        permuted = evidence.clone()
        permuted[ids] = evidence[permutation]
        allocations[POSITION_NULL] = collision_weights(evidence, pairs, canonical, support,
            valid, rival_evidence=permuted)[0]

    potentials, writes = {}, {}
    for name, allocation in allocations.items():
        potential = fixed_slot_potential(crops, count, coordinates, image_size,
            members, pairs, allocation, valid)
        potentials[name], writes[name] = potential, operator.double()@potential
    primary = writes[PRIMARY]
    stats = dict.fromkeys(DIAGNOSTICS, 0.)
    stats.update(canonical_weight_max_error=float((weights.gather(-1, canonical[pairs][..., None])-1).abs().max()),
        gauge_max_error=float(primary.sum(-1).abs().max()))
    if bool(valid.any()):
        stats.update(known_alias_fraction=float(known[valid].double().mean()),
            active_weight_fraction=float((weights[valid] < 1).double().mean()),
            collision_risk_mean=float((1-weights[valid]).mean()),
            potential_mean=float(potentials[PRIMARY][valid].abs().mean()))

    def matched(field):
        result, info = match_previous(field, primary, valid)
        stats['matched_norm_relative_error'] = max(stats['matched_norm_relative_error'], info['matched_previous_norm_relative_error'])
        stats['matched_unmatchable'] += info['matched_previous_unmatchable']
        return result

    values = {name: baseline.double()+field for name, field in writes.items() if name in methods}
    for raw, match in ((POOLED, MATCHED_POOLED), (CLASS_MEAN, MATCHED_CLASS_MEAN),
                       *zip(SHUFFLES, MATCHED_SHUFFLES)):
        if match in methods:
            values[match] = baseline.double()+matched(writes[raw])
    if SHUFFLED_WRITE in methods:
        values[SHUFFLED_WRITE] = baseline.double()+matched(permuted_operator(operator, valid).double()@potentials[PRIMARY])
    if DIRECT in methods:
        values[DIRECT] = baseline.double()+matched(potentials[PRIMARY])
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite collision prediction.')
    return values, stats


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS, cache=None):
    from benchmark_grouped_class_readout import CheckedGroupedCache
    from eval_geometry_vip_reliability import sample_broad
    from eval_rival_fine_full import prepare_wide, tile_coordinates
    from eval_sparse_alias_reuse import forbid_fine

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared collision methods required.')
    cache = CheckedGroupedCache() if cache is None else cache
    cache.verify = False
    resized = resize_geometry(image)
    height, width = resized.shape[-2:]
    local_branch = _BoundedBranch(geometry, 512, 4, 'prepare_image')
    wide_branch = _BoundedBranch(vip, 336, 4, 'crop_patch_features')
    wide, crops, _, count = prepare_wide(image, wide_branch, {'clean': (banks, queries)})
    texts = {p: F.normalize(bank.features.float(), dim=-1) for p, bank in banks.items()}
    requested = tuple(m for m in methods if m not in METHODS[:3])
    totals = {p: dict(tiles=0, **dict.fromkeys(DIAGNOSTICS, 0.)) for p in banks}
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
            for p, bank in banks.items():
                raw = cache.geometry((prepared.geometry_projected.float()@texts[p].T)[0], bank.parent_indices, bank.class_count)
                local = cache.geometry((features.float()@texts[p].T)[0], bank.parent_indices, bank.class_count)/.07
                broad = sample_broad(wide['clean', p], top, left, height, width).reshape_as(local)
                original = (raw/.07).double()
                baseline = local.double()+operator@(broad.double()-local.double())
                values = {'Geometry': raw, 'NoAdmission_Exact': original+operator@(broad.double()-original), BASELINE: baseline}
                if requested:
                    query = queries[p]
                    members = torch.stack([(query.parents == c).nonzero().flatten() for c in range(bank.class_count)])
                    canonical_ids = canonical_indices(query.class_names, query.aliases, query.parents)
                    canonical = canonical_ids-members[:, 0]
                    evidence = sample_raw(crops['clean', p], count, coordinates, (height, width), members)
                    current, diagnostics = scores(baseline, operator, relation, evidence, crops['clean', p], count,
                        coordinates, (height, width), members, baseline.topk(2, -1).indices, canonical, valid,
                        methods=requested)
                    values.update(current)
                    for field, value in diagnostics.items():
                        totals[p][field] += value
                for method in methods:
                    value = values[method]
                    dense = F.interpolate(value.T.reshape(1, bank.class_count, 32, 32), (512, 512), mode='bilinear', align_corners=False)[0]
                    if method == 'Geometry':
                        dense = dense/.07
                    ah, aw = min(512, height-top), min(512, width-left)
                    accumulators[p, method].add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], left, top)
                totals[p]['tiles'] += 1
        predictions = {p: {m: accumulators[p, m].finalize_resized(tuple(image.shape[-2:])) for m in methods} for p in banks}
    for row in totals.values():
        for key in DIAGNOSTICS:
            row[key] /= row['tiles']
        row.update(geometry_encodings=local_branch.calls, wide_encodings=wide_branch.calls,
            fine_forwards=0, additional_visual_forwards=0, additional_semantic_heads=0,
            maximum_support_neighbours=CAPACITY)
    return predictions, totals
