"""Canonical-referenced alias contrasts with the frozen bounded coupled reader."""
from contextlib import ExitStack
import torch
import torch.nn.functional as F

from .alias_action_capacity import reconstruction_operator
from .bounded_patch_only import PRIMARY as BASELINE, PROTOCOL as BASE_PROTOCOL, readout
from .bounded_physical_coupling import _BoundedBranch, geometry_windows, resize_geometry
from .calibrated_competitive_alias import profiled_logits
from .device_probability_accumulator import DeviceProbabilityAccumulator
from .gear_ov import _crop_at
from .inference import hann_blend_window
from .rival_alias_count import canonical_indices
from .stratified_soft_alias import crop_stencil


IMPLEMENTATION = 'geometry-bounded896-canonical-pair-alias-contrast-v1-20261005'
PRIMARY = 'CanonicalPair_RivalSoft'
CLASS_MEAN = 'CanonicalPair_ClassMean'
SHUFFLE = 'CanonicalPair_AliasShuffle'
CLASS_GATE = 'CanonicalPair_ClassGate'
POSITION_NULL = 'CanonicalPair_ResponseShuffle'
METHODS = ('Geometry', 'NoAdmission_Exact', BASELINE, PRIMARY, CLASS_MEAN,
           SHUFFLE, CLASS_GATE, POSITION_NULL)
PROTOCOL = {**BASE_PROTOCOL, 'alias_admission': 'canonical-referenced pair-specific soft use',
    'references': 'canonical-only local/wide agreement, margin confidence, current query excluded',
    'source': 'own-minus-rival wide raw-response means divided by pooled response deviation',
    'scope': 'local-winner versus wide-winner disputes only; no permanent word deletion',
    'weight': 'one minus rival Geometry support fraction times sigmoid of negative contrast; canonical1',
    'aggregation': 'weights outside exponent; normalized weighted LME with unchanged profiled evidence',
    'writer': 'antisymmetric dispute potential through unchanged original H',
    'fine_forwards': 0, 'additional_visual_forwards': 0, 'additional_semantic_heads': 0,
    'fitted_parameters': 0, 'controls': (CLASS_MEAN, SHUFFLE, CLASS_GATE, POSITION_NULL),
    'limits': 'pseudo-reference agreement and response separation do not certify correctness',
    'precedents': 'revisits stratified references and canonical alias increments, not a new kernel theorem'}


def canonical_references(local, wide, valid):
    if local.shape != wide.shape or local.ndim != 2 or valid.shape != local.shape[:1]:
        raise ValueError('Matching canonical class scores and valid queries required.')
    lp, wp = local.float().softmax(-1), wide.float().softmax(-1)
    lv, li = lp.topk(2, -1)
    wv, wi = wp.topk(2, -1)
    confidence = ((lv[:, 0]-lv[:, 1])*(wv[:, 0]-wv[:, 1])).clamp_min(0).sqrt()
    confidence *= valid & (li[:, 0] == wi[:, 0])
    return torch.zeros_like(lp).scatter_(1, li[:, :1], confidence[:, None])


def reference_contrast(evidence, references, pairs):
    """Compact class moments; leave-query-out means for just the two disputed classes."""
    if (evidence.ndim != 3 or references.shape != evidence.shape[:2]
            or pairs.shape != (len(evidence), 2)):
        raise ValueError('Matching query/class/alias responses, references and two-class pairs required.')
    n, classes, count = evidence.shape
    values, refs = evidence.double(), references.double()
    mass = refs.sum(0)
    first = (refs.T @ values.flatten(1)).reshape(classes, classes, count)
    second = (refs.T @ values.square().flatten(1)).reshape_as(first)
    rows = torch.arange(n, device=evidence.device)[:, None]
    current = values[rows, pairs]
    own, rival = pairs, pairs.flip(-1)

    def moments(cohort):
        removed = refs[rows, cohort]
        remaining = (mass[cohort]-removed).clamp_min(0)
        mean = (first[cohort, own]-removed[..., None]*current)/remaining[..., None].clamp_min(1e-12)
        square = (second[cohort, own]-removed[..., None]*current.square())/remaining[..., None].clamp_min(1e-12)
        return mean, (square-mean.square()).clamp_min(0), remaining > 1e-12

    mu_own, var_own, own_known = moments(own)
    mu_rival, var_rival, rival_known = moments(rival)
    contrast = (mu_own-mu_rival)/(var_own+var_rival).sqrt().clamp_min(1e-6)
    return contrast, own_known & rival_known


def rival_support(relation, references, pairs, valid):
    if relation.shape != (len(valid), len(valid)):
        raise ValueError('Matching original Geometry relation required.')
    weights = relation.float().masked_fill(~valid[None], 0.).clone()
    weights.fill_diagonal_(0.)
    support = weights @ references.float()
    paired = support.gather(1, pairs)
    total = paired.sum(-1)
    fractions = paired.flip(-1)/total[:, None].clamp_min(1e-12)
    return fractions, total > 1e-12


def weight_controls(weights, pairs, canonical, *, seed=20261005):
    shared, shuffled = weights.clone(), weights.clone()
    generator = torch.Generator().manual_seed(seed)
    for c in range(len(canonical)):
        ids = torch.arange(weights.shape[-1], device=weights.device)
        ids = ids[ids != canonical[c]]
        use = pairs == c
        selected = weights[use]
        class_weights, null_weights = selected.clone(), selected.clone()
        class_weights[:, ids] = selected[:, ids].mean(-1, keepdim=True)
        permutation = ids[torch.randperm(len(ids), generator=generator).to(ids.device)]
        null_weights[:, ids] = selected[:, permutation]
        shared[use], shuffled[use] = class_weights, null_weights
    return shared, shuffled


def alias_weights(evidence, references, relation, pairs, canonical, valid):
    contrast, known = reference_contrast(evidence, references, pairs)
    fraction, supported = rival_support(relation, references, pairs, valid)
    disputed = valid & (pairs[:, 0] != pairs[:, 1]) & supported
    risk = fraction.double()[..., None]*torch.sigmoid(-contrast)
    risk.masked_fill_(~(known & disputed[:, None])[..., None], 0.)
    risk.scatter_(-1, canonical[pairs][..., None], 0.)
    weights = (1-risk).clamp_min(1e-6)
    return weights, known & disputed[:, None], fraction


def sparse_pair_potential(crops, count, coordinates, image_size, members, pairs, weights, valid):
    delta = torch.zeros((len(valid), 2), device=weights.device, dtype=torch.float64)
    rows = torch.arange(len(valid), device=weights.device)[:, None, None]
    mass = weights.double().mean(-1)
    unchanged = (weights == 1).all(-1)
    for crop in crops:
        ids, coefficients = crop_stencil(crop, count, coordinates, image_size)
        evidence = profiled_logits(crop, members).double()
        probabilities = evidence.softmax(-1)[ids]
        selected = probabilities[rows, torch.arange(ids.shape[1], device=ids.device)[None, :, None], pairs[:, None]]
        retained = (selected*weights[:, None]).sum(-1)
        change = (retained.clamp_min(1e-300).log()-mass[:, None].log()).masked_fill(unchanged[:, None], 0.)
        delta += (change*coefficients.double()[..., None]).sum(1)
    contrast = .5*(delta[:, 0]-delta[:, 1])
    contrast.masked_fill_(~valid | (pairs[:, 0] == pairs[:, 1]), 0.)
    output = torch.zeros((len(valid), len(members)), device=weights.device, dtype=torch.float64)
    output.scatter_add_(1, pairs, torch.stack((contrast, -contrast), -1))
    return output


def sample_raw(crops, count, coordinates, image_size, members):
    result = coordinates.new_zeros((len(coordinates), *members.shape))
    for crop in crops:
        ids, coefficients = crop_stencil(crop, count, coordinates, image_size)
        result += (crop.alias_logits[:, members][ids]*coefficients[..., None, None]).sum(1)
    return result


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS, cache=None):
    from benchmark_grouped_class_readout import CheckedGroupedCache
    from eval_geometry_vip_reliability import sample_broad
    from eval_rival_fine_full import prepare_wide, tile_coordinates
    from eval_sparse_alias_reuse import forbid_fine

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared canonical-referenced methods required.')
    cache = CheckedGroupedCache() if cache is None else cache
    cache.verify = False
    resized = resize_geometry(image)
    height, width = resized.shape[-2:]
    local_branch = _BoundedBranch(geometry, 512, 4, 'prepare_image')
    wide_branch = _BoundedBranch(vip, 336, 4, 'crop_patch_features')
    wide, crops, _, count = prepare_wide(image, wide_branch, {'clean': (banks, queries)})
    texts = {p: F.normalize(bank.features.float(), dim=-1) for p, bank in banks.items()}
    extra = bool(set(methods)-set(METHODS[:3]))
    totals = {p: dict(tiles=0, dispute_fraction=0., known_pair_fraction=0., active_weight_fraction=0.,
                     reference_fraction=0., potential_mean=0.) for p in banks}
    blend = torch.from_numpy(hann_blend_window(512)).to(geometry.device)
    with forbid_fine(), ExitStack() as stack:
        accumulators = {(p, m): stack.enter_context(DeviceProbabilityAccumulator(
            bank.class_count, height, width, geometry.device)) for p, bank in banks.items() for m in methods}
        for top, left in geometry_windows(height, width):
            prepared = local_branch.prepare_image(_crop_at(resized, top, left, 512).to(geometry.device))
            coordinates = tile_coordinates(top, left, geometry.device)
            valid = (coordinates[:, 0] < height) & (coordinates[:, 1] < width)
            operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
            with geometry.backbone._autocast():
                features = readout(geometry.backbone.model.visual_model.head, prepared, 2.)[0]
            for p, bank in banks.items():
                raw = cache.geometry((prepared.geometry_projected.float() @ texts[p].T)[0], bank.parent_indices, bank.class_count)
                local_aliases = (features.float() @ texts[p].T)[0]
                local = cache.geometry(local_aliases, bank.parent_indices, bank.class_count)/.07
                broad = sample_broad(wide['clean', p], top, left, height, width).reshape_as(local)
                baseline = local.double()+operator@(broad.double()-local.double())
                original = (raw/.07).double()
                values = {'Geometry': raw, 'NoAdmission_Exact': original+operator@(broad.double()-original), BASELINE: baseline}
                if extra:
                    query = queries[p]
                    members = torch.stack([(query.parents == c).nonzero().flatten() for c in range(bank.class_count)])
                    if not torch.equal(members.flatten(), torch.arange(members.numel(), device=members.device)):
                        raise ValueError('Original class-contiguous alias order required.')
                    canonical_ids = canonical_indices(query.class_names, query.aliases, query.parents)
                    canonical = canonical_ids-members[:, 0]
                    evidence = sample_raw(crops['clean', p], count, coordinates, (height, width), members)
                    references = canonical_references(local_aliases[:, canonical_ids]/.07,
                        evidence[:, torch.arange(bank.class_count, device=valid.device), canonical], valid)
                    pairs = torch.stack((local.argmax(-1), broad.argmax(-1)), -1)
                    weights, known, fractions = alias_weights(evidence, references,
                        prepared.geometry_patch_conditional[0], pairs, canonical, valid)
                    shared, shuffled = weight_controls(weights, pairs, canonical)
                    allocations = {PRIMARY: weights, CLASS_MEAN: shared, SHUFFLE: shuffled}
                    if POSITION_NULL in methods:
                        ids = valid.nonzero().flatten()
                        generator = torch.Generator().manual_seed(20261005)
                        permutation = ids[torch.randperm(len(ids), generator=generator).to(ids.device)]
                        permuted = evidence.clone()
                        permuted[ids] = evidence[permutation]
                        allocations[POSITION_NULL] = alias_weights(permuted, references,
                            prepared.geometry_patch_conditional[0], pairs, canonical, valid)[0]
                    for method in methods:
                        if method in allocations:
                            potential = sparse_pair_potential(crops['clean', p], count, coordinates,
                                (height, width), members, pairs, allocations[method], valid)
                            values[method] = baseline+operator@potential
                            if method == PRIMARY:
                                totals[p]['potential_mean'] += float(potential[valid].abs().mean())
                        elif method == CLASS_GATE:
                            gate = torch.ones(len(valid), device=valid.device, dtype=torch.float64)
                            gate.masked_scatter_(known.all(-1), fractions[known.all(-1), 0].double())
                            values[method] = local.double()+operator@(gate[:, None]*(broad.double()-local.double()))
                    totals[p]['dispute_fraction'] += float((pairs[valid, 0] != pairs[valid, 1]).float().mean())
                    totals[p]['known_pair_fraction'] += float(known[valid].all(-1).float().mean())
                    totals[p]['active_weight_fraction'] += float((weights[valid] != 1).float().mean())
                    totals[p]['reference_fraction'] += float((references[valid].sum(-1) > 0).float().mean())
                for method in methods:
                    value = values[method]
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeError('Nonfinite canonical-referenced readout.')
                    dense = F.interpolate(value.T.reshape(1, bank.class_count, 32, 32), (512, 512),
                        mode='bilinear', align_corners=False)[0]
                    if method == 'Geometry':
                        dense = dense/.07
                    ah, aw = min(512, height-top), min(512, width-left)
                    accumulators[p, method].add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], left, top)
                totals[p]['tiles'] += 1
        predictions = {p: {m: accumulators[p, m].finalize_resized(tuple(image.shape[-2:])) for m in methods} for p in banks}
    for row in totals.values():
        for key in ('dispute_fraction', 'known_pair_fraction', 'active_weight_fraction', 'reference_fraction', 'potential_mean'):
            row[key] /= row['tiles']
        row.update(geometry_encodings=local_branch.calls, wide_encodings=wide_branch.calls,
                   fine_forwards=0, additional_visual_forwards=0, additional_semantic_heads=0)
    return predictions, totals
