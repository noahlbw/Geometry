"""Rival-conditioned lexical binding with unchanged visual observations."""
from contextlib import ExitStack
import time

import torch
import torch.nn.functional as F

from .alias_action_capacity import reconstruction_operator
from .bounded_canonical_pair_alias import sparse_pair_potential, weight_controls
from .bounded_patch_only import PRIMARY as BASELINE, PROTOCOL as BASE_PROTOCOL, readout
from .bounded_physical_coupling import _BoundedBranch, geometry_windows, resize_geometry
from .device_probability_accumulator import DeviceProbabilityAccumulator
from .gear_ov import _crop_at
from .inference import hann_blend_window
from .prompts import REMOTE_SENSING_TEMPLATES
from .rival_alias_count import canonical_indices


IMPLEMENTATION = 'geometry-bounded896-counterfactual-lexical-binding-v1-20261006'
PRIMARY = 'LexicalBinding_RivalSoft'
CLASS_MEAN = 'LexicalBinding_ClassMean'
SHUFFLE = 'LexicalBinding_AliasShuffle'
CANONICAL_ONLY = 'LexicalBinding_CanonicalOnly'
NO_GEOMETRY = 'LexicalBinding_NoGeometrySupport'
METHODS = ('Geometry', 'NoAdmission_Exact', BASELINE, PRIMARY, CLASS_MEAN,
           SHUFFLE, CANONICAL_ONLY, NO_GEOMETRY)
CHUNK = 64
PROTOCOL = {**BASE_PROTOCOL,
    'alias_admission': 'counterfactual lexical-binding contrast, query/class/rival soft use',
    'binding_phrase': '{owner}, described as {alias}',
    'binding_templates': list(REMOTE_SENSING_TEMPLATES),
    'source': 'existing native local descriptor compares the same alias under own/rival owners',
    'support': 'original-G valid-donor barycenter of native descriptors; no pseudo-labelled cohorts',
    'weight': '1-sigmoid(-direct/.07)*sigmoid(-supported/.07); canonical1; empty-support fallback1',
    'aggregation': 'outside-exponent normalized weighted LME; original wide salience untouched',
    'scope': 'unchanged coupled top2 classes; all original aliases remain globally available',
    'writer': 'antisymmetric pair potential through unchanged original H',
    'additional_visual_forwards': 0, 'additional_semantic_heads': 0, 'fine_forwards': 0,
    'binding_query_chunk': CHUNK, 'fitted_parameters': 0,
    'controls': (CLASS_MEAN, SHUFFLE, CANONICAL_ONLY, NO_GEOMETRY),
    'limits': 'linguistic binding and visual support are not semantic correctness certificates',
    'precedents': 'contextual prompting and contrastive scores have precedents; no priority claim'}


def binding_phrase(owner, alias):
    return f'{owner}, described as {alias}'


@torch.inference_mode()
def binding_bank(geometry, bank):
    key = (tuple(bank.class_names), tuple(bank.alias_names), tuple(bank.parent_indices.tolist()))
    store = getattr(geometry, '_lexical_binding_banks', None)
    if store is None:
        store = geometry._lexical_binding_banks = {}
    if key not in store:
        started = time.perf_counter()
        classes = len(bank.class_names)
        aliases = [[a for a, parent in zip(bank.alias_names, key[2]) if parent == c]
                   for c in range(classes)]
        count = len(aliases[0])
        if count < 2 or any(len(group) != count for group in aliases):
            raise ValueError('Equal, nontrivial class alias counts required.')
        groups = [[binding_phrase(owner, alias) for alias in aliases[c]]
                  for c in range(classes) for owner in bank.class_names]
        encoded, parents, _ = geometry.backbone.encode_text_aliases(
            groups, templates=REMOTE_SENSING_TEMPLATES)
        expected = torch.arange(classes*classes, device=parents.device).repeat_interleave(count)
        if not torch.equal(parents, expected):
            raise RuntimeError('Binding encoding changed original word order or count.')
        bound = encoded.float().reshape(classes, classes, count, -1)
        own = bound[torch.arange(classes, device=bound.device), torch.arange(classes, device=bound.device)]
        directions = own[:, None]-bound
        if not bool(torch.isfinite(directions).all()):
            raise RuntimeError('Nonfinite lexical binding directions.')
        store[key] = dict(directions=directions, setup_seconds=time.perf_counter()-started,
                          encoded_bindings=classes*classes*count,
                          direction_bytes=directions.numel()*directions.element_size())
    return store[key]


def supported_features(native, relation, valid):
    if native.ndim != 2 or relation.shape != (len(native), len(native)) or valid.shape != native.shape[:1]:
        raise ValueError('Matching native descriptors, Geometry and valid queries required.')
    mass = relation.float().masked_fill(~valid[None], 0.)
    total = mass.sum(-1)
    features = (mass@native.float())/total[:, None].clamp_min(1e-12)
    return features, valid & (total > 1e-12)


def binding_weights(native, supported, directions, pairs, canonical, known, *, canonical_only=False,
                    no_geometry=False):
    if (native.ndim != 2 or supported.shape != native.shape or directions.ndim != 4
            or directions.shape[0] != directions.shape[1] or directions.shape[-1] != native.shape[-1]
            or pairs.shape != (len(native), 2) or canonical.shape != directions.shape[:1]
            or known.shape != native.shape[:1]):
        raise ValueError('Matching descriptors, class/rival binding bank, pairs and support required.')
    result = torch.ones((len(native), 2, directions.shape[2]), device=native.device, dtype=torch.float64)
    for start in range(0, len(native), CHUNK):
        stop = min(start+CHUNK, len(native))
        selected = pairs[start:stop]
        contrast = directions[selected, selected.flip(-1)].float()
        if canonical_only:
            indices = canonical[selected][..., None, None].expand(-1, -1, 1, contrast.shape[-1])
            contrast = contrast.gather(2, indices).expand_as(contrast)
        direct = torch.einsum('nd,nskd->nsk', native[start:stop].float(), contrast)/.07
        context = direct if no_geometry else torch.einsum(
            'nd,nskd->nsk', supported[start:stop].float(), contrast)/.07
        weights = (1-torch.sigmoid(-direct.double())*torch.sigmoid(-context.double())).clamp_min(1e-6)
        active = known[start:stop] & (selected[:, 0] != selected[:, 1])
        weights.masked_fill_(~active[:, None, None], 1.)
        weights.scatter_(-1, canonical[selected][..., None], 1.)
        result[start:stop] = weights
    return result


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS, cache=None):
    from benchmark_grouped_class_readout import CheckedGroupedCache
    from eval_geometry_vip_reliability import sample_broad
    from eval_rival_fine_full import prepare_wide, tile_coordinates
    from eval_sparse_alias_reuse import forbid_fine

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared lexical binding methods required.')
    cache = CheckedGroupedCache() if cache is None else cache
    cache.verify = False
    resized = resize_geometry(image)
    height, width = resized.shape[-2:]
    local_branch = _BoundedBranch(geometry, 512, 4, 'prepare_image')
    wide_branch = _BoundedBranch(vip, 336, 4, 'crop_patch_features')
    wide, crops, _, count = prepare_wide(image, wide_branch, {'clean': (banks, queries)})
    texts = {p: F.normalize(bank.features.float(), dim=-1) for p, bank in banks.items()}
    extra = bool(set(methods)-set(METHODS[:3]))
    bindings = {p: binding_bank(geometry, bank) for p, bank in banks.items()} if extra else {}
    totals = {p: dict(tiles=0, active_weight_fraction=0., potential_mean=0.) for p in banks}
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
            native = prepared.native_projected[0].float()
            supported, known = supported_features(native, relation, valid) if extra else (None, None)
            for p, bank in banks.items():
                raw = cache.geometry((prepared.geometry_projected.float()@texts[p].T)[0], bank.parent_indices, bank.class_count)
                local = cache.geometry((features.float()@texts[p].T)[0], bank.parent_indices, bank.class_count)/.07
                broad = sample_broad(wide['clean', p], top, left, height, width).reshape_as(local)
                original = (raw/.07).double()
                baseline = local.double()+operator@(broad.double()-local.double())
                values = {'Geometry': raw, 'NoAdmission_Exact': original+operator@(broad.double()-original), BASELINE: baseline}
                if extra:
                    query = queries[p]
                    if tuple(query.aliases) != tuple(bank.alias_names):
                        raise RuntimeError('Local/context alias identity differs.')
                    members = torch.stack([(query.parents == c).nonzero().flatten() for c in range(bank.class_count)])
                    canonical = canonical_indices(query.class_names, query.aliases, query.parents)-members[:, 0]
                    pairs = baseline.topk(2, -1).indices
                    args = (native, supported, bindings[p]['directions'], pairs, canonical, known)
                    weights = binding_weights(*args)
                    allocations = {PRIMARY: weights}
                    if set(methods) & {CLASS_MEAN, SHUFFLE}:
                        allocations[CLASS_MEAN], allocations[SHUFFLE] = weight_controls(weights, pairs, canonical)
                    if CANONICAL_ONLY in methods:
                        allocations[CANONICAL_ONLY] = binding_weights(*args, canonical_only=True)
                    if NO_GEOMETRY in methods:
                        allocations[NO_GEOMETRY] = binding_weights(*args, no_geometry=True)
                    for method in methods:
                        if method in allocations:
                            potential = sparse_pair_potential(crops['clean', p], count, coordinates,
                                (height, width), members, pairs, allocations[method], valid)
                            values[method] = baseline+operator@potential
                            if method == PRIMARY:
                                totals[p]['potential_mean'] += potential[valid].abs().mean()
                    totals[p]['active_weight_fraction'] += (weights[valid] != 1).float().mean()
                for method in methods:
                    value = values[method]
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeError('Nonfinite lexical binding readout.')
                    dense = F.interpolate(value.T.reshape(1, bank.class_count, 32, 32), (512, 512), mode='bilinear', align_corners=False)[0]
                    if method == 'Geometry':
                        dense = dense/.07
                    ah, aw = min(512, height-top), min(512, width-left)
                    accumulators[p, method].add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], left, top)
                totals[p]['tiles'] += 1
        predictions = {p: {m: accumulators[p, m].finalize_resized(tuple(image.shape[-2:])) for m in methods} for p in banks}
    for p, row in totals.items():
        for key in ('active_weight_fraction', 'potential_mean'):
            row[key] = float(row[key]/row['tiles'])
        row.update(geometry_encodings=local_branch.calls, wide_encodings=wide_branch.calls,
            fine_forwards=0, additional_visual_forwards=0, additional_semantic_heads=0,
            binding_query_chunk=CHUNK, binding_text_setup_seconds=bindings[p]['setup_seconds'] if extra else 0.,
            binding_text_queries=bindings[p]['encoded_bindings'] if extra else 0,
            binding_direction_bytes=bindings[p]['direction_bytes'] if extra else 0)
    return predictions, totals
