"""Vocabulary-time entailment priors for competitive, fixed-slot alias use."""
from contextlib import ExitStack
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .alias_action_capacity import reconstruction_operator
from .bounded_canonical_pair_alias import weight_controls
from .bounded_covariance_alias import fixed_slot_potential
from .bounded_patch_only import PRIMARY as BASELINE, PROTOCOL as BASE_PROTOCOL, readout
from .bounded_physical_coupling import _BoundedBranch, geometry_windows, resize_geometry
from .device_probability_accumulator import DeviceProbabilityAccumulator
from .gear_ov import _crop_at
from .inference import hann_blend_window
from .rival_alias_count import canonical_indices
from .supported_positive_alias import permuted_operator
from .witness_cap_alias import match_previous


MODEL_ID = 'cross-encoder/nli-MiniLM2-L6-H768'
REVISION = 'b95119ce93d3e065de6214e38cd4a97b0f2f2c6d'
TEMPLATE = 'The object or surface is {phrase}.'
IMPLEMENTATION = 'geometry-bounded896-semantic-membership-fixed-slot-v1-20261006'
PRIMARY = 'SemanticMembership_Soft'
POOLED = 'SemanticMembership_PooledSource'
MATCHED_POOLED = 'SemanticMembership_MatchedPooledSource'
CLASS_MEAN = 'SemanticMembership_ClassMean'
MATCHED_CLASS_MEAN = 'SemanticMembership_MatchedClassMean'
SHUFFLES = tuple('SemanticMembership_AliasShuffle'+str(i) for i in range(3))
MATCHED_SHUFFLES = tuple('SemanticMembership_MatchedAliasShuffle'+str(i) for i in range(3))
RIVAL_COLLAPSED = 'SemanticMembership_RivalCollapsed'
MATCHED_RIVAL_COLLAPSED = 'SemanticMembership_MatchedRivalCollapsed'
SHUFFLED_WRITE = 'SemanticMembership_MatchedShuffledWrite'
DIRECT = 'SemanticMembership_DirectMatched'
METHODS = ('Geometry', 'NoAdmission_Exact', BASELINE, PRIMARY, POOLED,
    MATCHED_POOLED, CLASS_MEAN, MATCHED_CLASS_MEAN, *SHUFFLES, *MATCHED_SHUFFLES,
    RIVAL_COLLAPSED, MATCHED_RIVAL_COLLAPSED, SHUFFLED_WRITE, DIRECT)
DIAGNOSTICS = ('active_weight_fraction', 'weight_mean', 'potential_mean',
    'own_entailment_mean', 'rival_entailment_mean', 'unknown_mass_mean',
    'canonical_weight_max_error', 'matched_norm_relative_error',
    'matched_unmatchable', 'gauge_max_error')
PROTOCOL = {**BASE_PROTOCOL,
    'alias_admission': 'frozen NLI semantic membership, conditioned on current top2 competition',
    'semantic_model': MODEL_ID, 'semantic_revision': REVISION, 'semantic_template': TEMPLATE,
    'source': 'declared words and class names only; vocabulary-time NLI cache; no target images/masks',
    'weight': '(own entailment + unknown)/(own + rival entailment + unknown); unknown=(1-own)*(1-rival)',
    'aggregation': 'original fixed20 slots and salience; no survivor renormalization',
    'scope': 'unchanged bounded baseline top2 competitors; canonical protected; no permanent deletion',
    'writer': 'original H writes antisymmetric fixed-slot pair potential',
    'fine_forwards': 0, 'additional_visual_forwards': 0, 'additional_semantic_heads': 0,
    'per_image_nli_forwards': 0, 'fitted_parameters': 0, 'controls': METHODS[4:],
    'precedents': 'NLI, conditional weighting and ridge are inherited/familiar; no novelty claim from adding NLI',
    'limits': 'semantic membership is not pixel correctness; NLI posteriors are not truth or calibrated target probabilities'}


class SemanticSource:
    def __init__(self, path):
        data = json.loads(Path(path).read_text())
        expected = dict(model=MODEL_ID, revision=REVISION, template=TEMPLATE,
                        target_images_loaded=False, target_masks_loaded=False)
        if any(data.get(k) != v for k, v in expected.items()):
            raise ValueError('Frozen, image-free semantic source required.')
        self.entries, self.cache = {}, {}
        for entry in data['entries']:
            key = tuple(entry['pair'])
            values = torch.tensor(entry['probabilities'], dtype=torch.float64)
            if (len(key) != 2 or key in self.entries or values.shape != (3,)
                    or not bool(torch.isfinite(values).all()) or bool((values < 0).any())
                    or bool((values > 1).any()) or abs(float(values.sum())-1) > 1e-6):
                raise ValueError('Unique finite contradiction/entailment/neutral posteriors required.')
            self.entries[key] = float(values[1])
        self.setup = data['setup']

    def matrix(self, query, members, device):
        classes, aliases = tuple(query.class_names), tuple(query.aliases)
        key = classes, aliases, str(device)
        if key not in self.cache:
            flat = torch.tensor([[self.entries[a, c] for c in classes] for a in aliases],
                                dtype=torch.float64, device=device)
            self.cache[key] = flat[members]
        return self.cache[key]


def membership_weights(table, pairs, canonical, valid, *, pooled=False, collapsed=False):
    if (table.ndim != 3 or table.shape[0] != table.shape[-1] or table.shape[0] < 2
            or canonical.shape != table.shape[:1] or pairs.shape != (len(valid), 2)
            or valid.dtype != torch.bool or not bool(torch.isfinite(table).all())
            or bool((table < 0).any()) or bool((table > 1).any())):
        raise ValueError('Finite class/word/class entailment probabilities and query pairs required.')
    classes, slots, _ = table.shape
    if (bool((pairs < 0).any()) or bool((pairs >= classes).any())
            or bool((canonical < 0).any()) or bool((canonical >= slots).any()) or slots < 2):
        raise ValueError('Valid competitors and canonical slots required.')
    if pooled:
        keep = torch.arange(slots, device=table.device)[None] != canonical[:, None]
        table = (table*keep[..., None]).sum(1, keepdim=True)/(slots-1)
        table = table.expand(-1, slots, -1)
    ids = torch.arange(slots, device=table.device)
    own = table[pairs[..., None], ids, pairs[..., None]]
    rival = (table[pairs].sum(-1)-own)/(classes-1) if collapsed else table[
        pairs[..., None], ids, pairs.flip(-1)[..., None]]
    unknown = (1-own)*(1-rival)
    weights = ((own+unknown)/(own+rival+unknown)).clamp_min(1e-6)
    weights.masked_fill_(~(valid & (pairs[:, 0] != pairs[:, 1]))[:, None, None], 1.)
    weights.scatter_(-1, canonical[pairs][..., None], 1.)
    return weights, (own, rival, unknown)


@torch.inference_mode()
def scores(baseline, operator, table, crops, count, coordinates, image_size,
           members, pairs, canonical, valid, *, methods=METHODS[3:]):
    if not methods or not set(methods).issubset(METHODS[3:]):
        raise ValueError('Declared semantic membership endpoints required.')
    weights, posterior = membership_weights(table, pairs, canonical, valid)
    allocations = {PRIMARY: weights}
    if set(methods) & {POOLED, MATCHED_POOLED}:
        allocations[POOLED] = membership_weights(table, pairs, canonical, valid, pooled=True)[0]
    if set(methods) & {RIVAL_COLLAPSED, MATCHED_RIVAL_COLLAPSED}:
        allocations[RIVAL_COLLAPSED] = membership_weights(table, pairs, canonical, valid, collapsed=True)[0]
    if set(methods) & {CLASS_MEAN, MATCHED_CLASS_MEAN, *SHUFFLES, *MATCHED_SHUFFLES}:
        for i, name in enumerate(SHUFFLES):
            shared, shuffled = weight_controls(weights, pairs, canonical, seed=20261006+i)
            if i == 0:
                allocations[CLASS_MEAN] = shared
            allocations[name] = shuffled
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
        stats.update(active_weight_fraction=float((weights[valid] < 1).double().mean()),
            weight_mean=float(weights[valid].mean()), potential_mean=float(potentials[PRIMARY][valid].abs().mean()),
            **{name: float(value[valid].mean()) for name, value in zip(
                ('own_entailment_mean', 'rival_entailment_mean', 'unknown_mass_mean'), posterior)})

    def matched(field):
        result, info = match_previous(field, primary, valid)
        stats['matched_norm_relative_error'] = max(stats['matched_norm_relative_error'], info['matched_previous_norm_relative_error'])
        stats['matched_unmatchable'] += info['matched_previous_unmatchable']
        return result

    values = {name: baseline.double()+field for name, field in writes.items() if name in methods}
    for raw, match in ((POOLED, MATCHED_POOLED), (CLASS_MEAN, MATCHED_CLASS_MEAN),
                       (RIVAL_COLLAPSED, MATCHED_RIVAL_COLLAPSED), *zip(SHUFFLES, MATCHED_SHUFFLES)):
        if match in methods:
            values[match] = baseline.double()+matched(writes[raw])
    if SHUFFLED_WRITE in methods:
        values[SHUFFLED_WRITE] = baseline.double()+matched(permuted_operator(operator, valid).double()@potentials[PRIMARY])
    if DIRECT in methods:
        values[DIRECT] = baseline.double()+matched(potentials[PRIMARY])
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite semantic membership prediction.')
    return values, stats


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS, cache=None, semantic):
    from benchmark_grouped_class_readout import CheckedGroupedCache
    from eval_geometry_vip_reliability import sample_broad
    from eval_rival_fine_full import prepare_wide, tile_coordinates
    from eval_sparse_alias_reuse import forbid_fine

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared semantic membership methods required.')
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
            operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
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
                    table = semantic.matrix(query, members, geometry.device)
                    current, diagnostics = scores(baseline, operator, table, crops['clean', p], count,
                        coordinates, (height, width), members, baseline.topk(2, -1).indices, canonical, valid,
                        methods=requested)
                    values.update(current)
                    for field, value in diagnostics.items():
                        totals[p][field] += value
                for method in methods:
                    dense = F.interpolate(values[method].T.reshape(1, bank.class_count, 32, 32),
                                          (512, 512), mode='bilinear', align_corners=False)[0]
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
            fine_forwards=0, additional_visual_forwards=0, additional_semantic_heads=0, per_image_nli_forwards=0)
    return predictions, totals
