"""One global physical witness and fine-directed, sparse alias admission."""
from contextlib import ExitStack
import torch
import torch.nn.functional as F

from .alias_action_capacity import reconstruction_operator
from .bounded_canonical_pair_alias import sparse_pair_potential
from .bounded_detail_pair_alias import (detail_rgb, select_details, sample_evidence,
    sparse_weights, PRIMARY as WEIGHT_PRIMARY, CLASS_MEAN as WEIGHT_MEAN, SHUFFLE as WEIGHT_SHUFFLE)
from .bounded_patch_only import PRIMARY as BASELINE, PROTOCOL as BASE_PROTOCOL, readout
from .bounded_physical_coupling import _BoundedBranch, geometry_windows, resize_geometry
from .device_probability_accumulator import DeviceProbabilityAccumulator
from .gear_ov import _crop_at
from .inference import hann_blend_window
from .native_query_alias import query_relation
from .rival_alias_count import canonical_indices
from .stratified_soft_alias import WideCrop
from .tcpr import _finish_attention_block, _finish_head, _resolve_block_index


IMPLEMENTATION = 'geometry-bounded896-native-detail1-projected-alias-v1-20261006'
PRIMARY = 'Detail1_ProjectedSoft'
CLASS_MEAN = 'Detail1_ProjectedClassMean'
SHUFFLE = 'Detail1_ProjectedAliasShuffle'
OBSERVATION_MEAN = 'Detail1_ObservationMean'
PAIR_OBSERVATION = 'Detail1_PairObservation'
UNPROJECTED = 'Detail1_RivalSoft'
BUDGET_ONLY = 'Detail1_AliasBudgetOnly'
METHODS = ('Geometry', 'NoAdmission_Exact', BASELINE, PRIMARY, CLASS_MEAN, SHUFFLE,
           OBSERVATION_MEAN, PAIR_OBSERVATION, UNPROJECTED, BUDGET_ONLY)
PROTOCOL = {**BASE_PROTOCOL, 'alias_admission': 'fine-directed projected soft pair action',
    'detail_budget_per_complete_image': 1, 'fine_forwards': 1,
    'detail_source': 'one globally selected256 region, original RGB directly sampled to512',
    'detail_reader': 'native DINO.text path only; exact native_projected replay; no detail Geometry',
    'acquisition': 'unchanged global local/wide JS disagreement times original H column energy; top1',
    'weights': 'unchanged positive-wide/negative-native/negative-G contradiction; canonical1',
    'aggregation': 'outside-exponent weighted LME on original wide crop stencils',
    'projection': 'clip alias pair action onto0..native-minus-wide pair margin; no reverse/overshoot',
    'scope': 'original coupled top2 at observed donors only; no all-alias/all-rival tensor',
    'writer': 'antisymmetric pair potential through original H',
    'controls': (CLASS_MEAN, SHUFFLE, OBSERVATION_MEAN, PAIR_OBSERVATION, UNPROJECTED, BUDGET_ONLY),
    'fitted_parameters': 0,
    'precedents': 'bounded sparse adaptation of earlier fine-action projection, not a new projection theorem',
    'limits': 'native evidence and Geometry agreement do not certify correctness; H can transport outside observed donors'}


@torch.inference_mode()
def native_detail_features(geometry, rgb):
    backbone = geometry.backbone
    backbone._validate_rgb(rgb)
    visual, config = backbone.model.visual_model, geometry.config
    head = visual.head
    if head.training:
        raise ValueError('Frozen native detail head required.')
    normalized = (rgb.to(geometry.device, non_blocking=True)-backbone._imagenet_mean)/backbone._imagenet_std
    with backbone._autocast():
        cls, raw, registers = visual.get_backbone_features(normalized)
        current = torch.cat((cls.unsqueeze(1), registers, raw), 1)
        prefix = current.shape[1]-raw.shape[1]
        block_index = _resolve_block_index(head, config.head_block)
        for block in head.blocks[:block_index]:
            current = block(current)
        block, attention = head.blocks[block_index], head.blocks[block_index].attn
        qkv = attention.qkv(block.norm1(current))
        batch, tokens, _ = qkv.shape
        qkv = qkv.reshape(batch, tokens, 3, attention.num_heads, attention.qkv.in_features//attention.num_heads)
        query, key, value = (part.transpose(1, 2) for part in torch.unbind(qkv, 2))
        weights = ((query.float()@key.float().transpose(-1, -2))*attention.scale).softmax(-1).to(value.dtype)
        current = _finish_attention_block(block, current, weights@value)
        projected = _finish_head(head, current, block_index)
    return F.normalize(projected[:, prefix:].float(), dim=-1)


class _NativeDetailReader:
    def __init__(self, geometry):
        self.geometry, self.device = geometry, geometry.device

    def prepare_image(self, rgb):
        return native_detail_features(self.geometry, rgb)


def project_pair_action(request, target):
    if request.shape != target.shape or request.ndim != 1:
        raise ValueError('One matching requested and native-target action per donor required.')
    return target.sign()*torch.minimum(request.abs(), target.abs())*(request*target > 0)


def pair_field(action, pairs, classes):
    if pairs.shape != (len(action), 2):
        raise ValueError('One competitive pair per donor required.')
    output = torch.zeros((len(action), classes), device=action.device, dtype=torch.float64)
    output.scatter_add_(1, pairs, torch.stack((action*.5, -action*.5), -1).double())
    return output


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS, cache=None):
    from benchmark_grouped_class_readout import CheckedGroupedCache
    from eval_geometry_vip_reliability import sample_broad
    from eval_matched_contribution_alias import crop_from_features
    from eval_rival_fine_full import prepare_wide, tile_coordinates
    from eval_sparse_alias_reuse import forbid_fine

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared detail1 projected methods required.')
    cache = CheckedGroupedCache() if cache is None else cache
    cache.verify = False
    resized = resize_geometry(image)
    height, width = resized.shape[-2:]
    local_branch = _BoundedBranch(geometry, 512, 4, 'prepare_image')
    wide_branch = _BoundedBranch(vip, 336, 4, 'crop_patch_features')
    detail_branch = _BoundedBranch(_NativeDetailReader(geometry), 512, 1, 'prepare_image')
    wide, crops, _, count = prepare_wide(image, wide_branch, {'clean': (banks, queries)})
    texts = {p: F.normalize(bank.features.float(), dim=-1) for p, bank in banks.items()}
    extra = bool(set(methods)-set(METHODS[:3]))
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
            raw = cache.geometry((prepared.geometry_projected.float()@texts[p].T)[0], bank.parent_indices, bank.class_count)
            local = cache.geometry((features.float()@texts[p].T)[0], bank.parent_indices, bank.class_count)/.07
            broad = sample_broad(wide['clean', p], top, left, height, width).reshape_as(local)
            fields[p] = (local, broad)
            original = (raw/.07).double()
            values[p] = {'Geometry': raw, 'NoAdmission_Exact': original+operator@(broad.double()-original),
                         BASELINE: local.double()+operator@(broad.double()-local.double())}
        records.append(dict(top=top, left=left, prepared=prepared, coordinates=coordinates,
                            valid=valid, operator=operator, fields=fields, values=values))
    positions = select_details(records, height, width)[:1] if extra else ()
    fine_crops = {p: [] for p in banks}
    fine_count = torch.ones(height, width, device=geometry.device)
    with forbid_fine():
        for position in positions:
            top, left, h, w = position
            observed = detail_branch.prepare_image(detail_rgb(image, position, (height, width)).to(geometry.device))
            for p, query in queries.items():
                blank = WideCrop(image.new_empty(1, 1), image.new_empty(1), top, left, h, w, 32, 256)
                crop = crop_from_features(observed, query, blank)
                fine_crops[p].append(WideCrop(crop.alias_logits, crop.salience, top, left, h, w, 32, 256))
    totals = {p: dict(tiles=0, observed_fraction=0., active_weight_fraction=0.,
                     retained_action_fraction=0., potential_mean=0.) for p in banks}
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
                if extra and positions:
                    query = queries[p]
                    members = torch.stack([(query.parents == c).nonzero().flatten() for c in range(bank.class_count)])
                    canonical = canonical_indices(query.class_names, query.aliases, query.parents)-members[:, 0]
                    fine, coverage = sample_evidence(fine_crops[p], fine_count, coordinates, (height, width), members)
                    known = valid & (coverage > 0)
                    fine /= coverage.clamp_min(1e-6)[:, None, None]
                    support, supported_known = query_relation(record['prepared'].geometry_patch_conditional[0], known)
                    ids = (known & supported_known).nonzero().flatten()
                    if len(ids):
                        supported = (support[ids]@fine.flatten(1)).reshape(len(ids), *members.shape)
                        broad_evidence, _ = sample_evidence(crops['clean', p], count, coordinates[ids], (height, width), members)
                        pairs = values[BASELINE][ids].topk(2, -1).indices
                        allocations = sparse_weights(broad_evidence, fine[ids], supported, pairs, canonical,
                                                    torch.ones(len(ids), device=valid.device, dtype=torch.bool))
                        weights = {PRIMARY: allocations[WEIGHT_PRIMARY], CLASS_MEAN: allocations[WEIGHT_MEAN],
                                   SHUFFLE: allocations[WEIGHT_SHUFFLE]}
                        fine_class = fine[ids].double().logsumexp(-1)
                        innovation = fine_class-record['fields'][p][1][ids].double()
                        target = innovation.gather(1, pairs[:, :1])[:, 0]-innovation.gather(1, pairs[:, 1:])[:, 0]
                        requested = {}
                        for name in weights:
                            if name in methods or name == PRIMARY and any(m in methods for m in (UNPROJECTED, BUDGET_ONLY)):
                                potential = sparse_pair_potential(crops['clean', p], count, coordinates[ids],
                                    (height, width), members, pairs, weights[name], torch.ones(len(ids), device=valid.device, dtype=torch.bool))
                                requested[name] = potential.gather(1, pairs[:, :1])[:, 0]-potential.gather(1, pairs[:, 1:])[:, 0]
                        for method in methods:
                            if method in requested:
                                action = project_pair_action(requested[method], target)
                                potential = pair_field(action, pairs, bank.class_count)
                                if method == PRIMARY:
                                    active = requested[method] != 0
                                    totals[p]['retained_action_fraction'] += float((action[active] != 0).double().mean()) if bool(active.any()) else 1.
                                    totals[p]['potential_mean'] += float(potential.abs().mean())
                            elif method == UNPROJECTED:
                                potential = pair_field(requested[PRIMARY], pairs, bank.class_count)
                            elif method == BUDGET_ONLY:
                                action = target.sign()*torch.minimum(requested[PRIMARY].abs(), target.abs())
                                potential = pair_field(action, pairs, bank.class_count)
                            elif method == PAIR_OBSERVATION:
                                potential = pair_field(target*.5, pairs, bank.class_count)
                            elif method == OBSERVATION_MEAN:
                                potential = innovation*.5
                            else:
                                continue
                            values[method] = values[BASELINE]+operator[:, ids]@potential
                        totals[p]['active_weight_fraction'] += float((weights[PRIMARY] != 1).float().mean())
                for method in methods:
                    value = values.get(method, values[BASELINE])
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeError('Nonfinite projected detail1 readout.')
                    dense = F.interpolate(value.T.reshape(1, bank.class_count, 32, 32), (512, 512),
                        mode='bilinear', align_corners=False)[0]
                    if method == 'Geometry':
                        dense = dense/.07
                    ah, aw = min(512, height-top), min(512, width-left)
                    accumulators[p, method].add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], left, top)
                totals[p]['tiles'] += 1
                totals[p]['observed_fraction'] += float(known[valid].float().mean())
        predictions = {p: {m: accumulators[p, m].finalize_resized(tuple(image.shape[-2:])) for m in methods} for p in banks}
    for row in totals.values():
        for key in ('observed_fraction', 'active_weight_fraction', 'retained_action_fraction', 'potential_mean'):
            row[key] /= row['tiles']
        row.update(geometry_encodings=local_branch.calls, wide_encodings=wide_branch.calls,
                   fine_forwards=detail_branch.calls, additional_visual_forwards=detail_branch.calls,
                   selected_detail_regions=len(positions), detail_geometry_forwards=0)
    if detail_branch.calls != len(positions) or detail_branch.calls > 1:
        raise RuntimeError('Actual whole-image detail1 budget violated.')
    return predictions, totals
