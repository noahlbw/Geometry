"""Bounded independent semantic witness for joint fixed-slot alias use."""
from contextlib import ExitStack
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .alias_action_capacity import reconstruction_operator
from .bounded_canonical_pair_alias import weight_controls
from .bounded_patch_only import PRIMARY as BASELINE, PROTOCOL as BASE_PROTOCOL, readout
from .bounded_physical_coupling import _BoundedBranch, geometry_windows, resize_geometry
from .device_probability_accumulator import DeviceProbabilityAccumulator
from .gear_ov import _crop_at
from .inference import hann_blend_window
from .rival_alias_count import canonical_indices
from .stratified_soft_alias import crop_stencil


IMPLEMENTATION = 'geometry-bounded896-siglip256-joint-fixedslots-v1-20261006'
PRIMARY = 'SigWitness_JointSoft'
CLASS_MEAN = 'SigWitness_ClassMean'
SHUFFLE = 'SigWitness_AliasShuffle'
NO_GEOMETRY = 'SigWitness_NoGeometrySupport'
OBSERVATION_MEAN = 'SigWitness_ObservationMean'
METHODS = ('Geometry', 'NoAdmission_Exact', BASELINE, PRIMARY, CLASS_MEAN,
           SHUFFLE, NO_GEOMETRY, OBSERVATION_MEAN)
QUERY_BUDGET = 128
SOURCE = Path('/data/test/code/ovss_cafe_ped_v2_20260919/ckpt/SigLIP2_base256_region_20261001')
REPOSITORY = 'google/siglip2-base-patch16-256'
REVISION = '3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab'
SOURCE_SHA = '6125cacc01fa93bdc98a0c5101cefcd69b2ed1f8ab4f38d86f4ad5984f5dc863'
PROTOCOL = {**BASE_PROTOCOL,
    'alias_admission': 'independent SigLIP2 rival contradiction; joint local/wide fixed-slot attenuation',
    'semantic_source': dict(repository=REPOSITORY, revision=REVISION, weights_sha256=SOURCE_SHA,
        extra_full_image_encoding=1, input=[256, 256], patch=16, maximum_query_reads=QUERY_BUDGET),
    'selection': 'smallest baseline top2 probability gaps; global128 valid queries, shared across protocols',
    'support': 'original G donor rows projected through actual256/16 observer footprints; trained MAP pool',
    'weight': 'sigmoid(source own-alias/rival-LME margin)*sigmoid(-witness margin); canonical protected',
    'budget': 'sum of noncanonical attenuations at most one alias slot per query/class/source',
    'aggregation': 'weights outside exponent; original20 slots/salience kept, no survivor redistribution',
    'writer': '(I-H) local attenuation + H wide attenuation; original H unchanged',
    'additional_visual_forwards': 1, 'fine_forwards': 0, 'fitted_parameters': 0,
    'controls': (CLASS_MEAN, SHUFFLE, NO_GEOMETRY, OBSERVATION_MEAN),
    'limits': 'independent pretraining and Geometry support do not certify semantic correctness',
    'precedents': 'pretrained SigLIP2, MAP pooling, contradiction, attenuation and ridge writing are attributed; no priority claim'}


def project_support(relation_rows, top, left, image_size, valid):
    if relation_rows.ndim != 2 or relation_rows.shape[-1] != 1024 or valid.shape != (1024,):
        raise ValueError('Original32x32 donor rows and matching padding validity required.')
    height, width = image_size
    rows = relation_rows.float().masked_fill(~valid[None], 0.)
    yy = (torch.arange(256, device=rows.device)+.5)*height/256
    xx = (torch.arange(256, device=rows.device)+.5)*width/256
    y, x = torch.meshgrid(yy, xx, indexing='ij')
    grid = torch.stack((2*(x-left)/512-1, 2*(y-top)/512-1), -1)
    inside = ((y >= top) & (y < top+512) & (x >= left) & (x < left+512))
    sampled = F.grid_sample(rows.reshape(-1, 1, 32, 32), grid[None].expand(len(rows), -1, -1, -1),
                            mode='bilinear', padding_mode='border', align_corners=False)
    projected = F.avg_pool2d(sampled*inside[None, None], 16, stride=16).flatten(1)
    known = projected.sum(-1) > 0
    return projected, known


def cached_probe_pool(head, hidden, support):
    """Project fixed K/V once, then reuse them for all support-conditioned probes."""
    if hidden.shape[0] != 1 or hidden.ndim != 3 or support.ndim != 2 or support.shape[-1] != hidden.shape[1]:
        raise ValueError('One observer token field and matched query supports required.')
    if bool((support < 0).any()) or not bool(torch.isfinite(support).all()) or bool((support.sum(-1) <= 0).any()):
        raise ValueError('Finite nonempty support required.')
    attention = head.attention
    if (not attention._qkv_same_embed_dim or attention.bias_k is not None
            or attention.bias_v is not None or attention.add_zero_attn or attention.training):
        raise ValueError('Frozen native equal-dimension MAP attention required.')
    width = hidden.shape[-1]
    heads, channels = attention.num_heads, width//attention.num_heads
    q_weight, k_weight, v_weight = attention.in_proj_weight.split(width)
    biases = (None, None, None) if attention.in_proj_bias is None else attention.in_proj_bias.split(width)
    q = F.linear(head.probe, q_weight, biases[0]).reshape(1, heads, channels)[0]
    k = F.linear(hidden[0], k_weight, biases[1]).reshape(-1, heads, channels).transpose(0, 1)
    v = F.linear(hidden[0], v_weight, biases[2]).reshape(-1, heads, channels).transpose(0, 1)
    native = torch.einsum('hd,hnd->hn', q, k)/math.sqrt(channels)
    bias = support.float().clamp_min(1e-30).log().masked_fill(support == 0, -torch.inf)
    probabilities = (native[None]+bias[:, None].to(native.dtype)).softmax(-1)
    pooled = torch.einsum('rhn,hnd->rhd', probabilities, v).reshape(len(support), width)
    pooled = attention.out_proj(pooled)[:, None]
    return (pooled+head.mlp(head.layernorm(pooled)))[:, 0]


def contradiction_weights(source_margin, teacher_margin, pairs, canonical, known):
    if source_margin.shape != teacher_margin.shape or source_margin.shape[:2] != pairs.shape:
        raise ValueError('Matching source/witness alias margins and class pairs required.')
    risk = torch.sigmoid(source_margin.double())*torch.sigmoid(-teacher_margin.double())
    risk.masked_fill_(~known[:, None, None], 0.)
    risk.scatter_(-1, canonical[pairs][..., None], 0.)
    risk = risk/risk.sum(-1, keepdim=True).clamp_min(1.)
    return (1-risk).clamp_min(1e-6)


def fixed_local_delta(evidence, pairs, weights):
    rows = torch.arange(len(evidence), device=evidence.device)[:, None]
    selected = evidence[rows, pairs].double()
    delta = (selected+weights.double().log()).logsumexp(-1)-selected.logsumexp(-1)
    delta.masked_fill_((weights == 1).all(-1), 0.)
    return torch.zeros(evidence.shape[:2], device=evidence.device, dtype=torch.float64).scatter_(1, pairs, delta)


def sampled_wide(crops, count, coordinates, image_size, members, pairs, weights=None):
    classes = len(members)
    raw = coordinates.new_zeros((len(coordinates), classes, members.shape[-1]))
    delta = torch.zeros((len(coordinates), classes), device=coordinates.device, dtype=torch.float64)
    rows = torch.arange(len(coordinates), device=coordinates.device)[:, None, None]
    for crop in crops:
        ids, coefficients = crop_stencil(crop, count, coordinates, image_size)
        profile = crop.salience[members].softmax(-1)
        evidence = crop.alias_logits[:, members]*(profile/profile.mean(-1, keepdim=True))
        if weights is None:
            raw += (evidence[ids]*coefficients[..., None, None]).sum(1)
        else:
            probabilities = evidence.double().softmax(-1)[ids]
            selected = probabilities[rows, torch.arange(ids.shape[1], device=ids.device)[None, :, None], pairs[:, None]]
            change = (selected*weights[:, None]).sum(-1).clamp_min(1e-300).log().clamp_max(0.)
            change.masked_fill_((weights == 1).all(-1)[:, None], 0.)
            delta.scatter_add_(1, pairs, (change*coefficients.double()[..., None]).sum(1))
    return raw if weights is None else delta


def observer_for(geometry, banks):
    from .region_semantic_readout import FrozenRegionObserver

    identity = tuple((p, tuple(bank.class_names), tuple(bank.alias_names)) for p, bank in banks.items())
    saved = getattr(geometry, '_bounded_siglip_alias_observer', None)
    if saved is None:
        started = time.perf_counter()
        observer = FrozenRegionObserver(SOURCE, banks, geometry.device)
        expected = dict(repository=REPOSITORY, revision=REVISION, weights_sha256=SOURCE_SHA)
        if any(observer.manifest.get(k) != v for k, v in expected.items()):
            raise RuntimeError('Pinned independent semantic source changed.')
        if observer.model.config.vision_config.image_size != 256 or observer.model.config.vision_config.patch_size != 16:
            raise RuntimeError('Observer geometry differs from trained256/16.')
        saved = dict(observer=observer, identity=identity, setup_seconds=time.perf_counter()-started)
        geometry._bounded_siglip_alias_observer = saved
    if saved['identity'] != identity:
        raise ValueError('Observer belongs to different public class/alias banks.')
    return saved


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS, cache=None):
    from benchmark_grouped_class_readout import CheckedGroupedCache
    from eval_geometry_vip_reliability import sample_broad
    from eval_rival_fine_full import prepare_wide, tile_coordinates
    from eval_sparse_alias_reuse import forbid_fine

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared independent witness methods required.')
    cache = CheckedGroupedCache() if cache is None else cache
    cache.verify = False
    resized = resize_geometry(image)
    height, width = resized.shape[-2:]
    local_branch = _BoundedBranch(geometry, 512, 4, 'prepare_image')
    wide_branch = _BoundedBranch(vip, 336, 4, 'crop_patch_features')
    broad, crops, _, count = prepare_wide(image, wide_branch, {'clean': (banks, queries)})
    extra = bool(set(methods)-set(METHODS[:3]))
    texts = {p: F.normalize(bank.features.float(), dim=-1) for p, bank in banks.items()}
    tiles, priority = [], []
    with forbid_fine():
        for top, left in geometry_windows(height, width):
            prepared = local_branch.prepare_image(_crop_at(resized, top, left, 512).to(geometry.device))
            coordinates = tile_coordinates(top, left, geometry.device)
            valid = (coordinates[:, 0] < height) & (coordinates[:, 1] < width)
            operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
            with geometry.backbone._autocast():
                features = readout(geometry.backbone.model.visual_model.head, prepared, 2.)[0]
            fields, gaps = {}, []
            for p, bank in banks.items():
                raw = cache.geometry((prepared.geometry_projected.float()@texts[p].T)[0], bank.parent_indices, bank.class_count)
                aliases = (features.float()@texts[p].T)[0]
                local = cache.geometry(aliases, bank.parent_indices, bank.class_count)/.07
                wide = sample_broad(broad['clean', p], top, left, height, width).reshape_as(local)
                baseline = local.double()+operator@(wide.double()-local.double())
                original = (raw/.07).double()
                fields[p] = dict(local=local, wide=wide, aliases=aliases,
                    values={'Geometry': raw, 'NoAdmission_Exact': original+operator@(wide.double()-original), BASELINE: baseline})
                probabilities = baseline.softmax(-1).topk(2, -1).values
                gaps.append(probabilities[:, 0]-probabilities[:, 1])
            priority.append(torch.stack(gaps).amin(0).masked_fill(~valid, torch.inf))
            tiles.append(dict(top=top, left=left, coordinates=coordinates, valid=valid,
                              relation=prepared.geometry_patch_conditional[0], operator=operator, fields=fields))
        totals = {p: dict(tiles=len(tiles), witness_queries=0, active_weight_fraction=0.,
                          mean_local_action=0., mean_wide_action=0.) for p in banks}
        setup_seconds = 0.
        if extra:
            combined = torch.cat(priority)
            selected = combined.argsort(stable=True)[:min(QUERY_BUDGET, int(torch.isfinite(combined).sum()))]
            if not len(selected):
                raise RuntimeError('No valid semantic witness queries.')
            supports, locations = [], []
            for index, tile in enumerate(tiles):
                ids = selected[selected//1024 == index]%1024
                if not len(ids):
                    continue
                support, known = project_support(tile['relation'][ids], tile['top'], tile['left'], (height, width), tile['valid'])
                support[~known] = 1.
                supports.append(support)
                locations.append((index, ids, known))
            store = observer_for(geometry, banks)
            observer, setup_seconds = store['observer'], store['setup_seconds']
            hidden, _ = observer.visual(image[None].to(geometry.device))
            if hidden.shape[1] != 256:
                raise RuntimeError('Actual observer token count differs.')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                descriptors = cached_probe_pool(observer.model.vision_model.head, hidden, torch.cat(supports))
            descriptors = F.normalize(descriptors.float(), dim=-1)
            point_descriptors = None
            if NO_GEOMETRY in methods:
                coords = torch.cat([tiles[index]['coordinates'][ids] for index, ids, _ in locations])
                ys = (coords[:, 0]/height*16).long().clamp(0, 15)
                xs = (coords[:, 1]/width*16).long().clamp(0, 15)
                point = torch.zeros(len(coords), 256, device=geometry.device).scatter_(1, (ys*16+xs)[:, None], 1.)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    point_descriptors = F.normalize(cached_probe_pool(observer.model.vision_model.head, hidden, point).float(), dim=-1)
            offset = 0
            for index, ids, known in locations:
                tile = tiles[index]
                operator = tile['operator']
                for p, bank in banks.items():
                    fields = tile['fields'][p]
                    members = torch.stack([(bank.parent_indices == c).nonzero().flatten() for c in range(bank.class_count)])
                    if members.shape[-1] != 20:
                        raise ValueError('The frozen clean pool has exactly20 aliases/class.')
                    canonical_global = canonical_indices(bank.class_names, bank.alias_names, bank.parent_indices)
                    canonical = (members == canonical_global[:, None]).long().argmax(-1)
                    pairs = fields['values'][BASELINE][ids].topk(2, -1).indices
                    local_evidence = fields['aliases'][ids][:, members]/.07
                    wide_evidence = sampled_wide(crops['clean', p], count, tile['coordinates'][ids], (height, width), members, pairs)
                    rows = torch.arange(len(ids), device=ids.device)[:, None]
                    local_margin = local_evidence[rows, pairs]-fields['local'][ids].gather(1, pairs.flip(-1))[..., None]
                    wide_margin = wide_evidence[rows, pairs]-(fields['wide'][ids].gather(1, pairs.flip(-1))-math.log(20))[..., None]
                    semantic = (descriptors[offset:offset+len(ids)]@observer.texts[p].T)[:, members]/observer.semantic_temperature
                    teacher_classes = semantic.logsumexp(-1)-math.log(20)
                    teacher_margin = semantic[rows, pairs]-teacher_classes.gather(1, pairs.flip(-1))[..., None]
                    local_weights = contradiction_weights(local_margin, teacher_margin, pairs, canonical, known)
                    wide_weights = contradiction_weights(wide_margin, teacher_margin, pairs, canonical, known)
                    allocations = {PRIMARY: (local_weights, wide_weights)}
                    if set(methods) & {CLASS_MEAN, SHUFFLE}:
                        lc, ls = weight_controls(local_weights, pairs, canonical)
                        wc, ws = weight_controls(wide_weights, pairs, canonical)
                        allocations[CLASS_MEAN], allocations[SHUFFLE] = (lc, wc), (ls, ws)
                    if NO_GEOMETRY in methods:
                        point_evidence = (point_descriptors[offset:offset+len(ids)]@observer.texts[p].T)[:, members]/observer.semantic_temperature
                        point_class = point_evidence.logsumexp(-1)-math.log(20)
                        point_margin = point_evidence[rows, pairs]-point_class.gather(1, pairs.flip(-1))[..., None]
                        allocations[NO_GEOMETRY] = (contradiction_weights(local_margin, point_margin, pairs, canonical, known),
                                                   contradiction_weights(wide_margin, point_margin, pairs, canonical, known))
                    for method in methods:
                        if method in allocations:
                            lw, ww = allocations[method]
                            dl = torch.zeros_like(fields['values'][BASELINE])
                            db = torch.zeros_like(dl)
                            dl[ids] = fixed_local_delta(local_evidence, pairs, lw)
                            db[ids] = sampled_wide(crops['clean', p], count, tile['coordinates'][ids], (height, width), members, pairs, ww)
                            fields['values'][method] = fields['values'][BASELINE]+dl+operator@(db-dl)
                    if OBSERVATION_MEAN in methods:
                        change = torch.zeros_like(fields['values'][BASELINE])
                        change[ids] = .5*(teacher_classes.double()-fields['values'][BASELINE][ids])
                        fields['values'][OBSERVATION_MEAN] = fields['values'][BASELINE]+operator@change
                    totals[p]['witness_queries'] += len(ids)
                    totals[p]['active_weight_fraction'] += float(((local_weights < 1).double().mean()+(wide_weights < 1).double().mean())*.5)*len(ids)
                    totals[p]['mean_local_action'] += float(fixed_local_delta(local_evidence, pairs, local_weights).abs().mean())*len(ids)
                    totals[p]['mean_wide_action'] += float(sampled_wide(crops['clean', p], count, tile['coordinates'][ids], (height, width), members, pairs, wide_weights).abs().mean())*len(ids)
                offset += len(ids)
            for tile in tiles:
                for fields in tile['fields'].values():
                    for method in methods:
                        fields['values'].setdefault(method, fields['values'][BASELINE])
            if offset != len(selected) or offset > QUERY_BUDGET or any(p.requires_grad for p in observer.model.parameters()):
                raise RuntimeError('Actual global witness budget/frozen weights violated.')
        blend = torch.from_numpy(hann_blend_window(512)).to(geometry.device)
        with ExitStack() as stack:
            accumulators = {(p, m): stack.enter_context(DeviceProbabilityAccumulator(bank.class_count, height, width, geometry.device))
                            for p, bank in banks.items() for m in methods}
            for tile in tiles:
                ah, aw = min(512, height-tile['top']), min(512, width-tile['left'])
                for p, bank in banks.items():
                    for method in methods:
                        value = tile['fields'][p]['values'][method]
                        if not bool(torch.isfinite(value).all()):
                            raise RuntimeError('Nonfinite independent witness scores.')
                        dense = F.interpolate(value.T.reshape(1, bank.class_count, 32, 32), (512, 512), mode='bilinear', align_corners=False)[0]
                        if method == 'Geometry':
                            dense = dense/.07
                        accumulators[p, method].add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], tile['left'], tile['top'])
            predictions = {p: {m: accumulators[p, m].finalize_resized(tuple(image.shape[-2:])) for m in methods} for p in banks}
    for row in totals.values():
        for key in ('active_weight_fraction', 'mean_local_action', 'mean_wide_action'):
            row[key] /= max(row['witness_queries'], 1)
        row.update(geometry_encodings=local_branch.calls, wide_encodings=wide_branch.calls,
                   fine_forwards=0, additional_visual_forwards=int(extra), additional_semantic_heads=int(extra),
                   global_witness_query_budget=QUERY_BUDGET, observer_setup_seconds=setup_seconds)
    return predictions, totals
