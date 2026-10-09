"""Reuse the real Geometry backbone grid as a local semantic witness.

The additional head is the attributed matched VIPProxy_Two adaptation, not a
new proxy operator. No cropped-token or fine-RGB reconstruction is claimed.
"""
from contextlib import ExitStack

import torch
import torch.nn.functional as F

from .alias_action_capacity import reconstruction_operator
from .alias_budget_attribution import pooled_class_risk, uniform_class_action
from .bounded_patch_only import PRIMARY as BASELINE, PROTOCOL as BASE_PROTOCOL, readout
from .bounded_physical_coupling import _BoundedBranch, geometry_windows, resize_geometry
from .device_probability_accumulator import DeviceProbabilityAccumulator
from .gear_ov import _crop_at
from .inference import hann_blend_window
from .matched_contribution_alias import contribution_margins
from .matched_readout_controls import run_head
from .native_alias_noise import signed_potential
from .native_query_alias import native_risk
from .fine_alias_view import CONFIG
from .one_sided_alias_stream import fixed_slot_stream
from .rival_alias_count import canonical_indices
from .rival_alias_fast import cache_observations, sampled_cached, risk_statistics


IMPLEMENTATION = 'geometry-shared-local-proxy-soft-alias-v1-20261008'
PRIMARY = 'SharedLocal_Soft'
MEAN = 'SharedLocal_ObservationMean'
POOLED = 'SharedLocal_PooledClass'
SHUFFLED = 'SharedLocal_AliasShuffle'
METHODS = ('Geometry', 'NoAdmission_Exact', BASELINE, PRIMARY, MEAN, POOLED, SHUFFLED)
PROTOCOL = {**BASE_PROTOCOL, 'fine_forwards': 0,
    'semantic_source': 'same complete512 Geometry backbone tokens; attributed frozen VIPProxy_Two semantic head',
    'extra_backbone_encodings': 0, 'maximum_extra_semantic_heads': 4,
    'alias_admission': 'existing one-sided wide-positive/local-negative N/(P+N), canonical protected',
    'aggregation': 'fixed20 outside-exponent attenuation, no survivor redistribution',
    'writer': 'equal wide/local observation and half word potential through original H',
    'text': 'same ImageNet template bank; preaverage embeddings in FP32 before linear dot product',
    'fitted_parameters': 0, 'head_attribution': 'VIPProxy_Two is inherited, not a new operator claim',
    'limitation': 'local16-pixel backbone grid is not the old physical8 fine observation'}


def local_evidence(features, query, members):
    """Algebraic template mean; avoids a [patch,alias,template] intermediate."""
    text = query.features.float().mean(1)
    feature = features.float()[0]
    logits = (feature @ text.T)*40.
    mean = F.normalize(feature.mean(0), dim=-1)
    salience = mean @ F.normalize(text, dim=-1).T
    weights = salience[members].softmax(-1)*members.shape[1]
    return logits[:, members]*weights


@torch.inference_mode()
def shared_scores(local, operator, broad, evidence, wide, coordinates, valid,
                  members, canonical, parents, methods):
    baseline = local.double()+operator.double()@(broad.double()-local.double())
    witness = evidence.logsumexp(-1)
    innovation = .5*(witness.double()-broad.double())
    innovation.masked_fill_(~valid[:, None], 0.)
    positive = baseline+operator.double()@innovation
    values = {MEAN: positive} if MEAN in methods else {}
    stats = dict(canonical_risk_max=0., mean_absolute_admission_potential=0.,
                 normal_equation_max_error=0., gauge_max_error=0.,
                 retained_count_mean=float(members.shape[1]), deleted_alias_rival_fraction=0.)
    if set(methods) == {MEAN}:
        return values, stats
    if set(methods) & {PRIMARY, SHUFFLED}:
        wm = sampled_cached(wide, coordinates)
        lm = contribution_margins(evidence).masked_fill(~valid[:, None, None], 0.)
        risk = native_risk(wm, lm, lm, parents, canonical, valid, CONFIG)
        action, _ = fixed_slot_stream(wide, members, risk, valid)
        delta, consistency = signed_potential(action, valid)
        if PRIMARY in methods:
            values[PRIMARY] = positive+.5*(operator.double()@delta)
        stats.update({**risk_statistics(risk, delta, valid, members, canonical), **consistency})
        if SHUFFLED in methods:
            changed = risk.clone()
            generator = torch.Generator().manual_seed(20261008)
            for group in members:
                ids = group[~torch.isin(group, canonical)]
                permutation = torch.randperm(len(ids), generator=generator).to(ids.device)
                changed[:, ids] = risk[:, ids[permutation]]
            action, _ = fixed_slot_stream(wide, members, changed, valid)
            delta, _ = signed_potential(action, valid)
            values[SHUFFLED] = positive+.5*(operator.double()@delta)
    if POOLED in methods:
        risk = pooled_class_risk(broad, witness, valid)
        action = uniform_class_action(wide, members, canonical, risk, valid)
        delta, _ = signed_potential(action, valid)
        values[POOLED] = positive+.5*(operator.double()@delta)
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite shared-local prediction.')
    return values, stats


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS, cache=None):
    from benchmark_grouped_class_readout import CheckedGroupedCache
    from eval_geometry_vip_reliability import sample_broad
    from eval_rival_fine_full import prepare_wide, tile_coordinates
    from eval_sparse_alias_reuse import forbid_fine

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared shared-local arms required.')
    cache = CheckedGroupedCache() if cache is None else cache
    cache.verify = False
    resized = resize_geometry(image)
    height, width = resized.shape[-2:]
    local_branch = _BoundedBranch(geometry, 512, 4, 'prepare_image')
    wide_branch = _BoundedBranch(vip, 336, 4, 'crop_patch_features')
    wide, crops, _, count = prepare_wide(image, wide_branch, {'clean': (banks, queries)})
    texts = {p: F.normalize(bank.features.float(), dim=-1) for p, bank in banks.items()}
    active = tuple(m for m in methods if m in METHODS[3:])
    totals = {p: dict(tiles=0, canonical_risk_max=0., mean_absolute_admission_potential=0.,
        normal_equation_max_error=0., gauge_max_error=0., retained_count_mean=0.,
        deleted_alias_rival_fraction=0., extra_semantic_heads=0.) for p in banks}
    blend = torch.from_numpy(hann_blend_window(512)).to(geometry.device)
    with forbid_fine(), ExitStack() as stack:
        accumulators = {(p, m): stack.enter_context(DeviceProbabilityAccumulator(
            bank.class_count, height, width, geometry.device)) for p, bank in banks.items() for m in methods}
        for top, left in geometry_windows(height, width):
            prepared = local_branch.prepare_image(_crop_at(resized, top, left, 512).to(geometry.device))
            coordinates = tile_coordinates(top, left, geometry.device)
            valid = (coordinates[:, 0] < height) & (coordinates[:, 1] < width)
            operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
            head = geometry.backbone.model.visual_model.head
            with geometry.backbone._autocast():
                features = readout(head, prepared, 2.)[0]
                if active:
                    witness, _ = run_head(head, prepared.backbone_tokens,
                        prepared.backbone_tokens[:, prepared.prefix_tokens:],
                        prepared.geometry_patch_conditional, prepared.prefix_tokens,
                        'VIPProxy_Two', prepared.block_index)
            ah, aw = min(512, height-top), min(512, width-left)
            for p, bank in banks.items():
                raw = cache.geometry((prepared.geometry_projected.float()@texts[p].T)[0], bank.parent_indices, bank.class_count)
                local = cache.geometry((features.float()@texts[p].T)[0], bank.parent_indices, bank.class_count)/.07
                broad = sample_broad(wide['clean', p], top, left, height, width).reshape_as(local)
                original = (raw/.07).double()
                values = {'Geometry': raw, 'NoAdmission_Exact': original+operator@(broad.double()-original),
                          BASELINE: local.double()+operator@(broad.double()-local.double())}
                if active:
                    query = queries[p]
                    members = torch.stack([(query.parents == c).nonzero().flatten() for c in range(bank.class_count)])
                    canonical = canonical_indices(query.class_names, query.aliases, query.parents)
                    evidence = local_evidence(witness, query, members)
                    observations = () if set(active) == {MEAN} else cache_observations(
                        crops['clean', p], count, coordinates, (height, width), members)
                    changed, stats = shared_scores(local, operator, broad, evidence, observations,
                        coordinates, valid, members, canonical, query.parents, active)
                    values.update(changed)
                    for field, value in stats.items():
                        totals[p][field] += value
                    totals[p]['extra_semantic_heads'] += 1
                for method in methods:
                    dense = F.interpolate(values[method].T.reshape(1, bank.class_count, 32, 32), (512, 512),
                                          mode='bilinear', align_corners=False)[0]
                    if method == 'Geometry':
                        dense = dense/.07
                    accumulators[p, method].add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], left, top)
                totals[p]['tiles'] += 1
        predictions = {p: {m: accumulators[p, m].finalize_resized(tuple(image.shape[-2:])) for m in methods} for p in banks}
    for row in totals.values():
        for field in row:
            if field not in ('tiles', 'extra_semantic_heads'):
                row[field] /= row['tiles']
        row.update(geometry_encodings=local_branch.calls, wide_encodings=wide_branch.calls,
                   fine_forwards=0., additional_visual_forwards=0., native_resolution_encodings=0.)
    if local_branch.calls > 4 or wide_branch.calls > 4:
        raise RuntimeError('Backbone observation budget violated.')
    return predictions, totals
