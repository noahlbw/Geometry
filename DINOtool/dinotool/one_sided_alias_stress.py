"""Vocabulary stress of the unchanged one-sided rule, sharing visual observations."""
from contextlib import ExitStack

import torch
import torch.nn.functional as F

from .alias_action_capacity import reconstruction_operator
from .bounded_patch_only import readout
from .bounded_physical_coupling import _BoundedBranch, geometry_windows, resize_geometry
from .device_probability_accumulator import DeviceProbabilityAccumulator
from .gear_ov import _crop_at
from .inference import hann_blend_window
from .one_sided_alias_audit import (DIAGNOSTICS, HARD, MATCHED_SHUFFLES,
    OBSERVATION_MEAN, PRIMARY, scores)
from .rival_alias_count import canonical_indices
from .rival_alias_fast import cache_observations
from .rival_fine_full import tile_reference_coordinates
from .supported_positive_alias import cached_classes


IMPLEMENTATION = 'geometry-bounded896-existing-one-sided-vocabulary-stress-v1-20261006'
SCENARIOS = ('k20', 'k30', 'k40', 'wrong_parent', 'paraphrase')
METHODS = ('Geometry', 'NoAdmission_Exact', 'Geometry_PatchOnly2Coupled',
    PRIMARY, OBSERVATION_MEAN, HARD, 'OneSide_ClassMean', 'OneSide_MatchedClassMean',
    *MATCHED_SHUFFLES, 'OneSide_MatchedRivalCollapsed', 'OneSide_MatchedShuffledWrite')


def validate_variants(banks, variants, vocabulary):
    if tuple(variants) != SCENARIOS or set(vocabulary) != set(SCENARIOS):
        raise ValueError('All five frozen vocabulary scenarios required.')
    for scenario, (group, queries) in variants.items():
        if set(group) != set(banks) or set(queries) != set(banks):
            raise ValueError('All local protocols must be preserved.')
        count = int(scenario[1:]) if scenario.startswith('k') else 20
        for protocol, bank in banks.items():
            query, classes = queries[protocol], vocabulary[scenario][protocol]
            if (group[protocol] is not bank or list(query.class_names) != list(bank.class_names)
                    or list(query.class_names) != [c['name'] for c in classes]
                    or list(query.aliases) != [a for c in classes for a in c['synonyms']]
                    or any(len(c['synonyms']) != count for c in classes)
                    or any(int((query.parents == c).sum()) != count for c in range(bank.class_count))):
                raise ValueError('Frozen local bank, context words, parents or counts changed.')
            canonical_indices(query.class_names, query.aliases, query.parents)
            if scenario.startswith('k'):
                if any(c['synonyms'] != vocabulary['k40'][protocol][i]['synonyms'][:count]
                       for i, c in enumerate(classes)):
                    raise ValueError('Exact historical nested prefixes required.')


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, variants, *, execution, cache=None,
                  methods=METHODS):
    from benchmark_grouped_class_readout import CheckedGroupedCache
    from eval_geometry_vip_reliability import sample_broad
    from eval_rival_fine_full import observe_fine, prepare_wide, tile_coordinates

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared unchanged one-sided stress methods required.')
    extra = tuple(m for m in methods if m in METHODS[3:])
    if not extra:
        raise ValueError('Stress study requires the same fine-information route.')
    cache = CheckedGroupedCache() if cache is None else cache
    cache.verify = False
    resized = resize_geometry(image)
    height, width = resized.shape[-2:]
    local_branch = _BoundedBranch(geometry, 512, 4, 'prepare_image')
    wide_branch = _BoundedBranch(vip, 336, 4, 'crop_patch_features')
    reader = execution.reader(vip)
    wide, crops, _, count = prepare_wide(image, wide_branch, variants)
    flat_queries = {s+'__'+p: q for s, (_, qs) in variants.items() for p, q in qs.items()}
    texts = {p: F.normalize(bank.features.float(), dim=-1) for p, bank in banks.items()}
    fields = (*DIAGNOSTICS, 'canonical_risk_max', 'mean_absolute_admission_potential',
              'normal_equation_max_error', 'gauge_max_error', 'fine_coverage_max_error')
    totals = {s: {p: dict(tiles=0, **dict.fromkeys(fields, 0.)) for p in banks} for s in variants}
    blend = torch.from_numpy(hann_blend_window(512)).to(geometry.device)
    with ExitStack() as stack:
        accumulators = {(s, p, m): stack.enter_context(DeviceProbabilityAccumulator(
            bank.class_count, height, width, geometry.device))
            for s in variants for p, bank in banks.items() for m in methods}
        for top, left in geometry_windows(height, width):
            prepared = local_branch.prepare_image(_crop_at(resized, top, left, 512).to(geometry.device))
            coordinates = tile_coordinates(top, left, geometry.device)
            fine_coordinates = tile_reference_coordinates(coordinates, top, left)
            valid = (coordinates[:, 0] < height) & (coordinates[:, 1] < width)
            relation = prepared.geometry_patch_conditional[0]
            operator, _ = reconstruction_operator(relation, valid)
            with geometry.backbone._autocast():
                features = readout(geometry.backbone.model.visual_model.head, prepared, 2.)[0]
            fine, fine_count, costs = observe_fine(resized[:, top:top+512, left:left+512],
                vip, flat_queries, fine_coordinates, valid, feature_batch=reader)
            for p, bank in banks.items():
                raw = cache.geometry((prepared.geometry_projected.float()@texts[p].T)[0],
                                     bank.parent_indices, bank.class_count)
                local = cache.geometry((features.float()@texts[p].T)[0],
                                       bank.parent_indices, bank.class_count)/.07
                for s, (_, queries) in variants.items():
                    query = queries[p]
                    members = torch.stack([(query.parents == c).nonzero().flatten()
                                           for c in range(bank.class_count)])
                    canonical = canonical_indices(query.class_names, query.aliases, query.parents)
                    broad = sample_broad(wide[s, p], top, left, height, width).reshape_as(local)
                    values = {'Geometry': raw,
                        'NoAdmission_Exact': (raw/.07).double()+operator@(broad.double()-(raw/.07).double()),
                        METHODS[2]: local.double()+operator@(broad.double()-local.double())}
                    wc = cache_observations(crops[s, p], count, coordinates, (height, width), members)
                    fc = cache_observations(fine[s+'__'+p], fine_count, fine_coordinates, (512, 512), members)
                    fine_field = cached_classes(fc, fine_coordinates)
                    changed, stats = scores(local, operator, broad, fine_field, wc, fc,
                        coordinates, relation, valid, members, canonical, query.parents, methods=extra)
                    values.update(changed)
                    row = totals[s][p]
                    for field in fields:
                        row[field] += costs.get(field, stats.get(field, 0.))
                    row['tiles'] += 1
                    ah, aw = min(512, height-top), min(512, width-left)
                    for m in methods:
                        dense = F.interpolate(values[m].T.reshape(1, bank.class_count, 32, 32),
                            (512, 512), mode='bilinear', align_corners=False)[0]
                        if m == 'Geometry':
                            dense = dense/.07
                        accumulators[s, p, m].add(dense[:, :ah, :aw].softmax(0).float(),
                                                 blend[:ah, :aw], left, top)
        predictions = {s: {p: {m: accumulators[s, p, m].finalize_resized(tuple(image.shape[-2:]))
                              for m in methods} for p in banks} for s in variants}
    if (local_branch.calls != len(geometry_windows(height, width))
            or not 1 <= wide_branch.calls <= 4 or not 1 <= reader.calls <= 4*local_branch.calls <= 16):
        raise RuntimeError('Five vocabularies must share the actual4/4/16 visual budget.')
    for group in totals.values():
        for row in group.values():
            for field in fields:
                row[field] /= row['tiles']
            row.update(geometry_encodings=local_branch.calls, wide_encodings=wide_branch.calls,
                       fine_forwards=reader.calls, native_resolution_encodings=0)
    return predictions, totals
