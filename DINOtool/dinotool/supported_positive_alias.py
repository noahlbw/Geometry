"""Joint positive observation and Geometry-supported competitive alias action."""
from contextlib import ExitStack

import torch
import torch.nn.functional as F

from .alias_action_capacity import reconstruction_operator
from .bounded_fine_coverage import FineCoverageReader, alias_controls, fine_class_observation
from .bounded_patch_only import PRIMARY as BASELINE, PROTOCOL as BASE_PROTOCOL, readout
from .bounded_physical_coupling import _BoundedBranch, geometry_windows, resize_geometry
from .device_probability_accumulator import DeviceProbabilityAccumulator
from .fine_alias_view import CONFIG
from .gear_ov import _crop_at
from .inference import hann_blend_window
from .native_alias_noise import signed_potential
from .native_query_alias import native_risk, query_relation, support_margins
from .rival_alias_count import canonical_indices
from .rival_alias_fast import cache_observations, directed_cached, risk_statistics, sampled_cached
from .rival_fine_full import tile_reference_coordinates


IMPLEMENTATION = 'geometry-bounded896-supported-positive-alias-v1-20261006'
PRIMARY = 'SupportedPositive_Hard'
OBSERVATION_MEAN = 'SupportedPositive_ObservationMean'
CLASS_MEAN = 'SupportedPositive_ClassMean'
SHUFFLES = tuple('SupportedPositive_AliasShuffle'+str(i) for i in range(3))
NO_GEOMETRY_RISK = 'SupportedPositive_NoGeometryRisk'
WIDE_ONLY = 'SupportedPositive_WideOnly'
SHUFFLED_WRITE = 'SupportedPositive_ShuffledWrite'
METHODS = ('Geometry', 'NoAdmission_Exact', BASELINE, PRIMARY, OBSERVATION_MEAN,
           CLASS_MEAN, *SHUFFLES, NO_GEOMETRY_RISK, WIDE_ONLY, SHUFFLED_WRITE)
PROTOCOL = {**BASE_PROTOCOL,
    'fine_forwards': 16, 'maximum_fine_encodings_per_image': 16,
    'fine_source': 'four256-to512 quadrants per bounded Geometry tile; resized896 RGB, not native-image detail',
    'fine_reader': 'unchanged finite VIP proxy; batch-one exact cached-burst execution',
    'semantic_source': 'positive equal wide/fine observation plus competitive wide alias intervention',
    'alias_admission': 'wide advantage contradicted by fine query AND original-G supported fine margin',
    'canonical_protection': True, 'fitted_parameters': 0,
    'writer': 'half wide intervention plus positive fine innovation through unchanged original H',
    'positive_observation_fraction': .5,
    'controls': (OBSERVATION_MEAN, CLASS_MEAN, *SHUFFLES, NO_GEOMETRY_RISK, WIDE_ONLY, SHUFFLED_WRITE)}


def cached_classes(observations, coordinates):
    output = coordinates.new_zeros((len(coordinates), observations[0].evidence.shape[1]))
    for crop in observations:
        classes = crop.evidence.logsumexp(-1)
        output += (classes[crop.indices]*crop.coefficients[..., None]).sum(1)
    return output


def supported_risk(wide, fine, relation, valid, parents, canonical, *, use_geometry=True):
    weights, known = query_relation(relation, valid)
    supported = support_margins(weights, fine) if use_geometry else fine
    return native_risk(wide, fine, supported, parents, canonical, known, CONFIG)


def class_only_risk(risk, members, canonical):
    from .bounded_fine_coverage import CLASS_MEAN as OLD_CLASS_MEAN

    weights = alias_controls(risk, members, canonical)[OLD_CLASS_MEAN]
    changed = torch.zeros_like(risk, dtype=torch.float64)
    changed[:, members.flatten()] = (1-weights).reshape_as(changed)
    return changed


def permuted_operator(operator, valid):
    ids = valid.nonzero().flatten()
    generator = torch.Generator().manual_seed(CONFIG.random_seed)
    permutation = torch.arange(len(valid), device=valid.device)
    permutation[ids] = ids[torch.randperm(len(ids), generator=generator).to(valid.device)]
    return operator[permutation][:, permutation]


@torch.inference_mode()
def joint_scores(local, operator, broad, fine_field, wide, fine, coordinates,
                 relation, valid, members, canonical, parents, *, methods=METHODS[3:]):
    if not methods or not set(methods).issubset(METHODS[3:]):
        raise ValueError('Declared positive-observation/alias methods required.')
    baseline = local.double()+operator.double()@(broad.double()-local.double())
    innovation = .5*(fine_field.double()-broad.double())
    innovation.masked_fill_(~valid[:, None], 0.)
    positive = baseline+operator.double()@innovation
    values = {OBSERVATION_MEAN: positive} if OBSERVATION_MEAN in methods else {}
    if set(methods) == {OBSERVATION_MEAN}:
        return values, dict(canonical_risk_max=0., mean_absolute_admission_potential=0.,
                            retained_count_mean=float(members.shape[1]), deleted_alias_rival_fraction=0.,
                            normal_equation_max_error=0., gauge_max_error=0.)
    wide_margins = sampled_cached(wide, coordinates)
    fine_margins = sampled_cached(fine, coordinates).masked_fill(~valid[:, None, None], 0.)
    risk = supported_risk(wide_margins, fine_margins, relation, valid, parents, canonical)
    directed = directed_cached(wide, members, risk, valid)
    potential, consistency = signed_potential(directed, valid)
    if PRIMARY in methods:
        values[PRIMARY] = positive+.5*(operator.double()@potential)
    if WIDE_ONLY in methods:
        values[WIDE_ONLY] = baseline+operator.double()@potential
    if SHUFFLED_WRITE in methods:
        values[SHUFFLED_WRITE] = positive+.5*(permuted_operator(operator, valid).double()@potential)
    if NO_GEOMETRY_RISK in methods:
        ungrounded = supported_risk(wide_margins, fine_margins, relation, valid,
                                    parents, canonical, use_geometry=False)
        delta, _ = signed_potential(directed_cached(wide, members, ungrounded, valid), valid)
        values[NO_GEOMETRY_RISK] = positive+.5*(operator.double()@delta)
    if CLASS_MEAN in methods:
        changed = class_only_risk(risk, members, canonical)
        delta, _ = signed_potential(directed_cached(wide, members, changed, valid, 'weighted'), valid)
        values[CLASS_MEAN] = positive+.5*(operator.double()@delta)
    if any(name in methods for name in SHUFFLES):
        from .bounded_fine_coverage import SHUFFLES as OLD_SHUFFLES

        allocations = alias_controls(risk, members, canonical)
        for name, old in zip(SHUFFLES, OLD_SHUFFLES):
            if name in methods:
                delta, _ = signed_potential(directed_cached(wide, members, allocations[old], valid), valid)
                values[name] = positive+.5*(operator.double()@delta)
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite joint competitive scores.')
    return values, {**risk_statistics(risk, potential, valid, members, canonical), **consistency}


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS,
                  cache=None, execution=None, tile_scorer=None, allowed_methods=None,
                  observation_method=None, diagnostic_fields=(), observation_cacher=None):
    from benchmark_grouped_class_readout import CheckedGroupedCache
    from eval_geometry_vip_reliability import sample_broad
    from eval_rival_fine_full import observe_fine, prepare_wide, tile_coordinates

    allowed_methods = METHODS if allowed_methods is None else allowed_methods
    observation_method = OBSERVATION_MEAN if observation_method is None else observation_method
    tile_scorer = joint_scores if tile_scorer is None else tile_scorer
    observation_cacher = cache_observations if observation_cacher is None else observation_cacher
    if (tuple(allowed_methods[:3]) != METHODS[:3] or observation_method not in allowed_methods
            or not methods or not set(methods).issubset(allowed_methods)):
        raise ValueError('Declared supported-positive methods required.')
    extra = tuple(name for name in methods if name in allowed_methods[3:])
    cache = CheckedGroupedCache() if cache is None else cache
    cache.verify = False
    resized = resize_geometry(image)
    height, width = resized.shape[-2:]
    local_branch = _BoundedBranch(geometry, 512, 4, 'prepare_image')
    wide_branch = _BoundedBranch(vip, 336, 4, 'crop_patch_features')
    reader = (FineCoverageReader(vip) if execution is None else execution.reader(vip)) if extra else None
    wide, crops, _, count = prepare_wide(image, wide_branch, {'clean': (banks, queries)})
    texts = {p: F.normalize(bank.features.float(), dim=-1) for p, bank in banks.items()}
    totals = {p: dict(tiles=0, retained_count_mean=0., deleted_alias_rival_fraction=0.,
        canonical_risk_max=0., mean_absolute_admission_potential=0., normal_equation_max_error=0.,
        gauge_max_error=0., fine_coverage_max_error=0.) for p in banks}
    for row in totals.values():
        row.update(dict.fromkeys(diagnostic_fields, 0.))
    blend = torch.from_numpy(hann_blend_window(512)).to(geometry.device)
    with ExitStack() as stack:
        accumulators = {(p, m): stack.enter_context(DeviceProbabilityAccumulator(
            bank.class_count, height, width, geometry.device)) for p, bank in banks.items() for m in methods}
        for top, left in geometry_windows(height, width):
            prepared = local_branch.prepare_image(_crop_at(resized, top, left, 512).to(geometry.device))
            coordinates = tile_coordinates(top, left, geometry.device)
            fine_coordinates = tile_reference_coordinates(coordinates, top, left)
            valid = (coordinates[:, 0] < height) & (coordinates[:, 1] < width)
            relation = prepared.geometry_patch_conditional[0]
            operator, _ = reconstruction_operator(relation, valid)
            with geometry.backbone._autocast():
                features = readout(geometry.backbone.model.visual_model.head, prepared, 2.)[0]
            if extra:
                fine, fine_count, costs = observe_fine(resized[:, top:top+512, left:left+512],
                    vip, queries, fine_coordinates, valid, feature_batch=reader)
            for p, bank in banks.items():
                raw = cache.geometry((prepared.geometry_projected.float()@texts[p].T)[0], bank.parent_indices, bank.class_count)
                local = cache.geometry((features.float()@texts[p].T)[0], bank.parent_indices, bank.class_count)/.07
                broad = sample_broad(wide['clean', p], top, left, height, width).reshape_as(local)
                original = (raw/.07).double()
                baseline = local.double()+operator@(broad.double()-local.double())
                values = {'Geometry': raw, 'NoAdmission_Exact': original+operator@(broad.double()-original), BASELINE: baseline}
                if extra:
                    query = queries[p]
                    members = torch.stack([(query.parents == c).nonzero().flatten() for c in range(bank.class_count)])
                    canonical = canonical_indices(query.class_names, query.aliases, query.parents)
                    if set(extra) == {observation_method}:
                        wide_cache = fine_cache = ()
                        fine_field = fine_class_observation(fine[p], fine_count, fine_coordinates, members)
                    else:
                        wide_cache = observation_cacher(crops['clean', p], count, coordinates, (height, width), members)
                        fine_cache = observation_cacher(fine[p], fine_count, fine_coordinates, (512, 512), members)
                        fine_field = cached_classes(fine_cache, fine_coordinates)
                    changed, stats = tile_scorer(local, operator, broad, fine_field, wide_cache, fine_cache,
                        coordinates, relation, valid, members, canonical, query.parents, methods=extra)
                    values.update(changed)
                    for field in totals[p]:
                        if field != 'tiles':
                            totals[p][field] += costs.get(field, stats.get(field, 0.))
                ah, aw = min(512, height-top), min(512, width-left)
                for method in methods:
                    dense = F.interpolate(values[method].T.reshape(1, bank.class_count, 32, 32), (512, 512),
                                          mode='bilinear', align_corners=False)[0]
                    if method == 'Geometry':
                        dense = dense/.07
                    accumulators[p, method].add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], left, top)
                totals[p]['tiles'] += 1
        predictions = {p: {m: accumulators[p, m].finalize_resized(tuple(image.shape[-2:])) for m in methods} for p in banks}
    fine_calls = 0 if reader is None else reader.calls
    if (local_branch.calls != len(geometry_windows(height, width)) or not 1 <= wide_branch.calls <= 4
            or not (1 <= fine_calls <= 4*local_branch.calls <= 16 if extra else fine_calls == 0)):
        raise RuntimeError('Actual whole-image observation cap violated.')
    for row in totals.values():
        for field in row:
            if field != 'tiles':
                row[field] /= row['tiles']
        row.update(geometry_encodings=local_branch.calls, wide_encodings=wide_branch.calls,
            fine_forwards=fine_calls, additional_visual_forwards=fine_calls, native_resolution_encodings=0)
    return predictions, totals
