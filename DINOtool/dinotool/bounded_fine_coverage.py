"""Bounded transfer diagnosis of the established four-quadrant fine witness."""
from contextlib import ExitStack
import math

import torch
import torch.nn.functional as F

from .alias_action_capacity import reconstruction_operator
from .bounded_patch_only import PRIMARY as BASELINE, PROTOCOL as BASE_PROTOCOL, readout
from .bounded_physical_coupling import _BoundedBranch, geometry_windows, resize_geometry
from .calibrated_competitive_alias import profiled_logits
from .device_probability_accumulator import DeviceProbabilityAccumulator
from .fine_alias_view import CONFIG
from .fine_reference_admission import fine_patch_features
from .gear_ov import _crop_at
from .inference import hann_blend_window
from .native_alias_noise import hard_pair_observation, signed_potential
from .rival_alias_count import canonical_indices
from .rival_fine_full import retained_scores, tile_reference_coordinates
from .stratified_soft_alias import crop_stencil


IMPLEMENTATION = 'geometry-bounded896-legacy-fine-coverage-diagnosis-v1-20261006'
PRIMARY = 'BoundedFineCoverage_Hard'
CLASS_MEAN = 'BoundedFineCoverage_ClassMean'
SHUFFLES = tuple('BoundedFineCoverage_AliasShuffle'+str(i) for i in range(3))
OBSERVATION_MEAN = 'BoundedFineCoverage_ObservationMean'
METHODS = ('Geometry', 'NoAdmission_Exact', BASELINE, PRIMARY, CLASS_MEAN,
           *SHUFFLES, OBSERVATION_MEAN)
PROTOCOL = {**BASE_PROTOCOL,
    'alias_admission': 'unchanged legacy query/class/rival positive-risk hard admission',
    'fine_forwards': 16, 'maximum_fine_encodings_per_image': 16,
    'fine_source': 'four256-to512 quadrants per bounded Geometry tile; resized896 RGB, not original RGB detail',
    'fine_reader': 'unchanged finite VIP proxy with original frozen fine configuration',
    'fine_execution': 'serial batch-one sync-free observer; no graph setup or additional model',
    'coordinates': 'fine pitch8 in bounded896 coordinates, NOT native-image physical8',
    'writer': 'unchanged all-rival signed potential through original undoubled Geometry H',
    'controls': (CLASS_MEAN, *SHUFFLES, OBSERVATION_MEAN),
    'purpose': 'source coverage/reader transfer diagnosis; not a novel final method',
    'fitted_parameters': 0, 'extra_backbone_encodings': 'at most16 fixed512 fine views per complete image'}


class FineCoverageReader:
    def __init__(self, observer, limit=16, feature_reader=None):
        if limit != 16:
            raise ValueError('The frozen whole-image fine budget is16.')
        self.observer, self.calls, self.limit = observer, 0, limit
        self.feature_reader = feature_reader

    def __call__(self, observer, crops):
        if (observer is not self.observer or not 1 <= len(crops) <= 4
                or any(rgb.shape != (3, 512, 512) for rgb in crops)
                or self.calls+len(crops) > self.limit):
            raise RuntimeError('Actual fine RGB size/count budget exceeded.')
        self.calls += len(crops)
        if self.feature_reader is not None:
            output = self.feature_reader(observer, crops)
            if len(output) != len(crops):
                raise RuntimeError('Execution backend changed the fine crop count.')
            return output
        return tuple(fine_patch_features(observer, rgb, capture_safe=True) for rgb in crops)


def alias_controls(risk, members, canonical):
    grouped = risk[:, members]
    noncanonical = members != canonical[:, None]
    keep = (grouped == 0).double()
    shared = keep.clone()
    for c in range(len(members)):
        slots = noncanonical[c].nonzero().flatten()
        shared[:, c, slots] = keep[:, c, slots].mean(1, keepdim=True)
    controls = {CLASS_MEAN: shared}
    for index, name in enumerate(SHUFFLES):
        generator = torch.Generator().manual_seed(CONFIG.random_seed+index)
        changed = risk.clone()
        for c, group in enumerate(members):
            ids = group[noncanonical[c]]
            permutation = torch.randperm(len(ids), generator=generator).to(risk.device)
            changed[:, ids] = risk[:, ids[permutation]]
        controls[name] = changed
    if not torch.allclose(shared.sum(2), keep.sum(2), atol=1e-12, rtol=0):
        raise RuntimeError('Class-only control changed the pairwise alias budget.')
    return controls


def weighted_observation(crops, count, coordinates, image_size, members, weights, valid):
    if (weights.shape != (len(valid), *members.shape, len(members))
            or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any())
            or bool((weights.sum(2) <= 0).any())):
        raise ValueError('Nonnegative pair-specific alias allocations with positive mass required.')
    output = torch.zeros((len(valid), len(members), len(members)), device=weights.device, dtype=torch.float64)
    for start in range(0, len(valid), CONFIG.query_chunk):
        sl = slice(start, start+CONFIG.query_chunk)
        allocation = weights[sl].double()
        log_weight = allocation.log()
        offset = math.log(members.shape[-1])-allocation.sum(2).log()
        for crop in crops:
            ids, coefficient = crop_stencil(crop, count, coordinates[sl], image_size)
            evidence = (CONFIG.beta*profiled_logits(crop, members)[ids]).double()
            delta = ((evidence[..., None]+log_weight[:, None]).logsumexp(3)
                     -evidence.logsumexp(-1)[..., None]+offset[:, None])/CONFIG.beta
            delta.masked_fill_((allocation == 1).all(2)[:, None], 0.)
            output[sl] += (delta*coefficient.double()[..., None, None]).sum(1)
    output.masked_fill_(~valid[:, None, None], 0.)
    return output


def fine_class_observation(crops, count, coordinates, members):
    result = coordinates.new_zeros((len(coordinates), len(members)))
    for crop in crops:
        ids, coefficients = crop_stencil(crop, count, coordinates, (512, 512))
        classes = profiled_logits(crop, members).logsumexp(-1)
        result += (classes[ids]*coefficients[..., None]).sum(1)
    return result


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS, cache=None, execution=None):
    from benchmark_grouped_class_readout import CheckedGroupedCache
    from eval_geometry_vip_reliability import sample_broad
    from eval_rival_fine_full import observe_fine, prepare_wide, tile_coordinates

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared bounded fine-coverage methods required.')
    extra = bool(set(methods)-set(METHODS[:3]))
    cache = CheckedGroupedCache() if cache is None else cache
    cache.verify = False
    resized = resize_geometry(image)
    height, width = resized.shape[-2:]
    local_branch = _BoundedBranch(geometry, 512, 4, 'prepare_image')
    wide_branch = _BoundedBranch(vip, 336, 4, 'crop_patch_features')
    fine_reader = FineCoverageReader(vip) if execution is None else execution.reader(vip)
    score_reader = retained_scores if execution is None else execution.score_reader
    wide, crops, _, count = prepare_wide(image, wide_branch, {'clean': (banks, queries)})
    texts = {p: F.normalize(bank.features.float(), dim=-1) for p, bank in banks.items()}
    totals = {p: dict(tiles=0, retained_count_mean=0., deleted_alias_rival_fraction=0.,
        canonical_risk_max=0., mean_absolute_admission_potential=0.,
        normal_equation_max_error=0., gauge_max_error=0., fine_coverage_max_error=0.) for p in banks}
    blend = torch.from_numpy(hann_blend_window(512)).to(geometry.device)
    with ExitStack() as stack:
        accumulators = {(p, m): stack.enter_context(DeviceProbabilityAccumulator(
            bank.class_count, height, width, geometry.device)) for p, bank in banks.items() for m in methods}
        for top, left in geometry_windows(height, width):
            prepared = local_branch.prepare_image(_crop_at(resized, top, left, 512).to(geometry.device))
            coordinates = tile_coordinates(top, left, geometry.device)
            fine_coordinates = tile_reference_coordinates(coordinates, top, left)
            valid = (coordinates[:, 0] < height) & (coordinates[:, 1] < width)
            operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
            with geometry.backbone._autocast():
                features = readout(geometry.backbone.model.visual_model.head, prepared, 2.)[0]
            if extra:
                fine, fine_count, costs = observe_fine(resized[:, top:top+512, left:left+512],
                    vip, queries, fine_coordinates, valid, feature_batch=fine_reader)
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
                    scores, risk, stats = score_reader(local, operator, broad, crops['clean', p], count,
                        fine[p], fine_count, coordinates, fine_coordinates, valid, members, canonical,
                        query.parents, (height, width))
                    values[PRIMARY] = scores['RivalFineHard_Exact']
                    if any(name in methods for name in (CLASS_MEAN, *SHUFFLES)):
                        controls = alias_controls(risk, members, canonical)
                        for name, allocation in controls.items():
                            if name not in methods:
                                continue
                            if name == CLASS_MEAN:
                                directed = weighted_observation(crops['clean', p], count, coordinates,
                                    (height, width), members, allocation, valid)
                            else:
                                directed = hard_pair_observation(crops['clean', p], count, coordinates,
                                    (height, width), members, allocation, valid, CONFIG.beta, CONFIG.query_chunk)
                            potential, _ = signed_potential(directed, valid)
                            values[name] = baseline+operator@potential
                    if OBSERVATION_MEAN in methods:
                        fine_field = fine_class_observation(fine[p], fine_count, fine_coordinates, members)
                        innovation = .5*(fine_field.double()-broad.double())
                        innovation.masked_fill_(~valid[:, None], 0.)
                        values[OBSERVATION_MEAN] = baseline+operator@innovation
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
    if (local_branch.calls != len(geometry_windows(height, width)) or not 1 <= wide_branch.calls <= 4
            or not (1 <= fine_reader.calls <= 4*local_branch.calls <= 16 if extra else fine_reader.calls == 0)):
        raise RuntimeError('Actual whole-image observation budget differs.')
    for row in totals.values():
        for field in row:
            if field != 'tiles':
                row[field] /= row['tiles']
        row.update(geometry_encodings=local_branch.calls, wide_encodings=wide_branch.calls,
            fine_forwards=fine_reader.calls, additional_visual_forwards=fine_reader.calls, native_resolution_encodings=0)
    return predictions, totals
