"""Frozen all20, zero-fine admission: full inputs, shared controls, singleton timing."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_grouped_class_readout import CheckedGroupedCache
from benchmark_natural_sense_fast import measure, require_available_gpu
from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.device_probability_accumulator import DeviceProbabilityAccumulator
from dinotool.gear_ov import _crop_at
from dinotool.geometry_execution import GeometryExecution
from dinotool.inference import hann_blend_window, tile_starts
from dinotool.model import checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.shared_rival_soft_alias import (CONFIG, IMPLEMENTATION, METHODS, PRIMARY,
    alias_permutation, cache_wide, shared_scores)
from dinotool.sparse_alias_reuse import profile_aliases
from dinotool.stratified_soft_alias import WideCrop
from dinotool.vip_official_adapter import upstream_settings
from eval_bounded_alias_anchored import make_models
from eval_geometry_semantic_innovation import SETTINGS
from eval_gear_ov import digest, protocol
from eval_geometry_vip_reliability import sample_broad, summary
from eval_matched_contribution_alias import crop_from_features
from eval_rival_fine_full import check_frozen, frozen_state, save
from eval_sparse_alias_reuse import forbid_fine, layouts_for, predict_image as old_predict
from eval_vip_official_eight import PINNED_COMMIT
from run_region_semantic_suite_a800 import TOOL
import eval_rival_fine_full as reference


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work, *, methods=METHODS, cache=None,
                  output_size=None, wide_image=None):
    cache = CheckedGroupedCache() if cache is None else cache
    cache.verify = False
    layouts = layouts_for(queries)
    permutations = {p: alias_permutation(group) for p, group in layouts.items()}
    texts = {p: F.normalize(bank.features.float(), dim=-1) for p, bank in banks.items()}
    height, width = image.shape[-2:]
    wide, crops, _, count = reference.prepare_wide(image if wide_image is None else wide_image,
                                                  vip, {'clean': (banks, queries)})
    active = tuple(m for m in methods if m in METHODS[2:])
    observations = {p: cache_wide(crops['clean', p], layouts[p]) for p in banks} if active else {}
    blend = torch.from_numpy(hann_blend_window(512)).to(geometry.device)
    totals = {p: {'tiles': 0} for p in banks}
    with forbid_fine(), ExitStack() as stack:
        accumulators = {(p, m): stack.enter_context(DeviceProbabilityAccumulator(
            bank.class_count, height, width, geometry.device)) for p, bank in banks.items() for m in methods}
        for top in tile_starts(height, 512, 128):
            for left in tile_starts(width, 512, 128):
                prepared = geometry.prepare_image(_crop_at(image, top, left, 512).to(geometry.device))
                coordinates = reference.tile_coordinates(top, left, geometry.device)
                valid = (coordinates[:, 0] < height) & (coordinates[:, 1] < width)
                operator, residual = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
                ah, aw = min(512, height - top), min(512, width - left)
                for p, bank in banks.items():
                    raw = cache.geometry((prepared.geometry_projected.float() @ texts[p].T)[0],
                                         bank.parent_indices, bank.class_count)
                    local = raw / .07
                    broad = sample_broad(wide['clean', p], top, left, height, width).reshape_as(local)
                    values = {'Geometry': raw, 'NoAdmission_Exact': local.double() + operator.double() @ (broad.double() - local.double())}
                    diagnostic = dict(fine_forwards=0., additional_visual_forwards=0., operator_residual=residual)
                    if active:
                        blank = WideCrop(local.new_empty(1024, 1), local.new_empty(1), 0, 0, 512, 512, 32, 512)
                        source = {}
                        for name, features in (('native', prepared.native_projected), ('geometry', prepared.geometry_projected)):
                            crop = crop_from_features(features, queries[p], blank)
                            source[name] = profile_aliases(crop.alias_logits, crop.salience, layouts[p])
                        result, stats = shared_scores(local, operator, broad, observations[p], count, coordinates,
                            (height, width), source['native'], source['geometry'], valid, layouts[p],
                            methods=active, permutation=permutations[p])
                        values.update(result)
                        diagnostic.update(stats)
                    for method in methods:
                        dense = F.interpolate(values[method].T.reshape(1, bank.class_count, 32, 32), (512, 512),
                                              mode='bilinear', align_corners=False)[0]
                        if method == 'Geometry':
                            dense = dense / .07
                        accumulators[p, method].add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], left, top)
                    totals[p]['tiles'] += 1
                    for field, value in diagnostic.items():
                        totals[p][field] = totals[p].get(field, 0.) + value
        predictions = {p: {m: (accumulators[p, m].finalize_outputs(None, [0.] * bank.class_count, False)[0]
            if output_size is None else accumulators[p, m].finalize_resized(output_size))
            for m in methods} for p, bank in banks.items()}
    return predictions, {p: {k: v if k == 'tiles' else v / row['tiles'] for k, v in row.items()} for p, row in totals.items()}


def prepare(args):
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    models = make_models(args, specs)
    geometry, banks, vip, queries, checkpoints = models
    old = json.loads((TOOL / 'results/rival_fine_hard20_full_20261003/full' / args.dataset / 'merged.json').read_text())
    keys = [s.key for s in samples]
    if (keys != old['signature']['sample_keys'] or checkpoint_manifest(checkpoints) != old['signature']['checkpoints']
            or asdict(geometry.config) != old['signature']['gear']['geometry']
            or hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest() != old['signature']['vocabulary']['sha256']):
        raise RuntimeError('Original full sequence/checkpoint/Geometry/all20 vocabulary changed.')
    return samples, load_image, load_mask, models, old


@torch.inference_mode()
def smoke(args):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Preserve existing smoke output.')
    samples, load_image, _, models, old = prepare(args)
    geometry, banks, vip, queries = models[:4]
    state = frozen_state(geometry, vip)
    output.mkdir(parents=True)
    sample = samples[0]
    image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
    execution = GeometryExecution(geometry)
    rgb = _crop_at(image, 0, 0, 512).to(geometry.device)
    actual_prepared, expected_prepared = execution.prepare_image(rgb), geometry.prepare_image(rgb)
    for field in fields(expected_prepared):
        a, b = getattr(actual_prepared, field.name), getattr(expected_prepared, field.name)
        if isinstance(b, torch.Tensor):
            if not torch.equal(a, b):
                raise RuntimeError('Real pruned Geometry field changed: ' + field.name)
        elif a != b:
            raise RuntimeError('Real pruned Geometry metadata changed: ' + field.name)
    small = image[:, :512, :512]
    combined = predict_image(small, execution, banks, vip, queries, output)
    previous = old_predict(small, geometry, banks, vip, queries, output, methods=METHODS[:2])
    for p in banks:
        for m in METHODS[:2]:
            if not np.array_equal(combined[0][p][m], previous[0][p][m]):
                raise RuntimeError('Original Geometry/no-admission endpoint changed: ' + p + '/' + m)
    singleton = predict_image(small, execution, banks, vip, queries, output, methods=(PRIMARY,))
    if any(not np.array_equal(singleton[0][p][PRIMARY], combined[0][p][PRIMARY]) for p in banks):
        raise RuntimeError('Singleton primary differs from joint controls.')
    save(output / 'results.json', dict(status='complete', dataset=args.dataset, implementation=IMPLEMENTATION,
        sample_key=sample.key, prepared_fields_bitwise_equal=True, original_endpoints_exact=True,
        singleton_primary_exact=True, target_masks_loaded=False, fine_observer_forbidden=True,
        diagnostics=combined[1], **check_frozen(state, geometry, vip)))


@torch.inference_mode()
def evaluate(args, *, predictor=predict_image, implementation=IMPLEMENTATION, view_protocol=None):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Preserve existing full output.')
    samples, load_image, load_mask, models, old = prepare(args)
    geometry, banks, vip, queries, checkpoints = models
    state = frozen_state(geometry, vip)
    execution = GeometryExecution(geometry)
    cache = CheckedGroupedCache()
    cache.verify = False
    selected = samples[args.shard_index::args.num_shards]
    if not selected or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Invalid or empty full shard.')
    keys = [s.key for s in samples]
    signature = dict(implementation=implementation, dataset=args.dataset, methods=METHODS,
        classes={p: bank.class_names for p, bank in banks.items()},
        gear=dict(geometry=asdict(geometry.config), observation=asdict(SETTINGS), admission=asdict(CONFIG),
                  upstream_commit=PINNED_COMMIT, local_tiles=[512, 128], fine_forwards=0,
                  execution='pruned-equivalent-geometry/grouped-class/device-stitch'),
        competitive=dict(source='all-class leave-class-out LME; wide support AND native/Geometry contradiction',
                         aggregation='normalized weighted LME outside exponent; canonical protection'),
        vocabulary=old['signature']['vocabulary'], checkpoints=checkpoint_manifest(checkpoints),
        global_sample_count=len(keys), global_sample_keys_sha256=digest(keys), num_shards=args.num_shards,
        shard_index=args.shard_index, sample_keys=[s.key for s in selected],
        sample_keys_sha256=digest([s.key for s in selected]), config=vars(args))
    if view_protocol is not None:
        signature['gear'].update(input_protocol=view_protocol,
                                 local_tiles=view_protocol.get('local_tiles', [336, 224]),
                                 execution='pruned-geometry/resized-device-stitch')
    matrices = {p: {m: np.zeros((bank.class_count,) * 2, np.int64) for m in METHODS} for p, bank in banks.items()}
    per_image = {p + '__' + m: [] for p in banks for m in METHODS}
    ignored = dict.fromkeys(banks, 0)
    totals = {p: {'tiles': 0} for p in banks}
    output.mkdir(parents=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for number, sample in enumerate(selected, 1):
        image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
        predictions, diagnostic = predictor(image, execution, banks, vip, queries, output, cache=cache)
        for p, bank in banks.items():
            target = load_mask(sample, p, tuple(image.shape[-2:]))
            valid = (target >= 0) & (target < bank.class_count)
            ignored[p] += int((~valid).sum())
            for m in METHODS:
                encoded = target[valid].astype(np.int64) * bank.class_count + predictions[p][m][valid]
                cm = np.bincount(encoded, minlength=bank.class_count ** 2).reshape(bank.class_count, -1)
                matrices[p][m] += cm
                per_image[p + '__' + m].append(cm)
            tiles = diagnostic[p]['tiles']
            totals[p]['tiles'] += tiles
            for k, v in diagnostic[p].items():
                if k != 'tiles' and isinstance(v, (int, float)):
                    totals[p][k] = totals[p].get(k, 0.) + v * tiles
        if number % 5 == 0 or number == len(selected) or number == 1:
            result = dict(status='running', processed_images=number, total_images=len(selected), signature=signature,
                metrics={p: {m: summary(cm, banks[p].class_names, ignored[p]) for m, cm in group.items()}
                         for p, group in matrices.items()},
                diagnostics={p: {k: v if k == 'tiles' else v / row['tiles'] for k, v in row.items()} for p, row in totals.items()},
                wall_seconds=time.perf_counter() - started, peak_cuda_memory_mb=torch.cuda.max_memory_allocated() / 1048576,
                target_masks_used_only_after_prediction=True, fine_observer_forbidden=True)
            save(output / 'results.json', result)
            print(json.dumps(dict(dataset=args.dataset, processed=number, total=len(selected),
                miou={p: group[PRIMARY]['mean_iou_percent'] for p, group in result['metrics'].items()})), flush=True)
    np.savez_compressed(output / 'per_image_confusions.npz', sample_keys=np.asarray(signature['sample_keys']),
                        **{k: np.stack(v) for k, v in per_image.items()})
    result.update(status='complete', **check_frozen(state, geometry, vip))
    save(output / 'results.json', result)


@torch.inference_mode()
def benchmark(args, *, predictor=predict_image, implementation=IMPLEMENTATION, view_protocol=None):
    require_available_gpu(torch.device(args.device))
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Preserve existing timing output.')
    samples, load_image, _, models, old = prepare(args)
    geometry, banks, vip, queries = models[:4]
    execution = GeometryExecution(geometry)
    state = frozen_state(geometry, vip)
    output.mkdir(parents=True)
    vip_settings = upstream_settings(args.dataset)
    chosen = [samples[i] for i in sorted({0, len(samples) // 2, len(samples) - 1})]
    image_rows = []
    for sample in chosen:
        image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
        names = ('NoAdmission_Exact', PRIMARY, 'SharedRivalHard', 'VIP_All20')
        def call(name):
            if name == 'VIP_All20':
                return {p: vip.predict(image, q, vip_settings)[0] for p, q in queries.items()}
            return predictor(image, execution, banks, vip, queries, output, methods=(name,))[0]
        expected = {name: call(name) for name in names}
        timings = {name: {'seconds': [], 'peaks': []} for name in names}
        for repeat in range(args.repetitions):
            order = names[repeat % len(names):] + names[:repeat % len(names)]
            for name in order:
                result, timing = measure(call, (name,), torch.device(args.device), 1)
                for p in banks:
                    actual = result[p] if name == 'VIP_All20' else result[p][name]
                    previous = expected[name][p] if name == 'VIP_All20' else expected[name][p][name]
                    if not np.array_equal(actual, previous):
                        raise RuntimeError('Warmed singleton prediction changed: ' + name)
                timings[name]['seconds'].extend(timing['seconds'])
                timings[name]['peaks'].append(timing['peak_allocated_mib'])
        for timing in timings.values():
            timing['median_seconds'] = statistics.median(timing['seconds'])
            timing['peak_allocated_mib'] = max(timing.pop('peaks'))
        image_rows.append(dict(sample_key=sample.key, image_size=list(image.shape[-2:]), timings=timings))
        print(json.dumps(dict(dataset=args.dataset, sample_key=sample.key,
            milliseconds={m: row['median_seconds'] * 1000 for m, row in timings.items()})), flush=True)
    save(output / 'results.json', dict(status='complete', dataset=args.dataset, implementation=implementation,
        input_protocol=view_protocol,
        images=image_rows, source_vocabulary=old['signature']['vocabulary'], vip_settings=asdict(vip_settings),
        repetitions=args.repetitions, target_masks_loaded=False,
        scope='three fixed complete images, warmed synchronized rotated singleton order; same image words; different method views/settings',
        memory_scope='both backbones resident; shared-resident peaks, not standalone deployment',
        **check_frozen(state, geometry, vip)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--mode', choices=('smoke', 'full', 'benchmark'), default='full')
    args = parser.parse_args()
    {'smoke': smoke, 'full': evaluate, 'benchmark': benchmark}[args.mode](args)
