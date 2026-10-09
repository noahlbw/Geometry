"""Exact RS sparse execution: grouped Geometry classes and device stitching."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_grouped_class_readout import CheckedGroupedCache
from benchmark_natural_sense_fast import measure, require_available_gpu
from dinotool.device_probability_accumulator import DeviceProbabilityAccumulator
from dinotool.gear_ov import _crop_at
from dinotool.geometry_execution import GeometryExecution
from dinotool.inference import hann_blend_window, tile_starts
from dinotool.model import checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.sparse_alias_reuse import PRIMARY
from eval_bounded_alias_anchored import make_models
from eval_gear_ov import protocol
from eval_rival_fine_graph import bind
from eval_sparse_alias_reuse import forbid_fine, layouts_for, predict_image, tile, reference
from run_region_semantic_suite_a800 import TOOL


@torch.inference_mode()
def predict_device(image, geometry, banks, vip, queries, work, *, layouts, cache, methods=(PRIMARY,)):
    height, width = image.shape[-2:]
    storage = sum(bank.class_count + 1 for bank in banks.values()) * height * width * 4 * len(methods)
    if storage > 512 * 1024 * 1024:
        raise ValueError('RS execution pilot caps probability storage at512MiB.')
    fast_tile = bind(tile, alias_class_scores=cache.geometry)
    wide, crops, _, count = reference.prepare_wide(image, vip, {'clean': (banks, queries)})
    blend = torch.from_numpy(hann_blend_window(512)).to(geometry.device)
    totals = {p: {'tiles': 0} for p in banks}
    with forbid_fine(), ExitStack() as stack:
        accumulators = {(p, m): stack.enter_context(DeviceProbabilityAccumulator(
            bank.class_count, height, width, geometry.device)) for p, bank in banks.items() for m in methods}
        for top in tile_starts(height, 512, 128):
            for left in tile_starts(width, 512, 128):
                scores, diagnostics, fields = fast_tile(image, top, left, geometry, banks, vip, queries,
                    wide, crops, count, methods=methods, layouts=layouts, readers=None)
                ah, aw = min(512, height - top), min(512, width - left)
                for p, bank in banks.items():
                    for method, value in scores[p].items():
                        source = fields[p + '__raw'] if method == 'Geometry' else value
                        dense = F.interpolate(source.T.reshape(1, bank.class_count, 32, 32), (512, 512),
                            mode='bilinear', align_corners=False)[0]
                        if method == 'Geometry':
                            dense = dense / .07
                        accumulators[p, method].add(dense[:, :ah, :aw].softmax(0).float(), blend[:ah, :aw], left, top)
                    totals[p]['tiles'] += 1
                    for field, value in diagnostics[p].items():
                        totals[p][field] = totals[p].get(field, 0.) + value
        predictions = {p: {m: accumulators[p, m].finalize_outputs(None, [0.] * bank.class_count, False)[0]
            for m in methods} for p, bank in banks.items()}
    return predictions, {p: {field: value if field == 'tiles' else value / row['tiles']
        for field, value in row.items()} for p, row in totals.items()}


def require_equal(actual, expected):
    if actual[1] != expected[1] or set(actual[0]) != set(expected[0]):
        raise RuntimeError('RS grouped/device complete diagnostics or protocols changed.')
    for p, group in expected[0].items():
        if set(group) != set(actual[0][p]) or any(not np.array_equal(pred, actual[0][p][m]) for m, pred in group.items()):
            raise RuntimeError('RS complete image prediction changed: ' + p)


@torch.inference_mode()
def main(args):
    require_available_gpu(torch.device(args.device))
    output = Path(args.output_dir)
    if output.exists() or args.repetitions < 3:
        raise ValueError('New output and at least three warmed repeats required.')
    previous_root = TOOL / 'results/sparse_alias_reuse_20261005' / args.dataset
    previous = json.loads((previous_root / 'results.json').read_text())
    if previous['status'] != 'complete' or not previous['coverage_verified']:
        raise ValueError('Completed frozen RS source required.')
    samples, specs, load_image, _ = protocol(args, load_class_specs(args.vocabulary_config))
    models = make_models(args, specs)
    geometry, banks, vip, queries, checkpoints = models
    if (checkpoint_manifest(checkpoints) != previous['signature']['checkpoints']
            or asdict(geometry.config) != previous['signature']['geometry']
            or hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest()
                != previous['signature']['vocabulary_sha256']):
        raise RuntimeError('Frozen RS checkpoints, Geometry or all20 words changed.')
    sample_key = previous['benchmark']['sample_key']
    sample = {s.key: s for s in samples}[sample_key]
    image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
    if list(image.shape[-2:]) != previous['benchmark']['image_size']:
        raise RuntimeError('Complete RS image dimensions changed.')
    output.mkdir(parents=True)
    state = reference.frozen_state(geometry, vip)
    layouts, cache = layouts_for(queries), CheckedGroupedCache()
    inputs = (image, *models[:4], output)
    expected = predict_image(*inputs, methods=(PRIMARY,), layouts=layouts)
    actual = predict_device(*inputs, layouts=layouts, cache=cache)
    require_equal(actual, expected)
    with np.load(previous_root / 'benchmark_predictions.npz', allow_pickle=False) as old:
        for p in banks:
            if not np.array_equal(actual[0][p][PRIMARY], old[PRIMARY + '__' + p]):
                raise RuntimeError('Historical sparse complete prediction differs: ' + p)
    cache.verify = False
    geometry_views, preparation_checks = {}, {}
    if args.geometry_execution:
        geometry_views = {'PrunedDevice': GeometryExecution(geometry),
                          'GraphDevice': GeometryExecution(geometry, graph=True)}
        rgb = _crop_at(image, 0, 0, 512).to(geometry.device)
        for name, view in geometry_views.items():
            observed = [view.prepare_image(value) for value in (rgb, rgb.flip(-1))]
            for actual_prepared, value in zip(observed, (rgb, rgb.flip(-1))):
                expected_prepared = geometry.prepare_image(value)
                for field in fields(expected_prepared):
                    a, b = getattr(actual_prepared, field.name), getattr(expected_prepared, field.name)
                    if isinstance(b, torch.Tensor):
                        if not torch.equal(a, b):
                            raise RuntimeError('Real preparation field changed: ' + name + '/' + field.name)
                    elif a != b:
                        raise RuntimeError('Preparation metadata changed: ' + field.name)
            preparation_checks[name] = dict(all_fields_bitwise_equal=True, distinct_rgb_inputs=2,
                returned_results_survive_replays=True, graph_setup_seconds=view.setup_seconds)
        del observed, actual_prepared, expected_prepared, a, b, rgb
    # A combined endpoint check also exercises original Geometry interpolation order.
    small = image[:, :512, :512]
    methods = ('Geometry', 'NoAdmission_Exact', PRIMARY)
    require_equal(predict_device(small, *models[:4], output, layouts=layouts, cache=cache, methods=methods),
                  predict_image(small, *models[:4], output, methods=methods, layouts=layouts))
    for view in geometry_views.values():
        require_equal(predict_device(small, view, banks, vip, queries, output, layouts=layouts, cache=cache, methods=methods),
                      predict_image(small, *models[:4], output, methods=methods, layouts=layouts))
    calls = {
        'OriginalCPU': lambda: predict_image(*inputs, methods=(PRIMARY,), layouts=layouts),
        'GroupedDevice': lambda: predict_device(*inputs, layouts=layouts, cache=cache),
    }
    for name, view in geometry_views.items():
        calls[name] = lambda view=view: predict_device(image, view, banks, vip, queries, output, layouts=layouts, cache=cache)
        require_equal(calls[name](), expected)
    timings = {name: {'seconds': [], 'peaks': []} for name in calls}
    for repetition in range(args.repetitions):
        for name in tuple(calls) if repetition % 2 == 0 else tuple(calls)[::-1]:
            value, timing = measure(calls[name], (), geometry.device, 1)
            require_equal(value, expected)
            timings[name]['seconds'].extend(timing['seconds'])
            timings[name]['peaks'].append(timing['peak_allocated_mib'])
    for name, timing in timings.items():
        timing['median_seconds'] = statistics.median(timing['seconds'])
        timing['peak_allocated_mib'] = max(timing.pop('peaks'))
        print(json.dumps(dict(dataset=args.dataset, method=name, complete_image_ms=timing['median_seconds'] * 1000)), flush=True)
    np.savez_compressed(output / 'predictions.npz', **{p: group[PRIMARY] for p, group in actual[0].items()})
    reference.save(output / 'results.json', dict(status='complete', implementation='exact-sparse-rs-device-v1-20261005',
        dataset=args.dataset, sample_key=sample_key, image_size=list(image.shape[-2:]), signature=previous['signature'],
        timings=timings, complete_predictions_and_diagnostics_exact=True, historical_sparse_prediction_exact=True,
        geometry_execution_checks=preparation_checks,
        class_scores_bitwise_equal=True, geometry_noadmission_sparse_window_exact=True,
        checked_geometry_calls=cache.checked_calls['geometry'], reference_diagnostics=actual[1],
        counts={p: [int((q.parents == c).sum()) for c in range(len(q.class_names))] for p, q in queries.items()},
        probability_storage_bytes=sum(bank.class_count + 1 for bank in banks.values()) * image.shape[-2] * image.shape[-1] * 4,
        target_masks_loaded=False, semantic_rule_changed=False, additional_visual_forwards=0,
        scope='one fixed complete image/domain, unchanged RS wide path, warmed alternating singleton timing',
        **reference.check_frozen(state, geometry, vip)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--geometry-execution', action='store_true')
    main(parser.parse_args())
