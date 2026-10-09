"""Bitwise burst-execution replay and independent matched timing."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from dinotool.fine_alias_view import CONFIG
from dinotool.fine_observer_burst import IMPLEMENTATION, FineObserverBurstGraph
from dinotool.fine_observer_execution import FineObserverGraph
from dinotool.fine_reference_admission import fine_patch_features
from dinotool.model import checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.rival_competition_admission import retained_competition_scores
from eval_bounded_alias_anchored import make_models
from eval_gear_ov import protocol
from eval_geometry_vip_reliability import summary
from eval_rival_competition_admission import window as reference_window
from eval_rival_fine_graph import bind
import eval_rival_fine_full as reference
from run_region_semantic_suite_a800 import TOOL


METHODS = ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact')
BACKENDS = ('Eager', 'CropGraph', 'BurstGraph')
SOURCE = TOOL/'results/fine_observer_execution_20261005'
PROJECTED = TOOL/'results/rival_projected_vocabulary_stress_20261005'
FRAGMENTS = ((128, 128), (128, 400), (400, 128), (257, 511), (512, 512))


def observer_with(features=None, bulk=None):
    if bulk is not None:
        def observer(image, vip, queries, coordinates, valid):
            return reference.observe_fine(image, vip, queries, coordinates, valid, feature_batch=bulk)
        return observer
    return bind(reference.observe_fine, fine_patch_features=features)


def tile_with(observe, reader=None):
    def tile(*inputs, methods=METHODS):
        def competitive(*args):
            result = retained_competition_scores(*args, methods=methods)
            if reader is not None:
                reader(result[1])
            return result
        return bind(reference.predict_tile, observe_fine=observe, retained_scores=competitive)(*inputs)
    return tile


def window_with(observe, reader=None):
    return bind(reference_window, tile=tile_with(observe, reader))


def full_with(observe):
    return bind(reference.predict_image, predict_tile=tile_with(observe), METHODS=METHODS)


def prediction_difference(first, second):
    return {p: {m: int(np.count_nonzero(first[p][m] != second[p][m])) for m in METHODS} for p in first}


def require_equal(first, second, key):
    old_predictions, old_scores, old_diag = first[:3]
    predictions, scores, diag = second[:3]
    mismatches = prediction_difference(old_predictions, predictions)
    if diag != old_diag or any(v for group in mismatches.values() for v in group.values()) or any(
            not torch.equal(old_scores[p][m], scores[p][m]) for p in scores for m in METHODS):
        raise RuntimeError('Burst execution changed endpoint: '+key)
    return mismatches


def benchmark(functions, image, models, repetitions):
    endpoints = ('RivalFineHard_Exact', 'FineRivalProjected_Exact')
    rows = {m: {b: {'seconds': [], 'peak_allocated_mib': []} for b in BACKENDS} for m in endpoints}
    for m in endpoints:
        for function in functions.values():
            function(image, *models, (m,))
    for repeat in range(repetitions):
        order = BACKENDS if repeat % 2 == 0 else BACKENDS[::-1]
        for m in endpoints if repeat % 2 == 0 else endpoints[::-1]:
            for b in order:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                functions[b](image, *models, (m,))
                torch.cuda.synchronize()
                rows[m][b]['seconds'].append(time.perf_counter()-started)
                rows[m][b]['peak_allocated_mib'].append(torch.cuda.max_memory_allocated()/1048576)
    for methods in rows.values():
        for row in methods.values():
            row['median_seconds'] = statistics.median(row['seconds'])
            row['peak_allocated_mib'] = max(row['peak_allocated_mib'])
    return {'results': rows, 'repetitions': repetitions,
        'scope': 'one512 local window with full-image wide context; independent warmed alternating endpoints',
        'graphs_resident_in_every_arm': True, 'loading_text_decoding_masks_audits_excluded': True}


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Refusing existing burst study.')
    prior = json.loads((SOURCE/args.dataset/'results.json').read_text())
    projected_prior = json.loads((PROJECTED/args.dataset/'results.json').read_text())
    keys = prior['sample_keys']
    if prior['status'] != 'complete' or keys != projected_prior['sample_keys'] or len(keys) != 8:
        raise RuntimeError('Complete same eight-key execution and candidate sources required.')
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {sample.key: sample for sample in samples}
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    models = (geometry, banks, vip, queries)
    signature = {'checkpoints': checkpoint_manifest(checkpoints),
        'vocabulary_sha256': hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest(),
        'geometry': asdict(geometry.config), 'classes': {p: list(b.class_names) for p, b in banks.items()},
        'aliases': {p: list(b.alias_names) for p, b in banks.items()}}
    if signature != prior['signature']:
        raise RuntimeError('Frozen sources changed.')
    states = reference.frozen_state(geometry, vip)
    output.mkdir(parents=True)
    (output/'scores').mkdir()
    first_sample = lookup[keys[0]]
    first_image = load_image(first_sample.image_path if args.dataset == 'loveda' else first_sample)
    eager_observer = observer_with(fine_patch_features)
    eager = window_with(eager_observer)
    eager(first_image, *models, ('RivalFineHard_Exact',))
    torch.cuda.reset_peak_memory_stats()
    eager(first_image, *models, ('RivalFineHard_Exact',))
    before_graph_peak = torch.cuda.max_memory_allocated()/1048576
    crop_graph, burst = FineObserverGraph(vip), FineObserverBurstGraph(vip)
    observers = {'Eager': eager_observer, 'CropGraph': observer_with(crop_graph),
                 'BurstGraph': observer_with(bulk=burst)}
    matrices = {p: {m: np.zeros((b.class_count,)*2, np.int64) for m in METHODS} for p, b in banks.items()}
    confusions = {p+'__'+m: [] for p in banks for m in METHODS}
    checks, feature_checks = [], dict.fromkeys(BACKENDS[1:], 0)
    started = time.perf_counter()
    with np.load(SOURCE/args.dataset/'per_image_confusions.npz', allow_pickle=False) as history, np.load(
            PROJECTED/args.dataset/'per_image_confusions.npz', allow_pickle=False) as candidate_history:
        for index, key in enumerate(keys):
            sample = lookup[key]
            image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
            risks = []
            expected = window_with(eager_observer, lambda value: risks.append(value.clone()))(image, *models, METHODS)
            check = {'sample_key': key, 'backends': {}}
            for backend in BACKENDS[1:]:
                next_risk = [0]
                def compare_risk(actual):
                    if not torch.equal(actual, risks[next_risk[0]]):
                        raise RuntimeError('Fine backend changed hard admission risk.')
                    next_risk[0] += 1
                if backend == 'CropGraph':
                    def checked_feature(observer, rgb):
                        original, actual = fine_patch_features(observer, rgb), crop_graph(observer, rgb)
                        if not torch.equal(original, actual):
                            raise RuntimeError('Per-crop graph feature mismatch.')
                        feature_checks[backend] += 1
                        return actual
                    observe = observer_with(checked_feature)
                else:
                    def checked_bulk(observer, rgbs):
                        actual = burst(observer, rgbs)
                        for rgb, value in zip(rgbs, actual):
                            original = fine_patch_features(observer, rgb)
                            if not torch.equal(original, value):
                                raise RuntimeError('Burst feature mismatch: '+key)
                            feature_checks[backend] += 1
                        return actual
                    observe = observer_with(bulk=checked_bulk)
                actual = window_with(observe, compare_risk)(image, *models, METHODS)
                if next_risk[0] != len(risks):
                    raise RuntimeError('Endpoint admission coverage changed.')
                check['backends'][backend] = {'features_bitwise_exact': True, 'risk_bitwise_exact': True,
                    'scores_bitwise_exact': True, 'diagnostics_equal': True,
                    'prediction_mismatches': require_equal(expected, actual, backend+'/'+key)}
            checks.append(check)
            np.savez_compressed(output/'scores'/f'{index}.npz', sample_key=np.asarray(key),
                **{p+'__'+m: value.cpu().numpy() for p, methods in expected[1].items() for m, value in methods.items()})
            for p, bank in banks.items():
                target = load_mask(sample, p, tuple(image.shape[-2:]))[:512, :512]
                valid = (target >= 0) & (target < bank.class_count)
                for m in METHODS:
                    code = target[valid].astype(np.int64)*bank.class_count+expected[0][p][m][valid]
                    cm = np.bincount(code, minlength=bank.class_count**2).reshape(bank.class_count, bank.class_count)
                    old = candidate_history['k20__'+p+'__'+m][index] if m == METHODS[-1] else history[p+'__'+m][index]
                    if not np.array_equal(cm, old):
                        raise RuntimeError('Historical per-image endpoint confusion differs: '+p+'/'+m)
                    matrices[p][m] += cm
                    confusions[p+'__'+m].append(cm)
            result = {'status': 'running', 'implementation': IMPLEMENTATION, 'processed_images': index+1,
                'total_images': 8, 'sample_keys': keys[:index+1], 'signature': signature,
                'admission_config': asdict(CONFIG), 'methods': METHODS, 'backends': BACKENDS,
                'checks': checks, 'feature_checks': feature_checks, 'scores_persisted_before_masks': True,
                'historical_per_image_endpoints_exact': True,
                'metrics': {p: {m: summary(cm, banks[p].class_names, 0) for m, cm in methods.items()}
                            for p, methods in matrices.items()}}
            reference.save(output/'results.json', result)
            print(json.dumps({'dataset': args.dataset, 'processed': index+1, 'execution_exact': True}), flush=True)
    functions = {b: window_with(observe) for b, observe in observers.items()}
    result['benchmark'] = benchmark(functions, first_image, models, args.benchmark_repetitions)
    result['benchmark'].update(eager_before_graph_peak_allocated_mib=before_graph_peak,
        crop_graph_setup_seconds=crop_graph.setup_seconds,
        burst_setup_seconds_before_fragments=dict(burst.setup_seconds),
        reserved_mib_before_fragments=torch.cuda.memory_reserved()/1048576)
    fragments = []
    for height, width in FRAGMENTS:
        fragment = first_image[:, :height, :width]
        if fragment.shape[-2:] != (height, width):
            raise RuntimeError('Expected fragment is unavailable in fixed first input.')
        expected = functions['Eager'](fragment, *models, METHODS)
        actual = functions['BurstGraph'](fragment, *models, METHODS)
        fragments.append({'size': [height, width], 'diagnostics_equal': True,
            'scores_bitwise_exact': True, 'prediction_mismatches': require_equal(expected, actual, 'fragment'),
            'fine_crop_count': next(iter(actual[2].values()))['fine_forwards']})
    result['fragment_equivalence'] = fragments
    result['burst_setup_seconds_after_fragments'] = burst.setup_seconds
    if args.full_image_equivalence and args.dataset == 'vdd':
        expected, old_diag = full_with(observers['CropGraph'])(first_image, *models, output)
        actual, new_diag = full_with(observers['BurstGraph'])(first_image, *models, output)
        mismatches = prediction_difference(expected, actual)
        if old_diag != new_diag or any(v for group in mismatches.values() for v in group.values()):
            raise RuntimeError('Complete VDD burst equivalence failed.')
        result['full_image_equivalence'] = {'image_size': list(first_image.shape[-2:]),
            'prediction_mismatches': mismatches, 'diagnostics_equal': True}
    np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(keys),
                        **{key: np.stack(value) for key, value in confusions.items()})
    result.update(status='complete', coverage_verified=True, **reference.check_frozen(states, geometry, vip),
        crop_graph_replays=crop_graph.replays, burst_replays=burst.replays, wall_seconds=time.perf_counter()-started)
    reference.save(output/'results.json', result)
    print(json.dumps({'dataset': args.dataset, 'status': 'complete', 'benchmark': result['benchmark']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--benchmark-repetitions', type=int, default=7)
    parser.add_argument('--full-image-equivalence', action='store_true')
    main(parser.parse_args())
