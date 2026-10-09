"""Exact fine-observer execution replay on the fixed developed64 panel."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import FunctionType

import numpy as np
import torch

from dinotool.fine_alias_view import CONFIG
from dinotool.fine_observer_execution import (IMPLEMENTATION, FineObserverGraph, sync_free_patch_features)
from dinotool.fine_reference_admission import fine_patch_features
from dinotool.model import checkpoint_manifest
from dinotool.prompts import load_class_specs
from eval_bounded_alias_anchored import make_models
from eval_gear_ov import protocol
from eval_geometry_vip_reliability import summary
from eval_rival_alias_speed_soft import window as original_window, PREVIOUS
from eval_rival_fine_full import observe_fine, frozen_state, check_frozen, save, predict_tile, predict_image
from dinotool.rival_alias_fast import retained_scores_fast


METHODS = ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact')
BACKENDS = ('Eager', 'NoUnusedStats', 'CUDAGraph')


def bind(function, **scope):
    original = getattr(function, '__wrapped__', function)
    return FunctionType(original.__code__, dict(original.__globals__, **scope),
                        original.__name__, original.__defaults__)


def observer_with(features):
    return bind(observe_fine, fine_patch_features=features)


def window_with(features, reader=retained_scores_fast):
    return bind(original_window, observe_fine=observer_with(features), retained_scores_fast=reader)


def full_with(features):
    tile = bind(predict_tile, observe_fine=observer_with(features), retained_scores=retained_scores_fast)
    return bind(predict_image, predict_tile=tile)


def timing(function, image, models, repetitions):
    function(image, *models, ('RivalFineHard_Exact',))
    seconds, peaks, stages = [], [], []
    for _ in range(repetitions):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        _, _, _, _, fields = function(image, *models, ('RivalFineHard_Exact',))
        torch.cuda.synchronize()
        seconds.append(time.perf_counter()-started)
        peaks.append(torch.cuda.max_memory_allocated()/1048576)
        stages.append(fields)
    return {'seconds': seconds, 'median_seconds': statistics.median(seconds),
            'peak_allocated_mib': max(peaks), 'peak_reserved_mib': torch.cuda.max_memory_reserved()/1048576,
            'stage_median_seconds': {k: statistics.median(s.get(k, 0.) for s in stages)
                                     for k in set().union(*(s.keys() for s in stages))}}


@torch.inference_mode()
def main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Refusing existing output.')
    prior = json.loads((PREVIOUS/args.dataset/'merged.json').read_text())
    keys = prior['sample_keys']
    if prior['status'] != 'complete' or len(keys) != 8 or len(set(keys)) != 8:
        raise RuntimeError('Require the frozen eight-key developed panel.')
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {s.key: s for s in samples}
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    models = (geometry, banks, vip, queries)
    states = frozen_state(geometry, vip)
    if checkpoint_manifest(checkpoints) != prior['signature']['checkpoints']:
        raise RuntimeError('Historical checkpoints differ.')
    output.mkdir(parents=True)
    eager = window_with(fine_patch_features)
    first = lookup[keys[0]]
    first_image = load_image(first.image_path if args.dataset == 'loveda' else first)
    before_graph = timing(eager, first_image, models, args.benchmark_repetitions)
    graph = FineObserverGraph(vip)
    features = {'Eager': fine_patch_features, 'NoUnusedStats': sync_free_patch_features, 'CUDAGraph': graph}
    matrices = {p: {m: np.zeros((b.class_count,)*2, np.int64) for m in METHODS} for p, b in banks.items()}
    confusions = {p+'__'+m: [] for p in banks for m in METHODS}
    checks, feature_checks = [], dict.fromkeys(BACKENDS[1:], 0)
    started = time.perf_counter()
    result = {}
    for index, key in enumerate(keys):
        sample = lookup[key]
        image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
        risks = []
        def remember(*inputs):
            values, risk, stats = retained_scores_fast(*inputs)
            risks.append(risk.clone())
            return values, risk, stats
        expected, scores, diagnostics, _, _ = window_with(fine_patch_features, remember)(image, *models, METHODS, True)
        check = {'sample_key': key, 'backends': {}}
        for name in BACKENDS[1:]:
            next_risk = [0]
            def checked_reader(*inputs):
                values, risk, stats = retained_scores_fast(*inputs)
                if not torch.equal(risk, risks[next_risk[0]]):
                    raise RuntimeError('Fine execution changed risk: '+name+'/'+key)
                next_risk[0] += 1
                return values, risk, stats
            def checked_features(observer, rgb):
                baseline = fine_patch_features(observer, rgb)
                actual = features[name](observer, rgb)
                if not torch.equal(baseline, actual):
                    raise RuntimeError('Fine feature equivalence failed: '+name+'/'+key+'/'+str(float((baseline-actual).abs().max())))
                feature_checks[name] += 1
                return actual
            actual, new_scores, new_diag, _, _ = window_with(checked_features, checked_reader)(image, *models, METHODS, True)
            if next_risk[0] != len(risks) or new_diag != diagnostics:
                raise RuntimeError('Fine execution changed diagnostics.')
            mismatches = {p: {m: int(np.count_nonzero(expected[p][m] != actual[p][m])) for m in METHODS} for p in banks}
            if any(v for g in mismatches.values() for v in g.values()) or any(
                    not torch.equal(scores[p][m], new_scores[p][m]) for p in banks for m in METHODS):
                raise RuntimeError('Fine execution changed scores/predictions.')
            check['backends'][name] = {'features_bitwise_exact': True, 'risk_bitwise_exact': True,
                'scores_bitwise_exact': True, 'diagnostics_equal': True, 'prediction_mismatches': mismatches}
        checks.append(check)
        # Outputs and equivalence checks precede any mask access.
        cache = output/'scores'
        cache.mkdir(exist_ok=True)
        np.savez_compressed(cache/f'{index}.npz', sample_key=np.asarray(key),
                            **{p+'__'+m: s.cpu().numpy() for p, g in scores.items() for m, s in g.items()})
        for p, bank in banks.items():
            target = load_mask(sample, p, tuple(image.shape[-2:]))[:512, :512]
            valid = (target >= 0) & (target < bank.class_count)
            for m in METHODS:
                code = target[valid].astype(np.int64)*bank.class_count+expected[p][m][valid]
                cm = np.bincount(code, minlength=bank.class_count**2).reshape(bank.class_count, -1)
                matrices[p][m] += cm
                confusions[p+'__'+m].append(cm)
        result = {'status': 'running', 'implementation': IMPLEMENTATION, 'processed_images': index+1, 'total_images': 8,
            'sample_keys': keys[:index+1], 'methods': METHODS, 'backends': BACKENDS, 'admission_config': asdict(CONFIG),
            'signature': {'checkpoints': checkpoint_manifest(checkpoints),
                'vocabulary_sha256': hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest(),
                'geometry': asdict(geometry.config), 'classes': {p: b.class_names for p, b in banks.items()},
                'aliases': {p: b.alias_names for p, b in banks.items()}},
            'metrics': {p: {m: summary(cm, banks[p].class_names, 0) for m, cm in g.items()} for p, g in matrices.items()},
            'checks': checks, 'feature_checks': feature_checks, 'scores_persisted_before_masks': True}
        save(output/'results.json', result)
        print(json.dumps({'dataset': args.dataset, 'processed': index+1, 'execution_exact': True}), flush=True)
    benchmark = {name: {'seconds': [], 'stages': [], 'peaks': []} for name in BACKENDS}
    functions = {name: window_with(features[name]) for name in BACKENDS}
    for function in functions.values():
        function(first_image, *models, ('RivalFineHard_Exact',))
    for repeat in range(args.benchmark_repetitions):
        for name in BACKENDS if repeat % 2 == 0 else BACKENDS[::-1]:
            row = timing(functions[name], first_image, models, 1)
            benchmark[name]['seconds'] += row['seconds']
            benchmark[name]['stages'].append(row['stage_median_seconds'])
            benchmark[name]['peaks'].append(row['peak_allocated_mib'])
    for row in benchmark.values():
        row['median_seconds'] = statistics.median(row['seconds'])
        row['peak_allocated_mib'] = max(row.pop('peaks'))
        row['stage_median_seconds'] = {k: statistics.median(s.get(k, 0.) for s in row['stages'])
                                      for k in set().union(*(s.keys() for s in row['stages']))}
        del row['stages']
    result['benchmark'] = {'repetitions': args.benchmark_repetitions, 'scope': 'one512 window plus full-image wide context',
        'with_graph_resident': benchmark, 'eager_before_graph': before_graph,
        'graph_resident_reserved_mib': torch.cuda.memory_reserved()/1048576,
        'graph_setup_seconds': graph.setup_seconds, 'graph_replays': graph.replays,
        'notes': 'Independent arms, alternating warmed timing; equality audits excluded. All original four fine views remain.'}
    if args.full_image_equivalence and args.dataset == 'vdd':
        baseline, baseline_diag = full_with(fine_patch_features)(first_image, *models, output)
        actual, actual_diag = full_with(graph)(first_image, *models, output)
        mismatches = {p: {m: int(np.count_nonzero(baseline[p][m] != actual[p][m])) for m in METHODS} for p in banks}
        if any(v for g in mismatches.values() for v in g.values()) or baseline_diag != actual_diag:
            raise RuntimeError('Full-image fine graph equivalence failed.')
        result['full_image_equivalence'] = {'image_size': list(first_image.shape[-2:]),
            'prediction_mismatches': mismatches, 'diagnostics_equal': True}
    np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(keys),
                        **{k: np.stack(v) for k, v in confusions.items()})
    result.update(status='complete', coverage_verified=True, **check_frozen(states, geometry, vip),
                  wall_seconds=time.perf_counter()-started)
    save(output/'results.json', result)
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
