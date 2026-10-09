"""Fixed developed64 pilot and opt-in complete-image competition-admission evaluator."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from types import FunctionType

import numpy as np
import torch

from dinotool.model import checkpoint_manifest
from dinotool.prompts import load_class_specs
from dinotool.rival_competition_admission import IMPLEMENTATION, METHODS, PRIMARY, ALTERNATIVES, settings, retained_competition_scores, manifest_samples
from dinotool.semantic_correction_audit import transition_counts, transition_summary
from eval_bounded_alias_anchored import make_models
from eval_gear_ov import protocol
from eval_geometry_vip_reliability import summary
import eval_rival_fine_full as reference
from eval_rival_alias_speed_soft import StageTimer
from run_region_semantic_suite_a800 import TOOL


SOURCE = TOOL/'results/rival_alias_speed_soft_20261005'


@torch.inference_mode()
def tile(*args, methods=METHODS, **kwargs):
    original = reference.predict_tile.__wrapped__
    def reader(*reader_args):
        return retained_competition_scores(*reader_args, methods=methods)
    scope = dict(original.__globals__, retained_scores=reader)
    return FunctionType(original.__code__, scope, original.__name__, original.__defaults__)(*args, **kwargs)


@torch.inference_mode()
def predict_image(*args, methods=METHODS, **kwargs):
    original = reference.predict_image.__wrapped__
    def predictor(*tile_args):
        return tile(*tile_args, methods=methods)
    scope = dict(original.__globals__, predict_tile=predictor, METHODS=methods)
    return FunctionType(original.__code__, scope, original.__name__, original.__defaults__)(*args, **kwargs)


def full(args, methods, manifest_path=None):
    original = reference.main.__wrapped__
    manifest = json.loads(Path(manifest_path).read_text()) if manifest_path else None
    def selected_protocol(inputs, specs):
        samples, specs, load_image, load_mask = reference.protocol(inputs, specs)
        if manifest is not None:
            samples = manifest_samples(samples, manifest, inputs.dataset)
        return samples, specs, load_image, load_mask
    def predictor(*inputs):
        return predict_image(*inputs, methods=methods)
    def save(path, result):
        if 'signature' in result:
            result['signature']['gear']['competition_action'] = settings()
            if manifest is not None:
                result['signature']['pilot_manifest'] = manifest
        reference.save(path, result)
    scope = dict(original.__globals__, predict_image=predictor, protocol=selected_protocol,
                 METHODS=methods, IMPLEMENTATION=IMPLEMENTATION, save=save)
    with torch.inference_mode():
        FunctionType(original.__code__, scope, original.__name__, original.__defaults__)(args)


@torch.inference_mode()
def window(image, geometry, banks, vip, queries, methods):
    timer = StageTimer()
    wide, crops, _, count = timer.call('wide_observation', reference.prepare_wide, image, vip, {'clean': (banks, queries)})
    scores, diagnostics, fields = timer.call('tile_pipeline', tile, image, 0, 0, geometry, banks, vip, queries,
                                           wide, crops, count, methods=methods)
    predictions = {}
    for p, bank in banks.items():
        predictions[p] = {}
        for method, score in scores[p].items():
            raw = fields[p+'__raw'] if method == 'Geometry' else score
            dense = torch.nn.functional.interpolate(raw.T.reshape(1, bank.class_count, 32, 32), (512, 512),
                                                    mode='bilinear', align_corners=False)[0]
            if method == 'Geometry': dense = dense/.07
            predictions[p][method] = dense[:, :min(512, image.shape[-2]), :min(512, image.shape[-1])].argmax(0).cpu().numpy()
    return predictions, scores, diagnostics, timer.seconds


@torch.inference_mode()
def screen(args):
    output = Path(args.output_dir)
    if output.exists(): raise RuntimeError('Refusing existing output.')
    prior = json.loads((SOURCE/args.dataset/'results.json').read_text())
    if prior['status'] != 'complete' or prior['processed_images'] != 8: raise RuntimeError('Complete frozen source64 required.')
    samples, specs, load_image, load_mask = protocol(args, load_class_specs(args.vocabulary_config))
    lookup = {s.key: s for s in samples}
    geometry, banks, vip, queries, checkpoints = make_models(args, specs)
    if (checkpoint_manifest(checkpoints) != prior['signature']['checkpoints']
            or hashlib.sha256(Path(args.vocabulary_config).read_bytes()).hexdigest() != prior['signature']['vocabulary_sha256']
            or asdict(geometry.config) != prior['signature']['geometry']):
        raise RuntimeError('Frozen source identity changed.')
    states = reference.frozen_state(geometry, vip)
    output.mkdir(parents=True)
    (output/'scores').mkdir()
    per_image = {p+'__'+m: [] for p in banks for m in METHODS}
    matrices = {p: {m: np.zeros((bank.class_count,)*2, np.int64) for m in METHODS} for p, bank in banks.items()}
    transitions = {p: {m: np.zeros((bank.class_count,)*3, np.int64) for m in METHODS[2:]} for p, bank in banks.items()}
    diagnostics, ignored = [], dict.fromkeys(banks, 0)
    started = time.perf_counter()
    with np.load(SOURCE/args.dataset/'per_image_confusions.npz', allow_pickle=False) as old_confusions:
        for i, key in enumerate(prior['sample_keys']):
            sample = lookup[key]
            image = load_image(sample.image_path if args.dataset == 'loveda' else sample)
            predictions, scores, diag, stages = window(image, geometry, banks, vip, queries, METHODS)
            with np.load(SOURCE/args.dataset/'scores'/f'{i}.npz', allow_pickle=False) as old:
                for p in banks:
                    for m in ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact'):
                        if not np.array_equal(scores[p][m].cpu().numpy(), old[p+'__'+m]):
                            raise RuntimeError('Original score replay differs: '+p+'/'+m)
            np.savez_compressed(output/'scores'/f'{i}.npz', **{p+'__'+m: s.cpu().numpy() for p, group in scores.items() for m, s in group.items()})
            diagnostics.append({'sample_key': key, 'protocols': diag, 'shared_stages': stages})
            for p, bank in banks.items():
                target = load_mask(sample, p, tuple(image.shape[-2:]))[:512, :512]
                valid = (target >= 0) & (target < bank.class_count)
                ignored[p] += int((~valid).sum())
                for m in METHODS:
                    encoded = target[valid].astype(np.int64)*bank.class_count+predictions[p][m][valid]
                    cm = np.bincount(encoded, minlength=bank.class_count**2).reshape(bank.class_count, -1)
                    if m in ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact') and not np.array_equal(cm, old_confusions[p+'__'+m][i]):
                        raise RuntimeError('Original per-image control differs: '+p+'/'+m)
                    per_image[p+'__'+m].append(cm)
                    matrices[p][m] += cm
                    if m in transitions[p]: transitions[p][m] += transition_counts(predictions[p]['NoAdmission_Exact'], predictions[p][m], target, bank.class_count)
            result = {'status': 'running', 'implementation': IMPLEMENTATION, 'methods': METHODS, 'primary': PRIMARY,
                'processed_images': i+1, 'total_images': 8, 'sample_keys': prior['sample_keys'][:i+1], 'signature': prior['signature'],
                'competition_action': settings(), 'original_three_controls_exact': True, 'scores_persisted_before_masks': True,
                'metrics': {p: {m: summary(cm, banks[p].class_names, ignored[p]) for m, cm in group.items()} for p, group in matrices.items()},
                'transitions': {p: {m: {'counts': cm.tolist(), **transition_summary(cm)} for m, cm in group.items()} for p, group in transitions.items()},
                'diagnostics': diagnostics}
            reference.save(output/'results.json', result)
            print(json.dumps({'dataset': args.dataset, 'processed': i+1, 'miou': {p: {m: v['mean_iou_percent'] for m, v in group.items()} for p, group in result['metrics'].items()}}), flush=True)
    np.savez_compressed(output/'per_image_confusions.npz', sample_keys=np.asarray(prior['sample_keys']), **{k: np.stack(v) for k, v in per_image.items()})
    first = lookup[prior['sample_keys'][0]]
    image = load_image(first.image_path if args.dataset == 'loveda' else first)
    benchmark_methods = ('RivalFineHard_Exact', PRIMARY, *ALTERNATIVES)
    costs = {m: {'seconds': [], 'peak_allocated_mib': []} for m in benchmark_methods}
    for method in benchmark_methods: window(image, geometry, banks, vip, queries, (method,))
    for repeat in range(3):
        for method in (benchmark_methods if repeat%2 == 0 else benchmark_methods[::-1]):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            begun = time.perf_counter()
            window(image, geometry, banks, vip, queries, (method,))
            torch.cuda.synchronize()
            costs[method]['seconds'].append(time.perf_counter()-begun)
            costs[method]['peak_allocated_mib'].append(torch.cuda.max_memory_allocated()/1048576)
    for cost in costs.values():
        cost['median_seconds'] = statistics.median(cost['seconds'])
        cost['peak_allocated_mib'] = max(cost['peak_allocated_mib'])
    result.update(status='complete', coverage_verified=True, wall_seconds=time.perf_counter()-started,
                  benchmark={'sample_key': first.key, 'results': costs, 'scope': prior['benchmark']['scope']},
                  **reference.check_frozen(states, geometry, vip))
    reference.save(output/'results.json', result)


if __name__ == '__main__':
    if '--screen' in sys.argv:
        sys.argv.remove('--screen')
        parser = argparse.ArgumentParser(description=__doc__)
        for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
            parser.add_argument('--'+name, required=True)
        parser.add_argument('--sample-seed', type=int, default=20260923)
        parser.add_argument('--vdd-ontology', default='official')
        parser.add_argument('--device', default='cuda')
        screen(parser.parse_args())
    else:
        chosen = METHODS
        manifest_path = None
        if '--sample-manifest' in sys.argv:
            index = sys.argv.index('--sample-manifest')
            manifest_path = sys.argv[index+1]
            del sys.argv[index:index+2]
        if '--candidate' in sys.argv:
            index = sys.argv.index('--candidate')
            method = sys.argv[index+1]
            del sys.argv[index:index+2]
            if method not in (PRIMARY, *ALTERNATIVES): raise ValueError('A declared candidate, not a diagnostic control, is required.')
            chosen = ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact', method)
        full(reference.parse_args(), chosen, manifest_path)
