"""Frozen competition-action transfer panel with exact fine-observer execution."""
import argparse
import json
from pathlib import Path
import statistics
import time
from types import FunctionType

import torch

from dinotool.rival_competition_admission import (IMPLEMENTATION, retained_competition_scores,
                                                manifest_samples, settings)
from eval_rival_fine_graph import graph_features
import eval_rival_fine_full as reference
from eval_rival_competition_admission import window as reference_window


CANDIDATE = 'FineRivalProjected_Exact'
METHODS = ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact', CANDIDATE, 'FineBudgetOnly_Exact')


def bind(function, **scope):
    original = getattr(function, '__wrapped__', function)
    return FunctionType(original.__code__, dict(original.__globals__, **scope),
                        original.__name__, original.__defaults__)


@torch.inference_mode()
def tile(*args, methods=METHODS, graph=True, **kwargs):
    def reader(*inputs):
        return retained_competition_scores(*inputs, methods=methods)
    observe = bind(reference.observe_fine, fine_patch_features=graph_features) if graph else reference.observe_fine
    return bind(reference.predict_tile, retained_scores=reader, observe_fine=observe)(*args, **kwargs)


@torch.inference_mode()
def predict_image(*args, methods=METHODS, **kwargs):
    def predictor(*inputs):
        return tile(*inputs, methods=methods)
    return bind(reference.predict_image, predict_tile=predictor, METHODS=methods)(*args, **kwargs)


@torch.inference_mode()
def benchmark(image, models, repetitions=7):
    methods = ('HardEager', 'HardGraph', 'ProjectedGraph')
    def call(name):
        def predictor(*inputs, methods):
            return tile(*inputs, methods=methods, graph=name != 'HardEager')
        window = bind(reference_window, tile=predictor)
        chosen = CANDIDATE if name == 'ProjectedGraph' else 'RivalFineHard_Exact'
        return window(image, *models, (chosen,))
    for name in methods:
        call(name)
    rows = {m: {'seconds': [], 'peak_allocated_mib': []} for m in methods}
    for repeat in range(repetitions):
        for name in methods if repeat % 2 == 0 else methods[::-1]:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            call(name)
            torch.cuda.synchronize()
            rows[name]['seconds'].append(time.perf_counter()-started)
            rows[name]['peak_allocated_mib'].append(torch.cuda.max_memory_allocated()/1048576)
    for row in rows.values():
        row['median_seconds'] = statistics.median(row['seconds'])
        row['peak_allocated_mib'] = max(row['peak_allocated_mib'])
    return {'results': rows, 'repetitions': repetitions,
        'scope': 'one512 local window with full-image wide context; independent warmed alternating arms',
        'graph_resident_in_every_arm': True, 'loading_text_decoding_masks_excluded': True}


@torch.inference_mode()
def full(args, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if args.dataset == 'udd5' or manifest.get('selection_rule') != 'fixed-seed sample without replacement from key-disjoint pool':
        raise ValueError('This panel runs only seven new key-disjoint domains.')
    captured = {}
    def selected_protocol(inputs, specs):
        samples, specs, load_image, load_mask = reference.protocol(inputs, specs)
        return manifest_samples(samples, manifest, inputs.dataset), specs, load_image, load_mask
    def predictor(image, geometry, banks, vip, queries, work):
        if 'image' not in captured:
            captured.update(image=image, models=(geometry, banks, vip, queries))
        return predict_image(image, geometry, banks, vip, queries, work)
    def save(path, result):
        if 'signature' in result:
            result['signature']['gear']['competition_action'] = settings()
            result['signature']['pilot_manifest'] = manifest
        result['execution_backend'] = 'cached competition reader + original batch-one512 fine CUDA graph'
        if result['status'] == 'complete':
            result['status'] = 'finalizing'
        reference.save(path, result)
    bind(reference.main, predict_image=predictor, protocol=selected_protocol,
         METHODS=METHODS, IMPLEMENTATION=IMPLEMENTATION, save=save)(args)
    path = Path(args.output_dir)/'results.json'
    result = json.loads(path.read_text())
    if result['status'] != 'finalizing':
        raise RuntimeError('Full-image evaluation did not finish.')
    if args.shard_index == 0:
        result['execution_benchmark'] = benchmark(captured['image'], captured['models'])
    result['status'] = 'complete'
    reference.save(path, result)
    print(json.dumps({'status': 'complete', 'dataset': args.dataset, 'shard': args.shard_index,
                      'images': result['total_images']}), flush=True)


if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--sample-manifest', required=True)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    full(reference.parse_args(), known.sample_manifest)
