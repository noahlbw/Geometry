"""Frozen fine-only source on complete fixed inputs and independent cost arms."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

from dinotool.fine_responsibility_reader import IMPLEMENTATION, METHODS, PRIMARY, PROJECTED, WIDE, WITHIN, responsibility_scores
from dinotool.rival_competition_admission import manifest_samples
from eval_rival_fine_graph import bind, graph_features
from eval_rival_competition_admission import window as reference_window
import eval_rival_fine_full as reference
import eval_rival_competition_transfer as historical


@torch.inference_mode()
def tile(*args, methods=METHODS, **kwargs):
    observe = bind(reference.observe_fine, fine_patch_features=graph_features)
    def reader(*inputs):
        return responsibility_scores(*inputs, methods=methods)
    return bind(reference.predict_tile, observe_fine=observe, retained_scores=reader)(*args, **kwargs)


@torch.inference_mode()
def predict_image(*args, methods=METHODS, **kwargs):
    def predictor(*inputs):
        return tile(*inputs, methods=methods)
    return bind(reference.predict_image, METHODS=methods, predict_tile=predictor)(*args, **kwargs)


@torch.inference_mode()
def benchmark(image, models, repetitions=7):
    choices = {'HardEager': ('RivalFineHard_Exact', False), 'HardGraph': ('RivalFineHard_Exact', True),
        'ProjectedGraph': (PROJECTED, True), 'FineOnlyGraph': (PRIMARY, True),
        'WithinGraph': (WITHIN, True), 'WideOnlyGraph': (WIDE, True)}
    full = bind(reference_window, tile=tile)(image, *models, METHODS)[1]
    def call(name):
        method, graph = choices[name]
        def predictor(*inputs, methods):
            if method in (PRIMARY, WITHIN, WIDE):
                return tile(*inputs, methods=methods)
            return historical.tile(*inputs, methods=methods, graph=graph)
        return bind(reference_window, tile=predictor)(image, *models, (method,))
    for name, (method, _) in choices.items():
        scores = call(name)[1]
        if not all(torch.equal(scores[p][method], full[p][method]) for p in scores):
            raise RuntimeError('Real requested/all-arm benchmark score differs: '+name)
    rows = {name: {'seconds': [], 'peak_allocated_mib': []} for name in choices}
    for repeat in range(repetitions):
        for name in choices if repeat % 2 == 0 else tuple(choices)[::-1]:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            call(name)
            torch.cuda.synchronize()
            rows[name]['seconds'].append(time.perf_counter()-start)
            rows[name]['peak_allocated_mib'].append(torch.cuda.max_memory_allocated()/1048576)
    for row in rows.values():
        row['median_seconds'] = statistics.median(row['seconds'])
        row['peak_allocated_mib'] = max(row['peak_allocated_mib'])
    return {'results': rows, 'repetitions': repetitions, 'singleton_all_arm_scores_bitwise_equal': True,
        'scope': 'one512 local window + full-image wide context; warmed synchronized alternating independent arms',
        'graph_resident_in_every_arm': True, 'loading_text_decoding_masks_excluded': True}


@torch.inference_mode()
def full(args, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if len(manifest['sample_keys']) != (40 if args.dataset == 'udd5' else 16):
        raise ValueError('Original complete-input transfer manifest required.')
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
            result['signature']['pilot_manifest'] = manifest
            result['signature']['fine_responsibility'] = {
                'formula_frozen': 'original fine pair responsibility + protection + survivor-mass redistribution',
                'projection_posterior_H_unchanged': True, 'additional_visual_forwards': 0,
                'prior_robustness_gate_failed': True, 'not_automatic_final_model_promotion': True}
        result['execution_backend'] = 'requested-endpoint equivalent reader + original batch-one512 fine CUDA graph'
        if result['status'] == 'complete':
            result['status'] = 'finalizing'
        reference.save(path, result)
    bind(reference.main, predict_image=predictor, protocol=selected_protocol, METHODS=METHODS,
        IMPLEMENTATION=IMPLEMENTATION, save=save)(args)
    path = Path(args.output_dir)/'results.json'
    result = json.loads(path.read_text())
    if result['status'] != 'finalizing':
        raise RuntimeError('Complete-input responsibility evaluation did not finish.')
    result['execution_benchmark'] = benchmark(captured['image'], captured['models'])
    result['status'] = 'complete'
    reference.save(path, result)
    print(json.dumps({'status': 'complete', 'dataset': args.dataset, 'images': result['total_images']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--sample-manifest', required=True)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    full(reference.parse_args(), known.sample_manifest)
