"""Run the frozen64-window source pilot on authorized idle physical GPUs0-7."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.target_context_alias import IMPLEMENTATION, METHODS, TargetContextConfig
from run_bounded_alias_suite import save
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, SETTINGS, idle
from run_sat_geometry_transport_suite import alive, read_json


SOURCE = TOOL/'results/alias_action_capacity_audit_20261003'
ORDER = ('udd5', 'vdd', 'potsdam', 'oem', 'loveda', 'vaihingen', 'landcoverai', 'flair1')


def launch(root, dataset, gpu, smoke=False):
    if gpu not in range(8) or not idle(gpu):
        raise RuntimeError('Authorized physical GPU not idle.')
    _, data, vocabulary, _, _ = next(row for row in SETTINGS if row[0] == dataset)
    output = root/('smoke' if smoke else dataset)
    log = root/('smoke.log' if smoke else dataset+'.log')
    session = 'gtca03_'+('smoke' if smoke else dataset)
    if output.exists() or log.exists() or alive(session):
        raise RuntimeError('Existing output/session: '+session)
    args = [PYTHON, '-u', 'scripts/eval_target_context_alias.py', '--dataset', dataset,
        '--dinov3-repo', str(TOOL/'dinov3_hub'), '--checkpoint-dir', str(BASE/'ckpt/DINO'),
        '--data-root', str(Path('/data/test/datasets')/data), '--vocabulary-config', str(TOOL/'configs'/vocabulary),
        '--upstream-root', str(BASE/'third_party/VIP_official_5bd25ee'), '--output-dir', str(output),
        '--source-diagnostic', str(root/(dataset+'_samples.json')), '--num-shards', '1', '--shard-index', '0']
    if smoke:
        args.append('--smoke')
    env = ['env', 'CUDA_VISIBLE_DEVICES='+str(gpu), 'PYTHONPATH='+str(TOOL)+':'+str(TOOL)+'/scripts:'+str(THIRD),
           'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2', 'TOKENIZERS_PARALLELISM=false']
    shell = 'cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env)+' nice -n 10 '+shlex.join(args)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, shell], check=True)
    return {'dataset': dataset, 'gpu': gpu, 'session': session, 'output': str(output), 'log': str(log)}


def verify(root, dataset):
    row = read_json(root/dataset/'results.json')
    prior = read_json(SOURCE/dataset/'merged.json')
    if (not row or row['status'] != 'complete' or row['processed_images'] != 8 or row['total_images'] != 8
            or row['sample_keys'] != prior['sample_keys'] or row['signature'] != prior['signature']
            or row['implementation'] != IMPLEMENTATION or row['config'] != asdict(TargetContextConfig())
            or not row['source_pilot_only'] or not row['numerical_caches_precede_masks']
            or not row['target_masks_loaded'] or not row['weights_frozen'] or not row['head_weights_unchanged']):
        raise RuntimeError('Invalid completed source pilot.')
    with np.load(root/dataset/'per_image_confusions.npz', allow_pickle=False) as current, np.load(
            SOURCE/dataset/'per_image_audit.npz', allow_pickle=False) as previous:
        if current['sample_keys'].tolist() != prior['sample_keys'] or len(set(prior['sample_keys'])) != 8:
            raise RuntimeError('Changed/duplicate sample sequence.')
        for key, group in row['metrics'].items():
            if set(group) != set(METHODS):
                raise RuntimeError('Missing matched arms.')
            for method, metric in group.items():
                if not np.array_equal(current[key+'__'+method].sum(0), metric['confusion_matrix']):
                    raise RuntimeError('Confusion reconstruction mismatch.')
            for method in ('Geometry', 'Anchored_Exact', 'ClassRelativeReject_CG'):
                if not np.array_equal(current[key+'__'+method], previous['clean__'+key+'__'+method]):
                    raise RuntimeError('Original per-image controls changed: '+method)
    with np.load(root/dataset/'per_image_transitions.npz', allow_pickle=False) as arrays:
        for key, group in row['transitions'].items():
            base = np.asarray(row['metrics'][key]['Anchored_Exact']['confusion_matrix'])
            for method, value in group.items():
                counts = arrays[key+'__'+method].sum(0)
                if (not np.array_equal(counts, value['counts']) or not np.array_equal(counts.sum(1).T, base)
                        or not np.array_equal(counts.sum(0).T, row['metrics'][key][method]['confusion_matrix'])):
                    raise RuntimeError('Transition endpoints changed: '+method)
    for image in row['diagnostics'].values():
        if image['support_shuffle_spectrum_error'] != 0:
            raise RuntimeError('Spatial support spectrum changed.')
        for key in row['metrics']:
            if any(image[key][name] != 0 for name in ('alias_shuffle_spectrum_error', 'canonical_rejection', 'identity_writer_error')):
                raise RuntimeError('Retention/identity invariants failed.')
    if any(not (root/dataset/'numerical_cache'/f'{index}.npz').is_file() for index in range(8)):
        raise RuntimeError('Missing pre-mask caches.')
    row['coverage_verified'] = True
    row['exact_per_image_controls'] = ['Geometry', 'Anchored_Exact', 'ClassRelativeReject_CG']
    save(root/dataset/'merged.json', row)


def run(root):
    root.resolve().relative_to((TOOL/'results').resolve())
    if root.exists() or any(not idle(gpu) for gpu in range(8)):
        raise RuntimeError('Existing root or occupied GPU; no launch.')
    root.mkdir(parents=True)
    for dataset in ORDER:
        prior = read_json(SOURCE/dataset/'merged.json')
        if not prior or not prior['coverage_verified']:
            raise RuntimeError('Verified source required: '+dataset)
        save(root/(dataset+'_samples.json'), {'signature': {'samples': prior['sample_keys']}})
    save(root/'protocol.json', {'implementation': IMPLEMENTATION, 'config': asdict(TargetContextConfig()),
        'images': 64, 'window_only': True, 'source': str(SOURCE), 'physical_gpus': list(range(8)),
        'no_selector_fitting': True, 'no_full_rollout': True, 'methods': METHODS})
    started = time.time()
    smoke = launch(root, 'oem', 0, True)
    while alive(smoke['session']):
        time.sleep(5)
    row = read_json(root/'smoke/results.json')
    if not row or row['status'] != 'complete' or row['target_masks_loaded'] or not row['weights_frozen'] or not row['head_weights_unchanged']:
        save(root/'suite_results.json', {'status': 'failed', 'failures': {'smoke': Path(smoke['log']).read_text()[-6000:]}})
        raise RuntimeError('Mask-free source smoke failed.')
    pending, active, completed, failures = dict(enumerate(ORDER)), {}, [], {}
    while pending or active:
        for gpu, job in list(active.items()):
            if alive(job['session']):
                continue
            try:
                verify(root, job['dataset'])
                completed.append(job['dataset'])
            except Exception as error:
                failures[job['dataset']] = {'error': str(error), 'log_tail': Path(job['log']).read_text()[-6000:]}
            del active[gpu]
        if failures:
            pending.clear()
        for gpu, dataset in list(pending.items()):
            if idle(gpu):
                active[gpu] = launch(root, dataset, gpu)
                del pending[gpu]
        state = {'status': 'running' if pending or active else 'failed' if failures else 'complete',
            'active': list(active.values()), 'completed': completed, 'queued': list(pending.values()),
            'failures': failures, 'elapsed_seconds': time.time()-started}
        save(root/'suite_status.json', state)
        if pending or active:
            time.sleep(10)
    save(root/'suite_results.json', state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
