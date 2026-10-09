"""Run one predeclared window-only action diagnostic on idle physical GPUs0-7."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.alias_action_capacity import IMPLEMENTATION
from run_bounded_alias_suite import save
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, SETTINGS, idle
from run_sat_geometry_transport_suite import alive, read_json


SOURCE = TOOL / 'results/class_relative_alias_rejection_screen_20261003'
ORDER = ('udd5', 'vdd', 'potsdam', 'oem', 'loveda', 'vaihingen', 'landcoverai', 'flair1')


def launch(root, dataset, gpu, smoke=False):
    if gpu not in range(8) or not idle(gpu):
        raise RuntimeError('Authorized physical GPU not idle.')
    _, data, vocabulary, _, _ = next(row for row in SETTINGS if row[0] == dataset)
    output = root / ('smoke' if smoke else dataset)
    log = root / ('smoke.log' if smoke else dataset+'.log')
    session = 'gaac03_' + ('smoke' if smoke else dataset)
    if output.exists() or log.exists() or alive(session):
        raise RuntimeError('Existing audit output/session: '+session)
    args = [PYTHON, '-u', 'scripts/audit_alias_action_capacity.py', '--dataset', dataset,
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
    expected = prior['signature']['sample_keys'][:8]
    if (not row or row['status'] != 'complete' or row['processed_images'] != 8 or row['total_images'] != 8
            or row['sample_keys'] != expected or row['signature']['sample_keys'] != expected
            or row['implementation'] != IMPLEMENTATION or not row['audit_only'] or row['used_to_fit_selector']
            or not row['target_masks_loaded'] or not row['numerical_caches_precede_masks']
            or not row['weights_frozen'] or not row['head_weights_unchanged']):
        raise RuntimeError('Incomplete or invalid audit protocol.')
    for field in ('checkpoints', 'vocabulary'):
        if row['signature'][field] != prior['signature'][field]:
            raise RuntimeError('Changed input identity: '+field)
    for field in ('geometry', 'observation'):
        if row['signature'][field] != prior['signature']['gear'][field]:
            raise RuntimeError('Changed profile: '+field)
    if row['signature']['config'] != prior['signature']['gear']['excess_reject'] or row['vocabulary_stress'] != prior['vocabulary_stress']:
        raise RuntimeError('Changed fixed rule or stress words.')
    with np.load(root/dataset/'per_image_audit.npz', allow_pickle=False) as arrays:
        if arrays['sample_keys'].tolist() != expected or len(set(expected)) != 8:
            raise RuntimeError('Duplicate or changed audit sequence.')
        for scenario, protocols in row['metrics'].items():
            for key, group in protocols.items():
                for method, metric in group.items():
                    if not np.array_equal(arrays[scenario+'__'+key+'__'+method].sum(0), metric['confusion_matrix']):
                        raise RuntimeError('Per-image matrix reconstruction failed.')
    for index in range(8):
        if not (root/dataset/'numerical_cache'/f'{index}.npz').is_file():
            raise RuntimeError('Missing pre-mask numerical cache.')
    row['coverage_verified'] = True
    save(root/dataset/'merged.json', row)
    return row


def run(root):
    root.resolve().relative_to((TOOL/'results').resolve())
    if root.exists() or any(not idle(gpu) for gpu in range(8)):
        raise RuntimeError('Existing root or occupied GPU; no launch.')
    root.mkdir(parents=True)
    for dataset in ORDER:
        prior = read_json(SOURCE/dataset/'merged.json')
        if not prior or not prior['coverage_verified']:
            raise RuntimeError('Verified source required: '+dataset)
        save(root/(dataset+'_samples.json'), {'signature': {'samples': prior['signature']['sample_keys'][:8]}})
    save(root/'protocol.json', {'implementation': IMPLEMENTATION, 'audit_only': True, 'images': 64,
        'window_only': True, 'source': str(SOURCE), 'physical_gpus': list(range(8)),
        'no_selector_fitting': True, 'no_full_rollout': True,
        'note': 'Pointwise bounds are not joint oracle mIoU; label-assisted feasible policies are not deployable methods.'})
    started = time.time()
    smoke = launch(root, 'oem', 0, True)
    while alive(smoke['session']):
        time.sleep(5)
    row = read_json(root/'smoke/results.json')
    if not row or row['status'] != 'complete' or row['target_masks_loaded'] or not row['weights_frozen'] or not row['head_weights_unchanged']:
        save(root/'suite_results.json', {'status': 'failed', 'failures': {'smoke': Path(smoke['log']).read_text()[-6000:]}})
        raise RuntimeError('Mask-free smoke failed.')
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
