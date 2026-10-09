"""Replay the retained candidate, then evaluate eight full datasets on idle A800s."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.fine_alias_view import CONFIG
from dinotool.rival_fine_full import IMPLEMENTATION, METHODS
from eval_rival_fine_full import save
from merge_geometry_semantic_innovation import main as merge_model
from merge_vip_official_shards import merge as merge_vip
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, SETTINGS, idle, reference
from run_sat_geometry_transport_suite import alive, read_json


TOTALS = {d: total for d, _, _, _, total in SETTINGS}
SHARDS = {'loveda': 4, 'flair1': 4}


def launch(root, dataset, gpu, phase, shard=0, shards=1, vip=False):
    if not idle(gpu):
        raise RuntimeError(f'Physical GPU{gpu} occupied; no launch.')
    _, data, vocabulary, _, _ = next(s for s in SETTINGS if s[0] == dataset)
    name = 'vip_corrected_vaihingen' if vip else dataset
    output = root/phase/name/(f's{shard}' if phase == 'full' else 'replay')
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix('.log')
    session = f'grfh03_{phase}_{name}_s{shard}'
    if output.exists() or log.exists() or alive(session):
        raise RuntimeError('Existing output/session: '+session)
    args = [PYTHON, '-u', 'scripts/eval_vip_official_eight.py' if vip else 'scripts/eval_rival_fine_full.py',
        '--dataset', dataset, '--dinov3-repo', str(TOOL/'dinov3_hub'), '--checkpoint-dir', str(BASE/'ckpt/DINO'),
        '--data-root', str(Path('/data/test/datasets')/data), '--vocabulary-config', str(TOOL/'configs'/vocabulary),
        '--upstream-root', str(BASE/'third_party/VIP_official_5bd25ee'), '--output-dir', str(output),
        '--num-shards', str(shards), '--shard-index', str(shard)]
    if phase == 'replay':
        args.append('--replay')
    env = ['env', f'CUDA_VISIBLE_DEVICES={gpu}', f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
           'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2', 'TOKENIZERS_PARALLELISM=false']
    shell = f'cd {shlex.quote(str(TOOL))} && exec {shlex.join(env)} nice -n 10 {shlex.join(args)} > {shlex.quote(str(log))} 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, shell], check=True)
    return {'dataset': dataset, 'name': name, 'phase': phase, 'shard': shard, 'shards': shards,
            'gpu': gpu, 'session': session, 'output': str(output), 'log': str(log), 'vip': vip}


def verify_job(job):
    output = Path(job['output'])
    row = read_json(output/'results.json')
    if not row or row['status'] != 'complete' or row['processed_images'] != row['total_images']:
        raise RuntimeError('Incomplete job: '+job['session'])
    if job['phase'] == 'replay':
        if (row['target_masks_loaded'] or not row['weights_frozen'] or not row['head_weights_unchanged']
                or row['implementation'] != IMPLEMENTATION or row['processed_images'] != 8):
            raise RuntimeError('Mask-free full-model replay contract failed.')
    elif not job['vip']:
        if not row['weights_frozen'] or not row['head_weights_unchanged'] or not row['target_masks_used_only_after_prediction']:
            raise RuntimeError('Frozen evaluation contract failed.')
        with np.load(output/'per_image_confusions.npz', allow_pickle=False) as arrays:
            if arrays['sample_keys'].tolist() != row['signature']['sample_keys']:
                raise RuntimeError('Per-image sample sequence differs.')
            for p, ms in row['metrics'].items():
                for m, metric in ms.items():
                    if not np.array_equal(arrays[p+'__'+m].sum(0), metric['confusion_matrix']):
                        raise RuntimeError('Per-image confusion reconstruction differs.')
    return row


def verify_dataset(root, dataset, vip=False):
    name = 'vip_corrected_vaihingen' if vip else dataset
    shards = SHARDS.get(dataset, 1)
    inputs = [str(root/'full'/name/f's{s}') for s in range(shards)]
    output = root/'full'/name/'merged.json'
    if vip:
        merge_vip(inputs, str(output))
    else:
        merge_model(argparse.Namespace(inputs=inputs, output=str(output)))
    row = read_json(output)
    ref = read_json(reference(dataset))
    if (not row['coverage_verified'] or row['processed_images'] != TOTALS[dataset]
            or row['total_images'] != TOTALS[dataset]
            or row['signature']['sample_keys'] != ref['signature']['sample_keys']):
        raise RuntimeError('Full global sample coverage differs: '+name)
    if not vip:
        if (row['signature']['implementation'] != IMPLEMENTATION
                or row['signature']['vocabulary'] != ref['signature']['vocabulary']
                or row['signature']['checkpoints'] != ref['signature']['checkpoints']
                or row['signature']['gear']['admission'] != asdict(CONFIG)):
            raise RuntimeError('Frozen source/vocabulary/config differs: '+name)
        for p in row['metrics']:
            if not np.array_equal(row['metrics'][p]['Geometry']['confusion_matrix'], ref['metrics'][p]['Geometry']['confusion_matrix']):
                raise RuntimeError('Original full Geometry baseline changed: '+name+'/'+p)
        row.update(weights_frozen=True, head_weights_unchanged=True, exact_reference_geometry_confusion=True,
                   per_image_confusions_verified=True)
        save(output, row)
    return str(output)


def monitor(root, active, pending, completed, failures):
    verified_jobs = set()
    for d in TOTALS:
        for s in range(SHARDS.get(d, 1)):
            row = read_json(root/'full'/d/f's{s}'/'results.json')
            if row and row['status'] == 'complete':
                verified_jobs.add((d, s))
    while active or pending:
        for gpu, job in list(active.items()):
            if alive(job['session']):
                continue
            try:
                verify_job(job)
                verified_jobs.add((job['name'], job['shard']))
                if job['phase'] == 'replay':
                    completed[job['name']] = job['output']+'/results.json'
                elif all((job['name'], s) in verified_jobs for s in range(job['shards'])):
                    completed[job['name']] = verify_dataset(root, job['dataset'], job['vip'])
            except Exception as error:
                failures[job['session']] = {'error': str(error), 'log_tail': Path(job['log']).read_text()[-6000:]}
            del active[gpu]
        if failures:
            pending.clear()
        for dataset, vip, shard, shards in list(pending):
            free = next((g for g in range(4) if g not in active and idle(g)), None)
            if free is None:
                break
            job = launch(root, dataset, free, 'full', shard=shard, shards=shards, vip=vip)
            active[free] = job
            pending.remove((dataset, vip, shard, shards))
        save(root/'suite_status.json', {'status': 'running' if active or pending else 'failed' if failures else 'complete',
            'phase': next(iter(active.values()))['phase'] if active else 'full', 'active': list(active.values()),
            'pending': pending, 'completed': completed, 'failures': failures})
        if active or pending:
            time.sleep(15)


def main(root):
    root.resolve().relative_to((TOOL/'results').resolve())
    if root.exists() or any(not idle(g) for g in range(8)):
        raise RuntimeError('Existing root or occupied GPU; nothing launched.')
    root.mkdir(parents=True)
    save(root/'protocol.json', {'implementation': IMPLEMENTATION, 'config': asdict(CONFIG), 'methods': METHODS,
        'physical_gpus': list(range(8)), 'full_totals': TOTALS, 'fixed_aliases_per_class': 20,
        'fine_physical_side': 256, 'fine_encoder_side': 512, 'local_tile_side': 512, 'local_overlap': 128,
        'target_label_tuning': False, 'prior_window_replay_tolerance': 1e-4, 'operator_tolerance': 1e-10,
        'vip_comparator': 'Historical eight-domain VIP; separately rerun Vaihingen on corrected IRRG input.',
        'development_status': 'Exploratory full validation; method selected using prior developed windows.'})
    started = time.time()
    completed, failures = {}, {}
    active = {g: launch(root, s[0], g, 'replay') for g, s in enumerate(SETTINGS)}
    monitor(root, active, [], completed, failures)
    save(root/'replay_summary.json', {'completed': completed, 'failures': failures})
    if failures or len(completed) != 8:
        save(root/'suite_results.json', {'status': 'failed', 'phase': 'replay', 'failures': failures})
        return
    completed = {}
    active = {g: launch(root, d, g, 'full') for g, d in enumerate(('udd5', 'vdd', 'potsdam', 'oem'))}
    active.update({g: launch(root, 'flair1', g, 'full', shard=g-4, shards=4) for g in range(4, 8)})
    pending = [('loveda', False, s, 4) for s in range(4)] + [('vaihingen', False, 0, 1),
        ('landcoverai', False, 0, 1), ('vaihingen', True, 0, 1)]
    monitor(root, active, pending, completed, failures)
    save(root/'suite_results.json', {'status': 'failed' if failures else 'complete', 'completed': completed,
        'failures': failures, 'suite_wall_seconds': time.time()-started})


def resume_full(root):
    state = read_json(root/'suite_status.json')
    if not state or state['phase'] != 'full' or state['failures'] or (root/'suite_results.json').exists():
        raise RuntimeError('Only the active healthy full suite can be resumed.')
    active = {j['gpu']: j for j in state['active']}
    if any(j['dataset'] == 'loveda' for j in active.values()) or (root/'full/loveda/s0').exists():
        raise RuntimeError('LoveDA already started; cannot change sharding.')
    expected = [['loveda', False], ['vaihingen', False], ['landcoverai', False], ['vaihingen', True]]
    if state['pending'] != expected:
        raise RuntimeError('Scheduler transfer requires the original untouched pending queue.')
    pending = [('loveda', False, s, 4) for s in range(4)] + [('vaihingen', False, 0, 1),
        ('landcoverai', False, 0, 1), ('vaihingen', True, 0, 1)]
    protocol = read_json(root/'protocol.json')
    protocol['full_shards'] = {d: SHARDS.get(d, 1) for d in TOTALS}
    save(root/'protocol.json', protocol)
    started = root.stat().st_ctime
    monitor(root, active, pending, state['completed'], state['failures'])
    save(root/'suite_results.json', {'status': 'failed' if state['failures'] else 'complete',
        'completed': state['completed'], 'failures': state['failures'], 'suite_wall_seconds': time.time()-started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--resume-full', action='store_true')
    args = parser.parse_args()
    (resume_full if args.resume_full else main)(args.root)
