"""Queue finite static searches, freeze choices, then evaluate full natural sets."""
import argparse
import json
from itertools import zip_longest
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from eval_rival_fine_full import save
from eval_geometry_vip_reliability import summary
from eval_gear_ov import digest
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, idle
from run_sat_geometry_transport_suite import alive, read_json


PREFIX = 'gnss07'


def launch(root, dataset, entry, gpu, phase, shard=0, resume_search=False):
    output = root / phase / dataset if phase == 'search' else root / phase / dataset / ('s' + str(shard))
    log = output.with_suffix('.log')
    session = f'{PREFIX}_{phase}_{dataset}_s{shard}'
    if resume_search:
        log = output.with_suffix('.resume.log')
    if (output.exists() and not resume_search) or log.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Existing output/session or occupied GPU: ' + session)
    output.parent.mkdir(parents=True, exist_ok=True)
    args = [PYTHON, '-u', 'scripts/eval_natural_static_search.py', '--dataset', dataset,
        '--mode', phase, '--suite-root', str(root), '--data-root', entry['data_root'],
        '--output-dir', str(output), '--original-cache', str(TOOL / 'results/curated20_patchonly2_full_20261006/text_cache' / (dataset + '.pt')),
        '--dinov3-repo', str(TOOL / 'dinov3_hub'), '--checkpoint-dir', str(BASE / 'ckpt/DINO'),
        '--upstream-root', str(BASE / 'third_party/VIP_official_5bd25ee'),
        '--num-shards', str(entry['shards'] if phase == 'full' else 1), '--shard-index', str(shard)]
    if resume_search:
        args.append('--resume-search')
    env = ['env', f'CUDA_VISIBLE_DEVICES={gpu}', f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
        'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2', 'TOKENIZERS_PARALLELISM=false']
    shell = 'cd ' + shlex.quote(str(TOOL)) + ' && exec ' + shlex.join(env + args) + ' > ' + shlex.quote(str(log)) + ' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, shell], check=True)
    return dict(dataset=dataset, phase=phase, shard=shard, session=session, gpu=gpu, output=str(output), log=str(log))


def verify(job):
    row = read_json(Path(job['output']) / 'results.json')
    if not row or row['status'] != 'complete' or row['processed_images'] != row['total_images'] or not row['weights_frozen'] or not row['head_weights_unchanged']:
        raise RuntimeError('Worker exited without complete frozen results: ' + job['log'])
    if job['phase'] == 'search':
        if not read_json(Path(job['output']) / 'selection.json'):
            raise RuntimeError('Missing frozen search choice.')
    else:
        with np.load(Path(job['output']) / 'per_image_confusions.npz', allow_pickle=False) as arrays:
            if arrays['sample_keys'].tolist() != row['signature']['sample_keys']:
                raise RuntimeError('Per-image keys differ.')
            for m in row['signature']['methods']:
                if not np.array_equal(arrays[job['dataset'] + '__' + m].sum(0), row['metrics'][job['dataset']][m]['confusion_matrix']):
                    raise RuntimeError('Per-image confusion sums differ.')
    return row


def merge(root, dataset, entry):
    folders = [root / 'full' / dataset / ('s' + str(i)) for i in range(entry['shards'])]
    rows = [read_json(p / 'results.json') for p in folders]
    first = rows[0]['signature']
    keys = [k for items in zip_longest(*(r['signature']['sample_keys'] for r in rows)) for k in items if k is not None]
    if len(keys) != entry['total_images'] or len(set(keys)) != len(keys) or digest(keys) != first['global_sample_keys_sha256']:
        raise RuntimeError('Incomplete unique global coverage: ' + dataset)
    for row in rows:
        for field in ('implementation', 'dataset', 'methods', 'classes', 'gear', 'vocabulary', 'checkpoints', 'global_sample_count', 'global_sample_keys_sha256'):
            if row['signature'][field] != first[field]:
                raise RuntimeError('Shards differ in ' + field)
    names = first['classes'][dataset]
    matrices = {m: sum((np.asarray(r['metrics'][dataset][m]['confusion_matrix'], np.int64) for r in rows)) for m in first['methods']}
    if not np.array_equal(matrices['StaticTuned'].sum(1), matrices['VIP_Official_Finite'].sum(1)):
        raise RuntimeError('VIP and model scored different targets.')
    metrics = {m: summary(cm, names, sum(r['metrics'][dataset][m]['ignored_pixels'] for r in rows)) for m, cm in matrices.items()}
    split = {s: {m: np.zeros_like(cm) for m, cm in matrices.items()} for s in ('development', 'heldout')}
    dev = set(entry['development_keys'])
    for folder in folders:
        with np.load(folder / 'per_image_confusions.npz', allow_pickle=False) as arrays:
            for i, key in enumerate(arrays['sample_keys'].tolist()):
                group = 'development' if key in dev else 'heldout'
                for m in matrices:
                    split[group][m] += arrays[dataset + '__' + m][i]
    selected = read_json(root / 'search' / dataset / 'selection.json')
    if entry['development_source'] != 'ADE_training' and not np.array_equal(split['development']['StaticTuned'], selected['confusion_matrix']):
        raise RuntimeError('Frozen development choice does not replay in full evaluation.')
    path = root / 'full' / dataset / 'merged.json'
    if path.exists():
        raise RuntimeError('Existing merged result is preserved.')
    result = dict(status='complete', coverage_verified=True, processed_images=len(keys), total_images=len(keys),
        signature={**first, 'shard_index': None, 'sample_keys': keys}, metrics={dataset: metrics},
        split_metrics={s: {m: summary(cm, names, 0) for m, cm in group.items()} for s, group in split.items()},
        development_selection_replay_verified=True, scored_target_counts_match=True,
        parallel_wall_seconds=max(r['wall_seconds'] for r in rows),
        peak_cuda_memory_mb=max(r['peak_cuda_memory_mb'] for r in rows))
    save(path, result)
    return {m: v['mean_iou_percent'] for m, v in metrics.items()}


def main(root, resume=False):
    protocol = read_json(root / 'protocol.json')
    previous = read_json(root / 'suite_status.json')
    if previous and not resume:
        raise RuntimeError('Existing suite; inspect the saved state.')
    order = protocol['order']
    entries = protocol['datasets']
    pending = [(d, 'search', 0) for d in order]
    active, completed, merged = [], [], {}
    recovery = set()
    if resume:
        if not previous or previous['status'] != 'failed':
            raise RuntimeError('Resume requires an inspected failed suite.')
        save(root / 'suite_status.before_resume.json', previous)
        pending = [tuple(p) for p in previous['pending']]
        completed = previous['completed']
        merged = previous['merged']
        for job in previous['active']:
            row = read_json(Path(job['output']) / 'results.json')
            if alive(job['session']) or (row and row.get('status') == 'complete'):
                active.append(job)
            elif job['phase'] == 'search':
                pending.append((job['dataset'], 'search', 0))
                recovery.add(job['dataset'])
            else:
                raise RuntimeError('Full evaluation failure requires separate inspection: ' + job['log'])
        active_keys = {(j['dataset'], j['phase'], j['shard']) for j in active}
        pending = list(dict.fromkeys(p for p in pending if p not in active_keys))
        recovery.update(d for d, phase, _ in pending if phase == 'search' and (root / phase / d).exists())
    started = time.perf_counter()
    try:
        while pending or active:
            for job in active[:]:
                if alive(job['session']):
                    continue
                if resume and job['phase'] == 'search' and not job['log'].endswith('.resume.log'):
                    output = Path(job['output'])
                    tail = Path(job['log']).read_text()[-4000:]
                    if (output / 'stage_refinement.json').exists() and not (output / 'selection.json').exists() and 'ValueError: Circular reference detected' in tail:
                        active.remove(job)
                        if (job['dataset'], 'search', 0) not in pending:
                            pending.append((job['dataset'], 'search', 0))
                        recovery.add(job['dataset'])
                        continue
                verify(job)
                active.remove(job)
                completed.append(job)
                d = job['dataset']
                if job['phase'] == 'search':
                    pending.extend((d, 'full', s) for s in range(entries[d]['shards']))
                if job['phase'] == 'full' and sum(j['dataset'] == d and j['phase'] == 'full' for j in completed) == entries[d]['shards']:
                    merged[d] = merge(root, d, entries[d])
                    print(json.dumps(dict(dataset=d, full_complete=merged[d])), flush=True)
            for gpu in range(8):
                claimed = {(j['dataset'], j['phase'], j['shard']) for j in active + completed}
                pending = list(dict.fromkeys(p for p in pending if p not in claimed))
                if pending and not any(j['gpu'] == gpu for j in active) and idle(gpu):
                    d, phase, shard = pending.pop(0)
                    active.append(launch(root, d, entries[d], gpu, phase, shard, resume_search=d in recovery and phase == 'search'))
            save(root / 'suite_status.json', dict(status='running', active=active, pending=pending,
                completed=completed, merged=merged, wall_seconds=time.perf_counter()-started))
            if pending or active:
                time.sleep(10)
        save(root / 'suite_results.json', dict(status='complete', datasets=merged,
            wall_seconds=time.perf_counter()-started, total_protocol_images=sum(v['total_images'] for v in entries.values())))
        save(root / 'suite_status.json', dict(status='complete', active=[], pending=[], completed=completed, merged=merged))
    except Exception as exc:
        save(root / 'suite_status.json', dict(status='failed', error=str(exc), active=active, pending=pending, completed=completed, merged=merged))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    options = parser.parse_args()
    main(options.root, options.resume)
