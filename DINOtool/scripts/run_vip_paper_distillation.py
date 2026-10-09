"""Schedule mask-free VIP distillation and cache-paired evaluation on idle A800s."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import zip_longest
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.vip_paper_distillation import select_from_scores
from eval_vip_official_eight import Confusion, digest
from eval_vip_paper_distillation import ARMS, CONFIG, CONVENTIONS, DATASETS, IMPLEMENTATION, save
from merge_vip_official_shards import merge as merge_official
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, SETTINGS, idle


OFFICIAL = ('vdd', 'potsdam', 'vaihingen')
SHARDS = dict(loveda=2, udd5=1, oem=1, landcoverai=1, flair1=4, vdd=1, potsdam=1, vaihingen=1)
TOTALS = {d: n for d, _, _, _, n in SETTINGS}
EXCLUDED_SHARD_FIELDS = ('sample_keys', 'sample_keys_sha256', 'sample_count', 'shard_index')


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def alive(session):
    return subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0


def launch(root, dataset, phase, shard, gpu, official20=False):
    if not idle(gpu):
        raise RuntimeError(f'GPU{gpu} occupied; not launching.')
    _, data, vocab, _, _ = next(row for row in SETTINGS if row[0] == dataset)
    shards = 1 if phase in ('smoke', 'official') else SHARDS[dataset]
    output = root/phase/dataset/f's{shard}'
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix('.log')
    prefix = 'vpd20_03' if official20 else 'vpd03'
    session = f'{prefix}_{phase}_{dataset}_s{shard}'
    if output.exists() or log.exists() or alive(session):
        raise RuntimeError('Refusing existing output/session: '+session)
    args = [PYTHON, '-u', 'scripts/eval_vip_paper_distillation.py', '--phase', phase, '--dataset', dataset,
        '--dinov3-repo', str(TOOL/'dinov3_hub'), '--checkpoint-dir', str(BASE/'ckpt/DINO'),
        '--data-root', str(Path('/data/test/datasets')/data), '--vocabulary-config', str(TOOL/'configs'/vocab),
        '--upstream-root', str(BASE/'third_party/VIP_official_5bd25ee'), '--output-dir', str(output),
        '--reference-json', str(root/'references'/(dataset+'.json')), '--num-shards', str(shards),
        '--shard-index', str(shard)]
    if official20:
        args += ['--candidate-reference-json', str(root/'candidate_references'/(dataset+'.json'))]
    if phase == 'official':
        args = [PYTHON, '-u', 'scripts/eval_vip_official_eight.py', '--finite-empty-rows', '--dataset', dataset,
            '--dinov3-repo', str(TOOL/'dinov3_hub'), '--checkpoint-dir', str(BASE/'ckpt/DINO'),
            '--data-root', str(Path('/data/test/datasets')/data), '--upstream-root', str(BASE/'third_party/VIP_official_5bd25ee'),
            '--output-dir', str(output), '--num-shards', '1', '--shard-index', '0']
    if phase == 'evaluation':
        args += ['--source-dir', str(root/'selection'/dataset/f's{shard}'),
                 '--selection-json', str(root/'selection'/dataset/'selected.json')]
    env = ['env', f'CUDA_VISIBLE_DEVICES={gpu}', f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
           'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2', 'TOKENIZERS_PARALLELISM=false']
    shell = f'cd {shlex.quote(str(TOOL))} && exec {shlex.join(env)} nice -n 10 {shlex.join(args)} > {shlex.quote(str(log))} 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, shell], check=True)
    print(f'Started {session} on GPU{gpu}', flush=True)
    return dict(dataset=dataset, phase=phase, shard=shard, gpu=gpu, session=session, output=str(output), log=str(log))


def complete_shards(root, dataset, phase):
    paths = [root/phase/dataset/f's{s}'/'results.json' for s in range(SHARDS[dataset])]
    rows = [read(path) for path in paths]
    if any(not r or r['status'] != 'complete' or r['processed_images'] != r['total_images'] for r in rows):
        raise RuntimeError('Complete shards required: '+dataset+'/'+phase)
    first = rows[0]['signature']
    source = {k: v for k, v in first.items() if k not in EXCLUDED_SHARD_FIELDS}
    if first['implementation'] != IMPLEMENTATION or first['config'] != asdict(CONFIG):
        raise RuntimeError('Distillation rule changed.')
    keys = []
    for s, row in enumerate(rows):
        sig = row['signature']
        if (sig['shard_index'] != s or sig['num_shards'] != len(rows)
                or row['total_images'] != len(sig['sample_keys'])
                or digest(sig['sample_keys']) != sig['sample_keys_sha256']
                or source != {k: v for k, v in sig.items() if k not in EXCLUDED_SHARD_FIELDS}):
            raise RuntimeError('Shard signature differs: '+str(paths[s]))
    keys = [k for group in zip_longest(*(r['signature']['sample_keys'] for r in rows)) for k in group if k is not None]
    ref = read(root/'references'/(dataset+'.json'))
    if (len(set(keys)) != TOTALS[dataset] or keys != ref['signature']['sample_keys']
            or digest(keys) != first['global_sample_keys_sha256']):
        raise RuntimeError('Incomplete or changed full coverage: '+dataset)
    return rows, source, keys, ref


def freeze_selection(root, dataset):
    rows, source, keys, _ = complete_shards(root, dataset, 'selection')
    if any(row['target_masks_loaded'] or not row['weights_frozen'] for row in rows):
        raise RuntimeError('Selection contract violated.')
    banks = {}
    for p, meta in source['query_metadata'].items():
        state = {field: np.asarray([r['statistics'][p][field] for r in rows]).sum(0).tolist()
                 for field in ('vg_sum', 'sc_sum', 'observed_images')}
        if max(state['observed_images']) > len(keys):
            raise RuntimeError('Alias image count exceeds full coverage.')
        banks[p] = select_from_scores(state, meta['names'], meta['aliases'], meta['parents'], meta['cosine'], CONFIG)
        banks[p]['statistics'] = state
    output = root/'selection'/dataset/'selected.json'
    if output.exists():
        raise RuntimeError('Refusing to replace a frozen selection: '+str(output))
    save(output, dict(status='complete', coverage_verified=True, processed_images=len(keys), total_images=len(keys),
        sample_keys=keys, sample_keys_sha256=digest(keys), source_signature=source, banks=banks,
        target_masks_loaded=False, transductive=True, selection_parallel_wall_seconds=max(r['wall_seconds'] for r in rows),
        selection_aggregate_gpu_seconds=sum(r['wall_seconds'] for r in rows),
        selection_peak_cuda_memory_mb=max(r['peak_cuda_memory_mb'] for r in rows)))
    print(json.dumps(dict(dataset=dataset, selected_counts={p: b['selected_counts'] for p, b in banks.items()})), flush=True)
    return str(output)


def merge_evaluation(root, dataset):
    rows, source, keys, ref = complete_shards(root, dataset, 'evaluation')
    selected = read(root/'selection'/dataset/'selected.json')
    if any(not row['target_masks_loaded_only_after_prediction'] or row['selection'] != selected['banks'] for row in rows):
        raise RuntimeError('Evaluation selection/mask ordering changed.')
    if len(set(row['selection_sha256'] for row in rows)) != 1:
        raise RuntimeError('Selected vocabulary identity differs across evaluation shards.')
    metrics = {}
    for p, names in source['scored_classes'].items():
        metrics[p] = {}
        for arm in ARMS:
            collector = Confusion(tuple(names), len(source['query_metadata'][p]['names']))
            for s, row in enumerate(rows):
                metric = row['metrics'][p][arm]
                with np.load(root/'evaluation'/dataset/f's{s}'/'per_image_confusions.npz', allow_pickle=False) as arrays:
                    if (arrays['sample_keys'].tolist() != row['signature']['sample_keys'] or not np.array_equal(
                            arrays[p+'__'+arm].sum(0), metric['confusion_matrix'])):
                        raise RuntimeError('Per-image confusion verification failed.')
                collector.matrix += np.asarray(metric['confusion_matrix'], dtype=np.int64)
                collector.ignored += metric['ignored_pixels']
            metrics[p][arm] = collector.summary()
        baseline = metrics[p]['VIP_All20']
        if not np.array_equal(np.asarray(baseline['confusion_matrix']).sum(1),
                              np.asarray(ref['metrics'][p]['confusion_matrix']).sum(1)) or (
                baseline['ignored_pixels'] != ref['metrics'][p]['ignored_pixels']):
            raise RuntimeError('Historical scored target counts differ: '+dataset+'/'+p)
        if not np.array_equal(np.asarray(baseline['confusion_matrix']).sum(1),
                              np.asarray(metrics[p]['VIP_Distilled']['confusion_matrix']).sum(1)):
            raise RuntimeError('Scored target counts differ.')
    output = root/'evaluation'/dataset/'merged.json'
    if output.exists():
        raise RuntimeError('Refusing an existing merged evaluation.')
    save(output, dict(status='complete', coverage_verified=True, processed_images=len(keys), total_images=len(keys),
        sample_keys=keys, sample_keys_sha256=digest(keys), source_signature=source, selection=selected['banks'], metrics=metrics,
        historical_all20_confusion_equal={p: bool(np.array_equal(ms['VIP_All20']['confusion_matrix'],
            ref['metrics'][p]['confusion_matrix'])) for p, ms in metrics.items()}, per_image_confusions_verified=True,
        empty_proxy_rows=sum(r['empty_proxy_rows'] for r in [read(root/'selection'/dataset/f's{s}'/'results.json')
                             for s in range(SHARDS[dataset])]),
        observed_proxy_rows=sum(r['observed_proxy_rows'] for r in [read(root/'selection'/dataset/f's{s}'/'results.json')
                                for s in range(SHARDS[dataset])]),
        evaluation_parallel_wall_seconds=max(r['wall_seconds'] for r in rows),
        evaluation_aggregate_gpu_seconds=sum(r['wall_seconds'] for r in rows),
        evaluation_peak_cuda_memory_mb=max(r['peak_cuda_memory_mb'] for r in rows),
        **{k: v for k, v in selected.items() if k.startswith('selection_')}))
    print(json.dumps(dict(dataset=dataset, metrics={p: {a: m['mean_iou_percent'] for a, m in ms.items()}
                                                  for p, ms in metrics.items()})), flush=True)
    return str(output)


def verify_official(root, dataset):
    output = root/'official'/dataset/'merged.json'
    if output.exists():
        raise RuntimeError('Refusing an existing official merged result.')
    merged = merge_official([str(root/'official'/dataset/'s0')], str(output))
    ref = read(root/'references'/(dataset+'.json'))
    signature, old = merged['signature'], ref['signature']
    if (merged['processed_images'] != TOTALS[dataset] or merged['total_images'] != TOTALS[dataset]
            or signature['sample_keys'] != old['sample_keys'] or signature['implementation'] != 'vip-official-5bd25ee-finite-rows-20261003'
            or any(signature[k] != old[k] for k in ('settings', 'checkpoint_manifest', 'vocabulary_sha256', 'alias_counts'))):
        raise RuntimeError('Official input/vocabulary/settings/coverage differ: '+dataset)
    for p, metric in merged['metrics'].items():
        if not np.array_equal(np.asarray(metric['confusion_matrix']).sum(1),
                              np.asarray(ref['metrics'][p]['confusion_matrix']).sum(1)):
            raise RuntimeError('Official scored targets differ.')
    row = read(root/'official'/dataset/'s0'/'results.json')
    merged.update(empty_proxy_rows=row['empty_proxy_rows'], observed_proxy_rows=row['observed_proxy_rows'])
    save(output, merged)
    return str(output)


def run(root, official20=False):
    root.resolve().relative_to((TOOL/'results').resolve())
    datasets = ('vdd', 'potsdam') if official20 else DATASETS
    officials = () if official20 else OFFICIAL
    gpus = tuple(range(2, 8)) if official20 else tuple(range(8))
    if official20:
        SHARDS.update(vdd=2, potsdam=4)
    if (root/'protocol.json').exists() or any(not idle(g) for g in gpus):
        raise RuntimeError('Existing controller protocol or occupied GPU; nothing launched.')
    if any(not (root/'references'/(d+'.json')).is_file() for d in (*datasets, *officials)):
        raise RuntimeError('Missing historical comparator references.')
    if official20 and any(not (root/'candidate_references'/(d+'.json')).is_file() for d in datasets):
        raise RuntimeError('Missing verified full20 candidate references.')
    save(root/'protocol.json', dict(implementation=IMPLEMENTATION, config=asdict(CONFIG), conventions=CONVENTIONS,
        datasets=datasets, shards={d: SHARDS[d] for d in (*datasets, *officials)},
        totals={d: TOTALS[d] for d in (*datasets, *officials)}, candidate_count=20, fixed_vip_readout=True,
        official_short_query_numeric_replays=officials, physical_gpus=gpus,
        target_label_tuning=False, use_all_unlabeled_evaluation_images=True, arms=ARMS,
        source_paper='https://arxiv.org/html/2605.12325v2'))
    started = time.time()
    active, pending, failures, smoke_done, selections, completed = {}, [], {}, set(), {}, {}
    for gpu, dataset in zip(gpus, datasets):
        active[gpu] = launch(root, dataset, 'smoke', 0, gpu, official20)
    full_started = False
    verified = {'selection': set(), 'evaluation': set()}
    while active or pending or not full_started:
        for gpu, job in list(active.items()):
            if alive(job['session']):
                continue
            try:
                row = read(Path(job['output'])/'results.json')
                if not row or row['status'] != 'complete' or row['processed_images'] != row['total_images']:
                    raise RuntimeError('Worker exited without a complete result.')
                dataset, phase = job['dataset'], job['phase']
                if phase == 'smoke':
                    if row['target_masks_loaded'] or not row['weights_unchanged'] or not row['weights_frozen']:
                        raise RuntimeError('Unchanged mask-free smoke contract failed.')
                    smoke_done.add(dataset)
                elif phase == 'official':
                    completed[dataset] = verify_official(root, dataset)
                else:
                    verified[phase].add((dataset, job['shard']))
                    if all((dataset, s) in verified[phase] for s in range(SHARDS[dataset])):
                        if phase == 'selection':
                            selections[dataset] = freeze_selection(root, dataset)
                            pending.extend((dataset, 'evaluation', s) for s in range(SHARDS[dataset]))
                        else:
                            completed[dataset] = merge_evaluation(root, dataset)
            except Exception as error:
                log = Path(job['log'])
                failures[job['session']] = dict(error=str(error), log_tail=log.read_text()[-6000:] if log.exists() else 'missing log')
                print(json.dumps(failures[job['session']]), flush=True)
            del active[gpu]
        if failures:
            pending.clear()
            full_started = True
        elif not full_started and not active:
            if smoke_done != set(datasets):
                raise RuntimeError('Incomplete mask-free smoke.')
            full_started = True
            if official20:
                pending = [(d, 'selection', s) for d in datasets for s in range(SHARDS[d])]
            else:
                for g in range(4, 8):
                    active[g] = launch(root, 'flair1', 'selection', g-4, g)
                pending = [('loveda', 'selection', s) for s in range(2)] + [
                    ('udd5', 'selection', 0), ('oem', 'selection', 0), ('landcoverai', 'selection', 0)] + [
                    (d, 'official', 0) for d in OFFICIAL]
        for task in list(pending):
            free = next((g for g in gpus if g not in active and idle(g)), None)
            if free is None:
                break
            active[free] = launch(root, *task, free, official20)
            pending.remove(task)
        save(root/'suite_status.json', dict(status='running' if active or pending else 'failed' if failures else 'complete',
            active=list(active.values()), pending=pending, smoke_completed=sorted(smoke_done), selections=selections,
            completed=completed, failures=failures, elapsed_seconds=time.time()-started))
        if active or pending:
            time.sleep(15)
    save(root/'suite_results.json', dict(status='failed' if failures else 'complete', completed=completed,
        selections=selections, failures=failures, elapsed_seconds=time.time()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--official20', action='store_true', help='Only VDD/Potsdam fixed20 distillation on GPUs2-7.')
    args = parser.parse_args()
    run(args.root, args.official20)
