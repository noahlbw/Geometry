"""Frozen mask-free calibration, pilot comparison and conditional full evaluation."""
import argparse
import json
from itertools import zip_longest
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.natural_evaluation import ROOTS, TOTALS, discover_samples
from dinotool.natural_text_adaptation import IMPLEMENTATION, SEED, CALIBRATION_IMAGES
from eval_natural_text_adaptation import METHODS, save, digest
from eval_geometry_vip_reliability import summary
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, idle
from run_sat_geometry_transport_suite import alive, read_json


DATA = Path('/data/test/datasets/VIP_natural')
VOCAB = TOOL/'results/rival_fine_natural20_full_20261003/vocabularies'
SHARDS = dict(voc20=1, voc21=1, ade150=2, coco_stuff171=2, coco_object81=2)
PILOT_IMAGES = 16


def launch(root, dataset, gpu, phase, shard=0, shards=1):
    output = root/phase/dataset/f's{shard}'
    log = output.with_suffix('.log')
    session = f'gncal03_{phase}_{dataset}_s{shard}'
    if not idle(gpu) or output.exists() or log.exists() or alive(session):
        raise RuntimeError('Occupied GPU or existing output/session: '+session)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, '-u', 'scripts/eval_natural_text_adaptation.py',
        'select' if phase == 'selection' else 'evaluate', '--dataset', dataset,
        '--dinov3-repo', str(TOOL/'dinov3_hub'), '--checkpoint-dir', str(BASE/'ckpt/DINO'),
        '--upstream-root', str(BASE/'third_party/VIP_official_5bd25ee'),
        '--data-root', str(DATA/ROOTS[dataset]), '--source-vocabulary', str(VOCAB/(dataset+'_20.json')),
        '--cache-dir', str(root/'text_cache'/dataset), '--output-dir', str(output),
        '--num-shards', str(shards), '--shard-index', str(shard)]
    if phase != 'selection':
        command += ['--selection', str(root/'selection'/dataset/'s0/selection.json')]
    if phase == 'pilot':
        command += ['--max-images', str(PILOT_IMAGES)]
    env = ['env', f'CUDA_VISIBLE_DEVICES={gpu}', f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
           'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2', 'TOKENIZERS_PARALLELISM=false']
    shell = 'cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+['nice', '-n', '10', *command])+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, shell], check=True)
    job = dict(dataset=dataset, phase=phase, gpu=gpu, shard=shard, shards=shards, output=str(output), log=str(log), session=session)
    print(json.dumps(dict(started=session, gpu=gpu)), flush=True)
    return job


def verify(job):
    output = Path(job['output'])
    name = 'selection.json' if job['phase'] == 'selection' else 'results.json'
    row = read_json(output/name)
    if not row or row['status'] != 'complete' or not row['weights_frozen'] or not row['head_weights_unchanged'] or row['target_label_tuning']:
        raise RuntimeError('Incomplete or unfrozen job: '+job['session'])
    if job['phase'] == 'selection':
        if row['target_masks_loaded'] or row['implementation'] != IMPLEMENTATION:
            raise RuntimeError('Selection used masks or a changed implementation.')
    else:
        if row['processed_images'] != row['total_images'] or not row['target_masks_used_only_after_prediction']:
            raise RuntimeError('Incomplete evaluation coverage.')
        with np.load(output/'per_image_confusions.npz', allow_pickle=False) as arrays:
            if arrays['sample_keys'].tolist() != row['signature']['sample_keys']:
                raise RuntimeError('Changed sample sequence.')
            for protocol, methods in row['metrics'].items():
                for method, metric in methods.items():
                    if not np.array_equal(arrays[protocol+'__'+method].sum(0), metric['confusion_matrix']):
                        raise RuntimeError('Per-image confusion sum mismatch.')
    return row


def merge_results(inputs, path):
    rows = [read_json(Path(p)/'results.json') for p in inputs]
    first = rows[0]['signature']
    if any(r['status'] != 'complete' for r in rows) or len(rows) != first['num_shards']:
        raise RuntimeError('Incomplete merge inputs.')
    ordered = sorted(rows, key=lambda r: r['signature']['shard_index'])
    if [r['signature']['shard_index'] for r in ordered] != list(range(first['num_shards'])):
        raise RuntimeError('Missing or duplicate shard indices.')
    keys = [k for group in zip_longest(*(r['signature']['sample_keys'] for r in ordered)) for k in group if k is not None]
    if len(keys) != len(set(keys)) or len(keys) != first['global_sample_count'] or digest(keys) != first['global_sample_keys_sha256']:
        raise RuntimeError('Incomplete unique global sample coverage.')
    for row in rows:
        signature = row['signature']
        if row['processed_images'] != row['total_images'] or row['total_images'] != len(signature['sample_keys']):
            raise RuntimeError('Incomplete shard image coverage.')
        if not row['weights_frozen'] or not row['head_weights_unchanged'] or row['target_label_tuning']:
            raise RuntimeError('Changed model weights or label-tuned shard.')
        if digest(signature['sample_keys']) != signature['sample_keys_sha256']:
            raise RuntimeError('Changed per-shard sample sequence.')
        for field in ('implementation', 'dataset', 'methods', 'classes', 'gear', 'vocabulary', 'checkpoints', 'global_sample_keys_sha256'):
            if signature[field] != first[field]:
                raise RuntimeError('Changed shard identity: '+field)
        trim = lambda d: {k: v for k, v in d.items() if k not in ('output_dir', 'shard_index')}
        if trim(signature['config']) != trim(first['config']):
            raise RuntimeError('Changed shard evaluation settings.')
    metrics, diagnostics = {}, {}
    for protocol, names in first['classes'].items():
        metrics[protocol] = {}
        for method in first['methods']:
            sources = [r['metrics'][protocol][method] for r in rows]
            matrix = np.stack([s['confusion_matrix'] for s in sources]).sum(0)
            if matrix.shape[0] != len(names) or matrix.shape[1] not in (len(names), len(names)+1):
                raise RuntimeError('Unexpected scored or reject class count.')
            metrics[protocol][method] = summary(matrix, names, sum(s['ignored_pixels'] for s in sources))
        tiles = sum(r['diagnostics'][protocol]['tiles'] for r in rows)
        diagnostics[protocol] = {k: tiles if k == 'tiles' else sum(r['diagnostics'][protocol][k]*r['diagnostics'][protocol]['tiles'] for r in rows)/tiles
                                 for k in rows[0]['diagnostics'][protocol]}
    result = dict(status='complete', coverage_verified=True, processed_images=len(keys), total_images=len(keys),
        metrics=metrics, diagnostics=diagnostics, signature={**first, 'shard_index': None, 'sample_keys': keys, 'sample_keys_sha256': digest(keys)},
        parallel_wall_seconds=max(r['wall_seconds'] for r in rows), aggregate_gpu_seconds=sum(r['wall_seconds'] for r in rows),
        peak_cuda_memory_mb=max(r['peak_cuda_memory_mb'] for r in rows), weights_frozen=True, head_weights_unchanged=True,
        target_masks_used_only_after_prediction=True, target_label_tuning=False, per_image_confusions_verified=True)
    save(path, result)
    return result


def phase_run(root, datasets, phase, blocked):
    shards = {d: SHARDS.get(d, 1) if phase == 'full' else 1 for d in datasets}
    pending = []
    active, completed, failures, verified = {}, {}, {}, set()
    state = read_json(root/'suite_status.json') or {}
    old_active = {j['session']: j for j in state.get('active', [])}
    for dataset in datasets:
        for shard in range(shards[dataset]):
            session = f'gncal03_{phase}_{dataset}_s{shard}'
            output = root/phase/dataset/f's{shard}'
            if alive(session):
                if session not in old_active:
                    raise RuntimeError('Running worker missing its saved GPU assignment: '+session)
                job = old_active[session]
                active[job['gpu']] = job
            elif output.exists() or output.with_suffix('.log').exists():
                job = dict(dataset=dataset, phase=phase, output=str(output), log=str(output.with_suffix('.log')), session=session)
                try:
                    verify(job)
                    verified.add((dataset, shard))
                except Exception as error:
                    failures[session] = dict(error=str(error), log=job['log'])
            else:
                pending.append((dataset, shard))
        if all((dataset, s) in verified for s in range(shards[dataset])):
            if phase == 'selection':
                completed[dataset] = str(root/phase/dataset/'s0/selection.json')
            else:
                path = root/phase/dataset/'merged.json'
                if not path.exists():
                    merge_results([str(root/phase/dataset/f's{s}') for s in range(shards[dataset])], path)
                completed[dataset] = str(path)
    while pending or active:
        for gpu, job in list(active.items()):
            if alive(job['session']):
                continue
            try:
                verify(job)
                verified.add((job['dataset'], job['shard']))
                dataset = job['dataset']
                if all((dataset, s) in verified for s in range(shards[dataset])):
                    if phase == 'selection':
                        completed[dataset] = str(Path(job['output'])/'selection.json')
                    else:
                        path = root/phase/dataset/'merged.json'
                        row = merge_results([str(root/phase/dataset/f's{s}') for s in range(shards[dataset])], path)
                        expected = PILOT_IMAGES if phase == 'pilot' else TOTALS[dataset]
                        if not row['coverage_verified'] or row['processed_images'] != expected:
                            raise RuntimeError('Incomplete merged coverage: '+dataset)
                        row.update(weights_frozen=True, head_weights_unchanged=True,
                                   target_masks_used_only_after_prediction=True, target_label_tuning=False,
                                   per_image_confusions_verified=True)
                        save(path, row)
                        completed[dataset] = str(path)
                        print(json.dumps(dict(completed=dataset, phase=phase,
                            miou={m: v['mean_iou_percent'] for m, v in row['metrics'][dataset].items()})), flush=True)
            except Exception as error:
                failures[job['session']] = dict(error=str(error), log=job['log'],
                    log_tail=Path(job['log']).read_text()[-6000:])
            del active[gpu]
        if failures:
            pending.clear()
        for dataset, shard in list(pending):
            free = next((g for g in range(8) if g not in active and idle(g)), None)
            if free is None:
                break
            active[free] = launch(root, dataset, free, phase, shard, shards[dataset])
            pending.remove((dataset, shard))
        save(root/'suite_status.json', dict(status='running' if active or pending else 'failed' if failures else 'complete',
            phase=phase, active=list(active.values()), pending=pending, completed=completed, failures=failures, blocked=blocked))
        if active or pending:
            time.sleep(15)
    return completed, failures


def pilot_decision(completed):
    values, gains, losses = {}, [], []
    for dataset, path in completed.items():
        row = read_json(Path(path))
        group = row['metrics'][dataset]
        matrices = {m: np.asarray(x['confusion_matrix'], dtype=np.int64) for m, x in group.items()}
        classes = len(row['signature']['classes'][dataset])
        support = np.zeros(classes, dtype=bool)
        for cm in matrices.values():
            support |= cm.sum(1)+cm.sum(0)[:classes]-cm[np.arange(classes), np.arange(classes)] > 0
        score = {}
        for method, cm in matrices.items():
            tp = cm[np.arange(classes), np.arange(classes)]
            union = cm.sum(1)+cm.sum(0)[:classes]-tp
            iou = np.divide(tp, union, out=np.zeros(classes), where=union > 0)
            score[method] = float(iou[support].mean()*100)
        values[dataset] = score
        gains.append(score['Adaptive_RivalFine']-score['RivalFine_Pool'])
        losses.append(score['Adaptive_RivalFine']-score['Geometry_Pool'])
    passed = bool(gains) and float(np.mean(gains)) > 0 and min(gains+losses) >= -1.
    return dict(passed=passed, values=values, mean_gain_vs_unadapted_pool=float(np.mean(gains)),
                minimum_delta_vs_pool_or_geometry=min(gains+losses),
                rule='One global candidate; pilot mean gain >0 and no protocol loss >1pp versus pool reader or pool Geometry. No per-dataset result switching.')


def main(root, resume=False, full_diagnostic=False):
    root.resolve().relative_to((TOOL/'results').resolve())
    if root.exists() and not resume:
        raise RuntimeError('Existing adaptive experiment root; no relaunch.')
    datasets, blocked = [], {}
    for dataset in TOTALS:
        try:
            discover_samples(dataset, DATA/ROOTS[dataset])
            datasets.append(dataset)
        except (FileNotFoundError, ValueError) as error:
            blocked[dataset] = str(error)
    if not datasets:
        raise RuntimeError('No prepared natural-image dataset.')
    root.mkdir(parents=True, exist_ok=resume)
    protocol = dict(implementation=IMPLEMENTATION, datasets=datasets, blocked=blocked,
        seed=SEED, calibration_images=CALIBRATION_IMAGES, pilot_images=PILOT_IMAGES, methods=METHODS,
        full_totals={d: TOTALS[d] for d in datasets}, main_model_changed=False, target_label_tuning=False,
        vocabulary_sources='Canonical taxonomy + existing curated lexical synonyms + attributed official VIP queries; no synthetic twenty-word quota.',
        numerical_reference='Pinned VIP with Self-Value ONLY for empty proxy rows; not claimed equal to published table.',
        protocol_note='Our native visual views unchanged. VIP comparator retains its own official resize/settings; common input images and masks, not a pure head ablation.',
        pilot_note='Fixed16 development images; class support matched across arms. Gate decides full rollout only, not hyperparameters. All subsequent full scores exploratory.')
    if resume:
        if read_json(root/'protocol.json') != json.loads(json.dumps(protocol)):
            raise RuntimeError('Cannot resume with a changed frozen protocol.')
    else:
        save(root/'protocol.json', protocol)
    started = time.time()
    selections, failures = phase_run(root, datasets, 'selection', blocked)
    if failures:
        save(root/'suite_results.json', dict(status='failed', phase='selection', failures=failures, completed=selections, blocked=blocked))
        return
    pilot, failures = phase_run(root, datasets, 'pilot', blocked)
    if failures:
        save(root/'suite_results.json', dict(status='failed', phase='pilot', failures=failures, completed=pilot, blocked=blocked))
        return
    decision = pilot_decision(pilot)
    save(root/'pilot_decision.json', decision)
    print(json.dumps(dict(pilot_decision=decision)), flush=True)
    if not decision['passed'] and not full_diagnostic:
        save(root/'suite_results.json', dict(status='needs_analysis', phase='pilot', completed=pilot,
             decision=decision, blocked=blocked, failures={}, suite_wall_seconds=time.time()-started))
        return
    if full_diagnostic:
        save(root/'diagnostic_full_protocol.json', dict(mode='paired_full_measurement_not_pilot_promotion',
            failed_pilot_preserved=not decision['passed'], frozen_selection=True, model_changed=False,
            rationale='The fixed16 pilot has many absent classes. Measure the frozen word-pool and calibration candidate on full coverage against VIP without claiming a successful pilot or selecting per-dataset winners.'))
    completed, failures = phase_run(root, datasets, 'full', blocked)
    save(root/'suite_results.json', dict(status='failed' if failures else 'complete_available', phase='full',
         completed=completed, pilot=pilot, decision=decision, failures=failures, blocked=blocked,
         suite_wall_seconds=time.time()-started, complete_eight_protocols=not blocked))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--full-diagnostic', action='store_true')
    args = parser.parse_args()
    main(args.root, args.resume, args.full_diagnostic)
