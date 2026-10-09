"""Queue and verify the frozen eight-domain local-head swap on GPUs0-3."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

import numpy as np

from dinotool.coupled_readout_swap import IMPLEMENTATION, METHODS, READOUTS, COUPLED, FIXED_PROTOCOL
from dinotool.fine_alias_view import CONFIG
from eval_rival_fine_full import save
from merge_geometry_semantic_innovation import main as merge
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, SETTINGS, idle
from run_sat_geometry_transport_suite import alive, read_json


TOTALS = {d: total for d, _, _, _, total in SETTINGS}
SHARDS = {d: (4 if d in ('flair1', 'loveda') else 2 if d in ('potsdam', 'landcoverai') else 1) for d in TOTALS}
REFERENCE = TOOL/'results/rival_fine_hard20_full_20261003/full'
MATCHED = TOOL/'results/geometry_matched_readout_full_20261001'


def launch(root, dataset, gpu, phase, shard=0):
    if not idle(gpu):
        raise RuntimeError(f'GPU{gpu} occupied; no launch.')
    _, data, vocabulary, _, _ = next(s for s in SETTINGS if s[0] == dataset)
    shards = SHARDS[dataset] if phase == 'full' else 1
    output = root/phase/dataset/(f's{shard}' if phase == 'full' else 's0')
    log = output.with_suffix('.log')
    session = f'gcrs04_{phase}_{dataset}_s{shard}'
    if output.exists() or log.exists() or alive(session):
        raise RuntimeError('Existing output/session: '+session)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, '-u', 'scripts/eval_coupled_readout_swap.py', '--dataset', dataset,
        '--dinov3-repo', str(TOOL/'dinov3_hub'), '--checkpoint-dir', str(BASE/'ckpt/DINO'),
        '--data-root', str(Path('/data/test/datasets')/data), '--vocabulary-config', str(TOOL/'configs'/vocabulary),
        '--upstream-root', str(BASE/'third_party/VIP_official_5bd25ee'), '--output-dir', str(output),
        '--num-shards', str(shards), '--shard-index', str(shard)]
    if phase == 'smoke':
        command.append('--smoke')
    env = ['env', f'CUDA_VISIBLE_DEVICES={gpu}', f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
           'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2', 'TOKENIZERS_PARALLELISM=false']
    shell = f'cd {shlex.quote(str(TOOL))} && exec {shlex.join(env)} nice -n 10 {shlex.join(command)} > {shlex.quote(str(log))} 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, shell], check=True)
    job = dict(dataset=dataset, shard=shard, shards=shards, gpu=gpu, phase=phase, session=session,
               output=str(output), log=str(log))
    print(json.dumps({'launched': job}), flush=True)
    return job


def verify_job(job):
    output = Path(job['output'])
    row = read_json(output/'results.json')
    if (not row or row['status'] != 'complete' or row['processed_images'] != row['total_images']
            or not row['weights_frozen'] or not row['head_weights_unchanged']):
        raise RuntimeError('Incomplete/frozen-state failure: '+job['session'])
    if job['phase'] == 'smoke':
        if row['target_masks_loaded'] or any(row['historical_score_errors'].values()):
            raise RuntimeError('Mask-free exact numerical replay failed.')
    else:
        if not row['target_masks_used_only_after_prediction'] or not row['shared_context_admission_reconstruction']:
            raise RuntimeError('Prediction or shared evidence contract failed.')
        with np.load(output/'per_image_confusions.npz', allow_pickle=False) as arrays:
            if arrays['sample_keys'].tolist() != row['signature']['sample_keys']:
                raise RuntimeError('Per-image key sequence differs.')
            for p, methods in row['metrics'].items():
                for m, metric in methods.items():
                    if not np.array_equal(arrays[p+'__'+m].sum(0), metric['confusion_matrix']):
                        raise RuntimeError('Per-image confusion sum differs.')


def ordered_arrays(paths, expected):
    keys, fields = [], {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            keys.extend(data['sample_keys'].tolist())
            for name in data.files:
                if name != 'sample_keys':
                    fields.setdefault(name, []).append(data[name])
    if len(set(keys)) != len(keys) or set(keys) != set(expected):
        raise RuntimeError('Full per-image coverage differs.')
    lookup = {key: i for i, key in enumerate(keys)}
    order = [lookup[key] for key in expected]
    return {name: np.concatenate(values)[order] for name, values in fields.items()}


def verify_dataset(root, dataset):
    directory = root/'full'/dataset
    inputs = [directory/f's{s}' for s in range(SHARDS[dataset])]
    output = directory/'merged.json'
    existing = output.exists()
    if not existing:
        merge(argparse.Namespace(inputs=[str(path) for path in inputs], output=str(output)))
    row, ref, matched = (read_json(path) for path in (output, REFERENCE/dataset/'merged.json', MATCHED/dataset/'merged.json'))
    sig = row['signature']
    if (not row['coverage_verified'] or row['processed_images'] != TOTALS[dataset]
            or row['total_images'] != TOTALS[dataset] or sig['implementation'] != IMPLEMENTATION
            or sig['sample_keys'] != ref['signature']['sample_keys']):
        raise RuntimeError('Full coverage/signature differs: '+dataset)
    for field in ('global_sample_keys_sha256', 'vocabulary', 'checkpoints', 'classes'):
        if sig[field] != ref['signature'][field] or sig[field] != matched['signature'][field]:
            raise RuntimeError('Frozen identity differs: '+dataset+'/'+field)
    if (sig['gear']['admission'] != asdict(CONFIG)
            or any(sig['gear'][field] != ref['signature']['gear'][field]
                   for field in ('geometry', 'observation', 'upstream_commit', 'local_tiles', 'fine_physical_pixels_per_token'))):
        raise RuntimeError('Fixed model protocol differs: '+dataset)
    current = ordered_arrays([p/'per_image_confusions.npz' for p in inputs], sig['sample_keys'])
    old_shards = ref['signature']['num_shards']
    old = ordered_arrays([REFERENCE/dataset/f's{s}/per_image_confusions.npz' for s in range(old_shards)], sig['sample_keys'])
    near = ordered_arrays([MATCHED/dataset/'per_image_confusions.npz'], sig['sample_keys'])
    for p, methods in row['metrics'].items():
        for m, metric in methods.items():
            if not np.array_equal(current[p+'__'+m].sum(0), metric['confusion_matrix']):
                raise RuntimeError('Merged confusion reconstruction differs.')
        for m in ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact'):
            if not np.array_equal(current[p+'__'+m], old[p+'__'+m]):
                raise RuntimeError('Historical complete predictor differs: '+dataset+'/'+p+'/'+m)
        for m in READOUTS:
            if not np.array_equal(current[p+'__'+m], near[p+'__'+m]):
                raise RuntimeError('Historical local operator differs: '+dataset+'/'+p+'/'+m)
    if not existing:
        np.savez_compressed(directory/'per_image_confusions.npz', sample_keys=np.asarray(sig['sample_keys']), **current)
        row.update(weights_frozen=True, head_weights_unchanged=True, shared_context_admission_reconstruction=True,
                   exact_reference_coupled_per_image=True, exact_reference_local_operators_per_image=True,
                   per_image_confusions_verified=True)
        save(output, row)
    elif not all(row.get(field) for field in ('weights_frozen', 'head_weights_unchanged',
            'shared_context_admission_reconstruction', 'exact_reference_coupled_per_image',
            'exact_reference_local_operators_per_image', 'per_image_confusions_verified')):
        raise RuntimeError('Existing merge lacks verification flags: '+dataset)
    print(json.dumps({'verified': dataset, 'images': TOTALS[dataset]}), flush=True)
    return str(output)


def miou(metric):
    matrix = np.asarray(metric['confusion_matrix'], dtype=np.float64)
    tp = matrix.diagonal()
    union = matrix.sum(0) + matrix.sum(1) - tp
    return float(np.nanmean(np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0))*100)


def write_report(root, completed):
    rows = {d: read_json(Path(path)) for d, path in completed.items()}
    full = set(rows) == set(TOTALS)
    means = {m: [] for m in COUPLED}
    lines = ['# Fixed-context coupled local-head swap', '',
        'Status: '+('complete' if full else 'partial')+'. Shared frozen wide-view VIP observation, physical8 '
        'alias/rival admission and original Geometry reconstruction matrix. Only the local semantic-head '
        'operator changes. Fixed20 words, checkpoints, text templates, temperature, masks and assembly. '
        'SCLIP_Two and VIPProxy_Two are matched DINO.text adaptations, not official complete systems. '
        'Every coupled arm still uses the original raw-feature Geometry reconstruction graph.', '',
        'All eight domains informed earlier development; results are exploratory full validation. '
        'Corrected Vaihingen IRRG; LandCover.ai substitutes for unlabeled iSAID. LoveDA P/D share images; '
        'the domain mean counts LoveDA D once.', '',
        '| Dataset/protocol | Images | Geometry coupled | SCLIP coupled | VIPProxy coupled | SCLIP minus Geometry | VIPProxy minus Geometry |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    values = {}
    for d in TOTALS:
        if d not in rows:
            continue
        row = rows[d]
        values[d] = {}
        for p, methods in row['metrics'].items():
            scores = {m: miou(methods[m]) for m in COUPLED}
            values[d][p] = scores
            g, s, v = (scores[m] for m in COUPLED)
            lines.append(f'| {d}/{p} | {TOTALS[d]} | {g:.4f} | {s:.4f} | {v:.4f} | {s-g:+.4f} | {v-g:+.4f} |')
            if p == ('D' if d == 'loveda' else d):
                for m in COUPLED:
                    means[m].append(scores[m])
    averages = {m: float(np.mean(v)) for m, v in means.items()} if rows else {}
    if full:
        g, s, v = (averages[m] for m in COUPLED)
        lines.append(f'| Eight-domain mean | 20092 | {g:.4f} | {s:.4f} | {v:.4f} | {s-g:+.4f} | {v-g:+.4f} |')
        wins = {m: sum(values[d]['D' if d == 'loveda' else d][COUPLED[0]] >
                       values[d]['D' if d == 'loveda' else d][m] for d in TOTALS) for m in COUPLED[1:]}
        lines += ['', f'Geometry coupled wins {wins[COUPLED[1]]}/8 against SCLIP and '
                  f'{wins[COUPLED[2]]}/8 against VIPProxy. Mean differences are '
                  f'{g-s:+.4f}pp and {g-v:+.4f}pp, respectively. '
                  'This tests the local Geometry head within a fixed reconstruction; it does not remove all Geometry information.']
    lines += ['', '## Screening and coupling increments', '',
              '| Dataset/protocol | Local head | Local only | Coupled without admission | Coupled with admission | Admission delta |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for d, row in rows.items():
        for p, methods in row['metrics'].items():
            for name in READOUTS:
                unscreened, screened = ('NoAdmission_Exact', 'RivalFineHard_Exact') if name == 'Geometry' else (
                    name+'_NoAdmission', name+'_RivalFineHard')
                a, b, c = (miou(methods[m]) for m in (name, unscreened, screened))
                lines.append(f'| {d}/{p} | {name} | {a:.4f} | {b:.4f} | {c:.4f} | {c-b:+.4f} |')
    lines += ['', '## Per-class outcomes', '',
              '| Dataset/protocol | Class | Local head | Coupled IoU | Precision | Recall | Predicted area % |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for d, row in rows.items():
        for p, methods in row['metrics'].items():
            for name, method in zip(READOUTS, COUPLED):
                for item in methods[method]['per_class']:
                    fields = [item.get(f) for f in ('iou_percent', 'precision_percent', 'recall_percent', 'predicted_area_percent')]
                    lines.append(f'| {d}/{p} | {item["name"]} | {name} | '+
                                 ' | '.join('--' if x is None else f'{x:.4f}' for x in fields)+' |')
    lines += ['', '## Combined evaluation cost', '',
              '| Dataset | Max shard elapsed s | Sum shard elapsed s | Peak allocated MiB |',
              '| --- | ---: | ---: | ---: |']
    for d, row in rows.items():
        lines.append(f'| {d} | {row["parallel_wall_seconds"]:.2f} | {row["aggregate_gpu_seconds"]:.2f} | {row["peak_cuda_memory_mb"]:.2f} |')
    lines += ['', 'Timing covers nine shared-source arms and is not standalone latency. '
              'Original three-arm predictions and all three local-only operators must exactly replay historical '
              'per-image confusions. Every sample appears once, per-image matrices reproduce full metrics, '
              'target masks are loaded after predictions, and frozen weights remain unchanged.', '']
    (root/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    save(root/'comparison_summary.json', {'complete': full, 'values': values,
        'means': averages if full else {}, 'completed_datasets': list(rows), 'protocol': FIXED_PROTOCOL})


def monitor(root, phase, pending, completed, verified=None):
    active, verified, failures = {}, set() if verified is None else set(verified), {}
    while active or pending:
        for gpu, job in list(active.items()):
            if alive(job['session']):
                continue
            try:
                verify_job(job)
                verified.add((job['dataset'], job['shard']))
                if phase == 'smoke':
                    completed[job['dataset']] = job['output']+'/results.json'
                elif all((job['dataset'], s) in verified for s in range(SHARDS[job['dataset']])):
                    completed[job['dataset']] = verify_dataset(root, job['dataset'])
                    write_report(root, completed)
            except Exception as error:
                failures[job['session']] = {'error': str(error),
                    'log_tail': Path(job['log']).read_text()[-6000:] if Path(job['log']).exists() else 'Missing worker log.'}
            del active[gpu]
        if failures:
            pending.clear()
        for dataset, shard in list(pending):
            gpu = next((g for g in range(4) if g not in active and idle(g)), None)
            if gpu is None:
                break
            try:
                active[gpu] = launch(root, dataset, gpu, phase, shard)
                pending.remove((dataset, shard))
            except Exception as error:
                failures[f'launch_{dataset}_s{shard}'] = {'error': str(error)}
                pending.clear()
                break
        save(root/'suite_status.json', {'status': 'running' if active or pending else 'failed' if failures else 'complete',
            'phase': phase, 'active': list(active.values()), 'pending': pending, 'completed': completed, 'failures': failures})
        if active or pending:
            time.sleep(15)
    return failures


def main(root):
    root.resolve().relative_to((TOOL/'results').resolve())
    if root.exists():
        raise RuntimeError('Refusing existing experiment root.')
    if not any(idle(g) for g in range(4)):
        raise RuntimeError('No idle GPU0-3; nothing launched.')
    for d in TOTALS:
        for source in (REFERENCE/d/'merged.json', MATCHED/d/'merged.json'):
            row = read_json(source)
            if not row or not row['coverage_verified'] or row['processed_images'] != TOTALS[d]:
                raise RuntimeError('Missing complete reference: '+str(source))
    root.mkdir(parents=True)
    save(root/'protocol.json', {'implementation': IMPLEMENTATION, 'methods': METHODS, 'fixed_protocol': FIXED_PROTOCOL,
        'admission_config': asdict(CONFIG), 'physical_gpus': list(range(4)), 'full_totals': TOTALS,
        'shards': SHARDS, 'full_unique_images': sum(TOTALS.values()), 'fixed_aliases_per_class': 20,
        'target_label_tuning': False, 'reference_coupled': str(REFERENCE), 'reference_local_operators': str(MATCHED)})
    started = time.time()
    smoke_completed = {}
    failures = monitor(root, 'smoke', [('udd5', 0), ('oem', 0)], smoke_completed)
    save(root/'smoke_summary.json', {'completed': smoke_completed, 'failures': failures})
    if failures:
        save(root/'suite_results.json', {'status': 'failed', 'phase': 'smoke', 'failures': failures})
        return
    completed = {}
    first = [('udd5', 0), ('vdd', 0), ('potsdam', 0), ('oem', 0)]
    queue = first + [(d, s) for d in ('potsdam', 'vaihingen', 'landcoverai', 'loveda', 'flair1')
                     for s in range(SHARDS[d]) if (d, s) not in first]
    failures = monitor(root, 'full', queue, completed)
    write_report(root, completed)
    save(root/'suite_results.json', {'status': 'failed' if failures else 'complete', 'completed': completed,
        'failures': failures, 'suite_wall_seconds': time.time()-started})


def resume_full(root):
    root.resolve().relative_to((TOOL/'results').resolve())
    state, protocol = read_json(root/'suite_status.json'), read_json(root/'protocol.json')
    smoke = read_json(root/'smoke_summary.json')
    recovery = root/'report_path_recovery_20261004'
    if (not state or state['phase'] != 'full' or state['active'] or recovery.exists()
            or not smoke or smoke['failures'] or (root/'suite_results.json').exists()
            or protocol['implementation'] != IMPLEMENTATION or protocol['fixed_protocol'] != FIXED_PROTOCOL
            or protocol['admission_config'] != asdict(CONFIG) or protocol['shards'] != SHARDS):
        raise RuntimeError('Recovery requires the stopped original full suite with unchanged protocol.')
    if any(item['error'] != "'str' object has no attribute 'exists'" for item in state['failures'].values()):
        raise RuntimeError('Only the identified report-path failure may be recovered here.')
    pending, verified = [], set()
    for dataset in TOTALS:
        for shard in range(SHARDS[dataset]):
            output = root/'full'/dataset/f's{shard}'
            session = f'gcrs04_full_{dataset}_s{shard}'
            if alive(session):
                raise RuntimeError('An owned worker is still running: '+session)
            if output.exists():
                verify_job(dict(output=str(output), phase='full', session=session))
                verified.add((dataset, shard))
            elif output.with_suffix('.log').exists():
                raise RuntimeError('A previously attempted worker has no output; preserve it: '+session)
            else:
                pending.append((dataset, shard))
    completed = {}
    for dataset in TOTALS:
        if all((dataset, shard) in verified for shard in range(SHARDS[dataset])):
            completed[dataset] = verify_dataset(root, dataset)
    write_report(root, completed)
    recovery.mkdir()
    shutil.copy2(root/'suite_status.json', recovery/'original_failed_suite_status.json')
    save(recovery/'recovery.json', {'fixed_error': "read_json expected Path, received str in write_report",
        'preserved_complete_datasets': list(completed), 'pending_never_launched_shards': pending,
        'evaluation_code_unchanged': True, 'recovery_started_epoch': time.time()})
    started = root.stat().st_mtime
    failures = monitor(root, 'full', pending, completed, verified)
    write_report(root, completed)
    save(root/'suite_results.json', {'status': 'failed' if failures else 'complete', 'completed': completed,
        'failures': failures, 'recovery_elapsed_seconds': time.time()-started,
        'report_path_failure_recovered': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--resume-full', action='store_true')
    args = parser.parse_args()
    (resume_full if args.resume_full else main)(args.root)
