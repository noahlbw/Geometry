"""Schedule one frozen paired-vocabulary suite on idle A800 GPUs."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from dinotool.bounded_patch_only import IMPLEMENTATION, PRIMARY, PROTOCOL
from dinotool.natural_evaluation import ROOTS, TOTALS, discover_samples
from eval_curated20_patchonly2 import exact_specs
from eval_gear_ov import protocol
from eval_rival_fine_full import save
from merge_gear_ov_shards import merge
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, SETTINGS, idle
from run_sat_geometry_transport_suite import alive, read_json


PREFIX = 'gc20p06'
NATURAL_DATA = Path('/data/test/datasets/VIP_natural')
SHARDS = dict(flair1=4, loveda=4, context59=2, context60=2,
              ade150=2, coco_stuff171=4, coco_object81=4)


def inventory(root):
    manifest = read_json(root / 'vocabulary_manifest.json')
    ready, blocked = {}, {}
    for dataset, entry in manifest['datasets'].items():
        old = exact_specs(root / 'vocabularies' / (dataset + '_original20.json'))
        new = exact_specs(root / 'vocabularies' / (dataset + '_curated20.json'))
        if [s.name for s in old] != entry['class_names'] or [s.name for s in new] != entry['class_names']:
            raise RuntimeError('Class order changed: ' + dataset)
        if entry['family'] == 'remote_sensing':
            _, data, _, _, total = next(v for v in SETTINGS if v[0] == dataset)
            data_root = Path('/data/test/datasets') / data
            args = SimpleNamespace(dataset=dataset, data_root=str(data_root),
                                   sample_seed=20260923, vdd_ontology='official')
            samples = protocol(args, old)[0]
            if len(samples) != total:
                raise RuntimeError('Incomplete remote-sensing dataset: ' + dataset)
        else:
            data_root = NATURAL_DATA / ROOTS[dataset]
            total = TOTALS[dataset]
            try:
                samples = discover_samples(dataset, data_root)
            except (FileNotFoundError, ValueError) as error:
                blocked[dataset] = str(error)
                continue
        ready[dataset] = dict(family=entry['family'], data_root=str(data_root),
                              total_images=total, shards=SHARDS.get(dataset, 1))
    return dict(ready=ready, blocked=blocked)


def launch(root, dataset, entry, gpu, phase, shard=0):
    shards = 1 if phase == 'smoke' else entry['shards']
    output = root / phase / dataset / ('s' + str(shard))
    log = output.with_suffix('.log')
    session = f'{PREFIX}_{phase}_{dataset}_s{shard}'
    if output.exists() or log.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Occupied GPU or existing job/output: ' + session)
    output.parent.mkdir(parents=True, exist_ok=True)
    args = [PYTHON, '-u', 'scripts/eval_curated20_patchonly2.py', '--dataset', dataset,
        '--family', entry['family'], '--dinov3-repo', str(TOOL / 'dinov3_hub'),
        '--checkpoint-dir', str(BASE / 'ckpt/DINO'), '--data-root', entry['data_root'],
        '--upstream-root', str(BASE / 'third_party/VIP_official_5bd25ee'),
        '--original-vocabulary', str(root / 'vocabularies' / (dataset + '_original20.json')),
        '--curated-vocabulary', str(root / 'vocabularies' / (dataset + '_curated20.json')),
        '--text-cache', str(root / 'text_cache' / (dataset + '.pt')),
        '--output-dir', str(output), '--mode', phase,
        '--num-shards', str(shards), '--shard-index', str(shard)]
    env = ['env', f'CUDA_VISIBLE_DEVICES={gpu}', f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
           'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2', 'TOKENIZERS_PARALLELISM=false']
    shell = 'cd ' + shlex.quote(str(TOOL)) + ' && exec ' + shlex.join(env + args)
    shell += ' > ' + shlex.quote(str(log)) + ' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, shell], check=True)
    return dict(dataset=dataset, gpu=gpu, phase=phase, shard=shard, shards=shards,
                output=str(output), log=str(log), session=session)


def verify_job(job):
    folder = Path(job['output'])
    row = read_json(folder / 'results.json')
    if (not row or row['status'] != 'complete' or row['processed_images'] != row['total_images']
            or not row['weights_frozen'] or not row['head_weights_unchanged']):
        raise RuntimeError('Worker exited before complete frozen output: ' + job['session'])
    if job['phase'] == 'smoke':
        if row['target_masks_loaded'] or not row['paired_singleton_exact'] or row['implementation'] != IMPLEMENTATION:
            raise RuntimeError('Mask-free source smoke failed.')
        return row
    signature = row['signature']
    if (signature['implementation'] != IMPLEMENTATION or signature['gear']['input_protocol'] != PROTOCOL
            or signature['methods'] != [PRIMARY] or not row['target_masks_used_only_after_prediction']
            or row['target_label_tuning']
            or any(n != 20 for counts in signature['vocabulary']['counts'].values() for n in counts)):
        raise RuntimeError('Frozen model/vocabulary protocol changed.')
    with np.load(folder / 'per_image_confusions.npz', allow_pickle=False) as arrays:
        if arrays['sample_keys'].tolist() != signature['sample_keys']:
            raise RuntimeError('Per-image coverage changed.')
        for p, group in row['metrics'].items():
            if not np.array_equal(arrays[p + '__' + PRIMARY].sum(0), group[PRIMARY]['confusion_matrix']):
                raise RuntimeError('Per-image confusion reconstruction failed.')
    for diag in row['diagnostics'].values():
        if diag['geometry_encodings'] > 4 or diag['wide_encodings'] > 4 or diag['fine_forwards']:
            raise RuntimeError('Actual visual forward cap changed.')
    return row


def verify_dataset(root, dataset, entry):
    path = root / 'full' / dataset / 'merged.json'
    if path.exists():
        raise RuntimeError('Preserve existing merged output: ' + dataset)
    inputs = [str(root / 'full' / dataset / ('s' + str(s))) for s in range(entry['shards'])]
    row = merge(inputs, str(path))
    if not row['coverage_verified'] or row['processed_images'] != entry['total_images']:
        raise RuntimeError('Incomplete full coverage: ' + dataset)
    for p, group in row['metrics'].items():
        if not p.endswith('__original20'):
            continue
        paired = p.removesuffix('__original20') + '__curated20'
        a = np.asarray(group[PRIMARY]['confusion_matrix'])
        b = np.asarray(row['metrics'][paired][PRIMARY]['confusion_matrix'])
        if not np.array_equal(a.sum(1), b.sum(1)):
            raise RuntimeError('Paired scored targets differ.')
    if entry['family'] == 'remote_sensing':
        baseline = read_json(TOOL / 'results/bounded_patch_only_20261005/full' / dataset / 'merged.json')
        if (row['signature']['sample_keys'] != baseline['signature']['sample_keys']
                or row['signature']['checkpoints'] != baseline['signature']['checkpoints']):
            raise RuntimeError('Historical same-input identities changed.')
        for p, group in baseline['metrics'].items():
            if not np.array_equal(group[PRIMARY]['confusion_matrix'],
                    row['metrics'][p + '__original20'][PRIMARY]['confusion_matrix']):
                raise RuntimeError('Original20 frozen full prediction changed: ' + dataset + '/' + p)
    row.update(weights_frozen=True, head_weights_unchanged=True,
               per_image_confusions_verified=True, paired_scored_targets_equal=True)
    save(path, row)
    return str(path)


def monitor(root, info, pending, phase):
    active, completed, failures, done = {}, {}, {}, set()
    while active or pending:
        for gpu, job in list(active.items()):
            if alive(job['session']):
                continue
            try:
                verify_job(job)
                done.add((job['dataset'], job['shard']))
                d = job['dataset']
                if phase == 'smoke':
                    completed[d] = job['output'] + '/results.json'
                elif all((d, s) in done for s in range(info[d]['shards'])):
                    completed[d] = verify_dataset(root, d, info[d])
            except Exception as error:
                log = Path(job['log'])
                failures[job['session']] = dict(error=str(error),
                    log_tail=log.read_text(errors='replace')[-5000:] if log.exists() else 'No log')
            del active[gpu]
        if failures:
            pending.clear()
        for dataset, shard in list(pending):
            gpu = next((g for g in range(8) if g not in active and idle(g)), None)
            if gpu is None:
                break
            active[gpu] = launch(root, dataset, info[dataset], gpu, phase, shard)
            pending.remove((dataset, shard))
        save(root / 'suite_status.json', dict(status='running' if active or pending else
            'failed' if failures else 'complete', phase=phase, active=list(active.values()),
            pending=pending, completed=completed, failures=failures))
        if active or pending:
            time.sleep(15)
    return completed, failures


def main(root):
    root.resolve().relative_to((TOOL / 'results').resolve())
    if (root / 'suite_status.json').exists() or (root / 'suite_results.json').exists():
        raise RuntimeError('Existing run; inspect rather than restart.')
    started = time.time()
    prepared = inventory(root)
    info = prepared['ready']
    if not all(d in info for d, *_ in SETTINGS):
        raise RuntimeError('All eight remote-sensing datasets must be complete.')
    save(root / 'protocol.json', dict(implementation=IMPLEMENTATION, primary=PRIMARY,
        input_protocol=PROTOCOL, vocabulary_experiment='taxonomy-aware-curated20-v1-20261006',
        **prepared, target_label_tuning=False, model_rule_changed=False,
        alias_admission='none', local_templates='RS6 for remote sensing; ImageNet80 for natural',
        wide_templates='ImageNet80',
        comparison='Paired original20/curated20 with shared RGB encodings; no dataset winner switching',
        background='Retained model softmax argmax; no threshold calibration',
        development='Vocabulary design informed by prior development; not untouched validation'))
    smoke, failures = monitor(root, info, [(d, 0) for d in info], 'smoke')
    save(root / 'smoke_summary.json', dict(completed=smoke, failures=failures))
    if failures:
        save(root / 'suite_results.json', dict(status='failed', phase='smoke', failures=failures))
        return
    order = ['udd5', 'vdd', 'potsdam', 'oem', 'flair1', 'loveda', 'vaihingen', 'landcoverai']
    order += [d for d in info if d not in order]
    full, failures = monitor(root, info, [(d, s) for d in order for s in range(info[d]['shards'])], 'full')
    save(root / 'suite_results.json', dict(status='failed' if failures else 'complete',
        completed=full, failures=failures, blocked=prepared['blocked'],
        suite_wall_seconds=time.time() - started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=('inventory', 'run'), default='run')
    args = parser.parse_args()
    if args.mode == 'inventory':
        print(json.dumps(inventory(args.root)))
    else:
        main(args.root)
