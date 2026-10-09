"""Finite supervised development queue on idle GPUs; frozen full evaluation follows."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from dinotool.development_readout import IMPLEMENTATION, development_keys, profiles
from dinotool.natural_evaluation import ROOTS, TOTALS, discover_samples
from dinotool.prompts import ClassSpec
from eval_gear_ov import protocol
from eval_rival_fine_full import save
from merge_gear_ov_shards import merge
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, SETTINGS, idle
from run_sat_geometry_transport_suite import alive, read_json


PREFIX = 'gdro06'
ORDER = ('vdd', 'potsdam', 'voc21', 'context60', 'coco_object81', 'ade150',
         'context59', 'voc20', 'coco_stuff171')
SHARDS = dict(context59=2, context60=2, coco_object81=4, coco_stuff171=4, ade150=2)
OLD_ROOT = TOOL / 'results/curated20_patchonly2_full_20261006'


def inventory(root):
    inputs = read_json(root / 'inputs.json')
    entries = {}
    for dataset in ORDER:
        entry = inputs['datasets'][dataset]
        if entry['family'] == 'natural':
            data_root = Path('/data/test/datasets/VIP_natural') / ROOTS[dataset]
            samples = discover_samples(dataset, data_root)
            expected = TOTALS[dataset]
        else:
            _, data, _, _, expected = next(s for s in SETTINGS if s[0] == dataset)
            data_root = Path('/data/test/datasets') / data
            args = SimpleNamespace(dataset=dataset, data_root=str(data_root), sample_seed=20260923, vdd_ontology='official')
            specs = [ClassSpec(c['name'], tuple(c['synonyms'])) for c in entry['banks']['original']['classes']]
            samples = protocol(args, specs)[0]
        keys = [s.key for s in samples]
        if len(keys) != expected or len(set(keys)) != expected:
            raise RuntimeError('Incomplete validation coverage: ' + dataset)
        if dataset == 'ade150':
            training = [p.stem for p in sorted((data_root / 'images/training').glob('*.jpg'))
                if (data_root / 'annotations/training' / (p.stem + '.png')).is_file()]
            if len(training) != 20210 or set(training) & set(keys):
                raise RuntimeError('Complete disjoint ADE train/validation required.')
            dev = development_keys(training, 96)
            source, heldout = 'ADE_training', keys
        else:
            dev = development_keys(keys, min(64, max(8, expected // 5)))
            source, heldout = 'fixed_labeled_validation_development_subset', [k for k in keys if k not in set(dev)]
        classes = [c['name'] for c in entry['banks']['original']['classes']]
        for bank, record in entry['banks'].items():
            if [c['name'] for c in record['classes']] != classes:
                raise RuntimeError('Class order changed: ' + dataset + '/' + bank)
            if entry['family'] == 'remote_sensing' and any(len(set(c['synonyms'])) != 20 or len(c['synonyms']) != 20 for c in record['classes']):
                raise RuntimeError('Focused RS banks must be exactly20/class.')
        entries[dataset] = dict(**entry, data_root=str(data_root), total_images=expected,
            development_keys=dev, heldout_keys=heldout, development_source=source,
            shards=SHARDS.get(dataset, 1), search_profiles=len(profiles(entry['banks'])))
    return dict(**{k: v for k, v in inputs.items() if k != 'datasets'}, datasets=entries,
        implementation=IMPLEMENTATION, labels_select_hyperparameters=True,
        note='Weights frozen; supervised development, not label-free adaptation. '
             'Non-ADE full scores include tuning samples; the disjoint complement is reported separately. '
             'Prior experiments developed these domains; no untouched-test claim.')


def launch(root, dataset, entry, gpu, phase, shard=0):
    output = root / phase / dataset if phase != 'full' else root / phase / dataset / ('s' + str(shard))
    log = output.with_suffix('.log')
    session = f'{PREFIX}_{phase}_{dataset}_s{shard}'
    if output.exists() or log.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Existing output or occupied GPU: ' + session)
    output.parent.mkdir(parents=True, exist_ok=True)
    args = [PYTHON, '-u', 'scripts/eval_development_readout.py', '--dataset', dataset,
        '--mode', phase, '--suite-root', str(root), '--data-root', entry['data_root'],
        '--output-dir', str(output), '--original-cache', str(OLD_ROOT / 'text_cache' / (dataset + '.pt')),
        '--dinov3-repo', str(TOOL / 'dinov3_hub'), '--checkpoint-dir', str(BASE / 'ckpt/DINO'),
        '--upstream-root', str(BASE / 'third_party/VIP_official_5bd25ee'),
        '--num-shards', str(entry['shards'] if phase == 'full' else 1), '--shard-index', str(shard)]
    env = ['env', f'CUDA_VISIBLE_DEVICES={gpu}', f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
        'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2', 'TOKENIZERS_PARALLELISM=false']
    shell = 'cd ' + shlex.quote(str(TOOL)) + ' && exec ' + shlex.join(env + args)
    shell += ' > ' + shlex.quote(str(log)) + ' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, shell], check=True)
    return dict(dataset=dataset, phase=phase, shard=shard, output=str(output),
        log=str(log), session=session, gpu=gpu)


def verify_job(job):
    row = read_json(Path(job['output']) / 'results.json')
    if (not row or row['status'] != 'complete' or row['processed_images'] != row['total_images']
            or not row['weights_frozen'] or not row['head_weights_unchanged']):
        raise RuntimeError('Worker exited before frozen completion: ' + job['session'])
    if job['phase'] == 'smoke':
        if not row['exact_retained_prediction'] or row['target_masks_loaded']:
            raise RuntimeError('Reference/mask-free smoke failed.')
    elif job['phase'] == 'search':
        if not read_json(Path(job['output']) / 'selection.json'):
            raise RuntimeError('Missing frozen selection.')
    else:
        with np.load(Path(job['output']) / 'per_image_confusions.npz', allow_pickle=False) as data:
            if data['sample_keys'].tolist() != row['signature']['sample_keys']:
                raise RuntimeError('Per-image sample identity mismatch.')
            for method in ('Reference', 'Tuned'):
                if not np.array_equal(data[job['dataset'] + '__' + method].sum(0),
                        row['metrics'][job['dataset']][method]['confusion_matrix']):
                    raise RuntimeError('Per-image matrix sum mismatch.')
    return row


def verify_dataset(root, dataset, entry):
    path = root / 'full' / dataset / 'merged.json'
    if path.exists():
        raise RuntimeError('Existing merged output; preserve it.')
    folders = [root / 'full' / dataset / ('s' + str(s)) for s in range(entry['shards'])]
    row = merge([str(p) for p in folders], str(path))
    if not row['coverage_verified'] or row['processed_images'] != entry['total_images']:
        raise RuntimeError('Missing full unique coverage.')
    matrices = row['metrics'][dataset]
    old = read_json(OLD_ROOT / 'full' / dataset / 'merged.json')
    historical = old['metrics'][dataset + '__original20']['Geometry_PatchOnly2Coupled']['confusion_matrix']
    if not np.array_equal(matrices['Reference']['confusion_matrix'], historical):
        raise RuntimeError('Retained full reference changed: ' + dataset)
    if not np.array_equal(np.asarray(matrices['Reference']['confusion_matrix']).sum(1),
                          np.asarray(matrices['Tuned']['confusion_matrix']).sum(1)):
        raise RuntimeError('Paired scored target counts differ.')
    heldout = {m: np.zeros_like(np.asarray(matrices[m]['confusion_matrix'])) for m in matrices}
    names = row['signature']['classes'][dataset]
    wanted = set(entry['heldout_keys'])
    seen = set()
    for folder in folders:
        with np.load(folder / 'per_image_confusions.npz', allow_pickle=False) as data:
            keys = data['sample_keys'].tolist()
            mask = np.asarray([k in wanted for k in keys])
            seen.update(k for k in keys if k in wanted)
            for method in heldout:
                heldout[method] += data[dataset + '__' + method][mask].sum(0)
    if seen != wanted:
        raise RuntimeError('Incomplete disjoint holdout.')
    from eval_geometry_vip_reliability import summary
    row.update(exact_retained_confusion_replay=True, per_image_confusions_verified=True,
        paired_scored_targets_equal=True, tuning_labels_source=entry['development_source'],
        heldout_images=len(wanted), heldout_metrics={m: summary(cm, names, 0) for m, cm in heldout.items()})
    save(path, row)
    return str(path)


def main(root):
    root.resolve().relative_to((TOOL / 'results').resolve())
    if (root / 'suite_status.json').exists() or (root / 'protocol.json').exists():
        raise RuntimeError('Existing run; do not restart.')
    prepared = inventory(root)
    save(root / 'protocol.json', prepared)
    info = prepared['datasets']
    stages = {d: 'smoke' for d in ORDER}
    pending = [(d, 'smoke', 0) for d in ORDER]
    active, complete, failures, finished_shards = {}, {}, {}, set()
    started = time.time()
    while active or pending:
        for gpu, job in list(active.items()):
            if alive(job['session']):
                continue
            d, phase = job['dataset'], job['phase']
            try:
                verify_job(job)
                if phase == 'smoke':
                    stages[d] = 'search'
                    pending.append((d, 'search', 0))
                elif phase == 'search':
                    stages[d] = 'full'
                    pending.extend((d, 'full', s) for s in range(info[d]['shards']))
                else:
                    finished_shards.add((d, job['shard']))
                    if all((d, s) in finished_shards for s in range(info[d]['shards'])):
                        complete[d] = verify_dataset(root, d, info[d])
                        stages[d] = 'complete'
            except Exception as error:
                log = Path(job['log'])
                failures[job['session']] = dict(error=str(error),
                    log_tail=log.read_text(errors='replace')[-5000:] if log.exists() else 'No log')
            del active[gpu]
        if failures:
            pending.clear()
        # Start priority-domain searches before secondary smokes/full shards.
        pending.sort(key=lambda item: (0 if item[1] == 'search' else 1 if item[1] == 'smoke' else 2, ORDER.index(item[0]), item[2]))
        for d, phase, shard in list(pending):
            gpu = next((g for g in range(8) if g not in active and idle(g)), None)
            if gpu is None:
                break
            active[gpu] = launch(root, d, info[d], gpu, phase, shard)
            pending.remove((d, phase, shard))
        save(root / 'suite_status.json', dict(status='running' if active or pending else 'failed' if failures else 'complete',
            active=list(active.values()), pending=pending, stages=stages, completed=complete, failures=failures))
        if active or pending:
            time.sleep(15)
    save(root / 'suite_results.json', dict(status='failed' if failures else 'complete', completed=complete,
        failures=failures, suite_wall_seconds=time.time() - started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inventory', action='store_true')
    args = parser.parse_args()
    if args.inventory:
        print(json.dumps(inventory(args.root)))
    else:
        main(args.root)
