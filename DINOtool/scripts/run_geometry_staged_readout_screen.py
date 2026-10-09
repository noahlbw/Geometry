"""One finite fixed96 stage-allocation experiment on idle physical GPUs0-7."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.geometry_staged_readout import IMPLEMENTATION, METHODS, PRIMARY, settings
from run_region_semantic_suite_a800 import BASE, TOOL, PYTHON, THIRD, SETTINGS, idle
from run_sat_geometry_transport_suite import reference, exact_miou, read_json, alive
from run_geometry_region_strong_suite import verify_dataset


SCREEN = TOOL/'results/sat_geometry_transport_screen_20261002'


def shard_count(dataset):
    return 4 if dataset == 'udd5' else 1


def launch(root, setting, shard, gpu):
    dataset, data, vocab, *_ = setting
    output, log = root/dataset/f's{shard}', root/dataset/f's{shard}.log'
    session = f'gstage04_{dataset}_s{shard}'
    if output.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Existing job or occupied GPU: '+session)
    log.parent.mkdir(parents=True, exist_ok=True)
    args = [PYTHON, '-u', 'scripts/eval_geometry_staged_readout.py', '--dataset', dataset,
        '--dinov3-repo', str(TOOL/'dinov3_hub'), '--checkpoint-dir', str(BASE/'ckpt/DINO'),
        '--data-root', str(Path('/data/test/datasets')/data), '--vocabulary-config', str(TOOL/'configs'/vocab),
        '--upstream-root', str(BASE/'third_party/VIP_official_5bd25ee'), '--output-dir', str(output),
        '--source-diagnostic', str(root/f'{dataset}_samples.json'),
        '--num-shards', str(shard_count(dataset)), '--shard-index', str(shard)]
    env = ['env', f'CUDA_VISIBLE_DEVICES={gpu}', f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
           'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2', 'TOKENIZERS_PARALLELISM=false']
    shell = f'cd {shlex.quote(str(TOOL))} && exec {shlex.join(env+args)} > {shlex.quote(str(log))} 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, shell], check=True)
    print(f'Started {session} GPU{gpu}', flush=True)
    return {'dataset': dataset, 'shard': shard, 'session': session, 'output': str(output), 'log': str(log)}


def verify_one(root, dataset):
    row = verify_dataset(root, dataset, 'screen', shard_count=shard_count(dataset),
                         implementation=IMPLEMENTATION, semantic_source=None)
    old = read_json(reference(dataset))
    if row['signature']['gear'] != {'geometry': old['signature']['gear']['geometry'], 'primary': PRIMARY, **settings()}:
        raise RuntimeError('Frozen readout rule differs.')
    keys = row['signature']['sample_keys']
    with np.load(root/dataset/'per_image_confusions.npz', allow_pickle=False) as current, np.load(
            reference(dataset).parent/'per_image_confusions.npz', allow_pickle=False) as previous:
        lookup = {key: index for index, key in enumerate(previous['sample_keys'].tolist())}
        order = [lookup[key] for key in keys]
        for key in row['metrics']:
            field = key+'__Geometry_BlockPrefix'
            if not np.array_equal(current[field], previous[field][order]):
                raise RuntimeError('Historical BlockPrefix per-image replay differs.')
    row['exact_reference_matched_baselines'].append('Geometry_BlockPrefix')
    (root/dataset/'merged.json').write_text(json.dumps(row, indent=2)+'\n')
    return row


def run(root):
    root.resolve().relative_to((TOOL/'results').resolve())
    smoke = read_json(root/'smoke/results.json')
    if (not smoke or smoke['status'] != 'complete' or smoke['config'] != settings()
            or smoke['target_masks_loaded'] or not smoke['weights_frozen'] or not smoke['head_weights_unchanged']):
        raise RuntimeError('Verified actual-checkpoint smoke required.')
    if (root/'protocol.json').exists():
        raise RuntimeError('Refusing a previously launched screen.')
    if any(not idle(gpu) for gpu in range(8)):
        raise RuntimeError('All eight GPUs must be idle at suite launch.')
    manifests = {}
    for dataset, *_ in SETTINGS:
        manifest = read_json(SCREEN/f'{dataset}_samples.json')
        keys = manifest['signature']['samples']
        old = read_json(reference(dataset))
        if (len(keys) != (40 if dataset == 'udd5' else 8) or len(keys) != len(set(keys))
                or not old['coverage_verified'] or not set(keys).issubset(old['signature']['sample_keys'])):
            raise RuntimeError('Incomplete reference or fixed manifest: '+dataset)
        manifests[dataset] = manifest
    protocol = {'implementation': IMPLEMENTATION, 'config': settings(), 'methods': METHODS, 'primary': PRIMARY,
        'physical_gpus': list(range(8)), 'datasets': {key: len(row['signature']['samples']) for key, row in manifests.items()},
        'gate': 'mean>Geometry/BlockPrefix/SCLIP_Two/VIPProxy_Two; VDD/Potsdam no loss; protocol loss<=1pp; latency<=1.1x',
        'note': 'Fixed96 developed validation only. No automatic full rollout or control-to-primary switching.'}
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for dataset, manifest in manifests.items():
        (root/f'{dataset}_samples.json').write_text(json.dumps(manifest, indent=2)+'\n')
    by_name = {row[0]: row for row in SETTINGS}
    queue = [(by_name[dataset], shard) for dataset in ('vdd','potsdam','udd5','oem','loveda','vaihingen','landcoverai','flair1')
             for shard in range(shard_count(dataset))]
    active, completed, outcomes, failures = {}, set(), {}, {}
    started = time.perf_counter()
    while queue or active:
        for gpu, job in list(active.items()):
            if alive(job['session']):
                continue
            row = read_json(Path(job['output'])/'results.json')
            if (row and row['status'] == 'complete' and row['processed_images'] == row['total_images']
                    and (Path(job['output'])/'per_image_confusions.npz').exists()):
                completed.add((job['dataset'], job['shard']))
            else:
                failures[job['session']] = Path(job['log']).read_text()[-5000:]
            del active[gpu]
        for dataset in by_name:
            if dataset not in outcomes and all((dataset, s) in completed for s in range(shard_count(dataset))):
                try:
                    outcomes[dataset] = verify_one(root, dataset)
                except Exception as error:
                    failures[dataset] = str(error)
                    print('Verification failure', dataset, error, flush=True)
        if failures:
            queue.clear()
        for gpu in range(8):
            if queue and gpu not in active and idle(gpu):
                setting, shard = queue.pop(0)
                active[gpu] = launch(root, setting, shard, gpu)
        state = {'active': active, 'queued': [(row[0], shard) for row, shard in queue],
                 'completed': sorted(completed), 'verified_datasets': list(outcomes), 'failures': failures}
        temp = root/'suite_status.tmp'; temp.write_text(json.dumps(state, indent=2)+'\n'); temp.replace(root/'suite_status.json')
        if queue or active:
            time.sleep(10)
    if failures or len(outcomes) != 8:
        result = {'status': 'failed', 'failures': failures}
    else:
        values = {dataset: {key: {method: exact_miou(metric) for method, metric in group.items()}
                           for key, group in row['metrics'].items()} for dataset, row in outcomes.items()}
        means = {method: float(np.mean([groups['D' if dataset == 'loveda' else dataset][method]
                                       for dataset, groups in values.items()])) for method in METHODS}
        checks = {'mean_vs_'+method: means[PRIMARY] > means[method]
                  for method in ('Geometry','Geometry_BlockPrefix','SCLIP_Two','VIPProxy_Two')}
        for dataset, groups in values.items():
            for key, group in groups.items():
                checks['retain_'+dataset+'_'+key] = group[PRIMARY]-group['Geometry'] >= -1
            if dataset in ('vdd','potsdam'):
                checks['focus_'+dataset] = groups[dataset][PRIMARY] >= groups[dataset]['Geometry']
        ratio = smoke['deployed_window_cost'][PRIMARY]['window_median_seconds']/smoke['deployed_window_cost']['Geometry']['window_median_seconds']
        checks['latency_ratio_at_most_1_1'] = ratio <= 1.1
        result = {'status': 'complete', 'implementation': IMPLEMENTATION, 'images': 96, 'values': values,
                  'mean_miou': means, 'checks': checks, 'passed': all(checks.values()), 'latency_ratio': ratio}
    result['wall_seconds'] = time.perf_counter()-started
    (root/'suite_results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
