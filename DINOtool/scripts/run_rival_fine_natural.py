"""Run frozen natural-image evaluation only on idle GPUs, then merge coverage."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.natural_evaluation import ROOTS, TOTALS, discover_samples
from eval_rival_fine_natural import IMPLEMENTATION, save
from merge_gear_ov_shards import merge
from run_region_semantic_suite_a800 import BASE, TOOL, THIRD, PYTHON, idle
from run_sat_geometry_transport_suite import alive, read_json


DATA = Path('/data/test/datasets/VIP_natural')
PLANNED_SHARDS = dict(voc20=1, voc21=1, ade150=2, coco_stuff171=2, coco_object81=2)


def launch(root, dataset, gpu, phase, shard=0, shards=1):
    output = root/phase/dataset/f's{shard}'
    log = output.with_suffix('.log')
    session = f'gnat03_{phase}_{dataset}_s{shard}'
    if not idle(gpu) or output.exists() or log.exists() or alive(session):
        raise RuntimeError('Occupied GPU or existing output/session: '+session)
    output.parent.mkdir(parents=True, exist_ok=True)
    args = [PYTHON, '-u', 'scripts/eval_rival_fine_natural.py', '--dataset', dataset,
            '--dinov3-repo', str(TOOL/'dinov3_hub'), '--checkpoint-dir', str(BASE/'ckpt/DINO'),
            '--data-root', str(DATA/ROOTS[dataset]), '--vocabulary-config', str(root/'vocabularies'/(dataset+'_20.json')),
            '--upstream-root', str(BASE/'third_party/VIP_official_5bd25ee'), '--output-dir', str(output),
            '--text-cache', str(root/'text_cache'/(dataset+'.pt')),
            '--num-shards', str(shards), '--shard-index', str(shard)]
    if phase == 'smoke':
        args += ['--max-images', '1']
    env = ['env', f'CUDA_VISIBLE_DEVICES={gpu}', f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
           'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2', 'TOKENIZERS_PARALLELISM=false']
    shell = 'cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+['nice', '-n', '10', *args])+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, shell], check=True)
    print(json.dumps(dict(started=session, physical_gpu=gpu)), flush=True)
    return dict(dataset=dataset, gpu=gpu, phase=phase, shard=shard, shards=shards,
                output=str(output), log=str(log), session=session)


def verify_job(job):
    output = Path(job['output'])
    row = read_json(output/'results.json')
    if (not row or row['status'] != 'complete' or row['processed_images'] != row['total_images']
            or not row['weights_frozen'] or not row['head_weights_unchanged']
            or not row['target_masks_used_only_after_prediction'] or row['target_label_tuning']
            or row['signature']['implementation'] != IMPLEMENTATION):
        raise RuntimeError('Incomplete/changed frozen evaluation: '+job['session'])
    with np.load(output/'per_image_confusions.npz', allow_pickle=False) as arrays:
        if arrays['sample_keys'].tolist() != row['signature']['sample_keys']:
            raise RuntimeError('Per-image sample sequence differs.')
        for p, methods in row['metrics'].items():
            for method, metric in methods.items():
                if not np.array_equal(arrays[p+'__'+method].sum(0), metric['confusion_matrix']):
                    raise RuntimeError('Per-image confusion sums differ.')
    return row


def monitor(root, active, pending, completed, failures, blocked, phase, shards):
    verified = set()
    while active or pending:
        for gpu, job in list(active.items()):
            if alive(job['session']):
                continue
            try:
                verify_job(job)
                verified.add((job['dataset'], job['shard']))
                dataset = job['dataset']
                if all((dataset, s) in verified for s in range(shards[dataset])):
                    inputs = [str(root/phase/dataset/f's{s}') for s in range(shards[dataset])]
                    path = root/phase/dataset/'merged.json'
                    row = merge(inputs, str(path))
                    expected = 1 if phase == 'smoke' else TOTALS[dataset]
                    if not row['coverage_verified'] or row['processed_images'] != expected:
                        raise RuntimeError('Incorrect global sample coverage: '+dataset)
                    row.update(weights_frozen=True, head_weights_unchanged=True,
                               target_masks_used_only_after_prediction=True, target_label_tuning=False,
                               per_image_confusions_verified=True)
                    save(path, row)
                    completed[dataset] = str(path)
            except Exception as error:
                failures[job['session']] = dict(error=str(error), log_tail=Path(job['log']).read_text()[-6000:])
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


def main(root, reuse_from=None):
    root.resolve().relative_to((TOOL/'results').resolve())
    if root.exists():
        raise RuntimeError('Existing natural evaluation; inspect rather than relaunch.')
    available, blocked = [], {}
    for dataset in TOTALS:
        try:
            discover_samples(dataset, DATA/ROOTS[dataset])
            available.append(dataset)
        except (ValueError, FileNotFoundError) as error:
            blocked[dataset] = str(error)
    if not available:
        raise RuntimeError('No complete natural validation dataset is available.')
    root.mkdir(parents=True)
    if reuse_from:
        reuse_from.resolve().relative_to((TOOL/'results').resolve())
        old = read_json(reuse_from/'protocol.json')
        if not old or old['implementation'] != IMPLEMENTATION or old['datasets'] != available:
            raise RuntimeError('Recovery vocabulary/data identity differs.')
        (root/'vocabularies').symlink_to(reuse_from/'vocabularies', target_is_directory=True)
        (root/'text_cache').symlink_to(reuse_from/'text_cache', target_is_directory=True)
    else:
        subprocess.run([PYTHON, 'scripts/prepare_natural_vocabularies.py', '--output', str(root/'vocabularies')], check=True)
    shards = {d: PLANNED_SHARDS.get(d, 1) for d in available}
    save(root/'protocol.json', dict(implementation=IMPLEMENTATION, datasets=available,
        blocked=blocked, full_totals={d: TOTALS[d] for d in available}, full_shards=shards,
        physical_gpus=list(range(8)), aliases_per_class=20, expansion='natural-semantic20-v1-20261003',
        local_and_contextual_templates='openai_imagenet_template', model_rules_changed=False,
        target_label_tuning=False, smoke_gate='Execution, finite values, label mapping and coverage only; NO performance selection.',
        evaluation_protocol='Full validation labels at original RGB dimensions; not an upstream VIP benchmark reproduction.'))
    started = time.time()
    smoke_done, failures = {}, {}
    smoke_shards = {d: 1 for d in available}
    monitor(root, {}, [(d, 0) for d in available], smoke_done, failures, blocked, 'smoke', smoke_shards)
    if failures:
        save(root/'suite_results.json', dict(status='failed', phase='smoke', failures=failures, blocked=blocked))
        return
    completed = {}
    pending = [(d, s) for d in available for s in range(shards[d])]
    monitor(root, {}, pending, completed, failures, blocked, 'full', shards)
    save(root/'suite_results.json', dict(status='failed' if failures else 'complete_available',
         completed=completed, failures=failures, blocked=blocked,
         suite_wall_seconds=time.time()-started, complete_eight_protocols=not blocked))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reuse-from', type=Path)
    args = parser.parse_args()
    main(args.root, args.reuse_from)
