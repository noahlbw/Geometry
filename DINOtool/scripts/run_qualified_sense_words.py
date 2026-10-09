"""Finite paired word-input queue, after Context; idle physical GPUs4-7 only."""
import argparse
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from dinotool.natural_evaluation import ROOTS, TOTALS
from dinotool.qualified_sense_words import IMPLEMENTATION
from eval_rival_fine_full import save
from run_natural_text_adaptation import merge_results, verify as verify_evaluation
from run_region_semantic_suite_a800 import TOOL, BASE, PYTHON, THIRD, idle
from run_sat_geometry_transport_suite import alive, read_json


DATASETS = ('voc20', 'voc21', 'ade150', 'coco_stuff171', 'coco_object81', 'context59', 'context60')
GPUS = (4, 5, 6, 7)
PRECEDING = TOOL/'results/context_sense_adaptation_20261004'
SOURCE = TOOL/'results/natural_text_adaptation_20261003'
NATURAL = TOOL/'results/natural_sense_adaptation_v2_20261004'


def baseline_root(dataset):
    return PRECEDING if dataset.startswith('context') else NATURAL


def shards(dataset, phase):
    return 1 if phase == 'smoke' or dataset.startswith('voc') else 2


def verify(job):
    output = Path(job['output'])
    if job['phase'] == 'smoke':
        row = read_json(output/'smoke.json')
        if (not row or row['status'] != 'complete' or row['implementation'] != IMPLEMENTATION
                or row['target_masks_loaded'] or row['target_label_tuning'] or row['main_model_changed']
                or not row['weights_frozen'] or not row['head_weights_unchanged']
                or not row['source_equivalence_first_image_verified']
                or row['processed_images'] != 1 or row['total_images'] != 1):
            raise RuntimeError('Unverified mask-free paired-reader smoke: '+job['session'])
        return row
    row = verify_evaluation(job)
    if row['signature']['implementation'] != IMPLEMENTATION or not row['source_equivalence_first_image_verified']:
        raise RuntimeError('Changed word-comparison implementation or reader.')
    baseline_dir = baseline_root(job['dataset'])/'full'/job['dataset']/f's{job["shard"]}'
    original = read_json(baseline_dir/'results.json')
    sig, before = row['signature'], original['signature']
    for field in ('classes', 'checkpoints', 'global_sample_keys_sha256', 'sample_keys'):
        if sig[field] != before[field]:
            raise RuntimeError('Changed paired baseline identity: '+field)
    if (sig['gear']['baseline_vocabulary_sha256'] != before['gear']['candidate_vocabulary_sha256']
            or sig['gear']['profiles']['frozen'] != before['gear']['profiles']['default']):
        raise RuntimeError('Baseline words or default settings changed.')
    with np.load(output/'per_image_confusions.npz', allow_pickle=False) as ours, np.load(
            baseline_dir/'per_image_confusions.npz', allow_pickle=False) as old:
        for suffix in ('', '_NoThreshold'):
            if not np.array_equal(ours[job['dataset']+'__V2_Words'+suffix], old[job['dataset']+'__Sense_Default'+suffix]):
                raise RuntimeError('Per-image v2 baseline confusion differs: '+job['dataset']+suffix)
    return row


def launch(root, dataset, phase, shard, gpu):
    if gpu not in GPUS or not idle(gpu):
        raise RuntimeError('Only idle physical GPUs4-7 may run this study.')
    output = root/phase/dataset/f's{shard}'
    log, session = output.with_suffix('.log'), f'gqs04_{phase}_{dataset}_s{shard}'
    if output.exists() or log.exists() or alive(session):
        raise RuntimeError('Existing output, log or worker; no relaunch: '+session)
    output.parent.mkdir(parents=True, exist_ok=True)
    old_root = baseline_root(dataset)
    context = dataset.startswith('context')
    source_selection = old_root/'inputs'/(dataset+'_source_input.json') if context else SOURCE/'selection'/dataset/'s0/selection.json'
    source_vocabulary = old_root/'inputs'/(dataset+'_source_vocabulary.json') if context else TOOL/'results/rival_fine_natural20_full_20261003/vocabularies'/(dataset+'_20.json')
    source_cache = old_root/'source_text_cache'/dataset if context else SOURCE/'text_cache'/dataset
    old_cache = old_root/('candidate_text_cache' if context else 'text_cache')/dataset/'seg_template.pt'
    command = [PYTHON, '-u', 'scripts/eval_qualified_sense_words.py', '--dataset', dataset,
        '--dinov3-repo', str(TOOL/'dinov3_hub'), '--checkpoint-dir', str(BASE/'ckpt/DINO'),
        '--upstream-root', str(BASE/'third_party/VIP_official_5bd25ee'),
        '--data-root', str(Path('/data/test/datasets/VIP_natural')/ROOTS[dataset]),
        '--source-vocabulary', str(source_vocabulary), '--source-selection', str(source_selection),
        '--cache-dir', str(source_cache), '--candidate-vocabulary', str(root/'vocabularies'/(dataset+'.json')),
        '--candidate-cache-dir', str(root/'text_cache'/dataset), '--output-dir', str(output),
        '--baseline-vocabulary', str(old_root/'vocabularies'/(dataset+'.json')), '--baseline-cache', str(old_cache),
        '--num-shards', str(shards(dataset, phase)), '--shard-index', str(shard)]
    if phase == 'smoke':
        command += ['--smoke']
    env = ['env', f'CUDA_VISIBLE_DEVICES={gpu}', f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
           'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2', 'TOKENIZERS_PARALLELISM=false']
    shell = 'cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+['nice', '-n', '10', *command])+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, shell], check=True)
    return dict(dataset=dataset, phase=phase, shard=shard, gpu=gpu, session=session, output=str(output), log=str(log))


def main(root):
    root.resolve().relative_to((TOOL/'results').resolve())
    if (root/'suite_status.json').exists() or (root/'protocol.json').exists():
        raise RuntimeError('Existing qualified-word queue; no relaunch.')
    for dataset in DATASETS:
        if not (root/'vocabularies'/(dataset+'.json')).exists():
            raise RuntimeError('Frozen qualified vocabulary missing: '+dataset)
    save(root/'protocol.json', dict(implementation=IMPLEMENTATION, datasets=DATASETS, physical_gpus=GPUS,
        main_model_changed=False, profiles='Paired seg_template/tau1/tem1 at identical frozen source rejection; no profile search.',
        words='Fixed category-sense qualification/replacement; no quota or GT fitting.',
        baseline_rule='Require exact per-image agreement with the existing v2 default arm.',
        scheduling_dependency=str(PRECEDING), no_target_label_tuning=True))
    while True:
        dependency = read_json(PRECEDING/'suite_results.json')
        if dependency and dependency['status'] == 'complete':
            break
        if not alive('gctxsense04_controller'):
            dependency = read_json(PRECEDING/'suite_results.json')
            if dependency and dependency['status'] == 'complete':
                break
            failure = dict(error='Context controller exited before verified completion.', dependency_result=dependency)
            save(root/'suite_status.json', dict(status='failed', active=[], pending=[], failures={'dependency': failure}))
            save(root/'suite_results.json', dict(status='failed', failures={'dependency': failure}))
            return
        save(root/'suite_status.json', dict(status='waiting_for_preceding_suite', active=[], pending=list(DATASETS),
            failures={}, dependency=str(PRECEDING), allowed_gpus=GPUS))
        time.sleep(15)
    pending = [(d, p, s) for p in ('smoke', 'full') for d in DATASETS for s in range(shards(d, p))]
    done, active, completed, failures = set(), {}, {}, {}
    while pending or active:
        for gpu, job in list(active.items()):
            if alive(job['session']):
                continue
            try:
                verify(job)
                d, p = job['dataset'], job['phase']
                done.add((d, p, job['shard']))
                if all((d, p, s) in done for s in range(shards(d, p))):
                    path = root/p/d/('s0/smoke.json' if p == 'smoke' else 'merged.json')
                    if p == 'full':
                        row = merge_results([str(root/p/d/f's{s}') for s in range(shards(d, p))], path)
                        if row['processed_images'] != TOTALS[d]:
                            raise RuntimeError('Incorrect full image coverage: '+d)
                        row['matched_v2_per_image_confusions_verified'] = True
                        save(path, row)
                    completed[d+'/'+p] = str(path)
            except Exception as error:
                failures[job['session']] = dict(error=str(error), log=job['log'],
                    log_tail=subprocess.run(['tail', '-n', '30', job['log']], capture_output=True, text=True).stdout)
            del active[gpu]
        if failures:
            pending.clear()
        for d, p, s in list(pending):
            if p == 'full' and (d, 'smoke', 0) not in done:
                continue
            gpu = next((g for g in GPUS if g not in active and idle(g)), None)
            if gpu is None:
                break
            active[gpu] = launch(root, d, p, s, gpu)
            pending.remove((d, p, s))
        save(root/'suite_status.json', dict(status='failed' if failures else 'running' if pending or active else 'complete',
            active=list(active.values()), pending=pending, completed=completed, failures=failures, allowed_gpus=GPUS))
        if pending or active:
            time.sleep(15)
    save(root/'suite_results.json', dict(status='failed' if failures else 'complete', implementation=IMPLEMENTATION,
                                         completed=completed, failures=failures, allowed_gpus=GPUS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
