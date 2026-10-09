"""Finite dependency queue for frozen semantic inputs on physical GPUs 4-7 only."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from dinotool.natural_evaluation import ROOTS, TOTALS
from dinotool.natural_sense_calibration import IMPLEMENTATION
from eval_rival_fine_full import save
from run_natural_text_adaptation import merge_results, verify as verify_evaluation
from run_region_semantic_suite_a800 import TOOL, BASE, PYTHON, THIRD, idle
from run_sat_geometry_transport_suite import alive, read_json


DATASETS = ('voc20', 'voc21', 'ade150', 'coco_stuff171', 'coco_object81')
GPUS = (4, 5, 6, 7)
SOURCE = TOOL/'results/natural_text_adaptation_20261003'
SESSION = 'gnsense04_controller'


def shards(dataset, phase):
    return 2 if phase == 'full' and dataset in ('ade150', 'coco_stuff171', 'coco_object81') else 1


def verify(job):
    phase, output = job['phase'], Path(job['output'])
    if phase in ('smoke', 'selection'):
        row = read_json(output/'selection.json')
        if (not row or row['status'] != 'complete' or row['implementation'] != IMPLEMENTATION
                or row['target_masks_loaded'] or row['target_label_tuning'] or not row['weights_frozen']
                or not row['head_weights_unchanged'] or not row['source_equivalence_first_image_verified']
                or len(set(row['image_keys'])) != row['processed_images']
                or row['processed_images'] != (1 if phase == 'smoke' else 64)
                or row['smoke_only'] != (phase == 'smoke')):
            raise RuntimeError('Incomplete or changed image-only selection: '+job['session'])
    else:
        row = verify_evaluation(job)
        if row['signature']['implementation'] != IMPLEMENTATION or not row['source_equivalence_first_image_verified']:
            raise RuntimeError('Changed implementation or source reader: '+job['session'])
    return row


def launch(root, dataset, phase, shard, gpu):
    if gpu not in GPUS or not idle(gpu):
        raise RuntimeError('Only idle physical GPUs 4-7 are permitted.')
    output = root/phase/dataset/f's{shard}'
    log = output.with_suffix('.log')
    session = f'gnsense04_{phase}_{dataset}_s{shard}'
    if output.exists() or log.exists() or alive(session):
        raise RuntimeError('Existing output/session; no relaunch: '+session)
    output.parent.mkdir(parents=True, exist_ok=True)
    action = 'select' if phase in ('smoke', 'selection') else 'evaluate'
    command = [PYTHON, '-u', 'scripts/eval_natural_sense_adaptation.py', action,
        '--dataset', dataset, '--dinov3-repo', str(TOOL/'dinov3_hub'),
        '--checkpoint-dir', str(BASE/'ckpt/DINO'), '--upstream-root', str(BASE/'third_party/VIP_official_5bd25ee'),
        '--data-root', str(Path('/data/test/datasets/VIP_natural')/ROOTS[dataset]),
        '--source-vocabulary', str(TOOL/'results/rival_fine_natural20_full_20261003/vocabularies'/(dataset+'_20.json')),
        '--source-selection', str(SOURCE/'selection'/dataset/'s0/selection.json'),
        '--cache-dir', str(SOURCE/'text_cache'/dataset), '--candidate-vocabulary', str(root/'vocabularies'/(dataset+'.json')),
        '--candidate-cache-dir', str(root/'text_cache'/dataset), '--output-dir', str(output),
        '--num-shards', str(shards(dataset, phase)), '--shard-index', str(shard)]
    if phase in ('pilot', 'full'):
        command += ['--selection', str(root/'selection'/dataset/'s0/selection.json')]
    if phase in ('smoke', 'pilot'):
        command += ['--max-images', '1' if phase == 'smoke' else '16']
    env = ['env', f'CUDA_VISIBLE_DEVICES={gpu}', f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
           'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2', 'TOKENIZERS_PARALLELISM=false']
    shell = 'cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+['nice', '-n', '10', *command])+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, shell], check=True)
    return dict(dataset=dataset, phase=phase, shard=shard, gpu=gpu, session=session, output=str(output), log=str(log))


def main(root):
    root.resolve().relative_to((TOOL/'results').resolve())
    if (root/'suite_status.json').exists() or (root/'protocol.json').exists():
        raise RuntimeError('Queue already initialized; inspect instead of relaunching.')
    for dataset in DATASETS:
        if not (root/'vocabularies'/(dataset+'.json')).exists():
            raise RuntimeError('Frozen input vocabulary missing: '+dataset)
    root.mkdir(parents=True, exist_ok=True)
    save(root/'protocol.json', dict(implementation=IMPLEMENTATION, datasets=DATASETS, physical_gpus=GPUS,
        core_changed=False, calibration_images=64, pilot_images=16,
        phases=['one_image_source_equivalence', '64_image_mask_free_calibration', 'paired16', 'full'],
        full_expansion_rule='Structural/source-equivalence verification only; no labeled pilot winner selection.',
        template_rule='Default seg_template or frozen source template; source templates were previously searched.',
        thresholds_rule='Float full-image probabilities with conservative image-only foreground guards.',
        no_target_label_tuning=True))
    pending = [(d, p, s) for p in ('smoke', 'selection', 'pilot', 'full') for d in DATASETS for s in range(shards(d, p))]
    done, active, completed, failures = set(), {}, {}, {}
    previous = dict(selection='smoke', pilot='selection', full='pilot')
    while pending or active:
        for gpu, job in list(active.items()):
            if alive(job['session']):
                continue
            try:
                verify(job)
                d, p = job['dataset'], job['phase']
                done.add((d, p, job['shard']))
                if all((d, p, s) in done for s in range(shards(d, p))):
                    path = root/p/d/('s0/selection.json' if p in ('smoke', 'selection') else 'merged.json')
                    if p not in ('smoke', 'selection'):
                        row = merge_results([str(root/p/d/f's{s}') for s in range(shards(d, p))], path)
                        if row['processed_images'] != (16 if p == 'pilot' else TOTALS[d]):
                            raise RuntimeError('Incorrect complete global coverage: '+d)
                        row['source_equivalence_first_image_verified'] = True
                        save(path, row)
                    completed[d+'/'+p] = str(path)
            except Exception as error:
                failures[job['session']] = dict(error=str(error), log=job['log'],
                    log_tail=subprocess.run(['tail', '-n', '30', job['log']], capture_output=True, text=True).stdout)
            del active[gpu]
        if not failures:
            for d, p, s in list(pending):
                if p in previous and (d, previous[p], 0) not in done:
                    continue
                gpu = next((g for g in GPUS if g not in active and idle(g)), None)
                if gpu is None:
                    break
                job = launch(root, d, p, s, gpu)
                active[gpu] = job
                pending.remove((d, p, s))
                print(json.dumps(dict(started=job['session'], gpu=gpu)), flush=True)
        else:
            pending.clear()
        save(root/'suite_status.json', dict(status='failed' if failures else 'running' if pending or active else 'complete',
            allowed_gpus=GPUS, active=list(active.values()), pending=pending, completed=completed, failures=failures))
        if pending or active:
            time.sleep(15)
    save(root/'suite_results.json', dict(status='failed' if failures else 'complete', implementation=IMPLEMENTATION,
                                         completed=completed, failures=failures, allowed_gpus=GPUS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
