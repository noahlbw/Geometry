"""Finite Context transfer queue after the frozen five-protocol suite finishes."""
import argparse
from pathlib import Path
import shlex
import subprocess
import time

from dinotool.context_sense_evaluation import IMPLEMENTATION
from dinotool.natural_sense_calibration import IMPLEMENTATION as SELECTOR
from eval_rival_fine_full import save
from run_natural_text_adaptation import merge_results, verify as verify_evaluation
from run_region_semantic_suite_a800 import TOOL, BASE, PYTHON, THIRD, idle
from run_sat_geometry_transport_suite import alive, read_json


DATASETS, GPUS = ('context59', 'context60'), (4, 5, 6, 7)


def shards(phase):
    return 2 if phase == 'full' else 1


def verify(job):
    if job['phase'] in ('smoke', 'selection'):
        row = read_json(Path(job['output'])/'selection.json')
        expected = 1 if job['phase'] == 'smoke' else 64
        if (not row or row['status'] != 'complete' or row['implementation'] != SELECTOR
                or row['target_masks_loaded'] or row['target_label_tuning'] or not row['weights_frozen']
                or not row['head_weights_unchanged'] or not row['source_equivalence_first_image_verified']
                or row['processed_images'] != expected or len(set(row['image_keys'])) != expected
                or row['smoke_only'] != (job['phase'] == 'smoke')):
            raise RuntimeError('Unverified Context image-only calibration: '+job['session'])
        return row
    row = verify_evaluation(job)
    if row['signature']['implementation'] != IMPLEMENTATION or not row['source_equivalence_first_image_verified']:
        raise RuntimeError('Changed Context reader/implementation.')
    return row


def launch(root, dataset, phase, shard, gpu):
    if gpu not in GPUS or not idle(gpu):
        raise RuntimeError('Context requires an idle physical GPU4-7.')
    output = root/phase/dataset/f's{shard}'
    log, session = output.with_suffix('.log'), f'gctxsense04_{phase}_{dataset}_s{shard}'
    if output.exists() or log.exists() or alive(session):
        raise RuntimeError('Existing Context output/session; no relaunch.')
    output.parent.mkdir(parents=True, exist_ok=True)
    action = 'select' if phase in ('smoke', 'selection') else 'evaluate'
    command = [PYTHON, '-u', 'scripts/eval_context_sense_adaptation.py', action, '--dataset', dataset,
        '--dinov3-repo', str(TOOL/'dinov3_hub'), '--checkpoint-dir', str(BASE/'ckpt/DINO'),
        '--upstream-root', str(BASE/'third_party/VIP_official_5bd25ee'),
        '--data-root', str(Path('/data/test/datasets/VIP_natural/VOCdevkit/VOC2010')),
        '--source-vocabulary', str(root/'inputs'/(dataset+'_source_vocabulary.json')),
        '--source-selection', str(root/'inputs'/(dataset+'_source_input.json')),
        '--cache-dir', str(root/'source_text_cache'/dataset),
        '--candidate-vocabulary', str(root/'vocabularies'/(dataset+'.json')),
        '--candidate-cache-dir', str(root/'candidate_text_cache'/dataset), '--output-dir', str(output),
        '--num-shards', str(shards(phase)), '--shard-index', str(shard)]
    if phase in ('pilot', 'full'):
        command += ['--selection', str(root/'selection'/dataset/'s0/selection.json')]
    if phase in ('smoke', 'pilot'):
        command += ['--max-images', '1' if phase == 'smoke' else '16']
    env = ['env', f'CUDA_VISIBLE_DEVICES={gpu}', f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
           'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2', 'TOKENIZERS_PARALLELISM=false']
    shell = 'cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+['nice', '-n', '10', *command])+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, shell], check=True)
    return dict(dataset=dataset, phase=phase, shard=shard, gpu=gpu, session=session, output=str(output), log=str(log))


def main(root, after_suite):
    root.resolve().relative_to((TOOL/'results').resolve())
    if (root/'suite_status.json').exists() or (root/'protocol.json').exists():
        raise RuntimeError('Existing Context queue; no relaunch.')
    for dataset in DATASETS:
        for path in (root/'vocabularies'/(dataset+'.json'), root/'inputs'/(dataset+'_source_input.json')):
            if not path.exists():
                raise RuntimeError('Missing frozen Context input: '+str(path))
    save(root/'protocol.json', dict(implementation=IMPLEMENTATION, selector_implementation=SELECTOR,
        datasets=DATASETS, physical_gpus=GPUS, core_changed=False, calibration_images=64, pilot_images=16,
        source_profile='Fixed ImageNet semantic pool, tau=tem=1; threshold0 for Context59 and .1 for Context60. No fitted source calibration.',
        selector='Transferred actual full-image pseudo-witness rule from the already frozen natural suite; no new selection thresholds.',
        vip_reference='Official short Context city templates/settings/queries, explicit empty-row finite repair.',
        scheduling_dependency=str(after_suite), expected_full_images_per_protocol=5105,
        full_expansion_rule='Source equivalence and coverage checks only; no labeled winner selection.',
        no_target_label_tuning=True))
    while True:
        dependency = read_json(after_suite/'suite_results.json')
        if dependency and dependency['status'] == 'complete':
            break
        if not alive('gnsense04_controller'):
            dependency = read_json(after_suite/'suite_results.json')
            if dependency and dependency['status'] == 'complete':
                break
            failure = dict(error='Preceding five-protocol controller exited before verified completion.',
                           dependency=str(after_suite), dependency_result=dependency)
            save(root/'suite_status.json', dict(status='failed', active=[], pending=[], failures={'dependency': failure}))
            save(root/'suite_results.json', dict(status='failed', failures={'dependency': failure}))
            return
        save(root/'suite_status.json', dict(status='waiting_for_preceding_suite', active=[],
            pending=list(DATASETS), failures={}, dependency=str(after_suite), allowed_gpus=GPUS))
        time.sleep(15)
    pending = [(d, p, s) for p in ('smoke', 'selection', 'pilot', 'full') for d in DATASETS for s in range(shards(p))]
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
                if all((d, p, s) in done for s in range(shards(p))):
                    path = root/p/d/('s0/selection.json' if p in ('smoke', 'selection') else 'merged.json')
                    if p in ('pilot', 'full'):
                        row = merge_results([str(root/p/d/f's{s}') for s in range(shards(p))], path)
                        if row['processed_images'] != (16 if p == 'pilot' else 5105):
                            raise RuntimeError('Incomplete Context global image coverage.')
                        row['source_equivalence_first_image_verified'] = True
                        save(path, row)
                    completed[d+'/'+p] = str(path)
            except Exception as error:
                failures[job['session']] = dict(error=str(error), log=job['log'],
                    log_tail=subprocess.run(['tail', '-n', '30', job['log']], capture_output=True, text=True).stdout)
            del active[gpu]
        if failures:
            pending.clear()
        for d, p, s in list(pending):
            if p in previous and (d, previous[p], 0) not in done:
                continue
            gpu = next((g for g in GPUS if g not in active and idle(g)), None)
            if gpu is None:
                break
            active[gpu] = launch(root, d, p, s, gpu)
            pending.remove((d, p, s))
        save(root/'suite_status.json', dict(status='failed' if failures else 'running' if pending or active else 'complete',
            active=list(active.values()), pending=pending, completed=completed, failures=failures, allowed_gpus=GPUS))
        if active or pending:
            time.sleep(15)
    save(root/'suite_results.json', dict(status='failed' if failures else 'complete', implementation=IMPLEMENTATION,
                                         completed=completed, failures=failures, allowed_gpus=GPUS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--after-suite', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.after_suite)
