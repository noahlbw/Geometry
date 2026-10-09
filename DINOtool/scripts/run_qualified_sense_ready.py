"""Advance an unstarted qualified-word queue using per-protocol baseline readiness."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def waiting_queue(state, protocol):
    if (not state or state.get('status') != 'waiting_for_preceding_suite'
            or state.get('active') or state.get('completed') or state.get('failures')
            or not protocol or protocol.get('implementation') != 'natural-fixed-reader-qualified-queries-v1-20261004'
            or protocol.get('physical_gpus') != [4, 5, 6, 7] or protocol.get('main_model_changed')
            or not protocol.get('no_target_label_tuning')):
        raise RuntimeError('Only the unstarted, healthy waiting queue may advance.')


def baseline_ready(root, dataset, total, num_shards):
    path = root/'full'/dataset/'merged.json'
    if not path.is_file():
        return False
    row = json.loads(path.read_text())
    if (row.get('status') != 'complete' or row.get('processed_images') != total
            or row.get('total_images') != total or not row.get('coverage_verified')
            or not row.get('per_image_confusions_verified')
            or row['signature']['num_shards'] != num_shards):
        return False
    cache = 'candidate_text_cache' if dataset.startswith('context') else 'text_cache'
    required = [root/'vocabularies'/(dataset+'.json'), root/cache/dataset/'seg_template.pt']
    for shard in range(num_shards):
        required.extend(root/'full'/dataset/f's{shard}'/name
                        for name in ('results.json', 'per_image_confusions.npz'))
    return all(path.is_file() for path in required)


def yield_to_upstream(natural_state, context_state):
    if natural_state and natural_state.get('status') == 'running' and natural_state.get('pending'):
        return True
    if context_state and context_state.get('status') == 'waiting_for_preceding_suite':
        return bool(natural_state and natural_state.get('status') == 'complete')
    return bool(context_state and context_state.get('status') == 'running' and context_state.get('pending'))


def main(root, controller_log):
    from dinotool.natural_evaluation import TOTALS
    from eval_rival_fine_full import save
    from run_qualified_sense_words import (DATASETS, GPUS, IMPLEMENTATION, TOOL, NATURAL, PRECEDING, baseline_root,
        shards, verify, launch, merge_results, idle, alive, read_json)

    root.resolve().relative_to((TOOL/'results').resolve())
    waiting_queue(read_json(root/'suite_status.json'), read_json(root/'protocol.json'))
    if (root/'suite_results.json').exists() or (root/'smoke').exists() or (root/'full').exists():
        raise RuntimeError('Existing word-evaluation outputs; do not advance or relaunch.')
    if (root/'ready_scheduling.json').exists():
        raise RuntimeError('Existing scheduling transition; inspect instead of relaunching.')
    for dataset in DATASETS:
        if not (root/'vocabularies'/(dataset+'.json')).is_file():
            raise RuntimeError('Frozen vocabulary missing: '+dataset)
    save(root/'ready_scheduling.json', dict(status='started', implementation=IMPLEMENTATION,
        allowed_gpus=GPUS, original_protocol_unchanged=True, evaluation_rules_unchanged=True,
        transition='Unstarted whole-Context wait to verified per-protocol full-baseline readiness.',
        controller_log=str(controller_log)))
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
        ready = {d: baseline_ready(baseline_root(d), d, TOTALS[d], shards(d, 'full'))
                 for d in set(d for d, _, _ in pending)}
        priority = yield_to_upstream(read_json(NATURAL/'suite_status.json'), read_json(PRECEDING/'suite_status.json'))
        for d, p, s in list(pending):
            if priority or not ready[d] or p == 'full' and (d, 'smoke', 0) not in done:
                continue
            gpu = next((g for g in GPUS if g not in active and idle(g)), None)
            if gpu is None:
                break
            active[gpu] = launch(root, d, p, s, gpu)
            pending.remove((d, p, s))
        if pending and not active and not any(ready.values()):
            for d in ready:
                session = 'gctxsense04_controller' if d.startswith('context') else 'gnsense04_controller'
                if not alive(session):
                    failures['dependency/'+d] = dict(error='Required baseline controller exited before this protocol verified complete.',
                        baseline_root=str(baseline_root(d)), controller=session)
            if failures:
                pending.clear()
        status = 'failed' if failures else 'running' if active else 'waiting_for_ready_baseline_or_gpu' if pending else 'complete'
        save(root/'suite_status.json', dict(status=status, active=list(active.values()), pending=pending,
            completed=completed, failures=failures, allowed_gpus=GPUS,
            baseline_ready=ready, upstream_dispatch_priority=priority,
            controller_log=str(controller_log), scheduling='per_protocol_verified_full_baseline'))
        if pending or active:
            time.sleep(15)
    save(root/'suite_results.json', dict(status='failed' if failures else 'complete', implementation=IMPLEMENTATION,
        completed=completed, failures=failures, allowed_gpus=GPUS, scheduling='per_protocol_verified_full_baseline'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--controller-log', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.controller_log)
