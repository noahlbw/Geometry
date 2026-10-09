"""Resume only the preserved r3 dispatch queue; never relaunch existing outputs."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from eval_rival_fine_full import save
from run_alias_full_comparison_r3 import (PREFIX, launch, verify_job, verify_dataset,
    alive, idle, read_json)


def existing_job(root,d,entry,phase,shard):
    session = f'{PREFIX}_{phase}_{d}_s{shard}'
    output = root/phase/d/('s'+str(shard))
    if not alive(session):
        return None
    pid = int(subprocess.check_output(['tmux','display-message','-p','-t',session,'#{pane_pid}'],text=True))
    values = (Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
    gpu = next(int(value.split(b'=',1)[1]) for value in values if value.startswith(b'CUDA_VISIBLE_DEVICES='))
    return dict(dataset=d,gpu=gpu,phase=phase,shard=shard,session=session,
                output=str(output),log=str(output.with_suffix('.log')))


def main(root):
    state = read_json(root/'suite_status.json');protocol = read_json(root/'protocol.json')
    if not state or read_json(root/'suite_results.json'):
        raise RuntimeError('Missing continuation or already completed suite.')
    entries = protocol['datasets'];active = {};finished = set();completed = {};timings = {}
    pending = [];timing_pending = [];failures = dict(state.get('failures',{}))
    for d,entry in entries.items():
        for phase,shards in (('benchmark',1),('full',entry['shards'])):
            for shard in range(shards):
                job = existing_job(root,d,entry,phase,shard)
                folder = root/phase/d/('s'+str(shard))
                if job:
                    if job['gpu'] in active:
                        raise RuntimeError('Existing workers share a GPU; inspect before resuming.')
                    active[job['gpu']] = job
                elif folder.exists():
                    verify_job(dict(output=str(folder),phase=phase))
                    if phase == 'benchmark':
                        timings[d] = str(folder/'results.json')
                    else:
                        finished.add((d,shard))
                elif phase == 'benchmark':
                    timing_pending.append(d)
                else:
                    pending.append((d,shard))
        merged = root/'full'/d/'merged.json'
        if merged.exists():
            row = read_json(merged)
            if not row.get('coverage_verified') or not row.get('matched_patch_full_confusion_exact') or not row.get('paired_scored_targets_equal'):
                raise RuntimeError('Existing merged output unverified; preserve and inspect: '+d)
            completed[d] = str(merged)
        elif all((d,s) in finished for s in range(entry['shards'])):
            completed[d] = verify_dataset(root,d,entry)
    while pending or timing_pending or active:
        for gpu,job in list(active.items()):
            if alive(job['session']):
                continue
            try:
                verify_job(job);d = job['dataset']
                if job['phase'] == 'benchmark':
                    timings[d] = job['output']+'/results.json'
                else:
                    finished.add((d,job['shard']))
                    if d not in completed and all((d,s) in finished for s in range(entries[d]['shards'])):
                        completed[d] = verify_dataset(root,d,entries[d])
            except Exception as error:
                failures[job['session']] = dict(error=str(error),log=job['log'],
                    log_tail=Path(job['log']).read_text(errors='replace')[-7000:])
            del active[gpu]
        if failures:
            pending.clear();timing_pending.clear()
        if timing_pending and 7 not in active and idle(7):
            d = timing_pending.pop(0);active[7] = launch(root,d,entries[d],7,'benchmark')
        allowed = range(7) if timing_pending or any(j['phase']=='benchmark' for j in active.values()) else range(8)
        for d,s in list(pending):
            gpu = next((g for g in allowed if g not in active and idle(g)),None)
            if gpu is None:
                break
            active[gpu] = launch(root,d,entries[d],gpu,'full',s);pending.remove((d,s))
        save(root/'suite_status.json',dict(status='running',phase='full_and_timing',active=list(active.values()),
            pending=pending,timing_pending=timing_pending,completed=completed,timings=timings,failures=failures,
            resumed_preserving_workers=True))
        if pending or timing_pending or active:
            time.sleep(15)
    save(root/'suite_results.json',dict(status='failed' if failures else 'complete',completed=completed,
        timings=timings,failures=failures,blocked=protocol['blocked'],resumed_preserving_workers=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
