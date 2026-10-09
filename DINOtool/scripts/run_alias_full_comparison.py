"""Finite full suite; occupied GPUs and existing output are preserved."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from eval_alias_full_comparison_r3 import IMPLEMENTATION,PRIMARY_METHODS,HISTORICAL,BASELINE
from eval_rival_fine_full import save
from merge_gear_ov_shards import merge
from run_region_semantic_suite_a800 import BASE,TOOL,THIRD,PYTHON,idle
from run_sat_geometry_transport_suite import alive,read_json

PREFIX='gafc08r3'


def launch(root,d,entry,gpu,phase,shard=0):
    output=root/phase/d/('s'+str(shard));log=output.with_suffix('.log')
    session=f'{PREFIX}_{phase}_{d}_s{shard}'
    if output.exists() or log.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Occupied GPU or existing output: '+session)
    output.parent.mkdir(parents=True,exist_ok=True)
    args=[PYTHON,'-u','scripts/eval_alias_full_comparison_r3.py','--dataset',d,
        '--suite-root',str(root),'--data-root',entry['data_root'],'--vocabulary-config',entry['vocabulary_config'],
        '--dinov3-repo',str(TOOL/'dinov3_hub'),'--checkpoint-dir',str(BASE/'ckpt/DINO'),
        '--upstream-root',str(BASE/'third_party/VIP_official_5bd25ee'),'--output-dir',str(output),
        '--mode',phase,'--num-shards',str(entry['shards'] if phase=='full' else 1),
        '--shard-index',str(shard),'--repetitions','5']
    env=['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
        'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
    shell='cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    job=dict(dataset=d,gpu=gpu,phase=phase,shard=shard,session=session,output=str(output),log=str(log))
    if phase=='benchmark':
        state=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
        save(output.parent/'environment.json',dict(target_gpu=gpu,scope='one exclusive target GPU; other GPUs allowed',gpu_state=state))
    print(json.dumps(dict(event='launched',**job)),flush=True)
    return job


def verify_job(job):
    folder=Path(job['output']);row=read_json(folder/'results.json')
    if not row or row['status']!='complete' or not row['weights_frozen'] or not row['head_weights_unchanged']:
        raise RuntimeError('Exited before complete frozen results.')
    if job['phase']=='smoke':
        if row['target_masks_loaded'] or not row['singleton_exact'] or row['implementation']!=IMPLEMENTATION:
            raise RuntimeError('Smoke failed.')
    elif job['phase']=='benchmark':
        if row['target_masks_loaded'] or not row['benchmark_predictions_equal'] or row['implementation']!=IMPLEMENTATION:
            raise RuntimeError('Benchmark failed.')
        if len(row['images'])!=3 or any(len(t['seconds'])!=5 for im in row['images'] for t in im['timings'].values()):
            raise RuntimeError('Benchmark sampling changed.')
    else:
        if row['processed_images']!=row['total_images'] or not row['all_visual_budgets_verified']:
            raise RuntimeError('Full shard incomplete or budget failed.')
        if row['signature']['implementation']!=IMPLEMENTATION or not row['target_masks_used_only_after_prediction']:
            raise RuntimeError('Frozen inference protocol changed.')
        with np.load(folder/'per_image_confusions.npz',allow_pickle=False) as data:
            if data['sample_keys'].tolist()!=row['signature']['sample_keys']:
                raise RuntimeError('Per-image keys changed.')
            for p,group in row['metrics'].items():
                expected=None
                for name,value in group.items():
                    cm=data[p+'__'+name]
                    if not np.array_equal(cm.sum(0),value['confusion_matrix']):
                        raise RuntimeError('Per-image confusion aggregation failed.')
                    targets=cm.sum(-1)
                    if expected is not None and not np.array_equal(targets,expected):
                        raise RuntimeError('Paired scored target counts differ.')
                    expected=targets
    return row


def verify_dataset(root,d,entry):
    folders=[root/'full'/d/('s'+str(s)) for s in range(entry['shards'])]
    path=root/'full'/d/'merged.json'
    if path.exists():raise RuntimeError('Existing merged result; preserve it.')
    row=merge([str(f) for f in folders],str(path))
    if not row['coverage_verified'] or row['processed_images']!=entry['total_images']:
        raise RuntimeError('Incorrect full unique coverage.')
    old=read_json(Path(entry['matched_patch_result']))
    if row['signature']['sample_keys']!=old['signature']['sample_keys'] or row['signature']['checkpoints']!=old['signature']['checkpoints']:
        raise RuntimeError('Historical input/checkpoint identity changed.')
    for p,group in row['metrics'].items():
        old_p=p if entry['family']=='remote_sensing' else p+'__original20'
        if not np.array_equal(group[BASELINE]['confusion_matrix'],old['metrics'][old_p][BASELINE]['confusion_matrix']):
            raise RuntimeError('Matched PatchOnly2 full confusion differs: '+d+'/'+p)
    if entry.get('historical_result'):
        historical=read_json(Path(entry['historical_result']))
        if row['signature']['sample_keys']!=historical['signature']['sample_keys']:
            raise RuntimeError('Specified historical coverage differs.')
        if not np.array_equal(row['metrics'][d][HISTORICAL]['confusion_matrix'],
                historical['metrics'][d]['LocalBackgroundUnion']['confusion_matrix']):
            raise RuntimeError('Specified historical model confusion replay failed: '+d)
    # Replay developed panel predictions at their original ordered sample keys.
    if entry['family']=='remote_sensing':
        panel=TOOL/'results/shared_local_alias_20261008/full'/d/'s0'
        retained_panel=TOOL/'results/one_sided_stream_execution_20261008/full'/d/'s0'
        lookup={key:i for i,key in enumerate(row['signature']['sample_keys'])}
        full={key:np.empty((entry['total_images'],*np.asarray(value['confusion_matrix']).shape),np.int64)
              for p,group in row['metrics'].items() for name,value in group.items() for key in (p+'__'+name,)}
        for folder in folders:
            with np.load(folder/'per_image_confusions.npz',allow_pickle=False) as data:
                ids=[lookup[k] for k in data['sample_keys'].tolist()]
                for key in full:full[key][ids]=data[key]
        for folder in (panel,retained_panel):
            with np.load(folder/'per_image_confusions.npz',allow_pickle=False) as data:
                ids=[lookup[k] for k in data['sample_keys'].tolist()]
                for key in full:
                    if key in data and not np.array_equal(full[key][ids],data[key]):
                        raise RuntimeError('Preserved pilot confusion differs: '+d+'/'+key)
    row.update(weights_frozen=True,head_weights_unchanged=True,per_image_confusions_verified=True,
        paired_scored_targets_equal=True,matched_patch_full_confusion_exact=True,
        historical_full_confusion_exact=bool(entry.get('historical_result')),
        retained_pilot_replayed=entry['family']=='remote_sensing')
    save(path,row)
    return str(path)


def monitor_smokes(root,entries):
    reused=read_json(root/'protocol.json').get('reused_smokes',{})
    done={};active={};failures={}
    for d,path in reused.items():
        verify_job(dict(output=str(Path(path).parent),phase='smoke'))
        done[d]=path
    pending=[d for d in entries if d not in done]
    while pending or active:
        for gpu,job in list(active.items()):
            if alive(job['session']):continue
            try:verify_job(job);done[job['dataset']]=job['output']+'/results.json'
            except Exception as error:
                failures[job['session']]=dict(error=str(error),log=job['log'],
                    log_tail=Path(job['log']).read_text(errors='replace')[-7000:])
            del active[gpu]
        if failures:pending.clear()
        for d in list(pending):
            gpu=next((g for g in range(8) if g not in active and idle(g)),None)
            if gpu is None:break
            active[gpu]=launch(root,d,entries[d],gpu,'smoke');pending.remove(d)
        save(root/'suite_status.json',dict(status='running' if pending or active else 'failed' if failures else 'complete',
            phase='smoke',active=list(active.values()),pending=pending,completed=done,failures=failures))
        if pending or active:time.sleep(15)
    return done,failures


def main(root):
    root.resolve().relative_to((TOOL/'results').resolve())
    if (root/'suite_status.json').exists() or (root/'suite_results.json').exists():
        raise RuntimeError('Existing suite; no duplicate launch.')
    protocol=read_json(root/'protocol.json');entries=protocol['datasets'];started=time.time()
    done,failures=monitor_smokes(root,entries)
    save(root/'smoke_summary.json',dict(completed=done,failures=failures))
    if failures:
        save(root/'suite_results.json',dict(status='failed',phase='smoke',failures=failures));return
    order=list(entries)
    pending=[(d,s) for s in range(max(e['shards'] for e in entries.values())) for d in order if s<entries[d]['shards']]
    timing_pending=list(entries);active={};finished=set();completed={};timings={}
    while pending or timing_pending or active:
        for gpu,job in list(active.items()):
            if alive(job['session']):continue
            try:
                verify_job(job);d=job['dataset']
                if job['phase']=='benchmark':timings[d]=job['output']+'/results.json'
                else:
                    finished.add((d,job['shard']))
                    if all((d,s) in finished for s in range(entries[d]['shards'])):
                        completed[d]=verify_dataset(root,d,entries[d])
            except Exception as error:
                failures[job['session']]=dict(error=str(error),log=job['log'],
                    log_tail=Path(job['log']).read_text(errors='replace')[-7000:])
            del active[gpu]
        if failures:pending.clear();timing_pending.clear()
        if timing_pending and 7 not in active and idle(7):
            d=timing_pending.pop(0);active[7]=launch(root,d,entries[d],7,'benchmark')
        timing_active=any(j['phase']=='benchmark' for j in active.values())
        allowed=range(7) if timing_pending or timing_active else range(8)
        for d,s in list(pending):
            gpu=next((g for g in allowed if g not in active and idle(g)),None)
            if gpu is None:break
            active[gpu]=launch(root,d,entries[d],gpu,'full',s);pending.remove((d,s))
        save(root/'suite_status.json',dict(status='running' if pending or timing_pending or active else
            'failed' if failures else 'complete',phase='full_and_timing',active=list(active.values()),
            pending=pending,timing_pending=timing_pending,completed=completed,timings=timings,failures=failures))
        if pending or timing_pending or active:time.sleep(15)
    save(root/'suite_results.json',dict(status='failed' if failures else 'complete',completed=completed,
        timings=timings,failures=failures,blocked=protocol['blocked'],suite_wall_seconds=time.time()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
