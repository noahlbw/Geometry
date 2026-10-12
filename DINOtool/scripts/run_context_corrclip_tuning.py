"""Durable PC59/60 queue: freeze development choices, merge full runs, cost alone."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np
from eval_context_corrclip_tuning import read, save, TOOL, BASE
from eval_geometry_vip_reliability import summary
from run_efficient_geometry import alive, idle


def launch(root,job,gpu):
    d,mode=job['dataset'],job['mode'];suffix=f'_s{job["shard_index"]}' if mode=='full' else ''
    output=root/mode/d
    if mode=='full':output=output/f's{job["shard_index"]}'
    log=root/'logs'/f'{mode}_{d}{suffix}.log';session=f'gpcc12_{mode}_{d}{suffix}'
    if output.exists() or log.exists() or alive(session) or not idle(gpu):raise RuntimeError('Preserve output/worker: '+session)
    third=BASE/'third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c'
    command=['env','CUDA_VISIBLE_DEVICES='+str(gpu),'PYTHONPATH='+str(TOOL)+':'+str(TOOL/'scripts')+':'+str(third),
        'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false',
        '/data/miniconda3/envs/pfu/bin/python','-u','scripts/eval_context_corrclip_tuning.py',
        '--suite-root',str(root),'--dataset',d,'--mode',mode,'--num-shards',str(job['num_shards']),
        '--shard-index',str(job['shard_index'])]
    shell='cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(command)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    job.update(status='running',gpu=gpu,session=session,log=str(log),output=str(output),started=time.time())
    print(json.dumps(dict(event='launched',dataset=d,mode=mode,gpu=gpu)),flush=True)


def merge(root,d,entry,state):
    jobs=[j for j in state['jobs'].values() if j['dataset']==d and j['mode']=='full']
    if len(jobs)!=4 or not all(j['status']=='complete' for j in jobs):return
    output=root/'merged'/f'{d}.json'
    if output.exists():return
    rows=[read(Path(j['output'])/'results.json') for j in jobs]
    expected=rows[0]['global_sample_keys'];keys=[None]*len(expected);dev=set(rows[0]['choice']['development_keys'])
    arrays=[];split={k:np.zeros_like(rows[0]['metrics']['confusion_matrix'],dtype=np.int64) for k in ('development','complement')}
    for row,job in zip(rows,jobs):
        keys[row['shard_index']::row['num_shards']]=row['sample_keys']
        if any(row[f]!=rows[0][f] for f in ('choice','config','cache_identity','global_sample_keys')):
            raise RuntimeError('Shard identities differ '+d)
        with np.load(Path(job['output'])/'per_image_confusions.npz',allow_pickle=False) as packed:
            cm=packed['confusions'];raw_keys=packed['keys'].tolist()
            if raw_keys!=row['sample_keys'] or not np.array_equal(cm.sum(0),row['metrics']['confusion_matrix']):
                raise RuntimeError('Raw per-image confusion sum differs '+d)
            mask=np.asarray([k in dev for k in raw_keys])
            split['development']+=cm[mask].sum(0);split['complement']+=cm[~mask].sum(0);arrays.append(cm.sum(0))
    if len(keys)!=5105 or len(set(keys))!=5105 or keys!=expected:raise RuntimeError('Complete unique coverage differs '+d)
    matrix=sum(arrays);incumbent=np.asarray(entry['incumbent_metrics']['confusion_matrix'],np.int64)
    if not np.array_equal(matrix.sum(1),incumbent.sum(1)):raise RuntimeError('Scored target counts differ '+d)
    if not np.array_equal(split['development'],rows[0]['choice']['development_confusion']):
        raise RuntimeError('Frozen development choice did not replay '+d)
    names=[v['name'] for v in rows[0]['metrics']['per_class']]
    save(output,dict(status='complete',dataset=d,processed_images=len(keys),sample_keys=keys,metrics=summary(matrix,names,0),
        splits={k:summary(cm,names,0) for k,cm in split.items()},coverage_verified=True,per_image_sums_verified=True,
        development_replay_verified=True,scored_target_counts_match=True,choice=rows[0]['choice'],config=rows[0]['config'],
        cache_identity=rows[0]['cache_identity'],aggregate_gpu_seconds=sum(r['wall_seconds'] for r in rows)))
    print(json.dumps(dict(event='merged',dataset=d,miou=summary(matrix,names,0)['mean_iou_percent'])),flush=True)


def main(root):
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=read(root/'protocol.json');path=root/'state.json'
    for folder in ('logs','configs','vocabularies','merged'):(root/folder).mkdir(exist_ok=True)
    if path.exists():raise RuntimeError('Existing controller state preserved.')
    state=dict(status='running',jobs={f'search/{d}':dict(dataset=d,mode='search',num_shards=1,shard_index=0,status='pending') for d in protocol['order']})
    while True:
        for key,job in list(state['jobs'].items()):
            if job['status']=='running' and not alive(job['session']):
                try:
                    result=read(Path(job['output'])/'results.json')
                    if result['status']!='complete' or not result['weights_frozen'] or not result['head_weights_unchanged']:
                        raise RuntimeError('Incomplete worker or changed weights.')
                    job.update(status='complete',finished=time.time())
                    if job['mode']=='search':
                        for s in range(4):
                            state['jobs'][f'full/{job["dataset"]}/s{s}']=dict(dataset=job['dataset'],mode='full',num_shards=4,shard_index=s,status='pending')
                        state['jobs']['cost/'+job['dataset']]=dict(dataset=job['dataset'],mode='cost',num_shards=1,shard_index=0,status='pending')
                    if job['mode']=='cost' and (result['target_masks_loaded'] or not result['standalone_process']):
                        raise RuntimeError('Invalid standalone cost.')
                except Exception as error:job.update(status='failed',error=str(error))
        for d,e in protocol['datasets'].items():merge(root,d,e,state)
        failed=[k for k,j in state['jobs'].items() if j['status']=='failed']
        if failed:
            state['status']='failed';save(path,state);raise RuntimeError('Inspect exact logs: '+str(failed))
        if all(j['status']=='complete' for j in state['jobs'].values()):
            if not all((root/'merged'/f'{d}.json').exists() for d in protocol['order']):raise RuntimeError('Missing merged protocols.')
            state['status']='complete';save(path,state)
            save(root/'suite_results.json',dict(status='complete',protocols=2,protocol_images=10210,standalone_costs=2));return
        running=[j for j in state['jobs'].values() if j['status']=='running']
        pending=[j for j in state['jobs'].values() if j['status']=='pending']
        work=[j for j in pending if j['mode']!='cost']
        if work:
            used={j['gpu'] for j in running}
            for gpu in protocol['gpu_order']:
                if work and gpu not in used and idle(gpu):launch(root,work.pop(0),gpu)
        elif not running and pending:
            for gpu in protocol['gpu_order']:
                if idle(gpu):launch(root,pending[0],gpu);break
        save(path,state);time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite-root',required=True,type=Path);main(p.parse_args().suite_root)
