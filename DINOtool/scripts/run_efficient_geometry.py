"""Durable queue: sampled execution checks, paired full accuracy, isolated costs."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from eval_efficient_geometry import read, save, TOOL, BASE
from eval_geometry_vip_reliability import summary


def write_state(path, item):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(item,indent=2)+'\n');tmp.replace(path)


def alive(session):return subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0


def idle(gpu):
    raw=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    memory,usage=map(int,raw.split(','));return memory<100 and usage<5


def jobs(protocol):
    result={}
    for d,e in protocol['datasets'].items():
        result['cost/'+d]=dict(dataset=d,mode='cost',shard_index=0,num_shards=1,status='pending')
    for d,e in protocol['datasets'].items():
        count=(8 if e['total_images']>10000 else 4 if e['total_images']>=5000 else 2 if e['total_images']>=1000 else 1)
        for shard in range(count):
            result[f'full/{d}_s{shard}']=dict(dataset=d,mode='full',shard_index=shard,num_shards=count,status='pending')
    return result


def launch(root, key, job, gpu):
    mode,d=job['mode'],job['dataset'];suffix=f'_s{job["shard_index"]}' if mode=='full' else ''
    output=root/mode/(d+suffix+'.json');log=root/mode/(d+suffix+'.log');session='geff12_'+mode+'_'+d+suffix
    if output.exists() or log.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Preserve active/output state: '+session)
    log.parent.mkdir(parents=True,exist_ok=True)
    third=BASE/'third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c'
    command=['env','CUDA_VISIBLE_DEVICES='+str(gpu),'PYTHONPATH='+str(TOOL)+':'+str(TOOL/'scripts')+':'+str(third),
        'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false',
        '/data/miniconda3/envs/pfu/bin/python','-u',str(TOOL/'scripts/eval_efficient_geometry.py'),
        '--suite-root',str(root),'--dataset',d,'--mode',mode,'--num-shards',str(job['num_shards']),'--shard-index',str(job['shard_index'])]
    shell='cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(command)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    job.update(status='running',gpu=gpu,session=session,output=str(output),log=str(log),started=time.time())
    print(json.dumps(dict(event='launched',key=key,gpu=gpu)),flush=True)


def verify(job):
    path=Path(job['output'])
    if not path.exists():raise RuntimeError('Worker exited without complete output: '+job['log'])
    row=read(path)
    if row['status']!='complete':raise RuntimeError('Incomplete output '+str(path))
    if job['mode']=='full':
        packed=Path(row['per_image_confusion'])
        if hashlib.sha256(packed.read_bytes()).hexdigest()!=row['per_image_sha256']:raise RuntimeError('Raw confusion hash differs.')
        with np.load(packed,allow_pickle=False) as data:
            if list(data['keys'])!=row['sample_keys']:raise RuntimeError('Confusion keys differ.')
            for name, values in row['metrics'].items():
                for p, metric in values.items():
                    if not np.array_equal(data[name+'__'+p].sum(0),metric['confusion_matrix']):raise RuntimeError('Confusion sum differs.')
    return row


def merge(root,d,entry,state):
    selected=[j for j in state['jobs'].values() if j['dataset']==d and j['mode']=='full']
    if not all(j['status']=='complete' for j in selected):return
    output=root/'merged'/(d+'.json')
    if output.exists():return
    rows=[read(j['output']) for j in selected]
    reference=read(TOOL/'results/training_free_baselines_full_20261011/references'/f'{d}.json')
    expected=reference['coverage']['sample_keys'];keys=[None]*len(expected)
    for row in rows:keys[row['shard_index']::row['num_shards']]=row['sample_keys']
    if keys!=expected or len(keys)!=entry['total_images'] or len(set(keys))!=len(keys):raise RuntimeError('Unique complete coverage differs '+d)
    if any(r['config']!=rows[0]['config'] or r['cache_identity']!=rows[0]['cache_identity'] for r in rows):raise RuntimeError('Shard configuration differs.')
    metrics={};historical_match={}
    for name in rows[0]['metrics']:
        metrics[name]={}
        for p,m in rows[0]['metrics'][name].items():
            cm=sum(np.asarray(r['metrics'][name][p]['confusion_matrix'],np.int64) for r in rows)
            ref=np.asarray(next(iter(reference['results'][p].values()))['confusion_matrix'],np.int64)
            if not np.array_equal(cm.sum(1),ref.sum(1)):raise RuntimeError('Scored target count differs '+d+'/'+p)
            metrics[name][p]=summary(cm,tuple(v['name'] for v in m['per_class']),0)
            historical_match[name+'__'+p]=bool(np.array_equal(cm,entry['incumbent_metrics'][p]['confusion_matrix']))
    save(output,dict(status='complete',dataset=d,processed_images=len(keys),sample_keys=keys,
        coverage_verified=True,scored_target_counts_match=True,per_image_sums_verified=True,
        metrics=metrics,exact_historical_confusion=historical_match,
        changed_pixels={p:sum(r['changed_pixels'][p] for r in rows) for p in rows[0]['changed_pixels']},
        changed_keys=[v for r in rows for v in r['changed_keys']],config=rows[0]['config'],cache_identity=rows[0]['cache_identity'],
        aggregate_wall_seconds=sum(r['wall_seconds'] for r in rows)))


def main(args):
    root=Path(args.suite_root);lock=(root/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=read(root/'protocol.json');path=root/'state.json'
    state=read(path) if path.exists() else dict(status='running',jobs=jobs(protocol))
    while True:
        for key,job in state['jobs'].items():
            if job['status']=='running' and not alive(job['session']):
                try:verify(job);job.update(status='complete',finished=time.time())
                except Exception as error:job.update(status='failed',error=str(error))
        for d,e in protocol['datasets'].items():merge(root,d,e,state)
        failed=[k for k,j in state['jobs'].items() if j['status']=='failed']
        if failed:
            state['status']='failed';write_state(path,state)
            raise RuntimeError('Inspect exact logs; preserve running workers: '+str(failed))
        if all(j['status']=='complete' for j in state['jobs'].values()):
            state['status']='complete';write_state(path,state)
            save(root/'suite_results.json',dict(status='complete',protocols=len(protocol['datasets']),
                protocol_images=sum(e['total_images'] for e in protocol['datasets'].values()),
                datasets=list(protocol['datasets']),paired_full_evaluations=True,standalone_costs=15))
            return
        running=[j for j in state['jobs'].values() if j['status']=='running']
        pending=[(k,j) for k,j in state['jobs'].items() if j['status']=='pending']
        # Measure costs alone on this suite; then use all idle GPUs for paired full runs.
        costs=[v for v in pending if v[1]['mode']=='cost']
        if costs:
            if not running:
                for gpu in (4,5,6,7,0,2,3):
                    if idle(gpu):launch(root,*costs[0],gpu);break
        else:
            used={j['gpu'] for j in running}
            for gpu in (4,5,6,7,0,2,3):
                if pending and gpu not in used and idle(gpu):
                    key,job=pending.pop(0);launch(root,key,job,gpu)
        write_state(path,state);time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite-root',required=True);main(p.parse_args())
