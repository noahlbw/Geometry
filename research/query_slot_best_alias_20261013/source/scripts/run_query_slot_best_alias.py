"""Run isolated costs, then full sharded evaluation on idle A800 GPUs."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from eval_query_slot_best_alias import read, write
from eval_geometry_vip_reliability import summary


def alive(session):
    return subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode == 0


def idle(gpu):
    row = subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    memory,utilization = map(int,row.split(','))
    return memory < 100 and utilization < 5


def jobs(protocol):
    tasks = {}
    for d in protocol['datasets']:
        tasks['cost/'+d] = dict(dataset=d,mode='cost',shard_index=0,num_shards=1,status='pending')
    for d,e in protocol['datasets'].items():
        shards = 8 if e['total_images'] > 10000 else 4 if e['total_images'] >= 5000 else 2 if e['total_images'] >= 1000 else 1
        for i in range(shards):
            tasks[f'full/{d}_s{i}'] = dict(dataset=d,mode='full',shard_index=i,num_shards=shards,status='pending')
    return tasks


def launch(root,key,job,gpu,protocol):
    suffix = '_s'+str(job['shard_index']) if job['mode']=='full' else ''
    session = 'qsbest13_'+job['mode']+'_'+job['dataset']+suffix
    log = root/job['mode']/(job['dataset']+suffix+'.log')
    output = log.with_suffix('.json')
    if log.exists() or output.exists() or alive(session):
        raise RuntimeError('Preserve existing worker: '+session)
    if not idle(gpu):
        return
    env = ['env','CUDA_VISIBLE_DEVICES='+str(gpu), 'PYTHONPATH='+protocol['pythonpath'],
        'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
    cmd = [protocol['python'],'-u',str(root/'source/scripts/eval_query_slot_best_alias.py'),
        '--suite-root',str(root),'--dataset',job['dataset'],'--mode',job['mode'],
        '--num-shards',str(job['num_shards']),'--shard-index',str(job['shard_index']),
        '--checkpoint-dir',protocol['checkpoint_dir'],'--dinov3-repo',protocol['dinov3_repo'],
        '--upstream-root',protocol['upstream_root']]
    log.parent.mkdir(parents=True,exist_ok=True)
    shell = 'cd '+shlex.quote(protocol['base_root'])+' && exec '+shlex.join(env+cmd)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    job.update(status='running',gpu=gpu,session=session,output=str(output),log=str(log),started=time.time())
    print(json.dumps(dict(event='launched',key=key,gpu=gpu)),flush=True)


def merge(root,d,entry,state,protocol):
    selected = [j for j in state['jobs'].values() if j['dataset']==d and j['mode']=='full']
    if not all(j['status']=='complete' for j in selected):
        return
    output = root/'merged'/(d+'.json')
    if output.exists():
        return
    rows = [read(j['output']) for j in selected]
    reference = read(Path(protocol['base_root'])/'results/training_free_baselines_full_20261011/references'/f'{d}.json')
    expected = reference['coverage']['sample_keys']
    keys = [None]*len(expected)
    for row in rows:
        keys[row['shard_index']::row['num_shards']] = row['sample_keys']
    if keys != expected or len(keys)!=entry['total_images'] or len(set(keys))!=len(keys):
        raise RuntimeError('Unique complete coverage differs: '+d)
    if any(r['config']!=rows[0]['config'] or r['cache_identity']!=rows[0]['cache_identity'] for r in rows):
        raise RuntimeError('Shard configuration differs: '+d)
    metrics = {}
    for p,m in rows[0]['metrics'].items():
        cm = sum(np.asarray(row['metrics'][p]['confusion_matrix'],np.int64) for row in rows)
        ref = np.asarray(next(iter(reference['results'][p].values()))['confusion_matrix'],np.int64)
        if not np.array_equal(cm.sum(1),ref.sum(1)):
            raise RuntimeError('Scored target counts differ: '+d+'/'+p)
        metrics[p] = summary(cm,tuple(c['name'] for c in m['per_class']),0)
    write(output,dict(status='complete',dataset=d,processed_images=len(keys),sample_keys=keys,
        coverage_verified=True,scored_target_counts_match=True,per_image_sums_verified=True,
        metrics=metrics,config=rows[0]['config'],cache_identity=rows[0]['cache_identity'],
        original_profile=rows[0]['original_profile'],alias_source=rows[0]['alias_source'],both_modules_active=True,
        aggregate_wall_seconds=sum(row['wall_seconds'] for row in rows)))
    print(json.dumps(dict(event='merged',dataset=d,metrics={p:m['mean_iou_percent'] for p,m in metrics.items()})),flush=True)


def main(args):
    root = Path(args.suite_root)
    lock = (root/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol = read(root/'protocol.json')
    path = root/'state.json'
    state = read(path) if path.exists() else dict(status='running',jobs=jobs(protocol))
    while True:
        for key,job in state['jobs'].items():
            if job['status']=='running' and not alive(job['session']):
                try:
                    row = read(job['output'])
                    if row['status']!='complete' or not row['both_modules_active']:
                        raise RuntimeError('Incomplete or inactive worker')
                    job.update(status='complete',finished=time.time())
                except Exception as error:
                    job.update(status='failed',error=str(error))
        for d,e in protocol['datasets'].items():
            merge(root,d,e,state,protocol)
        failed = [k for k,j in state['jobs'].items() if j['status']=='failed']
        if failed:
            state['status']='failed'
            write(path,state)
            raise RuntimeError('Inspect worker failures: '+str(failed))
        if all(j['status']=='complete' for j in state['jobs'].values()):
            state['status']='complete'
            write(path,state)
            write(root/'suite_results.json',dict(status='complete',datasets=list(protocol['datasets']),
                source_images=sum(e['total_images'] for e in protocol['datasets'].values()),
                both_modules_active=True,full_miou=True,standalone_costs=True))
            return
        running = [j for j in state['jobs'].values() if j['status']=='running']
        pending = [(k,j) for k,j in state['jobs'].items() if j['status']=='pending']
        costs = [pair for pair in pending if pair[1]['mode']=='cost']
        if costs:
            if not running:
                for gpu in range(8):
                    if idle(gpu):
                        launch(root,*costs[0],gpu,protocol)
                        break
        else:
            used = {j['gpu'] for j in running}
            for gpu in range(8):
                if pending and gpu not in used and idle(gpu):
                    key,job = pending.pop(0)
                    launch(root,key,job,gpu,protocol)
        write(path,state)
        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite-root',required=True)
    main(parser.parse_args())
