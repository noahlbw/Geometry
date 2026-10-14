"""Schedule the frozen fixed-gain scan on idle A800 GPUs, preserving outputs."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def alive(session):
    return subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode == 0


def idle(gpu):
    row=subprocess.check_output(['nvidia-smi','--id='+str(gpu),
        '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    memory,utilization=map(int,row.split(','))
    return memory<100 and utilization<5


def jobs(protocol):
    counts={d:(8 if r['total_images']>=5000 else 4 if r['total_images']>=2000 else
               2 if r['total_images']>=1000 else 1) for d,r in protocol['datasets'].items()}
    # Interleave shards so every large dataset starts before one dataset
    # monopolizes all available workers.
    records={}
    for shard in range(max(counts.values())):
        for ds in protocol['datasets']:
            if shard<counts[ds]:
                records[f'{ds}_s{shard}']=dict(dataset=ds,shard_index=shard,
                    num_shards=counts[ds],status='pending')
    return records


def main(args):
    root=Path(args.experiment)
    lock=(root/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=read(root/'protocol.json')
    path=root/'state.json'
    state=read(path) if path.exists() else dict(status='running',jobs=jobs(protocol))
    source=Path(protocol['source'])
    pythonpath=':'.join(map(str,[source,source/'scripts',source.parents[2],source.parents[2]/'scripts',protocol['upstream_root']]))
    while True:
        for key,job in state['jobs'].items():
            if job['status']=='running' and not alive(job['session']):
                output=root/'full'/key/'results.json'
                try:
                    row=read(output)
                    if row['status']!='complete' or not row['both_modules_active']:
                        raise ValueError('Incomplete worker.')
                    job.update(status='complete',finished=time.time())
                except Exception as error:
                    job.update(status='failed',error=str(error))
        if any(j['status']=='failed' for j in state['jobs'].values()):
            state['status']='failed'
            write(path,state)
            raise RuntimeError('Inspect failed worker logs; completed shards are preserved.')
        if all(j['status']=='complete' for j in state['jobs'].values()):
            state['status']='complete'
            write(path,state)
            print('All fixed-step workers completed.',flush=True)
            return
        pending=[(k,j) for k,j in state['jobs'].items() if j['status']=='pending']
        used={j['gpu'] for j in state['jobs'].values() if j['status']=='running'}
        for gpu in args.gpus:
            if not pending or gpu in used or not idle(gpu):
                continue
            key,job=pending.pop(0)
            session='g0sweep14_'+key
            output=root/'full'/key
            log=root/'full'/(key+'.log')
            if output.exists() or log.exists() or alive(session):
                raise RuntimeError('Preserve existing worker: '+key)
            cmd=['env','CUDA_VISIBLE_DEVICES='+str(gpu),'PYTHONPATH='+pythonpath,
                'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2',
                protocol['python'],'-u',str(source/'scripts/eval_geov2_g0_sweep.py'),
                '--experiment',str(root),'--dataset',job['dataset'],'--mode','full',
                '--num-shards',str(job['num_shards']),'--shard-index',str(job['shard_index'])]
            shell='cd '+shlex.quote(str(source.parents[2]))+' && exec '+shlex.join(cmd)+' > '+shlex.quote(str(log))+' 2>&1'
            subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
            job.update(status='running',gpu=gpu,session=session,log=str(log),started=time.time())
            print(json.dumps(dict(event='launched',key=key,gpu=gpu)),flush=True)
        write(path,state)
        time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',required=True)
    p.add_argument('--gpus',type=int,nargs='+',default=list(range(8)))
    main(p.parse_args())
