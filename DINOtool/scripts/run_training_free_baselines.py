"""Durable, method-by-method official baseline queue, isolated outputs and GPUs."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np
from eval_training_free_baseline import digest,save,summary


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def alive(session):
    return subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0


def idle(gpu):
    line=subprocess.check_output(['nvidia-smi','--id='+str(gpu),
        '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    memory,usage=[int(v.strip()) for v in line.split(',')]
    return memory<100 and usage<5


def launch(root,method,dataset,phase,shard,shards,gpu,attempt=0):
    suffix=('_attempt'+str(attempt)) if attempt else ''
    output=root/method/dataset/phase/('s'+str(shard)+suffix);log=output.with_suffix('.log')
    safe_method=method.replace('-','').lower()
    session=f'gtf11_{safe_method}_{phase}_{dataset}_{shard}'+suffix
    if output.exists() or log.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Preserve existing worker/output or occupied GPU: '+session)
    output.parent.mkdir(parents=True,exist_ok=True)
    tool=Path(__file__).resolve().parents[1]
    command=['env','CUDA_VISIBLE_DEVICES='+str(gpu),'PYTHONPATH='+str(tool)+':'+str(tool/'scripts'),
        'OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','TOKENIZERS_PARALLELISM=false',
        str(tool/'.baseline_env_20261011/bin/python'),'-u',str(tool/'scripts/eval_training_free_baseline.py'),
        '--root',str(root),'--output',str(output),'--method',method,'--dataset',dataset,
        '--phase',phase,'--shards',str(shards),'--shard-index',str(shard)]
    shell='cd '+shlex.quote(str(tool))+' && exec '+shlex.join(command)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    return dict(method=method,dataset=dataset,phase=phase,shard=shard,shards=shards,gpu=gpu,
        session=session,output=str(output),log=str(log),attempt=attempt)


def verify(job):
    row=read(Path(job['output'])/'results.json')
    if not row or row['status']!='complete' or row['processed_images']!=row['total_images']:
        raise RuntimeError('Exited before complete verified coverage: '+job['log'])
    if job['phase']=='evaluate':
        with np.load(Path(job['output'])/'per_image_confusions.npz',allow_pickle=False) as data:
            for p,v in row['results'].items():
                if len(data[p])!=len(row['sample_keys']) or not np.array_equal(data[p].sum(0),v['confusion_matrix']):
                    raise RuntimeError('Per-image confusion does not sum: '+job['log'])
    if job['phase']=='cost' and not read(Path(job['output'])/'cost.json'):
        raise RuntimeError('Standalone cost missing: '+job['log'])
    return row


def merge(root,method,dataset,shards,completed_outputs=None):
    completed_outputs=completed_outputs or {}
    paths=[Path(completed_outputs.get(f'{method}/evaluate/{dataset}/{i}',
        str(root/method/dataset/'evaluate'/('s'+str(i))))) for i in range(shards)]
    rows=[read(path/'results.json') for path in paths]
    if any(not row or row['status']!='complete' for row in rows):return None
    manifest=read(root/'samples.json')[dataset];expected=[r['key'] for r in manifest['samples']]
    keys=[None]*len(expected)
    for i,row in enumerate(rows):
        keys[i::shards]=row['sample_keys']
        if row['signature']!=rows[0]['signature']:raise RuntimeError('Shard signatures differ')
    if keys!=expected or len(set(keys))!=len(keys) or digest(keys)!=manifest['global_sample_keys_sha256']:
        raise RuntimeError('Complete unique sample coverage mismatch')
    reference=read(root/'references'/(dataset+'.json'))
    if keys!=reference['coverage']['sample_keys']:raise RuntimeError('Coverage differs from verified VIP')
    config=read(root/'protocol.json')['datasets'][dataset]
    metrics={}
    for p,names in manifest['classes'].items():
        matrix=sum(np.asarray(r['results'][p]['confusion_matrix'],np.int64) for r in rows)
        ref=next(iter(reference['results'][p].values()))
        if not np.array_equal(matrix.sum(1),np.asarray(ref['confusion_matrix'],np.int64).sum(1)):
            raise RuntimeError('Scored target pixel counts differ from VIP: '+dataset+'/'+p)
        metrics[p]=summary(matrix,names,None if p=='P' else config['background_index'])
    result=dict(status='complete',coverage_verified=True,processed_images=len(keys),total_images=len(keys),
        signature=rows[0]['signature'],sample_keys=keys,sample_keys_sha256=digest(keys),results=metrics,
        scored_target_counts_match=True,aggregate_gpu_wall_seconds=sum(r['wall_seconds'] for r in rows),
        parallel_wall_seconds=max(r['wall_seconds'] for r in rows),
        peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
        peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows),
        execution_variants=[dict(shard=i,output=str(path),repair=row.get('execution_repairs'))
            for i,(path,row) in enumerate(zip(paths,rows))])
    path=root/method/dataset/'merged.json'
    if path.exists():
        previous=read(path)
        # Older verified files predate explicit null execution-provenance metadata.
        if previous!={k:v for k,v in result.items() if k in previous}:
            raise RuntimeError('Existing merged result differs; preserve it')
    else:save(path,result)
    return {p:m['mean_iou_percent'] for p,m in metrics.items()}


def main(args):
    root=Path(args.root)
    lock=(root/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=root/'suite_status.json'
    if state_path.exists():
        state=read(state_path)
        if not args.resume:raise RuntimeError('Use explicit resume; preserve active workers')
    else:
        state=dict(status='waiting_environment',active=[],completed=[],failures={},merged={},pending=[])
    protocol=read(root/'protocol.json')
    state.setdefault('completed_outputs',{})
    state.setdefault('attempts',{})
    if args.retry_failed_method:
        repairs=read(root/'source_repairs.json') or {}
        if repairs.get(args.retry_failed_method,{}).get('status')!='validated':
            raise RuntimeError('Retry requires a validated execution-only source repair')
        history=state.setdefault('failure_history',[])
        for failed in list(state['failures']):
            if failed.startswith(args.retry_failed_method+'/') and len(failed.split('/'))==4:
                history.append(dict(key=failed,failure=state['failures'].pop(failed),
                    recovered_with=repairs[args.retry_failed_method]['id']))
                state['attempts'][failed]=state['attempts'].get(failed,0)+1
        save(state_path,state)
    # Environment/weights preparation is independent of visual workers.
    while not (root/'environment_ready').exists() or not (root/'weights_ready').exists():
        if not alive('gtf11_setup') and not (root/'environment_ready').exists():
            state['status']='environment_failed';save(state_path,state);raise RuntimeError('Inspect setup.log')
        if not alive('gtf11_weights') and not (root/'weights_ready').exists():
            state['status']='weights_failed';save(state_path,state);raise RuntimeError('Inspect weights.log')
        save(state_path,state);time.sleep(20)
    for method in protocol['method_order']:
        # Third-party optional deps do not block the other four methods.
        if method in ('CorrCLIP','SegEarth-OV') and not (root/'extra_dependencies_ready').exists():
            while alive('gtf11_extra_setup'):time.sleep(20)
            if not (root/'extra_dependencies_ready').exists():
                state['failures'][method+'/dependencies']='Inspect extra_setup.log; other methods continue.'
                continue
        jobs=[]
        for phase in ('smoke','evaluate','cost'):
            for d in protocol['dataset_order']:
                shards=8 if phase=='evaluate' and protocol['datasets'][d]['total_images']>=1000 else 1
                jobs.extend(dict(method=method,dataset=d,phase=phase,shard=s,shards=shards) for s in range(shards))
        completed=set(state['completed']);pending=[j for j in jobs if key(j) not in completed and key(j) not in state['failures']
            and not any(key(a)==key(j) for a in state['active'])]
        while pending or state['active']:
            for job in state['active'][:]:
                if alive(job['session']):continue
                state['active'].remove(job)
                try:
                    verify(job);state['completed'].append(key(job))
                    state['completed_outputs'][key(job)]=job['output']
                except Exception as error:
                    tail=subprocess.run(['tail','-n','30',job['log']],capture_output=True,text=True).stdout
                    state['failures'][key(job)]=dict(error=str(error),log=job['log'],tail=tail)
                save(state_path,state)
            complete=set(state['completed'])
            for d in protocol['dataset_order']:
                n=8 if protocol['datasets'][d]['total_images']>=1000 else 1
                if all(f'{method}/evaluate/{d}/{s}' in complete for s in range(n)):
                    k=method+'/'+d
                    if k not in state['merged']:
                        try:state['merged'][k]=merge(root,method,d,n,state['completed_outputs'])
                        except Exception as error:state['failures'][k+'/merge']=dict(error=str(error))
            eligible=[]
            for j in pending[:]:
                smoke=f"{method}/smoke/{j['dataset']}/0"
                if j['phase']!='smoke' and smoke in state['failures']:
                    state['failures'][key(j)]=dict(error='Skipped because image-only smoke failed')
                    pending.remove(j);continue
                if j['phase']=='smoke' or smoke in complete:eligible.append(j)
            busy={j['gpu'] for j in state['active']}
            for gpu in range(8):
                if not eligible:break
                if gpu in busy or not idle(gpu):continue
                job=eligible.pop(0);pending.remove(job)
                state['active'].append(launch(root,gpu=gpu,attempt=state['attempts'].get(key(job),0),**job))
            state.update(status='running',current_method=method,pending=pending)
            save(state_path,state);time.sleep(20)
    expected=len(protocol['methods'])*len(protocol['datasets'])
    state['status']='complete' if len(state['merged'])==expected and not state['failures'] else 'finished_with_failures'
    save(state_path,state)
    save(root/'suite_results.json',dict(status=state['status'],datasets=state['merged'],failures=state['failures'],
        protocol_images_per_method=45200,total_protocol_images=45200*len(protocol['methods'])))


def key(job):
    return f"{job['method']}/{job['phase']}/{job['dataset']}/{job['shard']}"


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--retry-failed-method')
    main(parser.parse_args())
