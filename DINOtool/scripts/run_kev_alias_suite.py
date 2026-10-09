"""Idle-GPU queue: offline Kev admission, finite development, frozen full, timing."""
import argparse
from itertools import zip_longest
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np
from eval_rival_fine_full import save
from eval_geometry_vip_reliability import summary
from eval_gear_ov import digest
from dinotool.development_readout import score
from run_region_semantic_suite_a800 import BASE,TOOL,THIRD,PYTHON,idle
from run_sat_geometry_transport_suite import alive,read_json

PREFIX='gkev09'
KEV=BASE/'third_party/kev_20261009'


def launch(root,d,entry,gpu,phase,shard=0):
    output=root/('language' if phase=='language' else phase)/d
    if phase=='full':output=output/('s'+str(shard))
    log=output.with_suffix('.log');session=f'{PREFIX}_{phase}_{d}_s{shard}'
    if phase=='language' and output.exists() and not (output/'selection.json').exists() and not alive(session):
        # Resume only completed per-class language judgments; preserve the first log.
        log=output.with_suffix('.context_repair.log');session+='_context_repair'
    if (output.exists() and phase!='language') or (output/'selection.json').exists() or log.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Existing worker output/session or occupied GPU: '+session)
    output.parent.mkdir(parents=True,exist_ok=True)
    env=['env','CUDA_VISIBLE_DEVICES='+str(gpu),'PYTHONPATH='+str(TOOL)+':'+str(TOOL/'scripts')+':'+str(THIRD),
         'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
    if phase=='language':
        env[2]='PYTHONPATH='+str(KEV)+':'+str(TOOL)+':'+str(TOOL/'scripts')+':'+str(THIRD)
        env+=['HF_HOME='+str(root/'hf_cache'),'HF_HUB_OFFLINE=1','HF_HUB_DISABLE_XET=1']
        args=[PYTHON,'-u','scripts/select_kev_aliases.py','--root',str(root),'--dataset',d,'--kev-root',str(KEV)]
    else:
        args=[PYTHON,'-u','scripts/eval_kev_alias_search.py' if phase!='cost' else 'scripts/benchmark_kev_alias.py',
            '--dataset',d,'--data-root',entry['data_root'],'--suite-root',str(root),
            '--original-cache',str(TOOL/'results/curated20_patchonly2_full_20261006/text_cache'/(d+'.pt')),
            '--dinov3-repo',str(TOOL/'dinov3_hub'),'--checkpoint-dir',str(BASE/'ckpt/DINO'),
            '--upstream-root',str(BASE/'third_party/VIP_official_5bd25ee'),
            '--mode','search' if phase=='search' else 'full','--num-shards',str(entry['shards'] if phase=='full' else 1),
            '--shard-index',str(shard)]
        args+=['--output-dir',str(output)] if phase!='cost' else ['--output',str(output/'results.json')]
    shell='cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    return dict(dataset=d,phase=phase,shard=shard,gpu=gpu,session=session,log=str(log),output=str(output))


def verify(job):
    row=read_json(Path(job['output'])/'results.json')
    if not row or row['status']!='complete':raise RuntimeError('Worker exited without completion: '+job['log'])
    if job['phase'] in ('search','full'):
        if row['processed_images']!=row['total_images'] or not row['weights_frozen'] or not row['head_weights_unchanged']:
            raise RuntimeError('Incomplete/fitted worker: '+job['log'])
    if job['phase'] in ('search','language') and not read_json(Path(job['output'])/'selection.json'):
        raise RuntimeError('Missing frozen vocabulary/configuration: '+job['log'])
    if job['phase']=='full':
        with np.load(Path(job['output'])/'per_image_confusions.npz',allow_pickle=False) as data:
            if data['sample_keys'].tolist()!=row['signature']['sample_keys']:raise RuntimeError('Image keys differ.')
            for p,group in row['metrics'].items():
                for m,v in group.items():
                    if not np.array_equal(data[p+'__'+m].sum(0),v['confusion_matrix']):raise RuntimeError('Confusion sum differs.')
    return row


def merge(root,d,entry):
    folders=[root/'full'/d/('s'+str(i)) for i in range(entry['shards'])]
    rows=[read_json(f/'results.json') for f in folders];first=rows[0]['signature']
    keys=[k for group in zip_longest(*(r['signature']['sample_keys'] for r in rows)) for k in group if k is not None]
    if len(keys)!=entry['total_images'] or len(set(keys))!=len(keys) or digest(keys)!=first['global_sample_keys_sha256']:
        raise RuntimeError('Unique global coverage differs: '+d)
    for row in rows:
        for field in ('implementation','dataset','methods','classes','choice','checkpoints','vocabulary','global_sample_keys_sha256'):
            if row['signature'][field]!=first[field]:raise RuntimeError('Shards differ in '+field)
    matrices={p:{m:sum(np.asarray(r['metrics'][p][m]['confusion_matrix'],np.int64) for r in rows)
        for m in first['methods']} for p in first['classes']}
    split={s:{p:{m:np.zeros_like(cm) for m,cm in group.items()} for p,group in matrices.items()}
           for s in ('development','complement')}
    dev=set(entry['development_keys'])
    for folder in folders:
        with np.load(folder/'per_image_confusions.npz',allow_pickle=False) as data:
            # NpzFile indexing decompresses the complete member each time.
            # Read each member once, rather than once for every image.
            is_dev=np.asarray([key in dev and entry['development_source']!='ADE_training'
                               for key in data['sample_keys'].tolist()],dtype=bool)
            for p,methods in matrices.items():
                for m in methods:
                    images=data[p+'__'+m]
                    split['development'][p][m]+=images[is_dev].sum(0,dtype=np.int64)
                    split['complement'][p][m]+=images[~is_dev].sum(0,dtype=np.int64)
    previous=read_json(root.parent/'alias_finalization_20261009'/'full'/d/'merged.json')
    if previous is None:raise RuntimeError('Previous full result missing: '+d)
    replay={};local_error=wide_error=0.
    # Subset banks are rebuilt with a CPU mean. Verify the only old-model
    # representation difference is bounded float32 summation roundoff.
    import torch
    import torch.nn.functional as F
    old_cache=torch.load(root.parent/'alias_finalization_20261009'/'text_cache'/(d+'.pt'),map_location='cpu',weights_only=True)
    new_cache=torch.load(root/'text_cache'/(d+'.pt'),map_location='cpu',weights_only=True)
    for name,old_row in old_cache['encoded'].items():
        if name not in new_cache['encoded']:continue
        item=new_cache['encoded'][name]
        wide=new_cache['stores'][item['template']][item['indices']] if 'indices' in item else item['wide']
        local=F.normalize(wide.float().mean(1),dim=-1) if 'indices' in item else item['local']
        local_error=max(local_error,float((old_row['local']-local).abs().max()))
        wide_error=max(wide_error,float((old_row['wide']-wide).abs().max()))
    if local_error>2e-7 or wide_error!=0:raise RuntimeError('Old-model representation changed beyond float32 roundoff.')
    del old_cache,new_cache
    for p,methods in matrices.items():
        target=methods['KevTuned'].sum(1)
        if any(not np.array_equal(cm.sum(1),target) for cm in methods.values()):raise RuntimeError('Paired target counts differ.')
        old_matrix=np.asarray(previous['metrics'][p]['Uniform']['confusion_matrix'],np.int64)
        if not np.array_equal(methods['PreviousFinal'].sum(1),old_matrix.sum(1)):
            raise RuntimeError('Original final model target counts differ: '+d+'/'+p)
        l1=int(abs(methods['PreviousFinal']-old_matrix).sum())
        delta=abs(score(methods['PreviousFinal'])-score(old_matrix))
        if l1>max(2,2e-6*int(old_matrix.sum())) or delta>1e-4:
            raise RuntimeError('Original final model replay exceeds declared numerical tolerance: '+d+'/'+p)
        replay[p]=dict(exact=l1==0,confusion_l1=l1,miou_difference_pp=delta,
            local_vector_max_error=local_error,wide_vector_max_error=wide_error,
            interpretation='CPU/GPU float32 text-mean roundoff; raw replay retained, historical exact result remains the comparator.')
        if entry['development_source']!='ADE_training' and not np.array_equal(
            split['development'][p]['KevTuned'],first['choice']['development_confusions'][p]):
            raise RuntimeError('Frozen development choice does not replay: '+d+'/'+p)
    def metrics(groups):
        return {p:{m:summary(cm,first['classes'][p],0) for m,cm in group.items()} for p,group in groups.items()}
    output=root/'full'/d/'merged.json'
    if output.exists():
        existing=read_json(output)
        if (not existing or not existing.get('coverage_verified') or
            existing['signature']['sample_keys']!=keys or existing['signature']['choice']!=first['choice']):
            raise RuntimeError('Existing merged identity/coverage differs.')
        for p,methods in matrices.items():
            for m,cm in methods.items():
                if not np.array_equal(existing['metrics'][p][m]['confusion_matrix'],cm):
                    raise RuntimeError('Existing merged confusion differs.')
        return {p:{m:v['mean_iou_percent'] for m,v in group.items()} for p,group in existing['metrics'].items()}
    result=dict(status='complete',coverage_verified=True,processed_images=len(keys),total_images=len(keys),
        signature={**first,'shard_index':None,'sample_keys':keys},metrics=metrics(matrices),
        split_metrics={s:metrics(g) for s,g in split.items()},previous_model_replay_verified=True,
        previous_model_replay_exact=all(r['exact'] for r in replay.values()),previous_model_numerical_check=replay,
        development_replay_verified=True,scored_target_counts_match=True,
        parallel_wall_seconds=max(r['wall_seconds'] for r in rows),aggregate_gpu_seconds=sum(r['wall_seconds'] for r in rows),
        peak_cuda_memory_mb=max(r['peak_cuda_memory_mb'] for r in rows))
    save(output,result)
    return {p:{m:v['mean_iou_percent'] for m,v in group.items()} for p,group in result['metrics'].items()}


def main(root,resume=False,reference_repair=False,aggregation_repair=False):
    protocol=read_json(root/'protocol.json')
    prior=read_json(root/'suite_status.json')
    if prior and not resume:raise RuntimeError('Existing controller state is preserved.')
    if not read_json(root/'kev_smoke.json'):raise RuntimeError('Pinned Kev smoke required.')
    pending=[(d,'language',0) for d in protocol['order']]
    active=[];completed=[];merged={};failed={};started=time.perf_counter()
    if resume:
        if not prior:raise RuntimeError('No previous controller state to resume.')
        pending=[tuple(p) for p in prior['pending']];active=prior['active'];completed=prior['completed']
        merged=prior['merged'];failed=prior.get('failures',{}).copy()
        backup=root/('suite_status.before_aggregation_repair.json' if aggregation_repair else
                     'suite_status.before_reference_path_repair.json' if reference_repair else 'suite_status.before_context_repair.json')
        if not backup.exists():save(backup,prior)
        for session,item in list(failed.items()):
            job=item['job'];tail=Path(job['log']).read_text()[-6000:]
            if job['phase']=='language' and 'kev.model.ContextOverflow: branch too long:' in tail:
                pending.insert(0,(job['dataset'],'language',0));del failed[session]
            elif reference_repair and job['phase']=='full' and item['error']=="'NoneType' object is not subscriptable":
                # Inference is complete; only the old-reference path was wrong.
                entry=protocol['datasets'][job['dataset']]
                for shard in range(entry['shards']):
                    verify(dict(job,output=str(root/'full'/job['dataset']/('s'+str(shard)))))
                merged[job['dataset']]=merge(root,job['dataset'],entry)
                pending.append((job['dataset'],'cost',0));del failed[session]
    while pending or active:
        for job in active[:]:
            if alive(job['session']):continue
            active.remove(job);d=job['dataset'];phase=job['phase']
            try:
                verify(job);completed.append(job)
                if phase=='language':pending.append((d,'search',0))
                if phase=='search':pending.extend((d,'full',s) for s in range(protocol['datasets'][d]['shards']))
                if phase=='full' and sum(j['dataset']==d and j['phase']=='full' for j in completed)==protocol['datasets'][d]['shards']:
                    merged[d]=merge(root,d,protocol['datasets'][d]);pending.append((d,'cost',0))
                    print(json.dumps(dict(dataset=d,full_complete=merged[d])),flush=True)
            except Exception as exc:
                failed[job['session']]=dict(error=str(exc),job=job)
                pending=[p for p in pending if p[0]!=d]
                print(json.dumps(dict(failure=job['session'],error=str(exc),log=job['log'])),flush=True)
        for gpu in range(8):
            if pending and not any(j['gpu']==gpu for j in active) and idle(gpu):
                d,phase,shard=pending.pop(0)
                active.append(launch(root,d,protocol['datasets'][d],gpu,phase,shard))
        save(root/'suite_status.json',dict(status='running',active=active,pending=pending,completed=completed,
            merged=merged,failures=failed,wall_seconds=time.perf_counter()-started))
        if pending or active:time.sleep(10)
    row=dict(status='failed' if failed else 'complete',datasets=merged,failures=failed,
        total_protocol_images=sum(protocol['datasets'][d]['total_images'] for d in merged),
        wall_seconds=time.perf_counter()-started)
    save(root/'suite_results.json',row);save(root/'suite_status.json',dict(**row,active=[],pending=[],completed=completed))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--resume-context-repair',action='store_true')
    p.add_argument('--resume-reference-path-repair',action='store_true')
    p.add_argument('--resume-aggregation-repair',action='store_true');a=p.parse_args()
    main(a.root,a.resume_context_repair or a.resume_reference_path_repair or a.resume_aggregation_repair,
         a.resume_reference_path_repair,a.resume_aggregation_repair)
