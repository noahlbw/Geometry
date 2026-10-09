"""Run only frozen same-vocabulary VIP comparisons on genuinely idle GPUs."""
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
from eval_vip_official_eight import digest
from run_region_semantic_suite_a800 import TOOL, BASE, THIRD, PYTHON, idle
from run_sat_geometry_transport_suite import alive, read_json
from eval_vip_kev_same_vocabulary import METHODS
from dinotool.vip_official_adapter import upstream_aliases

PREFIX='gvkv10'


def launch(root,d,entry,gpu,phase,shard=0):
    output=root/phase/d/('s'+str(shard));log=output.with_suffix('.log')
    session=f'{PREFIX}_{phase}_{d}_s{shard}'
    if output.exists() or log.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Existing output/session or occupied GPU: '+session)
    output.parent.mkdir(parents=True,exist_ok=True)
    env=['env','CUDA_VISIBLE_DEVICES='+str(gpu),'PYTHONPATH='+str(TOOL)+':'+str(TOOL/'scripts')+':'+str(THIRD),
         'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
    command=env+[PYTHON,'-u','scripts/eval_vip_kev_same_vocabulary.py','--suite-root',str(root),
        '--source-root',read_json(root/'protocol.json')['source_root'],'--dataset',d,'--data-root',entry['data_root'],
        '--dinov3-repo',str(TOOL/'dinov3_hub'),'--checkpoint-dir',str(BASE/'ckpt/DINO'),
        '--upstream-root',str(BASE/'third_party/VIP_official_5bd25ee'),'--output-dir',str(output),
        '--phase',phase,'--num-shards',str(entry['shards'] if phase=='full' else 1),'--shard-index',str(shard)]
    shell='cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(command)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    return dict(dataset=d,phase=phase,shard=shard,gpu=gpu,session=session,output=str(output),log=str(log))


def verify(job):
    row=read_json(Path(job['output'])/'results.json')
    if (not row or row['status']!='complete' or not row['weights_frozen'] or not row['head_weights_unchanged']):
        raise RuntimeError('Worker exited without verified completion: '+job['log'])
    if job['phase']=='smoke' and row['target_masks_loaded']:raise RuntimeError('Smoke loaded masks.')
    if job['phase']=='cost':
        if not row['standalone_process'] or row['target_masks_loaded'] or len(row['images'])!=7:
            raise RuntimeError('Invalid standalone timing.')
    if job['phase']=='full':
        if row['processed_images']!=row['total_images']:raise RuntimeError('Incomplete full coverage.')
        with np.load(Path(job['output'])/'per_image_confusions.npz',allow_pickle=False) as data:
            if data['sample_keys'].tolist()!=row['signature']['sample_keys']:raise RuntimeError('Per-image keys differ.')
            for p,group in row['metrics'].items():
                for m,metric in group.items():
                    if not np.array_equal(data[p+'__'+m].sum(0),metric['confusion_matrix']):
                        raise RuntimeError('Per-image confusion sum differs.')
    return row


def merge(root,d,entry):
    folders=[root/'full'/d/('s'+str(i)) for i in range(entry['shards'])]
    rows=[read_json(f/'results.json') for f in folders];signature=rows[0]['signature']
    keys=[k for group in zip_longest(*(r['signature']['sample_keys'] for r in rows)) for k in group if k is not None]
    prior=read_json(Path(read_json(root/'protocol.json')['source_root'])/'full'/d/'merged.json')
    if (len(keys)!=entry['total_images'] or len(set(keys))!=len(keys)
            or keys!=prior['signature']['sample_keys'] or digest(keys)!=signature['global_sample_keys_sha256']):
        raise RuntimeError('Unique complete source coverage differs: '+d)
    for row in rows:
        for key in ('implementation','dataset','methods','classes','selected_profile','text_identity','template_counts',
                    'settings','checkpoints','upstream_commit','view_policy','global_sample_keys_sha256'):
            if row['signature'][key]!=signature[key]:raise RuntimeError('Shard identity differs: '+key)
    if signature['checkpoints']!=prior['signature']['checkpoints']:raise RuntimeError('Source checkpoints differ.')
    matrices={p:{m:sum(np.asarray(r['metrics'][p][m]['confusion_matrix'],np.int64) for r in rows)
                 for m in METHODS} for p in signature['classes']}
    split={s:{p:{m:np.zeros_like(cm) for m,cm in group.items()} for p,group in matrices.items()}
           for s in ('development','complement')}
    dev=set(entry['development_keys'])
    for folder in folders:
        with np.load(folder/'per_image_confusions.npz',allow_pickle=False) as data:
            in_dev=np.asarray([k in dev and entry['development_source']!='ADE_training'
                               for k in data['sample_keys'].tolist()],dtype=bool)
            for p,group in matrices.items():
                for m in group:
                    images=data[p+'__'+m]
                    split['development'][p][m]+=images[in_dev].sum(0,dtype=np.int64)
                    split['complement'][p][m]+=images[~in_dev].sum(0,dtype=np.int64)
    historical=read_json(root/'references'/(d+'.json')) or {};replay={}
    for p,group in matrices.items():
        target=np.asarray(prior['metrics'][p]['KevTuned']['confusion_matrix'],np.int64).sum(1)
        for m,cm in group.items():
            if not np.array_equal(cm.sum(1),target):raise RuntimeError('Scored targets differ: '+d+'/'+p)
            if p in historical:
                old=np.asarray(historical[p]['confusion_matrix'],np.int64)
                if not np.array_equal(old.sum(1),target):raise RuntimeError('Historical VIP target counts differ.')
                if d in ('vdd','potsdam'):
                    official=upstream_aliases(BASE/'third_party/VIP_official_5bd25ee'/'configs'/('cls_'+d+'.txt'))
                    selected=tuple(tuple(c['synonyms']) for c in entry['classes'])
                    same_words=selected==official
                    exact=np.array_equal(cm,old)
                    replay[m]=dict(exact=bool(exact),required=same_words,word_groups_equal=same_words,
                        confusion_l1=int(abs(cm-old).sum()),historical_groups=official,selected_groups=selected,
                        note='Pinned VIP splits only comma-space; the completed Kev source split all commas. '
                             'Only identical actual query groups require exact historical confusion replay.')
                    if same_words and not exact:raise RuntimeError('Unchanged official-short VIP replay differs: '+d+'/'+m)
    def metrics(groups):
        return {p:{m:summary(cm,tuple(signature['classes'][p]),sum(r['metrics'][p][m]['ignored_pixels'] for r in rows))
                   for m,cm in group.items()} for p,group in groups.items()}
    result=dict(status='complete',coverage_verified=True,processed_images=len(keys),total_images=len(keys),
                signature={**signature,'sample_keys':keys,'shard_index':None},metrics=metrics(matrices),
                split_metrics={s:metrics(group) for s,group in split.items()},
                scored_target_counts_match=True,source_checkpoints_verified=True,per_image_confusions_verified=True,
                unchanged_official_short_replay=replay,weights_frozen=True,head_weights_unchanged=True,
                parallel_wall_seconds=max(r['wall_seconds'] for r in rows),aggregate_gpu_seconds=sum(r['wall_seconds'] for r in rows),
                peak_cuda_memory_mb=max(r['peak_cuda_memory_mb'] for r in rows))
    path=root/'full'/d/'merged.json'
    if path.exists():
        existing=read_json(path)
        if (existing['signature']['sample_keys']!=keys or existing['signature']['text_identity']!=signature['text_identity']
                or any(existing['metrics'][p][m]['confusion_matrix']!=cm.tolist()
                       for p,group in matrices.items() for m,cm in group.items())):
            raise RuntimeError('Existing merged output differs; preserve it.')
        return {p:{m:metric['mean_iou_percent'] for m,metric in group.items()} for p,group in existing['metrics'].items()}
    save(path,result)
    return {p:{m:metric['mean_iou_percent'] for m,metric in group.items()} for p,group in result['metrics'].items()}


def main(root,resume=False):
    protocol=read_json(root/'protocol.json')
    prior=read_json(root/'suite_status.json')
    if prior and not resume:raise RuntimeError('Existing suite state preserved; inspect before recovery.')
    pending=[(d,'smoke',0) for d in protocol['order']];active=[];completed=[];merged={};failed={}
    started=time.perf_counter()
    if resume:
        if not prior:raise RuntimeError('No saved state to resume.')
        backup=root/'suite_status.before_query_parser_verification_repair.json'
        if not backup.exists():save(backup,prior)
        pending=[tuple(j) for j in prior['pending']];active=prior['active'];completed=prior['completed']
        merged=prior['merged'];failed=prior.get('failures',{}).copy()
        for session,item in list(failed.items()):
            job=item['job'];d=job['dataset']
            if job['phase']=='full' and item['error'].startswith('Unchanged official-short VIP replay differs:'):
                for shard in range(protocol['datasets'][d]['shards']):
                    verify(dict(job,output=str(root/'full'/d/('s'+str(shard)))))
                merged[d]=merge(root,d,protocol['datasets'][d]);pending.append((d,'cost',0));del failed[session]
                print(json.dumps(dict(dataset=d,recovered_completed_merge=merged[d])),flush=True)
    while pending or active:
        for job in active[:]:
            if alive(job['session']):continue
            active.remove(job);d=job['dataset']
            try:
                verify(job);completed.append(job)
                if job['phase']=='smoke':pending.extend((d,'full',s) for s in range(protocol['datasets'][d]['shards']))
                if job['phase']=='full' and sum(j['dataset']==d and j['phase']=='full' for j in completed)==protocol['datasets'][d]['shards']:
                    merged[d]=merge(root,d,protocol['datasets'][d]);pending.append((d,'cost',0))
                    print(json.dumps(dict(dataset=d,merged=merged[d])),flush=True)
            except Exception as exc:
                failed[job['session']]=dict(error=str(exc),job=job);pending=[p for p in pending if p[0]!=d]
                print(json.dumps(dict(failure=job['session'],error=str(exc),log=job['log'])),flush=True)
        for gpu in (4,5,6,7,0,1,2,3):
            if pending and not any(j['gpu']==gpu for j in active) and idle(gpu):
                d,phase,shard=pending.pop(0);active.append(launch(root,d,protocol['datasets'][d],gpu,phase,shard))
        save(root/'suite_status.json',dict(status='running',active=active,pending=pending,completed=completed,
                                         merged=merged,failures=failed,wall_seconds=time.perf_counter()-started))
        if pending or active:time.sleep(10)
    result=dict(status='failed' if failed else 'complete',datasets=merged,failures=failed,
                total_protocol_images=sum(protocol['datasets'][d]['total_images'] for d in merged),
                wall_seconds=time.perf_counter()-started)
    save(root/'suite_results.json',result);save(root/'suite_status.json',dict(**result,active=[],pending=[],completed=completed))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--resume-query-parser-verification-repair',action='store_true');args=parser.parse_args()
    main(args.root,args.resume_query_parser_verification_repair)
