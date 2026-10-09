"""Idle-GPU controller for four frozen paired full evaluations."""
import argparse
from pathlib import Path
import shlex
import subprocess
import time
import traceback

import numpy as np

from dinotool.semantic_contract_alias import BASE, PRIMARY
from eval_rival_fine_full import save
from merge_gear_ov_shards import merge
from run_region_semantic_suite_a800 import BASE as DATA, TOOL, THIRD, PYTHON, idle
from run_sat_geometry_transport_suite import alive, read_json


def launch(root, protocol, dataset, gpu, mode, shard):
    folder=root/'benchmark'/dataset if mode=='benchmark' else root/'full'/dataset/('s'+str(shard))
    log=root/(mode+'_'+dataset+'_s'+str(shard)+'.log');session='gsca09_'+mode+'_'+dataset+'_s'+str(shard)
    if folder.exists() or log.exists() or alive(session):raise RuntimeError('Existing output; preserve: '+session)
    if not idle(gpu):return None
    entry=protocol['datasets'][dataset]
    args=[PYTHON,'-u','scripts/eval_semantic_contract_alias.py','--dataset',dataset,'--suite-root',str(root),
        '--output-dir',str(folder),'--mode',mode,'--data-root',entry['data_root'],
        '--dinov3-repo',str(TOOL/'dinov3_hub'),'--checkpoint-dir',str(DATA/'ckpt/DINO'),
        '--upstream-root',str(DATA/'third_party/VIP_official_5bd25ee')]
    if mode=='full':args+=['--num-shards',str(entry['shards']),'--shard-index',str(shard)]
    env=['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
         'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
    shell='cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    return dict(dataset=dataset,mode=mode,shard=shard,gpu=gpu,session=session,output=str(folder),log=str(log))


def verify(root,protocol,d):
    entry=protocol['datasets'][d];paths=[root/'full'/d/('s'+str(i)) for i in range(entry['shards'])]
    dest=root/'full'/d/'merged.json'
    if dest.exists():raise RuntimeError('Existing merged result; preserve it.')
    row=merge([str(p) for p in paths],str(dest),diagnostic_weight='images')
    previous=read_json(Path(entry['reference']))
    if (not row['coverage_verified'] or row['processed_images']!=entry['total_images']
            or not previous or row['signature']['sample_keys']!=previous['signature']['sample_keys']
            or row['signature']['checkpoints']!=previous['signature']['checkpoints']):
        raise RuntimeError('Full inventory/checkpoints differ from previous: '+d)
    keys=row['signature']['sample_keys'];lookup={k:i for i,k in enumerate(keys)};nc=len(row['signature']['classes'][d])
    current=np.empty((len(keys),nc,nc),np.int64);seen=[]
    sums={name:np.zeros((nc,nc),np.int64) for name in (BASE,PRIMARY)}
    for path in paths:
        with np.load(path/'per_image_confusions.npz',allow_pickle=False) as a:
            sk=a['sample_keys'].tolist();seen+=sk
            current[[lookup[k] for k in sk]]=a[d+'__'+BASE]
            for name in sums:sums[name]+=a[d+'__'+name].sum(0)
    old_seen=[]
    for path in previous['shards']:
        with np.load(Path(path).parent/'per_image_confusions.npz',allow_pickle=False) as a:
            sk=a['sample_keys'].tolist();old_seen+=sk
            if not np.array_equal(current[[lookup[k] for k in sk]],a[d+'__'+entry['reference_method']]):
                raise RuntimeError('Original per-image baseline replay differs: '+d)
    if (len(seen)!=len(set(seen)) or set(seen)!=set(keys) or set(old_seen)!=set(keys)
            or any(not np.array_equal(sums[name],row['metrics'][d][name]['confusion_matrix']) for name in sums)
            or not np.array_equal(sums[BASE].sum(1),sums[PRIMARY].sum(1))):
        raise RuntimeError('Unique coverage/confusion/paired targets differ.')
    row.update(exact_predecessor_per_image_replayed=True,paired_scored_targets_equal=True,
               per_image_confusion_sums_verified=True)
    save(dest,row);return row


def main(root):
    if (root/'suite_status.json').exists():raise RuntimeError('Existing controller; preserve it.')
    protocol=read_json(root/'protocol.json');phase='mask_free_benchmarks'
    pending=[(d,'benchmark',0) for d in protocol['datasets']];active={};completed=[]
    try:
        while True:
            for gpu,job in list(active.items()):
                if alive(job['session']):continue
                worker=read_json(Path(job['output'])/'worker_status.json')
                if not worker or worker.get('status')!='complete':
                    raise RuntimeError('Worker exited without complete: '+job['session']+'; log='+job['log'])
                completed.append(job);del active[gpu]
            if not pending and not active:
                if phase=='mask_free_benchmarks':
                    phase='full_evaluation'
                    pending=[(d,'full',i) for d,entry in protocol['datasets'].items() for i in range(entry['shards'])]
                else:
                    datasets={d:dict(merged=str(root/'full'/d/'merged.json'),processed_images=verify(root,protocol,d)['processed_images'])
                              for d in protocol['datasets']}
                    save(root/'suite_results.json',dict(status='complete',outcome='full_verified',datasets=datasets,
                        processed_images=sum(entry['total_images'] for entry in protocol['datasets'].values())));return
            for gpu in protocol['gpus']:
                if not pending:break
                if gpu in active or not idle(gpu):continue
                d,mode,shard=pending[0];job=launch(root,protocol,d,gpu,mode,shard)
                if job is not None:active[gpu]=job;pending.pop(0)
            save(root/'suite_status.json',dict(status='running',phase=phase,active=list(active.values()),
                pending=pending,completed=completed));time.sleep(5)
    except Exception:
        save(root/'suite_status.json',dict(status='failed',phase=phase,active=list(active.values()),pending=pending,
             completed=completed,error=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
