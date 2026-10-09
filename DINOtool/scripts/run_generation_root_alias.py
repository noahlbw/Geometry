"""Two singleton cost gates, then ADE four shards and VDD one shard."""
import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess
import time
import traceback

import numpy as np

from dinotool.generation_root_alias import BASE, QUOTIENT, PRIMARY, SHUFFLE
from eval_rival_fine_full import save
from merge_gear_ov_shards import merge
from run_region_semantic_suite_a800 import BASE as DATA, TOOL, THIRD, PYTHON, idle
from run_sat_geometry_transport_suite import alive, read_json


def launch(root,dataset,gpu,mode,shard=0):
    entry=read_json(root/'protocol.json')['datasets'][dataset]
    folder=root/'benchmark'/dataset if mode=='benchmark' else root/'full'/dataset/('s'+str(shard))
    log=root/(mode+'_'+dataset+'_s'+str(shard)+'.log');session='ggra08_'+mode+'_'+dataset+'_s'+str(shard)
    if folder.exists() or log.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Existing output/session or occupied GPU: '+session)
    args=[PYTHON,'-u','scripts/eval_generation_root_alias.py','--dataset',dataset,'--suite-root',str(root),
        '--output-dir',str(folder),'--mode',mode,'--data-root',entry['data_root'],
        '--vocabulary-config',entry['vocabulary_config'],'--dinov3-repo',str(TOOL/'dinov3_hub'),
        '--checkpoint-dir',str(DATA/'ckpt/DINO'),'--upstream-root',str(DATA/'third_party/VIP_official_5bd25ee')]
    if mode=='full':args+=['--num-shards','4' if dataset=='ade150' else '1','--shard-index',str(shard)]
    env=['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
        'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
    shell='cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    return dict(gpu=gpu,session=session,output=str(folder),log=str(log),shard=shard,mode=mode,dataset=dataset)


def image_hashes(paths,dataset,method):
    values={}
    for path in paths:
        with np.load(path,allow_pickle=False) as archive:
            for key,cm in zip(archive['sample_keys'].tolist(),archive[dataset+'__'+method]):
                if key in values:raise RuntimeError('Duplicate reference archive key.')
                values[key]=hashlib.sha256(cm.astype(np.int64,copy=False).tobytes()).hexdigest()
    return values


def verify(root,protocol,dataset):
    count=4 if dataset=='ade150' else 1;shards=[root/'full'/dataset/('s'+str(i)) for i in range(count)]
    path=root/'full'/dataset/'merged.json'
    if path.exists():raise RuntimeError('Existing merged output; preserve it.')
    row=merge([str(p) for p in shards],str(path),diagnostic_weight='images')
    refs={BASE:(Path(protocol['datasets'][dataset]['exact_base_reference']),BASE)}
    if dataset=='ade150':refs[QUOTIENT]=(Path(protocol['exact_cq_reference']),'CQ')
    for method,(refpath,refmethod) in refs.items():
        ref=read_json(refpath)
        if (not ref or not ref.get('coverage_verified') or ref['processed_images']!=protocol['datasets'][dataset]['total_images']
                or ref['signature']['sample_keys']!=row['signature']['sample_keys']
                or ref['signature']['checkpoints']!=row['signature']['checkpoints']
                or ref['signature']['classes']!=row['signature']['classes']):
            raise RuntimeError('Full reference coverage/source identity differs: '+method)
        if method==BASE and ref['signature']['vocabulary']['sha256']!=row['signature']['vocabulary']['sha256']:
            raise RuntimeError('Frozen input vocabulary identity differs.')
        if ref['metrics'][dataset][refmethod]['confusion_matrix']!=row['metrics'][dataset][method]['confusion_matrix']:
            raise RuntimeError('Full reference confusion differs: '+method)
        expected=image_hashes([Path(p).parent/'per_image_confusions.npz' for p in ref['shards']],dataset,refmethod)
        actual=image_hashes([p/'per_image_confusions.npz' for p in shards],dataset,method)
        if expected!=actual:raise RuntimeError('Full per-image reference confusions differ: '+method)
    targets=[np.asarray(item['confusion_matrix']).sum(-1) for item in row['metrics'][dataset].values()]
    if any(not np.array_equal(targets[0],t) for t in targets[1:]):raise RuntimeError('Paired scored targets differ.')
    if dataset=='vdd':
        expected=image_hashes([p/'per_image_confusions.npz' for p in shards],dataset,BASE)
        if any(image_hashes([p/'per_image_confusions.npz' for p in shards],dataset,n)!=expected for n in (PRIMARY,QUOTIENT,SHUFFLE)):
            raise RuntimeError('VDD no-provenance fallback changed per-image confusions.')
    row.update(exact_base_per_image_replayed=True,exact_cq_per_image_replayed=dataset=='ade150',
        exact_no_provenance_fallback=dataset=='vdd',paired_scored_targets_equal=True)
    save(path,row);return row


def main(root):
    if (root/'suite_status.json').exists() or (root/'suite_results.json').exists():
        raise RuntimeError('Existing controller state; preserve it.')
    protocol=read_json(root/'protocol.json');active={};completed=[];failures={}
    pending=[('ade150','benchmark',0),('vdd','benchmark',0)];phase='mask_free_benchmarks'
    try:
        while True:
            for gpu,job in list(active.items()):
                if alive(job['session']):continue
                folder=Path(job['output']);worker=read_json(folder/'worker_status.json')
                if job['mode']=='benchmark':
                    timing=read_json(folder/'timing.json')
                    if worker and worker.get('status')=='cost_rejected':
                        failures[job['session']]=dict(job=job,reason='cost_rejected',timing=timing)
                    elif not worker or worker.get('status')!='complete' or not timing or not timing.get('cost_gate_passed'):
                        failures[job['session']]=dict(job=job,log_tail=Path(job['log']).read_text(errors='replace')[-7000:])
                    else:completed.append(job)
                else:
                    row=read_json(folder/'results.json');expected=500 if job['dataset']=='ade150' else 80
                    if worker and worker.get('status')=='complete' and row and row.get('status')=='complete' and row['processed_images']==row['total_images']==expected:
                        completed.append(job)
                    else:failures[job['session']]=dict(job=job,log_tail=Path(job['log']).read_text(errors='replace')[-7000:])
                del active[gpu]
            if failures:pending.clear()
            if not pending and not active:
                if failures:
                    cost=all(item.get('reason')=='cost_rejected' for item in failures.values())
                    save(root/'suite_results.json',dict(status='complete' if cost else 'failed',
                        outcome='cost_rejected' if cost else 'worker_failed',failures=failures,
                        target_masks_loaded=phase=='full'));return
                if phase=='mask_free_benchmarks':
                    pending=[('ade150','full',i) for i in range(4)]+[('vdd','full',0)];phase='full'
                else:break
            for d,mode,shard in list(pending):
                gpu=next((g for g in (7,6,5,4,3,2,1,0) if g not in active and idle(g)),None)
                if gpu is None:break
                active[gpu]=launch(root,d,gpu,mode,shard);pending.remove((d,mode,shard))
            save(root/'suite_status.json',dict(status='running',phase=phase,active=list(active.values()),
                pending=pending,completed=completed,failures=failures))
            time.sleep(10)
        save(root/'suite_status.json',dict(status='running',phase='merge_verify',completed=completed))
        rows={d:verify(root,protocol,d) for d in protocol['datasets']}
        save(root/'suite_results.json',dict(status='complete',outcome='full_verified',
            images=sum(row['processed_images'] for row in rows.values()),
            datasets={d:dict(images=row['processed_images'],merged=str(root/'full'/d/'merged.json')) for d,row in rows.items()}))
    except Exception:
        save(root/'suite_results.json',dict(status='failed',phase='controller',error=traceback.format_exc(),
            active=list(active.values()),completed=completed,failures=failures));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
