"""Two mask-free cost gates, then one frozen full VDD/ADE family-rule trial."""
import argparse
from pathlib import Path
import shlex
import subprocess
import time
import traceback

import numpy as np

from dinotool.family_lme_alias import CURRENT_BASE
from eval_family_lme_alias import FULL_METHODS
from eval_rival_fine_full import save
from merge_gear_ov_shards import merge
from run_generation_root_alias import image_hashes
from run_region_semantic_suite_a800 import BASE as DATA, TOOL, THIRD, PYTHON, idle
from run_sat_geometry_transport_suite import alive, read_json


def launch(root, dataset, gpu, mode, shard=0):
    entry=read_json(root/'protocol.json')['datasets'][dataset]
    folder=root/'benchmark'/dataset if mode=='benchmark' else root/'full'/dataset/('s'+str(shard))
    log=root/(mode+'_'+dataset+'_s'+str(shard)+'.log'); session='gflm09_'+mode+'_'+dataset+'_s'+str(shard)
    if folder.exists() or log.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Existing output/session or occupied GPU: '+session)
    args=[PYTHON,'-u','scripts/eval_family_lme_alias.py','--dataset',dataset,'--suite-root',str(root),
          '--output-dir',str(folder),'--mode',mode,'--data-root',entry['data_root'],
          '--vocabulary-config',entry['vocabulary_config'],'--dinov3-repo',str(TOOL/'dinov3_hub'),
          '--checkpoint-dir',str(DATA/'ckpt/DINO'),'--upstream-root',str(DATA/'third_party/VIP_official_5bd25ee')]
    if mode=='full': args+=['--num-shards','4' if dataset=='ade150' else '1','--shard-index',str(shard)]
    env=['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
         'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
    shell='cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    return dict(gpu=gpu,session=session,output=str(folder),log=str(log),shard=shard,mode=mode,dataset=dataset)


def verify(root, protocol, dataset):
    shards=[root/'full'/dataset/('s'+str(i)) for i in range(protocol['full_shards'][dataset])]
    path=root/'full'/dataset/'merged.json'
    if path.exists(): raise RuntimeError('Existing merge; preserve it.')
    row=merge([str(p) for p in shards],str(path),diagnostic_weight='images')
    refpath=Path(protocol['datasets'][dataset]['exact_base_reference']); ref=read_json(refpath)
    if (not ref or not ref.get('coverage_verified') or ref['processed_images']!=protocol['datasets'][dataset]['total_images']
        or ref['signature']['sample_keys']!=row['signature']['sample_keys']
        or ref['signature']['checkpoints']!=row['signature']['checkpoints']
        or ref['signature']['classes']!=row['signature']['classes']
        or ref['signature']['vocabulary']['sha256']!=row['signature']['vocabulary']['sha256']):
        raise RuntimeError('Reference sample/source identity differs.')
    if ref['metrics'][dataset]['Fixed20_TaskCoupled']['confusion_matrix']!=row['metrics'][dataset][CURRENT_BASE]['confusion_matrix']:
        raise RuntimeError('Aggregate Current20 baseline replay differs.')
    expected=image_hashes([Path(p).parent/'per_image_confusions.npz' for p in ref['shards']],dataset,'Fixed20_TaskCoupled')
    actual=image_hashes([p/'per_image_confusions.npz' for p in shards],dataset,CURRENT_BASE)
    if expected!=actual: raise RuntimeError('Per-image Current20 baseline replay differs.')
    targets=[np.asarray(row['metrics'][dataset][name]['confusion_matrix']).sum(-1) for name in FULL_METHODS]
    if any(not np.array_equal(targets[0],x) for x in targets[1:]): raise RuntimeError('Paired scored targets differ.')
    row.update(exact_current_base_per_image_replayed=True,paired_scored_targets_equal=True,
               candidate_bank_counts_exact20=True,full_unique_samples_verified=True)
    save(path,row); return row


def main(root):
    if (root/'suite_status.json').exists() or (root/'suite_results.json').exists():
        raise RuntimeError('Existing controller state; preserve it.')
    protocol=read_json(root/'protocol.json'); active={}; completed=[]; failures={}
    pending=[('ade150','benchmark',0),('vdd','benchmark',0)]; phase='mask_free_benchmarks'
    try:
        while True:
            for gpu,job in list(active.items()):
                if alive(job['session']): continue
                folder=Path(job['output']); worker=read_json(folder/'worker_status.json')
                if job['mode']=='benchmark':
                    timing=read_json(folder/'timing.json')
                    if worker and worker.get('status')=='cost_rejected':
                        failures[job['session']]=dict(job=job,reason='cost_rejected',timing=timing)
                    elif not worker or worker.get('status')!='complete' or not timing or not timing.get('cost_gate_passed'):
                        failures[job['session']]=dict(job=job,log_tail=Path(job['log']).read_text(errors='replace')[-7000:])
                    else: completed.append(job)
                else:
                    row=read_json(folder/'results.json'); expected=500 if job['dataset']=='ade150' else 80
                    if worker and worker.get('status')=='complete' and row and row.get('status')=='complete' and row['processed_images']==row['total_images']==expected:
                        completed.append(job)
                    else: failures[job['session']]=dict(job=job,log_tail=Path(job['log']).read_text(errors='replace')[-7000:])
                del active[gpu]
            if failures: pending.clear()
            if not pending and not active:
                if failures:
                    cost=all(item.get('reason')=='cost_rejected' for item in failures.values())
                    save(root/'suite_results.json',dict(status='complete' if cost else 'failed',
                         outcome='cost_rejected' if cost else 'worker_failed',failures=failures,target_masks_loaded=phase=='full')); return
                if phase=='mask_free_benchmarks':
                    pending=[('ade150','full',i) for i in range(4)]+[('vdd','full',0)]; phase='full'
                else: break
            for d,mode,shard in list(pending):
                gpu=next((g for g in (7,6,5,4,3,2,1,0) if g not in active and idle(g)),None)
                if gpu is None: break
                active[gpu]=launch(root,d,gpu,mode,shard); pending.remove((d,mode,shard))
            save(root/'suite_status.json',dict(status='running',phase=phase,active=list(active.values()),
                 pending=pending,completed=completed,failures=failures))
            time.sleep(10)
        rows={d:verify(root,protocol,d) for d in protocol['datasets']}
        save(root/'suite_results.json',dict(status='complete',outcome='full_verified',
             images=sum(r['processed_images'] for r in rows.values()),
             datasets={d:dict(images=r['processed_images'],merged=str(root/'full'/d/'merged.json')) for d,r in rows.items()}))
    except Exception:
        save(root/'suite_results.json',dict(status='failed',phase='controller',error=traceback.format_exc(),
             active=list(active.values()),completed=completed,failures=failures)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True); main(p.parse_args().root)
