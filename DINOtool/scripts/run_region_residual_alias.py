"""Cost gates, mask-free full-inventory mapping, then frozen paired scoring."""
import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess
import time
import traceback

import numpy as np

from dinotool import region_residual_alias as trial
from eval_region_residual_alias import FULL_METHODS, REPLAY
from eval_rival_fine_full import save
from merge_gear_ov_shards import merge
from run_generation_root_alias import image_hashes
from run_region_semantic_suite_a800 import BASE as DATA, TOOL, THIRD, PYTHON, idle
from run_sat_geometry_transport_suite import alive, read_json


def launch(root,dataset,gpu,mode,shard=0):
    entry=read_json(root/'protocol.json')['datasets'][dataset]
    folder=root/'benchmark'/dataset if mode=='benchmark' else root/mode/dataset/('s'+str(shard))
    log=root/(mode+'_'+dataset+'_s'+str(shard)+'.log'); session='grra09_'+mode+'_'+dataset+'_s'+str(shard)
    if folder.exists() or log.exists() or alive(session) or not idle(gpu):
        raise RuntimeError('Existing output/session or occupied GPU: '+session)
    args=[PYTHON,'-u','scripts/eval_region_residual_alias.py','--dataset',dataset,
        '--suite-root',str(root),'--output-dir',str(folder),'--mode',mode,'--data-root',entry['data_root'],
        '--vocabulary-config',entry['vocabulary_config'],'--dinov3-repo',str(TOOL/'dinov3_hub'),
        '--checkpoint-dir',str(DATA/'ckpt/DINO'),'--upstream-root',str(DATA/'third_party/VIP_official_5bd25ee')]
    if mode!='benchmark': args+=['--num-shards','4' if dataset=='ade150' else '1','--shard-index',str(shard)]
    env=['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
         'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
    shell='cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    return dict(gpu=gpu,session=session,output=str(folder),log=str(log),shard=shard,mode=mode,dataset=dataset)


def freeze_mapping(root,protocol,dataset):
    path=root/'prepass'/dataset; parts=[]; signatures=[]; seen=[]
    for i in range(protocol['full_shards'][dataset]):
        folder=path/('s'+str(i)); row=read_json(folder/'prepass.json'); archive=folder/'image_q.npz'
        if not row or row.get('status')!='complete' or row.get('target_masks_loaded'):
            raise RuntimeError('Mask-free prepass incomplete.')
        if row['archive_sha256']!=hashlib.sha256(archive.read_bytes()).hexdigest():
            raise RuntimeError('Prepass archive changed.')
        with np.load(archive,allow_pickle=False) as data:
            keys=data['sample_keys'].tolist(); q=data['q']
        if keys!=row['signature']['sample_keys'] or q.shape!=(len(keys),len(row['signature']['classes'][dataset]),20) or not np.isfinite(q).all():
            raise RuntimeError('Prepass image/alias inventory differs.')
        parts.extend(zip(keys,q)); signatures.append(row['signature']); seen.extend(keys)
    if len(seen)!=len(set(seen)) or len(seen)!=protocol['datasets'][dataset]['total_images']:
        raise RuntimeError('Prepass full unique coverage differs.')
    for sig in signatures[1:]:
        if any(sig[k]!=signatures[0][k] for k in ('checkpoints','classes','vocabulary','gear','global_sample_keys_sha256')):
            raise RuntimeError('Prepass source identities differ.')
    parts.sort(key=lambda p:p[0]); keys=[p[0] for p in parts]; q=np.stack([p[1] for p in parts])
    archive=path/'all_image_q.npz'; mapping=path/'image_mapping.json'
    if archive.exists() or mapping.exists(): raise RuntimeError('Existing frozen mapping; preserve it.')
    rng=np.random.default_rng(trial.SEED); identity=np.arange(len(keys))
    while True:
        donor=rng.permutation(len(keys))
        if np.all(donor!=identity): break
    np.savez_compressed(archive,sample_keys=np.asarray(keys),q=q)
    row=dict(mask_free_complete=True,processed_images=len(keys),recipient_keys=keys,
        donor_indices=donor.tolist(),donor_keys=[keys[i] for i in donor],seed=trial.SEED,
        source_signature=signatures[0],archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        target_masks_loaded=False,rule='Sorted full inventory, seeded bijection derangement; donor q retains class/alias identities.')
    save(mapping,row); return str(mapping)


def verify(root,protocol,dataset):
    shards=[root/'full'/dataset/('s'+str(i)) for i in range(protocol['full_shards'][dataset])]
    path=root/'full'/dataset/'merged.json'
    if path.exists(): raise RuntimeError('Existing merge; preserve it.')
    row=merge([str(p) for p in shards],str(path),diagnostic_weight='images')
    refpath=Path(protocol['predecessor'])/'full'/dataset/'merged.json'; ref=read_json(refpath)
    if not ref or any(ref['signature'][k]!=row['signature'][k] for k in
        ('sample_keys','checkpoints','classes','vocabulary','global_sample_keys_sha256')):
        raise RuntimeError('Predecessor full source identity differs.')
    for name,previous in REPLAY.items():
        if row['metrics'][dataset][name]['confusion_matrix']!=ref['metrics'][dataset][previous]['confusion_matrix']:
            raise RuntimeError('Predecessor aggregate replay differs: '+name)
        expected=image_hashes([Path(p).parent/'per_image_confusions.npz' for p in ref['shards']],dataset,previous)
        actual=image_hashes([p/'per_image_confusions.npz' for p in shards],dataset,name)
        if expected!=actual: raise RuntimeError('Predecessor per-image replay differs: '+name)
    targets=[np.asarray(row['metrics'][dataset][name]['confusion_matrix']).sum(-1) for name in FULL_METHODS]
    if any(not np.array_equal(targets[0],x) for x in targets[1:]): raise RuntimeError('Paired scored targets differ.')
    row.update(exact_predecessor_per_image_replayed=True,paired_scored_targets_equal=True,
               mask_free_full_inventory_image_derangement_verified=True,candidate_bank_counts_exact20=True)
    save(path,row); return row


def main(root):
    if (root/'suite_status.json').exists() or (root/'suite_results.json').exists():
        raise RuntimeError('Existing controller state; preserve it.')
    protocol=read_json(root/'protocol.json'); active={}; completed=[]; failures={}; frozen_maps={}; merged={}
    pending=[('ade150','benchmark',0),('vdd','benchmark',0)]; phase='mask_free_benchmarks'
    try:
        while True:
            for gpu,job in list(active.items()):
                if alive(job['session']): continue
                folder=Path(job['output']); worker=read_json(folder/'worker_status.json')
                if not worker or worker.get('status')!='complete':
                    failures[job['session']]=dict(job=job,reason='Worker exited before complete',worker=worker)
                elif job['mode']=='benchmark' and not (read_json(folder/'timing.json') or {}).get('cost_gate_passed'):
                    failures[job['session']]=dict(job=job,reason='Whole-image cost rejected')
                elif job['mode']=='prepass' and not (read_json(folder/'prepass.json') or {}).get('status')=='complete':
                    failures[job['session']]=dict(job=job,reason='Mask-free prepass missing')
                elif job['mode']=='full' and not (read_json(folder/'results.json') or {}).get('status')=='complete':
                    failures[job['session']]=dict(job=job,reason='Full results missing')
                else: completed.append(job)
                del active[gpu]
            if failures:
                save(root/'suite_status.json',dict(status='failed',phase=phase,failures=failures,active=list(active.values()),pending=pending)); return
            if not active and not pending:
                if phase=='mask_free_benchmarks':
                    phase='mask_free_prepass'
                    pending=[(d,'prepass',i) for d,n in protocol['full_shards'].items() for i in range(n)]
                elif phase=='mask_free_prepass':
                    frozen_maps={d:freeze_mapping(root,protocol,d) for d in protocol['datasets']}
                    phase='full_evaluation'
                    pending=[(d,'full',i) for d,n in protocol['full_shards'].items() for i in range(n)]
                else:
                    merged={d:dict(merged=str(root/'full'/d/'merged.json'),processed_images=verify(root,protocol,d)['processed_images']) for d in protocol['datasets']}
                    save(root/'suite_results.json',dict(status='complete',outcome='full_verified',datasets=merged,
                        image_maps=frozen_maps,processed_images=sum(p['processed_images'] for p in merged.values()))); return
            for gpu in range(8):
                if not pending: break
                if gpu not in active and idle(gpu):
                    d,mode,shard=pending.pop(0); active[gpu]=launch(root,d,gpu,mode,shard)
            save(root/'suite_status.json',dict(status='running',phase=phase,active=list(active.values()),pending=pending,
                completed=completed,image_maps=frozen_maps))
            time.sleep(5)
    except Exception:
        save(root/'suite_status.json',dict(status='failed',phase=phase,active=list(active.values()),pending=pending,
            traceback=traceback.format_exc())); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
