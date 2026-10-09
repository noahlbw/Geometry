"""Separate frozen11 scheduler; preserve outputs and recheck idle launch safely."""
import argparse
from pathlib import Path
import shlex
import subprocess
import time
import traceback

import numpy as np

from dinotool import family_presalience_readout as trial
from eval_rival_fine_full import save
from merge_gear_ov_shards import merge
from run_generation_root_alias import image_hashes
from run_region_semantic_suite_a800 import BASE as DATA,TOOL,THIRD,PYTHON,idle
from run_sat_geometry_transport_suite import alive,read_json


REPLAY={trial.CURRENT_BASE:'Current20_Base',trial.COMPLETE_BASE:'Complete20_UniformMass',
        trial.LEGACY:'Complete20_FamilySum',trial.VIP:'VIP_Complete20'}


def launch(root,dataset,gpu,mode,shard):
    folder=root/'benchmark'/dataset if mode=='benchmark' else root/'full'/dataset/('s'+str(shard))
    log=root/(mode+'_'+dataset+'_s'+str(shard)+'.log');session='gfpf09_'+mode+'_'+dataset+'_s'+str(shard)
    if folder.exists() or log.exists() or alive(session):raise RuntimeError('Existing worker; preserve it.')
    if not idle(gpu):return None
    entry=read_json(root/'protocol.json')['datasets'][dataset]
    args=[PYTHON,'-u','scripts/eval_family_presalience_alias.py','--dataset',dataset,'--suite-root',str(root),
        '--output-dir',str(folder),'--mode',mode,'--data-root',entry['data_root'],
        '--vocabulary-config',entry['vocabulary_config'],'--dinov3-repo',str(TOOL/'dinov3_hub'),
        '--checkpoint-dir',str(DATA/'ckpt/DINO'),'--upstream-root',str(DATA/'third_party/VIP_official_5bd25ee')]
    if mode=='full':args+=['--num-shards','4' if dataset=='ade150' else '1','--shard-index',str(shard)]
    env=['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
        'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
    shell='cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    return dict(dataset=dataset,mode=mode,shard=shard,gpu=gpu,session=session,output=str(folder),log=str(log))


def verify(root,protocol,dataset):
    shards=[root/'full'/dataset/('s'+str(i)) for i in range(protocol['full_shards'][dataset])]
    path=root/'full'/dataset/'merged.json'
    if path.exists():raise RuntimeError('Existing merge; preserve it.')
    row=merge([str(p) for p in shards],str(path),diagnostic_weight='images')
    ref=read_json(Path(protocol['predecessor'])/'full'/dataset/'merged.json')
    if not ref or any(ref['signature'][k]!=row['signature'][k] for k in ('sample_keys','checkpoints','classes','vocabulary')):
        raise RuntimeError('Predecessor source identity differs.')
    for name,previous in REPLAY.items():
        if row['metrics'][dataset][name]['confusion_matrix']!=ref['metrics'][dataset][previous]['confusion_matrix']:
            raise RuntimeError('Predecessor aggregate replay differs: '+name)
        expected=image_hashes([Path(p).parent/'per_image_confusions.npz' for p in ref['shards']],dataset,previous)
        actual=image_hashes([p/'per_image_confusions.npz' for p in shards],dataset,name)
        if expected!=actual:raise RuntimeError('Predecessor per-image replay differs: '+name)
    targets=[np.asarray(item['confusion_matrix']).sum(-1) for item in row['metrics'][dataset].values()]
    if any(not np.array_equal(targets[0],target) for target in targets[1:]):raise RuntimeError('Paired targets differ.')
    row.update(exact_predecessor_per_image_replayed=True,paired_scored_targets_equal=True,candidate_bank_counts_exact20=True)
    save(path,row);return row


def main(root):
    if (root/'suite_status.json').exists() or (root/'suite_results.json').exists():raise RuntimeError('Existing controller state.')
    protocol=read_json(root/'protocol.json');active={};completed=[]
    pending=[('vdd','benchmark',0),('ade150','benchmark',0)];phase='mask_free_benchmarks'
    try:
        while True:
            for gpu,job in list(active.items()):
                if alive(job['session']):continue
                folder=Path(job['output']);worker=read_json(folder/'worker_status.json')
                if not worker or worker.get('status')!='complete':
                    raise RuntimeError('Worker exited before complete: '+job['session'])
                completed.append(job);del active[gpu]
            if not pending and not active:
                if phase=='mask_free_benchmarks':
                    if any(not read_json(root/'benchmark'/d/'timing.json')['cost_gate_passed'] for d in protocol['datasets']):
                        raise RuntimeError('Cost gate rejected.')
                    phase='full_evaluation';pending=[(d,'full',i) for d,n in protocol['full_shards'].items() for i in range(n)]
                else:
                    datasets={d:dict(merged=str(root/'full'/d/'merged.json'),processed_images=verify(root,protocol,d)['processed_images']) for d in protocol['datasets']}
                    save(root/'suite_results.json',dict(status='complete',outcome='full_verified',datasets=datasets,processed_images=2080));return
            for gpu in range(8):
                if not pending:break
                if gpu in active or not idle(gpu):continue
                d,mode,shard=pending[0];job=launch(root,d,gpu,mode,shard)
                if job is not None:active[gpu]=job;pending.pop(0)
            save(root/'suite_status.json',dict(status='running',phase=phase,active=list(active.values()),pending=pending,completed=completed))
            time.sleep(5)
    except Exception:
        save(root/'suite_status.json',dict(status='failed',phase=phase,active=list(active.values()),pending=pending,
             completed=completed,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
