"""Reuse the verified idle-GPU scheduler for a separate fixed interaction trial."""
import argparse
from pathlib import Path
import shlex
import subprocess

import run_family_lme_alias as scheduler
from dinotool.family_mass_alias import METHODS, VIP


def launch(root, dataset, gpu, mode, shard=0):
    entry=scheduler.read_json(root/'protocol.json')['datasets'][dataset]
    folder=root/'benchmark'/dataset if mode=='benchmark' else root/'full'/dataset/('s'+str(shard))
    log=root/(mode+'_'+dataset+'_s'+str(shard)+'.log')
    session='gfmi09_'+mode+'_'+dataset+'_s'+str(shard)
    if folder.exists() or log.exists() or scheduler.alive(session) or not scheduler.idle(gpu):
        raise RuntimeError('Existing output/session or occupied GPU: '+session)
    args=[scheduler.PYTHON,'-u','scripts/eval_family_mass_alias.py','--dataset',dataset,
          '--suite-root',str(root),'--output-dir',str(folder),'--mode',mode,
          '--data-root',entry['data_root'],'--vocabulary-config',entry['vocabulary_config'],
          '--dinov3-repo',str(scheduler.TOOL/'dinov3_hub'),
          '--checkpoint-dir',str(scheduler.DATA/'ckpt/DINO'),
          '--upstream-root',str(scheduler.DATA/'third_party/VIP_official_5bd25ee')]
    if mode=='full': args+=['--num-shards','4' if dataset=='ade150' else '1','--shard-index',str(shard)]
    env=['env',f'CUDA_VISIBLE_DEVICES={gpu}',
         f'PYTHONPATH={scheduler.TOOL}:{scheduler.TOOL}/scripts:{scheduler.THIRD}',
         'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2',
         'TOKENIZERS_PARALLELISM=false']
    shell='cd '+shlex.quote(str(scheduler.TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    return dict(gpu=gpu,session=session,output=str(folder),log=str(log),
                shard=shard,mode=mode,dataset=dataset)


if __name__=='__main__':
    scheduler.launch=launch
    scheduler.FULL_METHODS=(*METHODS,VIP)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    scheduler.main(p.parse_args().root)
