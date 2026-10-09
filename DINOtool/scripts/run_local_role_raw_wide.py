"""Run a local-only fixed-rule ablation; replay the full unweighted baseline."""
import argparse
from pathlib import Path
import shlex
import subprocess
from types import FunctionType

import numpy as np

import run_local_role_transfer as previous
from run_region_semantic_suite_a800 import BASE as DATA, TOOL, THIRD, PYTHON, idle
from run_sat_geometry_transport_suite import alive, read_json


def launch(root,protocol,dataset,gpu,mode,shard):
    folder=root/'benchmark'/dataset if mode=='benchmark' else root/'full'/dataset/('s'+str(shard))
    log=root/(mode+'_'+dataset+'_s'+str(shard)+'.log')
    session='glrw09_'+mode+'_'+dataset+'_s'+str(shard)
    if folder.exists() or log.exists() or alive(session):
        raise RuntimeError('Existing output; preserve: '+session)
    if not idle(gpu):
        return None
    entry=protocol['datasets'][dataset]
    args=[PYTHON,'-u','scripts/eval_local_role_raw_wide.py','--dataset',dataset,'--suite-root',str(root),
        '--output-dir',str(folder),'--mode',mode,'--data-root',entry['data_root'],
        '--dinov3-repo',str(TOOL/'dinov3_hub'),'--checkpoint-dir',str(DATA/'ckpt/DINO'),
        '--upstream-root',str(DATA/'third_party/VIP_official_5bd25ee')]
    if mode=='full':
        args+=['--num-shards',str(entry['shards']),'--shard-index',str(shard)]
    env=['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={TOOL}:{TOOL}/scripts:{THIRD}',
        'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
    shell='cd '+shlex.quote(str(TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    return dict(dataset=dataset,mode=mode,shard=shard,gpu=gpu,session=session,output=str(folder),log=str(log))


def verify(root,protocol,dataset):
    row=previous.verify(root,protocol,dataset)
    reference=read_json(Path(protocol['paired_source'])/'full'/dataset/'merged.json')
    name='Same20_NoFamilyWeight'
    if row['metrics'][dataset][name]['confusion_matrix']!=reference['metrics'][dataset][name]['confusion_matrix']:
        raise RuntimeError('Full no-weight baseline aggregate replay differs.')
    for i in range(protocol['datasets'][dataset]['shards']):
        with np.load(root/'full'/dataset/('s'+str(i))/'per_image_confusions.npz',allow_pickle=False) as new:
            with np.load(Path(protocol['paired_source'])/'full'/dataset/('s'+str(i))/'per_image_confusions.npz',allow_pickle=False) as old:
                if not np.array_equal(new['sample_keys'],old['sample_keys']) or not np.array_equal(new[dataset+'__'+name],old[dataset+'__'+name]):
                    raise RuntimeError('Full no-weight baseline per-image replay differs.')
    row['no_weight_baseline_exact_full_replay']=True
    previous.save(root/'full'/dataset/'merged.json',row)
    return row


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    scope=dict(previous.main.__globals__,launch=launch,verify=verify)
    FunctionType(previous.main.__code__,scope)(p.parse_args().root)
