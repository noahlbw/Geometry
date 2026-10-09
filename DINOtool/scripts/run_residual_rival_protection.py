"""Reuse frozen PC60 queue checks with isolated protected-residual workers."""
import argparse
from pathlib import Path
import shlex
import subprocess
import numpy as np
import run_residual_ontology as queue
from eval_residual_rival_protection import METHODS


def launch(root,entry,gpu,phase,shard):
    phase='smoke' if phase=='prepare' else phase
    output=root/phase/('s'+str(shard));log=output.with_suffix('.log')
    session=f'grrp07_{phase}_s{shard}'
    if output.exists() or log.exists() or queue.alive(session) or not queue.idle(gpu):
        raise RuntimeError('Existing output/session or occupied GPU.')
    output.parent.mkdir(parents=True,exist_ok=True)
    args=[queue.PYTHON,'-u','scripts/eval_residual_rival_protection.py','--mode',phase,'--suite-root',str(root),
        '--data-root',entry['data_root'],'--output-dir',str(output),'--original-cache',str(queue.OLD/'text_cache/context60.pt'),
        '--dinov3-repo',str(queue.TOOL/'dinov3_hub'),'--checkpoint-dir',str(queue.BASE/'ckpt/DINO'),
        '--upstream-root',str(queue.BASE/'third_party/VIP_official_5bd25ee'),
        '--num-shards',str(entry['shards'] if phase=='full' else 1),'--shard-index',str(shard)]
    env=['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={queue.TOOL}:{queue.TOOL}/scripts:{queue.THIRD}',
        'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
    shell='cd '+shlex.quote(str(queue.TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    return dict(output=str(output),log=str(log),session=session,phase=phase,shard=shard,gpu=gpu)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    root=parser.parse_args().root
    queue.METHODS=METHODS;queue.launch=launch
    queue.main(root)
    row=queue.read_json(root/'merged.json')
    if row:
        old=queue.read_json(queue.TOOL/'results/residual_ontology_context60_20261007/merged.json')
        if not np.array_equal(row['metrics']['context60']['ResidualMax']['confusion_matrix'],old['metrics']['context60']['ResidualMax']['confusion_matrix']):
            raise RuntimeError('Unchanged residual-max arm failed exact replay.')
        row['exact_residual_max_confusion_replay']=True
        queue.save(root/'merged.json',row)
