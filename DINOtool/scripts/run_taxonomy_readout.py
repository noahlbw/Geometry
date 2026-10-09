"""Reuse verified queue/coverage checks with an isolated taxonomy evaluator."""
import argparse
from pathlib import Path
import shlex
import subprocess
import run_evidence_adaptive_readout as queue
from eval_taxonomy_readout import METHODS


def launch(root,d,entry,gpu,phase,shard=0):
    output=root/phase/d if phase=='smoke' else root/phase/d/('s'+str(shard))
    log=output.with_suffix('.log')
    session=f'gtr07_{phase}_{d}_s{shard}'
    if output.exists() or log.exists() or queue.alive(session) or not queue.idle(gpu):
        raise RuntimeError('Existing output or occupied GPU.')
    output.parent.mkdir(parents=True,exist_ok=True)
    args=[queue.PYTHON,'-u','scripts/eval_taxonomy_readout.py','--dataset',d,'--mode',phase,
        '--suite-root',str(root),'--data-root',entry['data_root'],'--output-dir',str(output),
        '--original-cache',str(queue.OLD/'text_cache'/(d+'.pt')),'--dinov3-repo',str(queue.TOOL/'dinov3_hub'),
        '--checkpoint-dir',str(queue.BASE/'ckpt/DINO'),'--upstream-root',str(queue.BASE/'third_party/VIP_official_5bd25ee'),
        '--num-shards',str(entry['shards'] if phase=='full' else 1),'--shard-index',str(shard)]
    env=['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={queue.TOOL}:{queue.TOOL}/scripts:{queue.THIRD}',
         'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
    shell='cd '+shlex.quote(str(queue.TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    return dict(dataset=d,phase=phase,shard=shard,output=str(output),log=str(log),session=session,gpu=gpu)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    queue.METHODS=METHODS
    queue.launch=launch
    queue.main(p.parse_args().root)
