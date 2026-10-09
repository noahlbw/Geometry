"""Queue frozen deployment verification without changing prior suites."""
import argparse
from pathlib import Path
import shlex
import subprocess
import numpy as np
import run_evidence_adaptive_readout as queue
from eval_final_adaptive_deployment import METHODS

original_verify=queue.verify_dataset


def verify_dataset(root,d,entry):
    path=original_verify(root,d,entry)
    row=queue.read_json(Path(path))
    if not np.array_equal(row['metrics'][d]['ScalarFrozen']['confusion_matrix'],entry['frozen_scalar_confusion']):
        raise RuntimeError('Previously frozen scalar route did not replay: '+d)
    checks=[]
    for shard in range(entry['shards']):
        checks.extend(queue.read_json(root/'full'/d/('s'+str(shard))/'deployment_checks.json')['samples'])
    if len(checks)!=entry['total_images'] or len({r['key'] for r in checks})!=len(checks):
        raise RuntimeError('Incomplete deployment checks.')
    row.update(exact_frozen_scalar_confusion_replay=True,
        deployment_changed_pixels=sum(r['changed_pixels'] for r in checks),
        maximum_probability_error=max(r['maximum_probability_error'] for r in checks),
        maximum_changed_gap=max(r['maximum_changed_gap'] for r in checks))
    queue.save(Path(path),row)
    return path


def launch(root,d,entry,gpu,phase,shard=0):
    output=root/phase/d if phase=='smoke' else root/phase/d/('s'+str(shard))
    log=output.with_suffix('.log')
    session=f'gfad07_{phase}_{d}_s{shard}'
    if output.exists() or log.exists() or queue.alive(session) or not queue.idle(gpu):
        raise RuntimeError('Existing output or occupied GPU.')
    output.parent.mkdir(parents=True,exist_ok=True)
    args=[queue.PYTHON,'-u','scripts/eval_final_adaptive_deployment.py','--dataset',d,'--mode',phase,
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
    p.add_argument('--datasets',nargs='+',choices=queue.ORDER,default=list(queue.ORDER))
    args=p.parse_args()
    queue.ORDER=tuple(args.datasets)
    queue.METHODS=METHODS
    queue.launch=launch
    queue.verify_dataset=verify_dataset
    queue.main(args.root)
