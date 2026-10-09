"""Run only the missing PC60 official comparator on idle GPUs."""
import argparse
from pathlib import Path
import shlex
import subprocess
import numpy as np
import run_evidence_adaptive_readout as queue
from eval_context60_official_finite import METHODS


def launch(root,d,entry,gpu,phase,shard=0):
    output=root/phase/d if phase=='smoke' else root/phase/d/('s'+str(shard))
    log=output.with_suffix('.log')
    session=f'gpcvip07_{phase}_s{shard}'
    if output.exists() or log.exists() or queue.alive(session) or not queue.idle(gpu):
        raise RuntimeError('Existing output or occupied GPU.')
    output.parent.mkdir(parents=True,exist_ok=True)
    args=[queue.PYTHON,'-u','scripts/eval_context60_official_finite.py','--mode',phase,
        '--suite-root',str(root),'--data-root',entry['data_root'],'--output-dir',str(output),
        '--dinov3-repo',str(queue.TOOL/'dinov3_hub'),'--checkpoint-dir',str(queue.BASE/'ckpt/DINO'),
        '--upstream-root',str(queue.BASE/'third_party/VIP_official_5bd25ee'),
        '--num-shards',str(entry['shards'] if phase=='full' else 1),'--shard-index',str(shard)]
    env=['env',f'CUDA_VISIBLE_DEVICES={gpu}',f'PYTHONPATH={queue.TOOL}:{queue.TOOL}/scripts:{queue.THIRD}',
        'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']
    shell='cd '+shlex.quote(str(queue.TOOL))+' && exec '+shlex.join(env+args)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,shell],check=True)
    return dict(dataset=d,phase=phase,shard=shard,output=str(output),log=str(log),session=session,gpu=gpu)


original_verify_job=queue.verify_job
def verify_job(job):
    if job['phase']!='smoke':
        return original_verify_job(job)
    row=queue.read_json(Path(job['output'])/'results.json')
    if not row or row['status']!='complete' or row['target_masks_loaded'] or not row['weights_frozen'] or not row['head_weights_unchanged']:
        raise RuntimeError('Official mask-free smoke failed.')


def verify_dataset(root,d,entry):
    path=root/'full'/d/'merged.json'
    if path.exists():
        raise RuntimeError('Existing merged output.')
    folders=[root/'full'/d/('s'+str(i)) for i in range(entry['shards'])]
    row=queue.merge([str(f) for f in folders],str(path))
    prior=queue.read_json(Path(entry['candidate_result']))
    if not row['coverage_verified'] or row['processed_images']!=5105 or row['signature']['global_sample_keys_sha256']!=prior['signature']['global_sample_keys_sha256']:
        raise RuntimeError('Candidate cohort mismatch.')
    if row['signature']['checkpoints']!=prior['signature']['checkpoints'] or row['signature']['classes'][d]!=prior['signature']['classes'][d]:
        raise RuntimeError('Checkpoint/class order mismatch.')
    target=np.asarray(prior['metrics'][d]['Deployed']['confusion_matrix']).sum(1)
    for m in METHODS:
        if not np.array_equal(target,np.asarray(row['metrics'][d][m]['confusion_matrix']).sum(1)):
            raise RuntimeError('Scored target counts mismatch.')
    row.update(candidate_cohort_verified=True,paired_scored_targets_equal=True,per_image_confusions_verified=True)
    queue.save(path,row)
    return str(path)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    queue.ORDER=('context60',)
    queue.METHODS=METHODS
    queue.launch=launch
    queue.verify_job=verify_job
    queue.verify_dataset=verify_dataset
    queue.main(p.parse_args().root)
