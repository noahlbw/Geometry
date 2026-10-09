"""Queue one frozen envelope rule across all five target protocols."""
import argparse
from pathlib import Path
import shlex
import subprocess
import numpy as np
import run_evidence_adaptive_readout as queue
from eval_evidence_envelope import METHODS

EVALUATOR='scripts/eval_evidence_envelope.py'
PREFIX='geen07'


def launch(root,d,entry,gpu,phase,shard=0):
    output=root/phase/d if phase=='smoke' else root/phase/d/('s'+str(shard))
    log=output.with_suffix('.log');session=f'{PREFIX}_{phase}_{d}_s{shard}'
    if output.exists() or log.exists() or queue.alive(session) or not queue.idle(gpu):
        raise RuntimeError('Existing output or occupied GPU.')
    output.parent.mkdir(parents=True,exist_ok=True)
    args=[queue.PYTHON,'-u',EVALUATOR,'--dataset',d,'--mode',phase,
        '--suite-root',str(root),'--data-root',entry['data_root'],'--output-dir',str(output),
        '--original-cache',str(queue.OLD/'text_cache'/(d+'.pt')),'--dinov3-repo',str(queue.TOOL/'dinov3_hub'),
        '--checkpoint-dir',str(queue.BASE/'ckpt/DINO'),'--upstream-root',str(queue.BASE/'third_party/VIP_official_5bd25ee'),
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
    if not row or row['status']!='complete' or row['target_masks_loaded'] or not (row.get('shared_visual_observations') or row.get('shared_geometry_observations')) or not row['weights_frozen'] or not row['head_weights_unchanged']:
        raise RuntimeError('Shared mask-free smoke failed.')


def verify_dataset(root,d,entry):
    path=root/'full'/d/'merged.json'
    if path.exists():
        raise RuntimeError('Existing merge.')
    row=queue.merge([str(root/'full'/d/('s'+str(i))) for i in range(entry['shards'])],str(path))
    prior=queue.read_json(Path(entry['candidate_result']))
    if not row['coverage_verified'] or row['processed_images']!=entry['total_images'] or row['signature']['global_sample_keys_sha256']!=prior['signature']['global_sample_keys_sha256']:
        raise RuntimeError('Cohort mismatch.')
    if row['signature']['checkpoints']!=prior['signature']['checkpoints'] or row['signature']['classes'][d]!=prior['signature']['classes'][d]:
        raise RuntimeError('Checkpoint/class order mismatch.')
    old=np.asarray(prior['metrics'][d]['Deployed']['confusion_matrix'])
    if not np.array_equal(old,row['metrics'][d]['Frozen']['confusion_matrix']):
        raise RuntimeError('Frozen candidate did not replay.')
    target=old.sum(1)
    for m in METHODS:
        if not np.array_equal(target,np.asarray(row['metrics'][d][m]['confusion_matrix']).sum(1)):
            raise RuntimeError('Scored target mismatch.')
    row.update(exact_frozen_confusion_replay=True,paired_scored_targets_equal=True,per_image_confusions_verified=True)
    queue.save(path,row)
    return str(path)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    queue.METHODS=METHODS;queue.launch=launch;queue.verify_job=verify_job;queue.verify_dataset=verify_dataset
    queue.main(p.parse_args().root)
