"""Idle-GPU queue; preserve current task-candidate confusion exactly."""
import argparse
from pathlib import Path
import numpy as np
import run_evidence_envelope as runner
from eval_semantic_mass_wide import METHODS


def verify_dataset(root,d,entry):
    path=root/'full'/d/'merged.json'
    if path.exists():raise RuntimeError('Existing merge.')
    row=runner.queue.merge([str(root/'full'/d/('s'+str(i))) for i in range(entry['shards'])],str(path))
    prior=runner.queue.read_json(Path(entry['candidate_result']))
    if not row['coverage_verified'] or row['processed_images']!=entry['total_images']:
        raise RuntimeError('Incomplete unique coverage.')
    for field in ('global_sample_keys_sha256','checkpoints','classes'):
        if row['signature'][field]!=prior['signature'][field]:raise RuntimeError('Identity mismatch: '+field)
    old=np.asarray(prior['metrics'][d]['NaturalShortEdge']['confusion_matrix'])
    if not np.array_equal(old,row['metrics'][d]['Frozen']['confusion_matrix']):
        raise RuntimeError('Selected task candidate did not replay.')
    for m in METHODS:
        if not np.array_equal(old.sum(1),np.asarray(row['metrics'][d][m]['confusion_matrix']).sum(1)):
            raise RuntimeError('Scored target mismatch.')
    row.update(exact_frozen_confusion_replay=True,paired_scored_targets_equal=True,per_image_confusions_verified=True)
    runner.queue.save(path,row)
    return str(path)


def verify_job(job):
    runner.verify_job(job)
    if job['phase']=='smoke':
        row=runner.queue.read_json(Path(job['output'])/'results.json')
        if not row.get('singleton_probability_replay'):raise RuntimeError('Singleton smoke absent.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    runner.EVALUATOR='scripts/eval_semantic_mass_wide.py';runner.PREFIX='gsmw07';runner.METHODS=METHODS
    runner.queue.METHODS=METHODS;runner.queue.launch=runner.launch
    runner.queue.verify_job=verify_job;runner.queue.verify_dataset=verify_dataset
    runner.queue.main(p.parse_args().root)
