import argparse
from pathlib import Path
import numpy as np
import run_evidence_envelope as runner
from eval_two_witness_correction import METHODS


def verify_dataset(root,d,entry):
    path=root/'full'/d/'merged.json'
    if path.exists():raise RuntimeError('Existing merged result.')
    row=runner.queue.merge([str(root/'full'/d/('s'+str(i))) for i in range(entry['shards'])],str(path))
    prior=runner.queue.read_json(Path(entry['candidate_result']))
    if not row['coverage_verified'] or row['processed_images']!=entry['total_images']:raise RuntimeError('Incomplete coverage.')
    for field in ('global_sample_keys_sha256','checkpoints','classes','vocabulary','residual_identity'):
        if row['signature'][field]!=prior['signature'][field]:raise RuntimeError('Identity mismatch: '+field)
    old=np.asarray(prior['metrics'][d]['LocalBackgroundUnion']['confusion_matrix'])
    if not np.array_equal(old,row['metrics'][d]['Frozen']['confusion_matrix']):raise RuntimeError('Retained candidate replay failed.')
    transitions=None
    for i in range(entry['shards']):
        r=runner.queue.read_json(root/'full'/d/('s'+str(i))/'results.json')
        t=np.asarray(r['readout_transitions']['TwoWitness'],dtype=np.int64)
        transitions=t if transitions is None else transitions+t
    if not np.array_equal(transitions[:,2]-transitions[:,3],np.diag(row['metrics'][d]['TwoWitness']['confusion_matrix'])-np.diag(old)):
        raise RuntimeError('Transition correctness mismatch.')
    for m in METHODS:
        if not np.array_equal(old.sum(1),np.asarray(row['metrics'][d][m]['confusion_matrix']).sum(1)):
            raise RuntimeError('Different scored targets.')
    row.update(exact_frozen_confusion_replay=True,paired_scored_targets_equal=True,per_image_confusions_verified=True,
        readout_transitions={'TwoWitness':transitions.tolist()})
    runner.queue.save(path,row);return str(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    runner.EVALUATOR='scripts/eval_two_witness_correction.py';runner.PREFIX='gtwc07';runner.METHODS=METHODS
    runner.queue.METHODS=METHODS;runner.queue.launch=runner.launch
    runner.queue.verify_job=runner.verify_job;runner.queue.verify_dataset=verify_dataset
    runner.queue.main(p.parse_args().root)
