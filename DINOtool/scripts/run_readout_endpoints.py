"""Coverage and additive per-class transport transition verification."""
import argparse
from pathlib import Path
import numpy as np
import run_evidence_envelope as runner
from run_semantic_mass_wide import verify_dataset as verify_reference
from eval_readout_endpoints import METHODS
from dinotool.readout_diagnostics import FIELDS


def verify_dataset(root,d,entry):
    path=verify_reference(root,d,entry)
    row=runner.queue.read_json(Path(path))
    count=len(row['signature']['classes'][d])
    audit={name:np.zeros((count,len(FIELDS)),np.int64) for name in ('LocalEndpoint','WideEndpoint')}
    for i in range(entry['shards']):
        shard=runner.queue.read_json(root/'full'/d/('s'+str(i))/'results.json')
        for name in audit:audit[name]+=np.asarray(shard['readout_transitions'][name],dtype=np.int64)
    frozen=np.asarray(row['metrics'][d]['Frozen']['confusion_matrix'])
    for name,values in audit.items():
        cm=np.asarray(row['metrics'][d][name]['confusion_matrix'])
        if not np.array_equal(values[:,0],cm.sum(1)) or not np.array_equal(values[:,5],cm.diagonal()) or not np.array_equal(values[:,6],frozen.diagonal()):
            raise RuntimeError('Transport audit/confusion mismatch.')
        if not np.array_equal(values[:,1],values[:,2:5].sum(1)) or not np.array_equal(values[:,2]-values[:,3],values[:,6]-values[:,5]):
            raise RuntimeError('Transport partition mismatch.')
    row['readout_transition_fields']=FIELDS
    row['readout_transitions']={k:v.tolist() for k,v in audit.items()}
    row['readout_transition_counts_verified']=True
    runner.queue.save(Path(path),row)
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    runner.EVALUATOR='scripts/eval_readout_endpoints.py';runner.PREFIX='grep07';runner.METHODS=METHODS
    # The shared verifier checks all selected methods, not just the first study.
    import run_semantic_mass_wide as reference
    reference.METHODS=METHODS
    runner.queue.METHODS=METHODS;runner.queue.launch=runner.launch
    runner.queue.verify_job=runner.verify_job;runner.queue.verify_dataset=verify_dataset
    runner.queue.main(p.parse_args().root)
