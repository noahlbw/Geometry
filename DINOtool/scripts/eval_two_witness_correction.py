"""One fixed foreground veto; backgrounds and original candidate are preserved."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import eval_evidence_envelope as evaluator
from dinotool.two_witness_correction import IMPLEMENTATION
from dinotool.readout_diagnostics import transition_counts

METHODS=('Frozen','TwoWitness')


def audit_update(audit,target,results,count):
    new=transition_counts(target,results['Frozen'],results['TwoWitness'],count)
    audit['TwoWitness']=(np.asarray(audit.get('TwoWitness',np.zeros_like(new)),dtype=np.int64)+new).tolist()
    return audit


def smoke(models,source,results):
    a,b=models['Frozen'],models['TwoWitness']
    _,_,reference=a.predict_observations(source,return_probability=True)
    saved=b.correction;b.correction='frozen'
    try:
        replay=b.predict_observations(source,return_probability=True)[2]
        if not torch.equal(reference,replay):raise RuntimeError('Candidate changed retained path.')
    finally:b.correction=saved
    return dict(exact_retained_probability_replay=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','original-cache','dinov3-repo','checkpoint-dir','upstream-root'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=('smoke','full'),required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20260923);p.add_argument('--vdd-ontology',default='official')
    args=p.parse_args()
    entry=json.loads((Path(args.suite_root)/'protocol.json').read_text())['datasets'][args.dataset]
    wide='natural_short336_cap672' if entry['family']=='natural' else 'long448'
    evaluator.METHODS=METHODS;evaluator.IMPLEMENTATION=IMPLEMENTATION
    evaluator.ARM_OPTIONS=tuple(dict(wide_policy=wide,local_background='union_max',correction=c) for c in ('frozen','two_witness'))
    evaluator.OBSERVATION_OPTIONS={'wide_policy':wide}
    evaluator.SMOKE_VALIDATOR=smoke;evaluator.RESULT_AUDITOR=audit_update
    evaluator.main(args)
