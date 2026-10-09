"""Matched endpoints and transport correctness audit, not a parameter grid."""
import argparse
import json
from pathlib import Path
import numpy as np
import eval_evidence_envelope as evaluator
from dinotool.readout_diagnostics import transition_counts

METHODS=('Frozen','LocalEndpoint','WideEndpoint')


def audit_update(audit,target,results,count):
    for name in ('LocalEndpoint','WideEndpoint'):
        new=transition_counts(target,results[name],results['Frozen'],count)
        audit[name]=(np.asarray(audit.get(name,np.zeros_like(new)),dtype=np.int64)+new).tolist()
    return audit


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
    evaluator.METHODS=METHODS
    evaluator.IMPLEMENTATION='geometry-selected-task-matched-endpoint-audit-v1-20261007'
    evaluator.ARM_OPTIONS=tuple(dict(wide_policy=wide,correction=c) for c in ('frozen','local_endpoint','wide_endpoint'))
    evaluator.OBSERVATION_OPTIONS={'wide_policy':wide};evaluator.RESULT_AUDITOR=audit_update
    evaluator.main(args)
