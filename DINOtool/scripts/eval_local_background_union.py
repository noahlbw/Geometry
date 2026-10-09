"""Frozen local residual-union max; foreground/wide/coupling unchanged."""
import argparse
import json
from pathlib import Path
import torch
import eval_evidence_envelope as evaluator

METHODS=('Frozen','LocalBackgroundUnion')


def validate_controls(models,source,results):
    a=models['Frozen'];b=models['LocalBackgroundUnion']
    _,_,pa=a.predict_observations(source,return_probability=True)
    _,diagnostic,pb=b.predict_observations(source,return_probability=True)
    applied=diagnostic['local_background_union_applied']
    if not applied and not torch.equal(pa,pb):raise RuntimeError('Unchanged control probability differs.')
    return dict(local_union_applied=applied,unchanged_control_probability_replay=not applied)


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
    evaluator.IMPLEMENTATION='geometry-local-background-concept-union-v1-20261007'
    evaluator.ARM_OPTIONS=({'wide_policy':wide},{'wide_policy':wide,'local_background':'union_max'})
    evaluator.OBSERVATION_OPTIONS={'wide_policy':wide};evaluator.SMOKE_VALIDATOR=validate_controls
    evaluator.main(args)
