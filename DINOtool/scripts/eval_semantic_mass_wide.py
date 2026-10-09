"""One frozen lexical-group aggregation against selected task deployment."""
import argparse
import json
from pathlib import Path
import torch
import eval_evidence_envelope as evaluator
from dinotool.semantic_mass_wide import SemanticMassWide

METHODS=('Frozen','SemanticMass')


def validate_singletons(models,source,results):
    frozen=models['Frozen']
    grouped=models['SemanticMass']
    reference=frozen.predict_observations(source,return_probability=True)[2]
    saved=grouped.semantic_readers
    grouped.semantic_readers={k:SemanticMassWide(q,grouped.background,singleton=True)
                              for k,q in grouped.queries.items()}
    try:
        replay=grouped.predict_observations(source,return_probability=True)[2]
    finally:
        grouped.semantic_readers=saved
    if not torch.equal(reference,replay):
        raise RuntimeError('Singleton real-image probabilities did not exactly replay.')
    return dict(singleton_probability_replay=True,
                semantic_group_counts={k:[len(set(g)) for g in r.groups] for k,r in saved.items()},
                frozen_alias_counts={k:[len(g) for g in r.groups] for k,r in saved.items()})


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
    evaluator.IMPLEMENTATION='geometry-frozen-lexical-semantic-mass-wide-v1-20261007'
    evaluator.ARM_OPTIONS=({'wide_policy':wide}, {'wide_policy':wide,'wide_aggregation':'semantic_mass'})
    evaluator.OBSERVATION_OPTIONS={'wide_policy':wide}
    evaluator.SMOKE_VALIDATOR=validate_singletons
    evaluator.main(args)
