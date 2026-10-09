"""Isolated coverage-soft variant of the frozen taxonomy readout evaluator."""
import argparse
import eval_taxonomy_readout as evaluator
from dinotool.coverage_soft_alias import IMPLEMENTATION,coverage_scores


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','original-cache','dinov3-repo','checkpoint-dir','upstream-root'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=('smoke','full'),required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20260923)
    p.add_argument('--vdd-ontology',default='official')
    evaluator.IMPLEMENTATION=IMPLEMENTATION
    evaluator.competition_scores=coverage_scores
    evaluator.main(p.parse_args())
