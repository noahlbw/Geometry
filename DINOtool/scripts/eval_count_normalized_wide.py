"""Use the verified paired evaluator with one opt-in count-only correction."""
import argparse
import eval_evidence_envelope as pair
from dinotool.count_normalized_wide import IMPLEMENTATION

METHODS=('Frozen','CountNormalizedWide')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','original-cache','dinov3-repo','checkpoint-dir','upstream-root'):p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=('smoke','full'),required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20260923);p.add_argument('--vdd-ontology',default='official')
    pair.METHODS=METHODS;pair.IMPLEMENTATION=IMPLEMENTATION
    pair.ARM_OPTIONS=({'wide_aggregation':'inherited'},{'wide_aggregation':'count_normalized'})
    pair.main(p.parse_args())
