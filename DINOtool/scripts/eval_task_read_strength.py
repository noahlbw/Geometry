import argparse
import eval_evidence_envelope as pair
from dinotool.task_read_strength import IMPLEMENTATION

METHODS=('Frozen','NaturalStrength3')
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Task prior: strength3 replaces natural strength2 only.')
    for name in ('dataset','data-root','suite-root','output-dir','original-cache','dinov3-repo','checkpoint-dir','upstream-root'):p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=('smoke','full'),required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20260923);p.add_argument('--vdd-ontology',default='official')
    pair.METHODS=METHODS;pair.IMPLEMENTATION=IMPLEMENTATION
    pair.ARM_OPTIONS=({'read_policy':'retained'},{'read_policy':'natural_strength3'})
    pair.main(p.parse_args())
