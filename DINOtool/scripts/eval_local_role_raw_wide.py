"""Bind local-only ablation into the already frozen paired evaluator."""
import argparse

import eval_local_role_transfer as harness
from dinotool import local_role_raw_wide as trial


if __name__=='__main__':
    harness.trial = trial
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','dinov3-repo','checkpoint-dir','upstream-root'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=('benchmark','full'),required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--sample-seed',type=int,default=20260923)
    p.add_argument('--vdd-ontology',default='official')
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--repetitions',type=int,default=5)
    harness.main(p.parse_args())
