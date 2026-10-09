"""Run the frozen class-mass interaction through the verified paired harness.

Only this process's experiment bindings change. The earlier evaluator sources,
model/readout policies, output roots, and finished results remain untouched.
"""
import argparse

import eval_family_lme_alias as harness
from dinotool import family_mass_alias as trial


def bind_contract():
    for name in ('IMPLEMENTATION', 'METHODS', 'CURRENT_BASE', 'COMPLETE_BASE',
                 'PRIMARY', 'SHUFFLE', 'VIP', 'prepare_plan', 'predict'):
        setattr(harness, name, getattr(trial, name))
    harness.FULL_METHODS = (*trial.METHODS, trial.VIP)


if __name__ == '__main__':
    bind_contract()
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','dinov3-repo',
                 'checkpoint-dir','upstream-root','vocabulary-config'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--mode', choices=('benchmark','full'), required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--sample-seed', type=int, default=20260923)
    p.add_argument('--vdd-ontology', default='official')
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--repetitions', type=int, default=5)
    harness.main(p.parse_args())
