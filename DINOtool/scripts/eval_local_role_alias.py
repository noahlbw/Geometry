"""Bind fixed local-role priors into the existing paired full/timing harness."""
import argparse
import json
from pathlib import Path

import eval_family_lme_alias as harness
from dinotool import local_role_alias as trial


def bind_contract():
    for name in ('IMPLEMENTATION','METHODS','CURRENT_BASE','COMPLETE_BASE','PRIMARY','SHUFFLE','VIP','prepare_plan','predict'):
        setattr(harness, name, getattr(trial, name))
    harness.FULL_METHODS = (*trial.METHODS, trial.VIP)
    original_banks = harness.input_banks
    def input_banks(root, dataset, *args):
        banks, queries, plans, identity, cost = original_banks(root, dataset, *args)
        meta = json.loads((Path(root)/'vocabularies'/(dataset+'.json')).read_text())
        for key in banks:
            # Presence of frozen semantic_roots is generation provenance. No guesses.
            roots = meta[key].get('semantic_roots')
            plans[key] = trial.apply_recorded_roles(plans[key], banks[key], roots)
        return banks, queries, plans, identity, cost
    harness.input_banks = input_banks


if __name__ == '__main__':
    bind_contract(); p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','dinov3-repo','checkpoint-dir','upstream-root','vocabulary-config'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--mode', choices=('benchmark','full'), required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--sample-seed',type=int,default=20260923)
    p.add_argument('--vdd-ontology', default='official'); p.add_argument('--num-shards', type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0); p.add_argument('--repetitions',type=int,default=5)
    harness.main(p.parse_args())
