"""Frozen scale-free rank source, with exact old/fine-only score replay."""
import numpy as np

from dinotool.rival_rank_density import IMPLEMENTATION, METHODS, PRIMARY, FINE_ONLY, OLD_DENSITY, rank_scores
from eval_rival_fine_graph import bind
from eval_geometry_semantic_innovation import parse_args
from run_region_semantic_suite_a800 import TOOL
import eval_rival_fine_support as reference


PRIOR = TOOL/'results/rival_conditional_density_20261005'


def checked_reader(dataset, scenario, protocol):
    index = 0

    def reader(*inputs, methods=METHODS):
        nonlocal index
        values, details = rank_scores(*inputs, methods=methods)
        if methods == METHODS:
            with np.load(PRIOR/dataset/'scores'/f'{index}.npz', allow_pickle=False) as old:
                for name in (FINE_ONLY, OLD_DENSITY):
                    if not np.array_equal(values[name].cpu().numpy(), old[scenario+'__'+protocol+'__'+name]):
                        raise RuntimeError('Original source score changed before masks: '+name)
            details['frozen_fine_only_scores_bitwise_exact'] = True
            details['old_density_scores_bitwise_exact'] = True
            index += 1
        return values, details

    return reader


if __name__ == '__main__':
    args = parse_args()
    factory = lambda variants: {s: {p: checked_reader(args.dataset, s, p) for p in queries}
        for s, (_, queries) in variants.items()}
    bind(reference.main, IMPLEMENTATION=IMPLEMENTATION, METHODS=METHODS, PRIMARY=PRIMARY,
         REPLAY=reference.REPLAY)(args, context_factory=factory)
