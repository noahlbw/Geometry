"""Replay the frozen fine-only endpoint and evaluate its own matched controls."""
import numpy as np

from dinotool.fine_responsibility_audit import IMPLEMENTATION, METHODS, PRIMARY, audit_scores
from eval_rival_fine_graph import bind
from eval_geometry_semantic_innovation import parse_args
from run_region_semantic_suite_a800 import TOOL
import eval_rival_fine_support as reference


PRIOR = TOOL/'results/rival_responsibility_intersection_20261005'


def checked_reader(dataset, scenario, protocol):
    index = 0

    def reader(*inputs, methods=METHODS):
        nonlocal index
        values, details = audit_scores(*inputs, methods=methods)
        if methods == METHODS:
            with np.load(PRIOR/dataset/'scores'/f'{index}.npz', allow_pickle=False) as prior:
                if not np.array_equal(values[PRIMARY].cpu().numpy(), prior[scenario+'__'+protocol+'__'+PRIMARY]):
                    raise RuntimeError('Previously frozen fine-only score changed before masks.')
            details['frozen_fine_only_scores_bitwise_exact'] = True
            index += 1
        return values, details

    return reader


def context_factory(dataset, variants):
    return {s: {p: checked_reader(dataset, s, p) for p in queries}
            for s, (_, queries) in variants.items()}


if __name__ == '__main__':
    args = parse_args()
    bind(reference.main, IMPLEMENTATION=IMPLEMENTATION, METHODS=METHODS, PRIMARY=PRIMARY,
         REPLAY=reference.REPLAY)(args, context_factory=lambda variants: context_factory(args.dataset, variants))
