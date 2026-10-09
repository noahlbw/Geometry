"""Frozen pair-conditional cross-view alias responsibility intersection."""
from functools import partial

import torch

from dinotool.rival_matched_support import semantic_rivals
from dinotool.rival_responsibility_intersection import IMPLEMENTATION, METHODS, PRIMARY, intersection_scores
from eval_rival_fine_graph import bind
from eval_geometry_semantic_innovation import parse_args
import eval_rival_fine_support as reference


def context_factory(variants):
    contexts = {}
    for scenario, (_, queries) in variants.items():
        contexts[scenario] = {}
        for protocol, query in queries.items():
            members = torch.stack([(query.parents == c).nonzero().flatten() for c in range(len(query.class_names))])
            contexts[scenario][protocol] = partial(intersection_scores, matches=semantic_rivals(query.features, members))
    return contexts


if __name__ == '__main__':
    bind(reference.main, IMPLEMENTATION=IMPLEMENTATION, METHODS=METHODS, PRIMARY=PRIMARY,
         REPLAY=reference.REPLAY)(parse_args(), context_factory=context_factory)
