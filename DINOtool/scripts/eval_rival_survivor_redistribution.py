"""Frozen survivor-mass and canonical-prior-preserving word allocation."""
from functools import partial

import torch

from dinotool.rival_matched_support import semantic_rivals
from dinotool.rival_survivor_redistribution import IMPLEMENTATION, METHODS, PRIMARY, redistribution_scores
from eval_rival_fine_graph import bind
from eval_geometry_semantic_innovation import parse_args
import eval_rival_fine_support as reference


def context_factory(variants):
    contexts = {}
    for scenario, (_, queries) in variants.items():
        contexts[scenario] = {}
        for protocol, query in queries.items():
            members = torch.stack([(query.parents == c).nonzero().flatten() for c in range(len(query.class_names))])
            matches = semantic_rivals(query.features, members)
            contexts[scenario][protocol] = partial(redistribution_scores, matches=matches)
    return contexts


if __name__ == '__main__':
    bind(reference.main, IMPLEMENTATION=IMPLEMENTATION, METHODS=METHODS, PRIMARY=PRIMARY,
         REPLAY=reference.REPLAY)(parse_args(), context_factory=context_factory)
