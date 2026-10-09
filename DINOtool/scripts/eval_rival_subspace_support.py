"""Frozen rival text-span response weights on existing observed features."""
from functools import partial

import torch

from dinotool.rival_subspace_support import IMPLEMENTATION, METHODS, PRIMARY, rival_subspaces, subspace_scores
from eval_rival_fine_graph import bind
from eval_geometry_semantic_innovation import parse_args
import eval_rival_fine_support as reference


def context_factory(variants):
    contexts = {}
    for scenario, (_, queries) in variants.items():
        contexts[scenario] = {}
        for protocol, query in queries.items():
            members = torch.stack([(query.parents == c).nonzero().flatten() for c in range(len(query.class_names))])
            source = rival_subspaces(query.features, members)
            contexts[scenario][protocol] = partial(subspace_scores, source=source)
    return contexts


if __name__ == '__main__':
    bind(reference.main, IMPLEMENTATION=IMPLEMENTATION, METHODS=METHODS, PRIMARY=PRIMARY,
         REPLAY=reference.REPLAY)(parse_args(), context_factory=context_factory)
