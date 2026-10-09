"""Opt-in cached reader with the unchanged batch-one fine observer on a CUDA graph."""
from types import FunctionType

import torch

from dinotool.fine_observer_execution import IMPLEMENTATION, FineObserverGraph
from dinotool.rival_alias_fast import retained_scores_fast
import eval_rival_fine_full as reference


def bind(function, **scope):
    original = getattr(function, '__wrapped__', function)
    return FunctionType(original.__code__, dict(original.__globals__, **scope),
                        original.__name__, original.__defaults__)


def graph_features(observer, rgb):
    graph = getattr(observer, '_frozen_fine_execution_graph', None)
    if graph is None:
        graph = FineObserverGraph(observer)
        observer._frozen_fine_execution_graph = graph
    return graph(observer, rgb)


@torch.inference_mode()
def predict_tile_graph(*args, **kwargs):
    observe = bind(reference.observe_fine, fine_patch_features=graph_features)
    return bind(reference.predict_tile, observe_fine=observe, retained_scores=retained_scores_fast)(*args, **kwargs)


@torch.inference_mode()
def predict_image_graph(*args, **kwargs):
    return bind(reference.predict_image, predict_tile=predict_tile_graph)(*args, **kwargs)


def graph_save(path, value):
    reference.save(path, dict(value, execution_backend={
        'implementation': IMPLEMENTATION, 'reader': 'cached-equivalent',
        'fine_observer': 'batch-one512 CUDA graph; finite-value validation retained',
        'observations_and_model_rules_unchanged': True}))


if __name__ == '__main__':
    with torch.inference_mode():
        bind(reference.main, predict_image=predict_image_graph, save=graph_save)(reference.parse_args())
