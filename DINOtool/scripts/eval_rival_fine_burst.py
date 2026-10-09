"""Opt-in burst execution; every original fine crop and rule is preserved."""
import torch

from dinotool.fine_observer_burst import IMPLEMENTATION, FineObserverBurstGraph
from dinotool.rival_alias_fast import retained_scores_fast
from eval_rival_fine_graph import bind
import eval_rival_fine_full as reference


def burst_features(observer, crops):
    graph = getattr(observer, '_frozen_fine_burst_graph', None)
    if graph is None:
        graph = FineObserverBurstGraph(observer)
        observer._frozen_fine_burst_graph = graph
    return graph(observer, crops)


def observe_burst(image, observer, queries, coordinates, valid):
    return reference.observe_fine(image, observer, queries, coordinates, valid, feature_batch=burst_features)


@torch.inference_mode()
def predict_tile_burst(*args, **kwargs):
    return bind(reference.predict_tile, observe_fine=observe_burst, retained_scores=retained_scores_fast)(*args, **kwargs)


@torch.inference_mode()
def predict_image_burst(*args, **kwargs):
    return bind(reference.predict_image, predict_tile=predict_tile_burst)(*args, **kwargs)


def save(path, result):
    reference.save(path, dict(result, execution_backend={
        'implementation': IMPLEMENTATION, 'reader': 'cached-equivalent',
        'fine_observer': 'one burst replay of original batch-one512 forwards; all-crop finite validation',
        'observations_and_model_rules_unchanged': True}))


if __name__ == '__main__':
    with torch.inference_mode():
        bind(reference.main, predict_image=predict_image_burst, save=save)(reference.parse_args())
