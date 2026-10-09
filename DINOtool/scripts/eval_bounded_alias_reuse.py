"""Frozen eight-domain developed-panel test of bounded branch-union reuse."""
import argparse

import torch

from dinotool.bounded_alias_reuse import IMPLEMENTATION, METHODS, PRIMARY, bounded_scores
from eval_rival_fine_graph import bind
import eval_sparse_alias_reuse as reference


def tile(*args, methods=METHODS, layouts=None):
    return reference.tile(*args, methods=methods, layouts=layouts, readers={PRIMARY: bounded_scores})


def window(*args, methods=METHODS):
    return bind(reference.window, tile=tile)(*args, methods=methods)


def predict_image(*args, methods=(PRIMARY,), layouts=None):
    return reference.predict_image(*args, methods=methods, layouts=layouts, readers={PRIMARY: bounded_scores})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--repetitions', type=int, default=3)
    benchmark = bind(reference.complete_image_benchmark, PRIMARY=PRIMARY, predict_image=predict_image)
    with torch.inference_mode():
        bind(reference.main, IMPLEMENTATION=IMPLEMENTATION, METHODS=METHODS, PRIMARY=PRIMARY,
             window=window, complete_image_benchmark=benchmark)(parser.parse_args())
