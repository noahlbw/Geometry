"""Opt-in equivalent RS backend; retained model and evaluator remain untouched."""
from types import FunctionType

import torch

from dinotool.rival_alias_fast import retained_scores_fast
from eval_rival_fine_full import predict_tile as reference_tile, predict_image as reference_image


@torch.inference_mode()
def predict_tile_fast(*args, **kwargs):
    original = reference_tile.__wrapped__
    scope = dict(original.__globals__, retained_scores=retained_scores_fast)
    return FunctionType(original.__code__, scope, original.__name__, original.__defaults__)(*args, **kwargs)


@torch.inference_mode()
def predict_image_fast(*args, **kwargs):
    original = reference_image.__wrapped__
    scope = dict(original.__globals__, predict_tile=predict_tile_fast)
    return FunctionType(original.__code__, scope, original.__name__, original.__defaults__)(*args, **kwargs)


if __name__ == '__main__':
    import eval_rival_fine_full as reference
    reference.predict_image = predict_image_fast
    reference.main(reference.parse_args())
