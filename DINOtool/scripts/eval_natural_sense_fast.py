"""Opt-in execution backend; the frozen evaluation script remains untouched."""
import argparse
from types import FunctionType

import torch
import torch.nn.functional as F

from dinotool.inference import tile_starts
from dinotool.natural_variable_alias_fast import retained_variable_scores_fast
from dinotool.stratified_soft_alias import WideCrop
from eval_matched_contribution_alias import crop_from_features
from eval_matched_head_fov import resize_rgb
from eval_natural_sense_adaptation import predict as reference_predict, TOTALS


@torch.inference_mode()
def prepare_wide_observations(image, vip, variants):
    resized = resize_rgb(image, 448)
    height, width = resized.shape[-2:]
    crops = {(scenario, key): [] for scenario, (_, queries) in variants.items() for key in queries}
    count = torch.zeros(height, width, device=vip.device)
    for top in tile_starts(height, 336, 224):
        for left in tile_starts(width, 336, 224):
            ah, aw = min(336, height-top), min(336, width-left)
            rgb = F.pad(resized[:, top:top+ah, left:left+aw], (0, 336-aw, 0, 336-ah))
            features = vip.crop_patch_features(rgb)
            blank = WideCrop(features.new_empty(441, 1), features.new_empty(1), top, left, ah, aw)
            for scenario, (_, queries) in variants.items():
                for key, query in queries.items():
                    crops[scenario, key].append(crop_from_features(features, query, blank))
            count[top:top+ah, left:left+aw] += 1
    if not bool((count > 0).all()):
        raise ValueError('Incomplete wide observation coverage.')
    return {}, crops, {}, count


@torch.inference_mode()
def predict_fast(*args, **kwargs):
    # Reuse exactly the frozen predictor with scoped dependency substitution.
    original = reference_predict.__wrapped__
    scope = dict(original.__globals__, prepare_wide=prepare_wide_observations,
                 retained_variable_scores=retained_variable_scores_fast)
    return FunctionType(original.__code__, scope, original.__name__, original.__defaults__)(*args, **kwargs)


if __name__ == '__main__':
    import eval_natural_sense_adaptation as reference

    reference.predict = predict_fast
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('evaluate',))
    parser.add_argument('--dataset', choices=tuple(TOTALS), required=True)
    for name in ('dinov3-repo', 'checkpoint-dir', 'upstream-root', 'data-root', 'source-vocabulary',
                 'cache-dir', 'output-dir', 'source-selection', 'candidate-vocabulary', 'candidate-cache-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    args.execution_backend = 'cached-exact-v1'
    reference.main(args)
