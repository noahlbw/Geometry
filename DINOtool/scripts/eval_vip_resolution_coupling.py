"""Default bounded-resolution Geometry+soft-alias coupling; never native tiling."""
import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from dinotool.geometry_execution import GeometryExecution
from dinotool.vip_resolution_coupling import (IMPLEMENTATION, PRIMARY, VIEW_PROTOCOL,
    crop_windows, padded_crop, predict_image, resize_for_vip)
from eval_rival_fine_full import check_frozen, frozen_state, save
from eval_shared_rival_soft_full import evaluate, prepare


@torch.inference_mode()
def smoke(args):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Preserve existing smoke output.')
    samples, load_image, _, models, _ = prepare(args)
    geometry, banks, vip, queries = models[:4]
    state = frozen_state(geometry, vip)
    image = load_image(samples[0].image_path if args.dataset == 'loveda' else samples[0])
    execution = GeometryExecution(geometry)
    resized = resize_for_vip(image)
    window = crop_windows(*resized.shape[-2:])[0]
    crop = padded_crop(resized, window)[None].to(geometry.device)
    actual, original = execution.prepare_image(crop), geometry.prepare_image(crop)
    for field in fields(original):
        a, b = getattr(actual, field.name), getattr(original, field.name)
        if isinstance(b, torch.Tensor):
            if not torch.equal(a, b):
                raise RuntimeError('336px pruned Geometry changed: ' + field.name)
        elif a != b:
            raise RuntimeError('336px Geometry metadata changed: ' + field.name)
    if (actual.grid_height, actual.grid_width) != (21, 21):
        raise RuntimeError('Geometry was not evaluated at the real336px input.')
    output.mkdir(parents=True)
    predictions, diagnostics = predict_image(image, execution, banks, vip, queries)
    singleton, _ = predict_image(image, execution, banks, vip, queries, methods=(PRIMARY,))
    for p in banks:
        if not np.array_equal(predictions[p][PRIMARY], singleton[p][PRIMARY]):
            raise RuntimeError('Singleton primary changed.')
        if any(a.shape != tuple(image.shape[-2:]) for a in predictions[p].values()):
            raise RuntimeError('Original prediction size not restored.')
        if diagnostics[p]['geometry_encodings'] > 4 or diagnostics[p]['native_resolution_encodings']:
            raise RuntimeError('Visual budget/native-resolution guard failed.')
    save(output / 'results.json', dict(status='complete', dataset=args.dataset,
        implementation=IMPLEMENTATION, input_protocol=VIEW_PROTOCOL, diagnostics=diagnostics,
        sample_key=samples[0].key, prepared_fields_bitwise_equal=True, singleton_primary_exact=True,
        target_masks_loaded=False, **check_frozen(state, geometry, vip)))
    print('Passed resized-image protocol and real336px Geometry smoke: ' + args.dataset, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config',
                 'upstream-root', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    args = parser.parse_args()
    if args.mode == 'smoke':
        smoke(args)
    else:
        evaluate(args, predictor=predict_image, implementation=IMPLEMENTATION, view_protocol=VIEW_PROTOCOL)
