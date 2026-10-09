"""Evaluate or time the single frozen bounded896 candidate."""
import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from dinotool.bounded_physical_coupling import IMPLEMENTATION, PRIMARY, VIEW_PROTOCOL
from dinotool.bounded_physical_coupling import predict_image as bounded_predict
from dinotool.gear_ov import _crop_at
from dinotool.geometry_execution import GeometryExecution
from eval_rival_fine_full import check_frozen, frozen_state, save
from eval_shared_rival_soft_full import benchmark, evaluate, prepare, predict_image as native_predict


def predict_image(*args, **kwargs):
    return bounded_predict(*args, local_predictor=native_predict, **kwargs)


@torch.inference_mode()
def smoke(args):
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError('Preserve existing smoke output.')
    samples, load_image, _, models, _ = prepare(args)
    geometry, banks, vip, queries = models[:4]
    state = frozen_state(geometry, vip)
    image = load_image(samples[0].image_path if args.dataset == 'loveda' else samples[0])
    rgb = _crop_at(image, 0, 0, 512).to(geometry.device)
    execution = GeometryExecution(geometry)
    actual, original = execution.prepare_image(rgb), geometry.prepare_image(rgb)
    for field in fields(original):
        a, b = getattr(actual, field.name), getattr(original, field.name)
        if isinstance(b, torch.Tensor):
            if not torch.equal(a, b):
                raise RuntimeError('Pruned Geometry field changed: ' + field.name)
        elif a != b:
            raise RuntimeError('Geometry metadata changed: ' + field.name)
    output.mkdir(parents=True)
    predictions, diagnostics = predict_image(image, execution, banks, vip, queries, output)
    singleton, _ = predict_image(image, execution, banks, vip, queries, output, methods=(PRIMARY,))
    if any(not np.array_equal(predictions[p][PRIMARY], singleton[p][PRIMARY]) for p in banks):
        raise RuntimeError('Singleton primary differs from simultaneous controls.')
    save(output / 'results.json', dict(status='complete', dataset=args.dataset,
        implementation=IMPLEMENTATION, input_protocol=VIEW_PROTOCOL, diagnostics=diagnostics,
        sample_key=samples[0].key, prepared_fields_bitwise_equal=True,
        singleton_primary_exact=True, target_masks_loaded=False, **check_frozen(state, geometry, vip)))


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
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--mode', choices=('smoke', 'full', 'benchmark'), default='smoke')
    args = parser.parse_args()
    if args.mode == 'smoke':
        smoke(args)
    elif args.mode == 'benchmark':
        benchmark(args, predictor=predict_image, implementation=IMPLEMENTATION, view_protocol=VIEW_PROTOCOL)
    else:
        evaluate(args, predictor=predict_image, implementation=IMPLEMENTATION, view_protocol=VIEW_PROTOCOL)
