"""A fixed 96-image same-information audit of the already evaluated final candidate."""
import argparse

import eval_bounded_contrast_coupling as evaluator
from dinotool.bounded_coupling_controls import IMPLEMENTATION, METHODS, PROTOCOL, predict_image
from eval_bounded_local_contrast import smoke


original_inputs = evaluator.inputs


def panel_inputs(args):
    output, samples, *rest = original_inputs(args)
    if args.dataset != 'udd5':
        samples = [samples[round(i * (len(samples) - 1) / 7)] for i in range(8)]
    if len({s.key for s in samples}) != len(samples):
        raise RuntimeError('Panel sample keys must be unique.')
    return output, samples, *rest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--full-domain', action='store_true')
    parser.add_argument('--mode', choices=('smoke', 'full'), default='full')
    args = parser.parse_args()
    options = dict(predictor=predict_image, implementation=IMPLEMENTATION, view_protocol=PROTOCOL)
    if args.mode == 'smoke':
        smoke(args, methods=METHODS, **options)
    else:
        if not args.full_domain:
            evaluator.inputs = panel_inputs
        evaluator.evaluate(args, methods=METHODS, competitive=dict(alias_admission='none',
            fixed_aliases_per_class=20, panel='full domain' if args.full_domain else 'UDD5 full40; eight evenly spaced complete images/other domain',
            controls='same frozen local/wide scores; equal mean; valid-token H correspondence permutation'), **options)
