"""Fixed patch-only strength controls; no class or domain-specific selection."""
import argparse

from dinotool.bounded_patch_only import IMPLEMENTATION, METHODS, PROTOCOL, predict_image
from eval_bounded_contrast_coupling import benchmark, evaluate
from eval_bounded_local_contrast import smoke


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'dinov3-repo', 'checkpoint-dir', 'data-root', 'vocabulary-config', 'upstream-root', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample-seed', type=int, default=20260923)
    parser.add_argument('--vdd-ontology', default='official')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--mode', choices=('smoke', 'full', 'benchmark'), default='full')
    args = parser.parse_args()
    options = dict(predictor=predict_image, implementation=IMPLEMENTATION, view_protocol=PROTOCOL)
    if args.mode == 'smoke':
        smoke(args, methods=METHODS, **options)
    elif args.mode == 'full':
        evaluate(args, methods=METHODS, competitive=dict(alias_admission='none', fixed_aliases_per_class=20,
            local_readout=PROTOCOL['local_readout'], reconstruction=PROTOCOL['reconstruction']), **options)
    else:
        benchmark(args, names=(*METHODS[1:], 'VIP_All20'), **options)
