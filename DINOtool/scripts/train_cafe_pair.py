#!/usr/bin/env python3
"""Train DINO pair fusion using the existing source-only COCO/OEM engine."""
from __future__ import annotations

import argparse

from train_cafe_ped import main, parse_args
from dinotool.cafe_pair import CafePair, CafePairConfig


def extend_parser(parser, arm_argument):
    parser.description = __doc__
    arm_argument.choices = ("pair_fusion",)
    arm_argument.default = "pair_fusion"
    arm_argument.required = False
    # Apply --warmup-steps 0 to matched PCA controls too. This keeps every
    # trainable group active from update one without changing the old engine.
    parser.set_defaults(warmup_steps=0)
    group = parser.add_argument_group("Pair comparison fusion")
    group.add_argument("--relation-dim", type=int, default=32)
    group.add_argument("--pair-kernel", type=int, default=3)
    group.add_argument("--pair-chunk", type=int, default=16)
    group.add_argument("--recompute-pairs", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--legacy-class-attention", action="store_true",
                       help="Explicit compatibility control; default attends over the query axis.")


def model_config(args) -> CafePairConfig:
    config = CafePairConfig(relation_dim=args.relation_dim, kernel_size=args.pair_kernel,
                           pair_chunk=args.pair_chunk, recompute_pairs=args.recompute_pairs,
                           corrected_class_attention=not args.legacy_class_attention)
    config.validate()
    return config


def build_pair(cafe, args):
    return CafePair(cafe, model_config(args))


if __name__ == "__main__":
    arguments = parse_args(extend_parser=extend_parser)
    model_config(arguments)
    main(arguments, model_factory=build_pair)
