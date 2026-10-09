#!/usr/bin/env python3
"""Train DINO-RC with the locked COCOStuff/OpenEarthMap source-only protocol."""
from __future__ import annotations

import argparse
from functools import partial

from train_cafe_ped import main, parse_args
from dinotool.cafe_relational_parallel import (
    CafeRelationalParallel,
    RelationalParallelConfig,
    relational_parallel_loss,
)


def extend_parser(parser, arm_argument):
    parser.description = __doc__
    arm_argument.choices = ("relational_parallel",)
    arm_argument.default = "relational_parallel"
    arm_argument.required = False
    # Relation heads and CAFe branches must be optimized together from update
    # one; a separate decoder-only warmup would change the comparison protocol.
    parser.set_defaults(warmup_steps=0)
    group = parser.add_argument_group("relational parallel reconstruction")
    group.add_argument("--relation-dim", type=int, default=32)
    group.add_argument("--spatial-weight", type=float, default=0.05)
    group.add_argument("--class-weight", type=float, default=0.05)
    group.add_argument("--solver-steps", type=int, default=4)
    group.add_argument("--solver-step-size", type=float, default=2.0 / 2.85)
    group.add_argument("--relation-weight", type=float, default=0.05)
    group.add_argument("--legacy-class-attention", action="store_true",
                       help="Compatibility control; by default class attention operates over queries.")


def model_config(args) -> RelationalParallelConfig:
    config = RelationalParallelConfig(
        relation_dim=args.relation_dim,
        spatial_weight=args.spatial_weight,
        class_weight=args.class_weight,
        solver_steps=args.solver_steps,
        solver_step_size=args.solver_step_size,
        relation_weight=args.relation_weight,
        corrected_class_attention=not args.legacy_class_attention,
    )
    config.validate()
    return config


def build_relational_parallel(cafe, args):
    return CafeRelationalParallel(cafe, model_config(args))


if __name__ == "__main__":
    arguments = parse_args(extend_parser=extend_parser)
    model_config(arguments)
    main(
        arguments,
        model_factory=build_relational_parallel,
        loss_function=partial(relational_parallel_loss, relation_weight=arguments.relation_weight),
        forward_kwargs_factory=lambda target: {"relation_target": target},
    )
