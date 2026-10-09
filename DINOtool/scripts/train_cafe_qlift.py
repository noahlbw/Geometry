#!/usr/bin/env python3
"""Train Q-Lift-DINO with the locked COCO/OEM source protocol."""
from __future__ import annotations

from train_cafe_ped import main, parse_args
from dinotool.cafe_qlift import CafeQLift
from dinotool.qlift import QLiftConfig


def extend_parser(parser, arm_argument):
    parser.description = __doc__
    arm_argument.choices = ("skip", "haar", "image_lift", "query_lift")
    arm_argument.default = "query_lift"
    arm_argument.required = False
    group = parser.add_argument_group("Q-Lift decoder")
    defaults = QLiftConfig()
    group.add_argument("--qlift-levels", type=int, default=defaults.levels)
    group.add_argument("--qlift-guide-dim", type=int, default=defaults.guide_dim)
    group.add_argument("--qlift-hidden-dim", type=int, default=defaults.hidden_dim)
    group.add_argument("--qlift-kernel", type=int, default=defaults.kernel)
    group.add_argument("--qlift-query-chunk", type=int, default=defaults.query_chunk)
    group.add_argument("--qlift-tune-visual-blocks", type=int, default=defaults.tune_visual_blocks,
                       choices=(0, 2))


def model_config(args) -> QLiftConfig:
    config = QLiftConfig(
        arm=args.arm,
        levels=args.qlift_levels,
        guide_dim=args.qlift_guide_dim,
        hidden_dim=args.qlift_hidden_dim,
        kernel=args.qlift_kernel,
        query_chunk=args.qlift_query_chunk,
        tune_visual_blocks=args.qlift_tune_visual_blocks,
    )
    config.validate()
    return config


def build_qlift(cafe, args):
    return CafeQLift(cafe, model_config(args))


if __name__ == "__main__":
    arguments = parse_args(extend_parser=extend_parser)
    model_config(arguments)
    main(arguments, model_factory=build_qlift)
