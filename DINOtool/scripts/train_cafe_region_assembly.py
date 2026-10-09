#!/usr/bin/env python3
"""Train region assembly with the unchanged locked COCO/OEM source pipeline."""
from __future__ import annotations

from functools import partial

from train_cafe_ped import main, parse_args
from dinotool.cafe_region_assembly import CafeRegionAssembly, assembly_loss
from dinotool.region_assembly import RegionAssemblyConfig


def extend_parser(parser, arm_argument):
    parser.description = __doc__
    arm_argument.choices = ("flat_regions", "image_assembly", "query_assembly")
    arm_argument.default = "query_assembly"
    arm_argument.required = False
    group = parser.add_argument_group("Region assembly (legacy PED capacity flags do not apply)")
    defaults = RegionAssemblyConfig()
    for name in ("dim", "heads", "regions", "parents", "proposal_layers", "rounds", "stages", "query_chunk"):
        group.add_argument("--assembly-" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    group.add_argument("--membership-weight", type=float, default=0.1)


def model_config(args):
    config = RegionAssemblyConfig(arm=args.arm, **{
        name: getattr(args, "assembly_" + name)
        for name in ("dim", "heads", "regions", "parents", "proposal_layers", "rounds", "stages", "query_chunk")
    })
    config.validate()
    if args.membership_weight <= 0:
        raise ValueError("Use positive membership supervision for the zero-start regional decoder")
    return config


def build_assembly(cafe, args):
    return CafeRegionAssembly(cafe, model_config(args))


if __name__ == "__main__":
    arguments = parse_args(extend_parser=extend_parser)
    model_config(arguments)
    main(arguments, model_factory=build_assembly,
         loss_function=partial(assembly_loss, membership_weight=arguments.membership_weight),
         forward_kwargs={"return_aux": True})
