#!/usr/bin/env python3
"""Train the source-only PSDR addressing probe under the locked CAFe recipe."""
from __future__ import annotations

from functools import partial

from train_cafe_ped import main, parse_args
from dinotool.cafe_psdr import CafePSDRProbe, psdr_probe_loss
from dinotool.pixel_query_evidence import PixelQueryEvidenceConfig


def extend_parser(parser, arm_argument):
    parser.description = __doc__
    arm_argument.choices = ("psdr_probe",)
    arm_argument.default = "psdr_probe"
    arm_argument.required = False
    defaults = PixelQueryEvidenceConfig()
    group = parser.add_argument_group("PSDR mechanism probe")
    group.add_argument("--address-mode", choices=("image_only", "query_only", "pixel_only", "pixel_query"), default=defaults.address_mode)
    group.add_argument("--content-mode", choices=("region_mean", "member_geometry", "flat_attention"), default=defaults.content_mode)
    group.add_argument("--evidence-heads", type=int, default=defaults.heads)
    group.add_argument("--evidence-grids", type=int, nargs="+", default=defaults.grids)
    group.add_argument("--regions-per-read", type=int, default=defaults.regions_per_read)
    group.add_argument("--samples-per-region", type=int, default=defaults.samples_per_region)
    group.add_argument("--evidence-rounds", type=int, default=defaults.rounds)
    group.add_argument("--evidence-query-chunk", type=int, default=defaults.query_chunk)
    group.add_argument("--evidence-visual-blocks", type=int, default=defaults.visual_blocks)
    group.add_argument("--utility-weight", type=float, default=0.0)


def model_config(args) -> PixelQueryEvidenceConfig:
    config = PixelQueryEvidenceConfig(address_mode=args.address_mode, content_mode=args.content_mode,
        dim=args.evidence_dim, heads=args.evidence_heads, grids=tuple(args.evidence_grids),
        regions_per_read=args.regions_per_read, samples_per_region=args.samples_per_region,
        rounds=args.evidence_rounds, query_chunk=args.evidence_query_chunk, visual_blocks=args.evidence_visual_blocks)
    config.validate()
    if args.utility_weight < 0:
        raise ValueError("utility-weight must be nonnegative")
    return config


def build(cafe, args):
    return CafePSDRProbe(cafe, model_config(args))


if __name__ == "__main__":
    args = parse_args(extend_parser=extend_parser)
    model_config(args)
    main(args, model_factory=build, loss_function=partial(psdr_probe_loss, utility_weight=args.utility_weight),
         forward_kwargs={"return_utility": args.utility_weight > 0})
