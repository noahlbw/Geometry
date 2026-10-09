#!/usr/bin/env python3
"""Train matched CAFe/PCA-DINO arms on the locked source-only COCO/OEM protocol."""
from __future__ import annotations

from functools import partial

from train_cafe_ped import main, parse_args
from dinotool.cafe_pca import CafePCA, PCADINOConfig, pca_loss


def extend_parser(parser, arm_argument):
    parser.description = __doc__
    arm_argument.choices = ("serial", "parallel", "pca_epl", "pca_epl_fod")
    arm_argument.default = "pca_epl_fod"
    arm_argument.required = False
    group = parser.add_argument_group("PCA-DINO")
    group.add_argument("--experts", type=int, default=4)
    group.add_argument("--reduction-ratio", type=int, default=4)
    group.add_argument("--corrected-class-attention", action="store_true",
                       help="Use explicit query-axis SDPA; this is reported as a separate implementation control.")
    group.add_argument("--fod-weight", type=float,
                       help="FOD coefficient. Defaults to 0.001 for pca_epl_fod, matching the released PCA-Seg code.")


def model_config(args) -> PCADINOConfig:
    config = PCADINOConfig(
        arm=args.arm,
        experts=args.experts,
        reduction_ratio=args.reduction_ratio,
        stages=6,
        corrected_class_attention=args.corrected_class_attention,
    )
    config.validate()
    if args.fod_weight is None:
        args.fod_weight = 0.001 if args.arm == "pca_epl_fod" else 0.0
    if args.fod_weight < 0:
        raise ValueError("fod-weight must be nonnegative")
    if args.arm == "pca_epl_fod" and args.fod_weight == 0:
        raise ValueError("pca_epl_fod requires a positive fod-weight")
    if args.arm != "pca_epl_fod" and args.fod_weight != 0:
        raise ValueError("Only pca_epl_fod may use a nonzero fod-weight")
    return config


def build_pca(cafe, args):
    return CafePCA(cafe, model_config(args))


if __name__ == "__main__":
    arguments = parse_args(extend_parser=extend_parser)
    model_config(arguments)
    main(
        arguments,
        model_factory=build_pca,
        loss_function=partial(pca_loss, fod_weight=arguments.fod_weight),
        forward_kwargs={"return_aux": arguments.arm == "pca_epl_fod"},
    )
