#!/usr/bin/env python3
"""Evaluate bounded-anchor pairwise evidence reconstruction."""
from __future__ import annotations

from dinotool.contrastive_context import ContrastiveContextConfig
from dinotool.contrastive_context_v2 import METHODS, PairwiseReconciliationSegmenter
from eval_grounded_context import main, parse_args


if __name__ == "__main__":
    args = parse_args()
    if args.methods == "G_all20_uniform,BoundedUnion":
        args.methods = ",".join(METHODS)
    main(args, model_type=PairwiseReconciliationSegmenter,
         config_type=ContrastiveContextConfig, methods=METHODS,
         implementation="contrastive-context-v2-20260929")
