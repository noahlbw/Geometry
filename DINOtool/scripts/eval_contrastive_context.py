#!/usr/bin/env python3
"""Evaluate pairwise contrastive context against frozen Geometry/BoundedUnion."""
from __future__ import annotations

from dinotool.contrastive_context import (
    METHODS, ContrastiveContextConfig, ContrastiveContextSegmenter,
)
from eval_grounded_context import main, parse_args


if __name__ == "__main__":
    args = parse_args()
    if args.methods == "G_all20_uniform,BoundedUnion":
        args.methods = ",".join(METHODS)
    main(args, model_type=ContrastiveContextSegmenter,
         config_type=ContrastiveContextConfig, methods=METHODS,
         implementation="contrastive-context-v1-20260929")
