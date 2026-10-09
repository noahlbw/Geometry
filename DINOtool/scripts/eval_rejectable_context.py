#!/usr/bin/env python3
"""Evaluate the frozen rejectable bounded-context proposal on RS OVSS data."""
from __future__ import annotations

from dinotool.rejectable_context import METHODS, RejectableContextConfig, RejectableContextSegmenter
from eval_grounded_context import main, parse_args


if __name__ == "__main__":
    args = parse_args()
    if args.output_temperature != RejectableContextConfig.posterior_temperature:
        raise ValueError("The posterior and evaluator temperatures must match.")
    main(args, model_type=RejectableContextSegmenter,
         config_type=RejectableContextConfig, methods=METHODS,
         implementation="rejectable-context-v1-20260928")
