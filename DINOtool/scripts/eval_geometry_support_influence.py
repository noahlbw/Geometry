#!/usr/bin/env python3
"""Full-image Geometry-grounded native semantic score influence, without VIP."""
from dataclasses import asdict
import time

import torch

from dinotool.geometry_support_erasure import IMPLEMENTATION, full_text_queries, read_support_erasure
from dinotool.support_conditioned_geometry import SupportConfig
from eval_frozen_semantic_path import main
from eval_geometry_semantic_innovation import parse_args


METHODS = ("Geometry", "ScoreGain", "ScoreInfluence")
CONFIG = SupportConfig()


def make_reader(geometry, banks):
    queries = {key: full_text_queries(geometry.backbone, bank) for key, bank in banks.items()}

    @torch.inference_mode()
    def read_scores(geometry, prepared, banks, texts, valid, rgb):
        started = time.perf_counter()
        scores, totals = {}, {}
        for key, bank in banks.items():
            output, diagnostics = read_support_erasure(geometry, prepared, bank, queries[key], rgb, valid, CONFIG)
            scores[key] = {method: output[method] for method in METHODS}
            values = {field: float(value) for field, value in diagnostics.items()
                      if isinstance(value, (int, float))}
            effective = diagnostics.get("support_effective_tokens", [])
            values["mean_effective_support_tokens"] = sum(effective)/max(len(effective), 1)
            for method in METHODS[1:]:
                values[method+"_changed_patch_fraction"] = float(
                    ((output[method].argmax(-1) != output["Geometry"].argmax(-1)) & valid).sum()/valid.sum().clamp_min(1))
            for field, value in values.items():
                totals[field] = totals.get(field, 0.)+value/len(banks)
        totals["observation_writeback_seconds"] = time.perf_counter()-started
        return scores, totals

    return read_scores


if __name__ == "__main__":
    main(parse_args(), methods=METHODS, implementation=IMPLEMENTATION, primary="ScoreInfluence",
         reader_factory=make_reader, reader_uses_rgb=True, readout_settings={
             "support": asdict(CONFIG),
             "observation": "actual native S(original RGB)-S(Geometry-support-erased RGB)",
             "operator": "average original Geometry relation rows; valid centers only; row-sum one",
             "erasure": "H/max(H); valid-window RGB channel mean; unchanged 512px FOV",
             "primary": "ScoreInfluence",
             "reconstruction": "min_delta .5||delta||^2+.5||D(H delta-Delta/f)||^2; z=g+delta",
             "normalization": "D=1/||H_row||_2; f=sum(H)/(max(H)*valid_token_count)",
             "control": "ScoreGain uses the same native observations and inverse without dividing Delta by f",
             "templates": "same six RS prompts, full trained text descriptor, all20; normalized LME .07",
             "precision": "unchanged fp32 weights and bf16 AMP",
             "protocol": "unchanged Geometry tile512/stride128, interpolation before temperature, Hann probability blend",
             "selection": "fixed image-only support rule, no labels, alias deletion, fitted strength or domain routing"})
