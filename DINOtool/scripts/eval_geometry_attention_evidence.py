#!/usr/bin/env python3
"""Fixed attention-only semantic observation and anchored Geometry writeback."""
import torch

from dinotool.geometry_attention_evidence import IMPLEMENTATION, METHODS, read_attention_evidence
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.geometry_semantic_innovation import anchored_innovation
from eval_frozen_semantic_path import main
from eval_geometry_semantic_innovation import parse_args


def make_reader(geometry, banks):
    @torch.inference_mode()
    def read_scores(geometry, prepared, banks, texts, valid):
        with geometry.backbone._autocast():
            features, diagnostics = read_attention_evidence(geometry.backbone.model.visual_model.head, prepared)
        error = float((features["Native"]-prepared.native_projected).abs().max())
        if error != 0.:
            raise ValueError(f"Native donor replay differs: {error}")
        diagnostics["native_replay_max_error"] = error
        rows = {}
        corrections, residuals = [], []
        for key, bank in banks.items():
            local = alias_class_scores(prepared.geometry_projected.float() @ texts[key].T,
                                       bank.parent_indices, bank.class_count)
            scores = {"Geometry": local}
            for method, feature in features.items():
                observed = alias_class_scores(feature.float() @ texts[key].T,
                                              bank.parent_indices, bank.class_count)
                scores[method] = torch.where(valid[..., None], observed, local)
            scores["MeanLogitEvidence"] = .5*(local+scores["EvidenceGeometry"])
            anchored, current = anchored_innovation(local/.07, scores["EvidenceGeometry"]/.07,
                                                    prepared.geometry_patch_conditional, valid)
            scores["AnchoredEvidence"] = .07*anchored
            corrections.append(current["mean_absolute_innovation"])
            residuals.append(current["solver_relative_residual"])
            rows[key] = scores
        diagnostics["anchored_mean_absolute_innovation"] = sum(corrections)/len(corrections)
        diagnostics["anchored_solver_relative_residual"] = max(residuals)
        return rows, diagnostics

    return read_scores


if __name__ == "__main__":
    main(parse_args(), methods=METHODS, implementation=IMPLEMENTATION, primary="AnchoredEvidence",
         reader_factory=make_reader, readout_settings={
             "observation": "sum_l projected Geometry-attention(native_l), without explicit residual or MLP",
             "donors": "exact unedited native Q/K/V and MLP states at both frozen head blocks",
             "normalization": "unchanged learned final LayerNorm and projection",
             "anchor": "min_z .5||z-g||^2+.5||A(z-b_attention)||^2; fixed unit weights",
             "templates": "same six RS prompts and all20 alias strings; normalized LME .07",
             "precision": "unchanged fp32 weights and bf16 AMP; no extra crops or model",
             "writeback": "same anchored rule for all classes/domains; no fitted gate or density calibration"})
