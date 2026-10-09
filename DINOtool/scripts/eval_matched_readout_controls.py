"""Full-set matched readout controls with per-image confusion matrices."""
from __future__ import annotations

import torch

from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.matched_readout_controls import IMPLEMENTATION, METHODS, OPERATOR_NOTES, run_head
from eval_frozen_semantic_path import main as evaluate
from eval_geometry_semantic_innovation import parse_args


def make_reader(geometry, banks):
    @torch.inference_mode()
    def reader(geometry, prepared, banks, texts, valid):
        head = geometry.backbone.model.visual_model.head
        features = {"Geometry": prepared.geometry_projected, "Native": prepared.native_projected}
        diagnostics = {"geometry_replay_max_error": 0., "proxy_empty_rows": 0., "proxy_rows": 0.}
        with geometry.backbone._autocast():
            for method in METHODS:
                if method in features:
                    continue
                current, observed = run_head(head, prepared.backbone_tokens,
                                            prepared.backbone_tokens[:, prepared.prefix_tokens:],
                                            prepared.geometry_patch_conditional, prepared.prefix_tokens,
                                            method, prepared.block_index)
                features[method] = current
                for field, value in observed.items():
                    diagnostics[field] += value
        if any(not bool(torch.isfinite(feature).all()) for feature in features.values()):
            raise ValueError("Nonfinite matched readout features.")
        return {key: {method: alias_class_scores(feature.float() @ texts[key].T,
                     bank.parent_indices, bank.class_count) for method, feature in features.items()}
                for key, bank in banks.items()}, diagnostics
    return reader


if __name__ == "__main__":
    args = parse_args()
    evaluate(args, methods=METHODS, implementation=IMPLEMENTATION, primary="Geometry",
             reader_factory=make_reader, save_per_image=True, readout_settings={
                 "precision": "FP32 frozen weights and common bf16 AMP",
                 "text": "same fixed20 aliases/six RS templates/normalized LME .07",
                 "view": "native-resolution512 tiles/128 overlap/Hann probability blending",
                 "operators": OPERATOR_NOTES,
                 "note": "DINO.text operator adaptations; not complete official CLIP systems. "
                         "No target-label parameter selection; author constants retained."})
