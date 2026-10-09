import torch

from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.matched_neighborhood_controls import IMPLEMENTATION, METHODS, OPERATOR_NOTES, run_neighborhood
from eval_frozen_semantic_path import main
from eval_geometry_semantic_innovation import parse_args


def make_reader(geometry, banks):
    @torch.inference_mode()
    def reader(geometry, prepared, banks, texts, valid):
        features = {"Geometry": prepared.geometry_projected}
        with geometry.backbone._autocast():
            for method in METHODS[1:]:
                features[method] = run_neighborhood(geometry.backbone.model.visual_model.head, prepared, method)
        if any(not bool(torch.isfinite(value).all()) for value in features.values()):
            raise ValueError("Nonfinite locality control")
        return {key: {method: alias_class_scores(feature @ texts[key].T, bank.parent_indices, bank.class_count)
                for method, feature in features.items()} for key, bank in banks.items()}, {}
    return reader


if __name__ == "__main__":
    main(parse_args(), methods=METHODS, implementation=IMPLEMENTATION, primary="Geometry",
         reader_factory=make_reader, save_per_image=True, readout_settings={
             "operators": OPERATOR_NOTES, "precision": "FP32 weights, common bf16 AMP",
             "text": "same20 aliases/six RS templates/normalized LME .07",
             "view": "same512/128 native-resolution windows and Hann probability blending",
             "note": "NACLIP is a DINO.text operator adaptation, not complete official CLIP/PAMR reproduction."})
