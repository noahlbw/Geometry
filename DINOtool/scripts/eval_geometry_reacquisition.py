"""Evaluate one fixed complete Geometry/backbone rereading candidate."""
import torch

from dinotool.geometry_reacquisition import (
    GeometryReacquisition, IMPLEMENTATION, METHODS, PRIMARY, ReacquisitionConfig,
)
from dinotool.geometry_readout_trace import alias_class_scores
from eval_frozen_semantic_path import main as evaluate
from eval_geometry_semantic_innovation import parse_args


def make_reader(geometry, banks):
    rereader = GeometryReacquisition(geometry.backbone)

    @torch.inference_mode()
    def reader(geometry, prepared, banks, texts, valid):
        features, diagnostics = rereader.read(prepared)
        scores = {key: {method: alias_class_scores(feature.float() @ texts[key].T,
                                      bank.parent_indices, bank.class_count)
                        for method, feature in features.items()} for key, bank in banks.items()}
        for group in scores.values():
            group["MeanLogit_Reacquired"] = .5 * (group["Geometry"] + group["Reacquired_Native"])
        return scores, diagnostics

    return reader


if __name__ == "__main__":
    evaluate(parse_args(), methods=METHODS, implementation=IMPLEMENTATION, primary=PRIMARY,
             reader_factory=make_reader, save_per_image=True,
             readout_settings=ReacquisitionConfig().signature(),
             signature_note="Frozen Geometry-conditioned real-patch backbone rereading. "
             "VIPProxy_Two and SCLIP_Two are attributed matched operator controls, not the primary. "
             "One fixed rule across all eight development datasets. Masks enter metrics after prediction.")
