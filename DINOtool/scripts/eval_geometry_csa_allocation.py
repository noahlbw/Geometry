"""Matched Geometry/CSA group-allocation and strength study."""
import torch

from dinotool.geometry_csa_allocation import IMPLEMENTATION, METHODS, PRIMARY, settings, read_factors
from dinotool.geometry_readout_trace import alias_class_scores
from eval_frozen_semantic_path import main as evaluate
from eval_geometry_semantic_innovation import parse_args


def factory(geometry, banks):
    @torch.inference_mode()
    def reader(geometry, prepared, banks, texts, valid):
        features, diagnostics = read_factors(geometry, prepared)
        scores = {key: {method: alias_class_scores(feature.float() @ texts[key].T,
                       bank.parent_indices, bank.class_count) for method, feature in features.items()}
                  for key, bank in banks.items()}
        return scores, diagnostics
    return reader


if __name__ == "__main__":
    evaluate(parse_args(), methods=METHODS, implementation=IMPLEMENTATION, primary=PRIMARY,
        reader_factory=factory, save_per_image=True, readout_settings=settings(),
        signature_note="CSA borrowed from SCLIP. Geometry relation unchanged; group allocations and read strength isolated. "
        "New arms keep native prefix queries. Fixed264 developed panel reused; no untouched-validation or novelty claim. "
        "No target fitting or per-domain winner routing; masks only after complete prediction.")
