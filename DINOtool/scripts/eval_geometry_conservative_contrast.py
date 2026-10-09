"""Frozen Geometry contrast-compensation mechanism study."""
import torch

from dinotool.geometry_conservative_contrast import IMPLEMENTATION, METHODS, PRIMARY, settings, read_factors
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
        signature_note="Contrast compensation study, not certified semantics or established novelty. "
        "Original Geometry read retained; added Value correction centered at each evolving block. "
        "Same developed264 panel; fixed20/view/text. No target fitting, per-domain routing or automatic full rollout.")
