"""Fixed stage-specific Geometry readout with allocation factorial controls."""
import torch

from dinotool.geometry_staged_readout import IMPLEMENTATION, METHODS, PRIMARY, settings, read_staged
from dinotool.geometry_readout_trace import alias_class_scores
from eval_frozen_semantic_path import main as evaluate
from eval_geometry_semantic_innovation import parse_args


def factory(geometry, banks):
    @torch.inference_mode()
    def reader(geometry, prepared, banks, texts, valid):
        features, diagnostics = read_staged(geometry, prepared)
        scores = {}
        for key, bank in banks.items():
            scores[key] = {}
            for method, feature in features.items():
                value = alias_class_scores(feature.float() @ texts[key].T, bank.parent_indices, bank.class_count)
                scores[key][method] = value
                margin = value.topk(2, dim=-1).values
                diagnostics[key+"__"+method+"__top2_margin"] = float((margin[..., 0]-margin[..., 1])[valid].mean())
        return scores, diagnostics
    return reader


if __name__ == "__main__":
    evaluate(parse_args(), methods=METHODS, implementation=IMPLEMENTATION, primary=PRIMARY,
        reader_factory=factory, save_per_image=True, readout_settings=settings(),
        signature_note="Original Geometry relation with context-preserving block0 and patch-only unit-mass block1. "
        "No sparse VIP relation or normalized-prefix increment. Same-source reversed stage and allocation factorial controls. "
        "Published dense-readout principles are attributed; staging is not claimed as a novel attention primitive. "
        "Fixed96 developed validation, no fitted parameter or target labels in inference.")
