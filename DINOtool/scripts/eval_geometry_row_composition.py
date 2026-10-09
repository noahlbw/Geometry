"""Frozen native-row composition with exact matched readout controls."""
import torch

from dinotool.geometry_row_composition import IMPLEMENTATION, METHODS, PRIMARY, settings, read_compositions
from dinotool.geometry_readout_trace import alias_class_scores
from eval_frozen_semantic_path import main as evaluate
from eval_geometry_semantic_innovation import parse_args


def factory(geometry, banks):
    @torch.inference_mode()
    def reader(geometry, prepared, banks, texts, valid):
        features, diagnostics = read_compositions(geometry, prepared)
        scores = {}
        for key, bank in banks.items():
            scores[key] = {}
            for method, feature in features.items():
                value = alias_class_scores(feature.float() @ texts[key].T, bank.parent_indices, bank.class_count)
                scores[key][method] = value
                top = value.topk(2, dim=-1).values
                diagnostics[key+"__"+method+"__top2_margin"] = float((top[..., 0]-top[..., 1])[valid].mean())
        return scores, diagnostics
    return reader


if __name__ == "__main__":
    evaluate(parse_args(), methods=METHODS, implementation=IMPLEMENTATION, primary=PRIMARY,
        reader_factory=factory, save_per_image=True, readout_settings=settings(),
        signature_note="Whole native self-conditional rows composed by original Geometry. "
        "FPRead isolates numerical changes; budget/special 2x2 uses original patch conditional. "
        "Full versus GroupCompose changes donor conditional weights, not a complete three-factor factorial. "
        "UDD5 full40 developed; other domains32 excluding old8, not untouched validation. "
        "No masks before prediction; no fitted parameters or semantic-reliability interpretation of native mass.")
