"""Frozen nonlinear response transport; same source/text/view across all arms."""
import torch

from dinotool.geometry_semantic_response import IMPLEMENTATION, METHODS, PRIMARY, settings, read_responses
from dinotool.geometry_readout_trace import alias_class_scores
from eval_frozen_semantic_path import main as evaluate
from eval_geometry_semantic_innovation import parse_args


def factory(geometry, banks):
    @torch.inference_mode()
    def reader(geometry, prepared, banks, texts, valid):
        features, diagnostics = read_responses(geometry, prepared)
        scores = {}
        for key, bank in banks.items():
            scores[key] = {}
            for method, feature in features.items():
                value = alias_class_scores(feature.float() @ texts[key].T, bank.parent_indices, bank.class_count)
                scores[key][method] = value
                top = value.topk(2, dim=-1).values
                diagnostics[key+'__'+method+'__top2_margin'] = float((top[...,0]-top[...,1])[valid].mean())
        return scores, diagnostics
    return reader


if __name__ == '__main__':
    evaluate(parse_args(), methods=METHODS, implementation=IMPLEMENTATION, primary=PRIMARY,
        reader_factory=factory, save_per_image=True, readout_settings=settings(),
        signature_note='Query baseline retained; Geometry transports frozen attention-induced MLP responses. '
        'DonorBefore holds attention increments fixed to isolate response-order changes. '
        'Not post-head smoothing or attention-only transport. Same20 aliases/native view/templates. '
        'Fixed96 developed validation; labels only after prediction, no fitted parameters or first-in-literature claim.')
