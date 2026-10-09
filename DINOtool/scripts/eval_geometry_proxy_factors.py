"""Matched support/weight/path transplantation; predictions precede masks."""
import torch

from dinotool.geometry_proxy_factors import IMPLEMENTATION, METHODS, PRIMARY, settings, read_factors
from dinotool.geometry_readout_trace import alias_class_scores
from eval_frozen_semantic_path import main as evaluate
from eval_geometry_semantic_innovation import parse_args


def factory(geometry, banks):
    @torch.inference_mode()
    def reader(geometry, prepared, banks, texts, valid):
        features, diagnostics = read_factors(geometry, prepared)
        scores = {}
        for key, bank in banks.items():
            scores[key] = {}
            for method, feature in features.items():
                value = alias_class_scores(feature.float() @ texts[key].T, bank.parent_indices, bank.class_count)
                scores[key][method] = value
                margin = value.topk(2, dim=-1).values
                diagnostics[key+'__'+method+'__top2_margin'] = float((margin[..., 0]-margin[..., 1])[valid].mean())
        return scores, diagnostics
    return reader


if __name__ == '__main__':
    evaluate(parse_args(), methods=METHODS, implementation=IMPLEMENTATION, primary=PRIMARY,
        reader_factory=factory, save_per_image=True, readout_settings=settings(),
        signature_note='VIP global-mean support transplanted into unchanged Geometry conditional reading. '
        'Support, relative weighting/spatial term and full VIP prefix/mass pathway are attributed controls. '
        'Same20 vocabulary/view/templates/scoring and frozen weights; developed fixed96 validation. '
        'No target labels in prediction or parameter fitting. This transplant is not claimed novel.')
