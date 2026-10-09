"""Evaluate the frozen per-head Geometry budget with matched unchanged20 text."""
import torch

from dinotool.geometry_semantic_budget import IMPLEMENTATION, METHODS, PRIMARY, BudgetConfig, read_semantic_budget
from dinotool.geometry_readout_trace import alias_class_scores
from eval_frozen_semantic_path import main as evaluate
from eval_geometry_semantic_innovation import parse_args


def factory(geometry, banks):
    @torch.inference_mode()
    def reader(geometry, prepared, banks, texts, valid):
        features, diagnostics = read_semantic_budget(geometry, prepared, valid)
        scores = {}
        for key, bank in banks.items():
            scores[key] = {}
            for method, feature in features.items():
                value = alias_class_scores(feature.float() @ texts[key].T, bank.parent_indices, bank.class_count)
                scores[key][method] = value
                top = value.topk(2, dim=-1).values
                diagnostics[key+"__"+method+"__top2_margin"] = float((top[..., 0]-top[..., 1]).mean())
        return scores, diagnostics
    return reader


if __name__ == "__main__":
    evaluate(parse_args(), methods=METHODS, implementation=IMPLEMENTATION, primary=PRIMARY,
        reader_factory=factory, save_per_image=True, readout_settings=BudgetConfig().signature(),
        signature_note="Frozen per-head CSA constrained by original Geometry expected transport cost. "
        "CSA and KL projection are attributed standard machinery. Same-shell comparisons isolate patch relations. "
        "No alias changes, fitted threshold, extra encoder or enlarged view. Developed fixed96 screen, labels evaluation-only.")
