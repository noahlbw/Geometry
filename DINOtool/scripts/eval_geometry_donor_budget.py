"""Fixed head-role donor budget and unchanged all20 Geometry evaluation."""
import torch

from dinotool.geometry_donor_budget import IMPLEMENTATION, METHODS, PRIMARY, DonorBudgetConfig, read_donor_budget
from dinotool.geometry_readout_trace import alias_class_scores
from eval_frozen_semantic_path import main as evaluate
from eval_geometry_semantic_innovation import parse_args


def factory(geometry, banks):
    @torch.inference_mode()
    def reader(geometry, prepared, banks, texts, valid):
        features, diagnostics = read_donor_budget(geometry, prepared, valid)
        scores = {key: {method: alias_class_scores(feature.float() @ texts[key].T,
                          bank.parent_indices, bank.class_count) for method, feature in features.items()}
                  for key, bank in banks.items()}
        for group in scores.values():
            group["MeanLogit_DonorBudget"] = .5*(group["Geometry"]+group[PRIMARY])
        return scores, diagnostics
    return reader


if __name__ == "__main__":
    evaluate(parse_args(), methods=METHODS, implementation=IMPLEMENTATION, primary=PRIMARY,
        reader_factory=factory, save_per_image=True, readout_settings=DonorBudgetConfig().signature(),
        signature_note="Frozen single-encoder exploratory head-role/Geometry coupling. "
            "Native visual donor marginals, not class quotas. Standard KL/Sinkhorn inference, no new learned weights. "
            "No label fitting, alias deletion, external teacher or per-domain winner. "
            "SCLIP_Two/VIPProxy_Two are matched adaptations, not full official systems.")
