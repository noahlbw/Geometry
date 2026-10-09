"""Frozen Geometry/attention-profile coupling with unchanged all20 evaluation."""
import torch

from dinotool.geometry_role_compatibility import IMPLEMENTATION, METHODS, PRIMARY, RoleConfig, read_role_compatibility
from dinotool.geometry_readout_trace import alias_class_scores
from eval_frozen_semantic_path import main as evaluate
from eval_geometry_semantic_innovation import parse_args


def factory(geometry, banks):
    @torch.inference_mode()
    def reader(geometry, prepared, banks, texts, valid):
        features, diagnostics = read_role_compatibility(geometry, prepared, valid)
        scores = {key: {method: alias_class_scores(feature.float() @ texts[key].T,
                          bank.parent_indices, bank.class_count) for method, feature in features.items()}
                  for key, bank in banks.items()}
        for group in scores.values():
            group["MeanLogit_RoleCompatible"] = .5*(group["Geometry"]+group[PRIMARY])
        return scores, diagnostics
    return reader


if __name__ == "__main__":
    evaluate(parse_args(), methods=METHODS, implementation=IMPLEMENTATION, primary=PRIMARY,
        reader_factory=factory, save_per_image=True, readout_settings=RoleConfig().signature(),
        signature_note="Frozen attention-profile/Geometry exploratory coupling. No donor/class quotas, "
            "alias deletion, external teacher, fitted gate or per-domain winner. Hellinger kernels are standard machinery. "
            "SCLIP_Two/VIPProxy_Two are matched adaptations, not complete official systems.")
