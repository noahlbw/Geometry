"""Fixed semantic-unit re-encoding and detail-preserving Geometry coupling."""
import torch

from dinotool.geometry_unit_readout import IMPLEMENTATION, METHODS, PRIMARY, UnitConfig, read_unit_geometry
from dinotool.geometry_readout_trace import alias_class_scores
from eval_frozen_semantic_path import main as evaluate
from eval_geometry_semantic_innovation import parse_args


def factory(geometry, banks):
    @torch.inference_mode()
    def reader(geometry, prepared, banks, texts, valid):
        features, diagnostics = read_unit_geometry(geometry, prepared, valid)
        scores = {key: {method: alias_class_scores(feature.float() @ texts[key].T,
                          bank.parent_indices, bank.class_count) for method, feature in features.items()}
                  for key, bank in banks.items()}
        for group in scores.values():
            group["MeanLogit_Unit"] = .5 * (group["Geometry"] + group["UnitOnly"])
        return scores, diagnostics
    return reader


if __name__ == "__main__":
    evaluate(parse_args(), methods=METHODS, implementation=IMPLEMENTATION, primary=PRIMARY,
        reader_factory=factory, save_per_image=True, readout_settings=UnitConfig().signature(),
        signature_note="Frozen Geometry semantic-unit re-encoding with fine descriptor detail lift. "
            "No teacher, crop, alias deletion, fitted threshold or domain routing. MST partition and "
            "coarse/detail reconstruction are established machinery. VIPProxy is not full official VIP.")
