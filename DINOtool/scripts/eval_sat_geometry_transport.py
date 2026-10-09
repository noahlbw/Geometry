"""One frozen complete Geometry/paired aerial-feature transport candidate."""
import torch

from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.sat_geometry_transport import IMPLEMENTATION, METHODS, PRIMARY, SATGeometryTransport, SATTransportConfig
from eval_frozen_semantic_path import main as evaluate
from eval_geometry_semantic_innovation import parse_args


def factory(geometry, banks):
    transport = SATGeometryTransport(geometry.backbone)

    @torch.inference_mode()
    def reader(geometry, prepared, banks, texts, valid, rgb):
        features, diagnostics = transport.read(prepared, rgb, valid)
        scores = {key: {method: alias_class_scores(feature.float() @ texts[key].T,
                            bank.parent_indices, bank.class_count) for method, feature in features.items()}
                  for key, bank in banks.items()}
        for group in scores.values():
            group["MeanLogit_SATTransport"] = .5*(group["Geometry"]+group["SAT_UniformTransport"])
        return scores, diagnostics
    return reader


if __name__ == "__main__":
    evaluate(parse_args(), methods=METHODS, implementation=IMPLEMENTATION, primary=PRIMARY,
             reader_factory=factory, reader_uses_rgb=True, save_per_image=True,
             readout_settings=SATTransportConfig().signature(),
             signature_note="Frozen SAT/LVD paired-coordinate feature transport, original Geometry graph/head/text. "
             "Window-local orthogonal alignment states, not trained network parameters. "
             "SCLIP_Two/VIPProxy_Two are attributed matched operator controls, not complete systems. "
             "Masks enter metrics after every image prediction; one rule across development domains.")
