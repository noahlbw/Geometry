"""Frozen joint semantic decoder, original Geometry and same-source controls."""
import torch

from dinotool.geometry_semantic_demixing import (IMPLEMENTATION, METHODS, PRIMARY,
    DemixConfig, JointSemanticDecoder, read_semantic_demixing)
from eval_frozen_semantic_path import main as evaluate
from eval_geometry_semantic_innovation import parse_args


def factory(geometry, banks):
    decoders = {key: JointSemanticDecoder(bank.features, bank.parent_indices, bank.class_count)
                for key, bank in banks.items()}

    @torch.inference_mode()
    def reader(geometry, prepared, banks, texts, valid):
        return read_semantic_demixing(geometry, prepared, banks, texts, valid, decoders)
    return reader


if __name__ == "__main__":
    evaluate(parse_args(), methods=METHODS, implementation=IMPLEMENTATION, primary=PRIMARY,
        reader_factory=factory, save_per_image=True, readout_settings=DemixConfig().signature(),
        signature_note="Single frozen encoder, all20 aliases retained. Joint nonnegative semantic explanation, "
            "original Geometry applied to residual observations, not label diffusion. No target-label fitting, "
            "teacher, crop or domain routing. NNLS/FISTA have precedent, not a novelty claim. "
            "VIPProxy_Two is not full official VIP. All eight domains are development data.")
