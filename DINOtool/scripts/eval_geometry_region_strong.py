"""Evaluate one complete frozen Geometry-supported strong-observer candidate."""
import json

import torch
import torch.nn.functional as F

from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.geometry_region_strong import (
    BASELINES, IMPLEMENTATION, METHODS, PRIMARY, SOURCE,
    FrozenStrongRegionObserver, StrongRegionConfig)
from dinotool.matched_readout_controls import run_head
from dinotool.region_semantic_readout import restricted_pool
from eval_frozen_semantic_path import main as evaluate
from eval_geometry_semantic_innovation import parse_args


def factory(geometry, banks):
    observer = FrozenStrongRegionObserver(SOURCE, banks, geometry.device)
    replay_checked = False

    @torch.inference_mode()
    def reader(geometry, prepared, banks, texts, valid, rgb):
        nonlocal replay_checked
        if not replay_checked:
            hidden, descriptor = observer.visual(rgb)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                replay = restricted_pool(observer.model.vision_model.head, hidden,
                                         torch.ones(hidden.shape[:2], device=hidden.device))
            error = float((F.normalize(replay.float(), dim=-1)-descriptor).abs().max())
            if error > 3e-3:
                raise ValueError(f"Actual trained full-support pooling differs: {error}")
            reader.pool_replay_error = error
            replay_checked = True
        scores, diagnostics = observer(geometry, prepared, banks, texts, valid, rgb)
        diagnostics["trained_pool_replay_max_error"] = reader.pool_replay_error
        head = geometry.backbone.model.visual_model.head
        with geometry.backbone._autocast():
            features = {method: run_head(head, prepared.backbone_tokens,
                prepared.backbone_tokens[:, prepared.prefix_tokens:], prepared.geometry_patch_conditional,
                prepared.prefix_tokens, method, prepared.block_index)[0] for method in BASELINES[1:]}
        for key, bank in banks.items():
            for method, feature in features.items():
                scores[key][method] = alias_class_scores(feature.float() @ texts[key].T,
                                                        bank.parent_indices, bank.class_count)
        return scores, diagnostics
    return reader


if __name__ == "__main__":
    settings = StrongRegionConfig().signature()
    settings["semantic_source"] = json.loads((SOURCE/"region_source_manifest.json").read_text())
    evaluate(parse_args(), methods=METHODS, implementation=IMPLEMENTATION, primary=PRIMARY,
        reader_factory=factory, reader_uses_rgb=True, save_per_image=True, readout_settings=settings,
        signature_note="Frozen two-encoder full candidate, no alias deletion or target-label fitting. "
            "Same-information regional fusion control; unchanged original Geometry. "
            "Borrowed SigLIP2 observation, not an original backbone. "
            "SCLIP_Two/VIPProxy_Two are matched operator controls, not complete official systems. "
            "All eight datasets are exploratory development, not untouched validation.")
