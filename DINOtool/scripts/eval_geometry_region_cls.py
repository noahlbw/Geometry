#!/usr/bin/env python3
"""Geometry-supported virtual CLS and full DINO.text semantic observations."""
import torch
import torch.nn.functional as F

from dinotool.geometry_region_cls import IMPLEMENTATION, METHODS, region_descriptors
from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.geometry_semantic_innovation import anchored_innovation
from dinotool.prompts import REMOTE_SENSING_TEMPLATES
from eval_frozen_semantic_path import main
from eval_geometry_semantic_innovation import parse_args


@torch.inference_mode()
def text_branches(geometry, bank):
    backbone = geometry.backbone
    prompts = [template.format(label=alias) for alias in bank.alias_names for template in REMOTE_SENSING_TEMPLATES]
    full, cls = [], []
    for start in range(0, len(prompts), 64):
        tokens = backbone.tokenize(prompts[start:start+64]).to(backbone.device)
        with backbone._autocast():
            encoded = backbone.model.encode_text(tokens, normalize=False).float()
        full.append(F.normalize(encoded, dim=-1))
        cls.append(F.normalize(encoded[:, :encoded.shape[-1]//2], dim=-1))
    return {key: F.normalize(torch.cat(values).reshape(len(bank.alias_names), len(REMOTE_SENSING_TEMPLATES), -1).mean(1), dim=-1)
            for key, values in (("RegionFull", full), ("RegionCLS", cls))}


def make_reader(geometry, banks):
    queries = {key: text_branches(geometry, bank) for key, bank in banks.items()}

    @torch.inference_mode()
    def read_scores(geometry, prepared, banks, texts, valid):
        with geometry.backbone._autocast():
            descriptors, usable, diagnostics = region_descriptors(geometry.backbone.model.visual_model.head, prepared, valid)
        rows = {}
        correction_mean = 0.
        for key, bank in banks.items():
            local = alias_class_scores(prepared.geometry_projected.float() @ texts[key].T, bank.parent_indices, bank.class_count)
            scores = {"Geometry": local}
            for method, features in descriptors.items():
                text = queries[key]["RegionCLS" if method == "RegionCLS" else "RegionFull"]
                raw = alias_class_scores(features @ text.T, bank.parent_indices, bank.class_count)
                scores[method] = torch.where(usable[..., None], raw, local)
            scores["MeanLogitRegion"] = .5*(local+scores["RegionFull"])
            anchored, current = anchored_innovation(local/.07, scores["RegionFull"]/.07,
                                                    prepared.geometry_patch_conditional, usable)
            scores["AnchoredRegion"] = .07*anchored
            correction_mean += current["mean_absolute_innovation"]
            rows[key] = scores
        diagnostics["anchored_mean_absolute_innovation"] = correction_mean/len(banks)
        return rows, diagnostics

    return read_scores


if __name__ == "__main__":
    main(parse_args(), methods=METHODS, implementation=IMPLEMENTATION, primary="AnchoredRegion",
         reader_factory=make_reader, readout_settings={
             "observation": "Geometry-conditioned virtual CLS + same-support native pooled patches; full text encoding",
             "counterfactual_control": "unconditioned native global descriptor, broadcast",
             "anchor": "min_z .5||z-g||^2+.5||A(z-b_region)||^2",
             "templates": "same six RS prompts for exact same 20 aliases; full/CLS text branches",
             "precision": "unchanged fp32 weights and bf16 AMP, one backbone, no extra crops",
             "writeback": "unit-weight anchored rule, same for all domains; no threshold or selection"})
