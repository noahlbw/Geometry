from __future__ import annotations

import torch


def dinov3_vitl16_dinotxt_tet1280d20h24l(
    *,
    pretrained: bool = True,
    weights: str,
    backbone_weights: str,
    bpe_path_or_url: str,
    check_hash: bool = False,
):
    del check_hash
    from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l as factory

    model, tokenizer = factory(pretrained=False, bpe_path_or_url=bpe_path_or_url)
    if not pretrained:
        return model, tokenizer
    backbone_state = torch.load(backbone_weights, map_location="cpu", weights_only=True)
    model.visual_model.backbone.load_state_dict(backbone_state, strict=True)
    text_state = torch.load(weights, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(text_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    non_backbone_missing = [key for key in incompatible.missing_keys if not key.startswith("visual_model.backbone.")]
    if unexpected or non_backbone_missing:
        raise RuntimeError(
            "Original DINO.text checkpoint is incompatible with the pinned model: "
            f"missing={non_backbone_missing}, unexpected={unexpected}"
        )
    model.eval()
    return model, tokenizer

