"""Pinned VIP observation with identity fallback only for empty support rows."""
import torch
import torch.nn.functional as F

from .vip_official_adapter import VIPOfficialAdapter


def finite_proxy_attention(head, attention, tokens, raw, *, count_empty=True):
    """VIP 5bd25ee proxy equations; an all-masked row reads its own Value."""
    batch, _, channels = tokens.shape
    prefix, patches = tokens[:, :5].clone(), tokens[:, 5:]
    qkv = attention.qkv(patches).reshape(batch, patches.shape[1], 3,
                                         attention.num_heads, channels//attention.num_heads)
    query, key, value = [part.transpose(1, 2).squeeze(0).contiguous() for part in qkv.unbind(2)]
    normalized = F.normalize(raw.permute(0, 2, 1), dim=1)
    similarity = torch.einsum("b c m, b c n -> b m n", normalized, normalized)
    similarity = (similarity-similarity.mean()*1.5)*.5
    similarity[similarity <= 0.] = -torch.inf
    empty = ~torch.isfinite(similarity).any(-1)
    diagonal = torch.eye(patches.shape[1], device=raw.device, dtype=torch.bool)[None]
    similarity = similarity.masked_fill(empty[..., None] & diagonal, 0.)
    weights = similarity.to(query.dtype).unsqueeze(1).repeat(1, attention.num_heads, 1, 1)
    weights = weights.reshape(batch*attention.num_heads, patches.shape[1], patches.shape[1]).softmax(-1)
    value = value.reshape(batch*attention.num_heads, head.patch_size, head.patch_size,
                          channels//attention.num_heads).permute(0, 3, 1, 2)
    value = F.interpolate(value, size=(head.patch_size, head.patch_size), mode="bilinear", align_corners=False)
    value = value.permute(0, 2, 3, 1).reshape(batch*attention.num_heads, patches.shape[1], -1)
    read = (weights @ value).transpose(0, 1).contiguous().view(batch, -1, channels)
    result = torch.cat((prefix, attention.proj_drop(attention.proj(read))), dim=1)
    return result, int(empty.sum()) if count_empty else None, empty.numel()


class FiniteVIPObserver(VIPOfficialAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.empty_rows = 0
        self.observed_rows = 0

    @torch.inference_mode()
    def crop_patch_features(self, rgb):
        if rgb.shape != (3, 336, 336):
            raise ValueError("Expected a 336x336 RGB crop.")
        image = rgb[None].to(self.device)
        image = ((image-self.backbone._imagenet_mean)/self.backbone._imagenet_std).half()
        visual = self.backbone.model.visual_model
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cls, raw, registers = visual.get_backbone_features(image)
            tokens = torch.cat((cls[:, None], registers, raw), dim=1)
            if registers.shape[1] != 4:
                raise ValueError("The pinned proxy requires four register tokens.")
            for block in visual.head.blocks:
                attention, empty, rows = finite_proxy_attention(visual.head, block.attn, block.norm1(tokens), raw)
                self.empty_rows += empty
                self.observed_rows += rows
                tokens = tokens+block.ls1(attention)
                tokens = tokens+block.ls2(block.mlp(block.norm2(tokens)))
            projected = visual.head.linear_projection(visual.head.ln_final(tokens))
            result = F.normalize(projected[:, 5:].float(), dim=-1)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Nonfinite observer features after empty-row fallback.")
        return result
