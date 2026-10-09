"""Geometry-supported counterfactual reads of the frozen VIP semantic head."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .finite_vip_observer import FiniteVIPObserver, finite_proxy_attention


IMPLEMENTATION = "geometry-paired-context-readout-v1-20261002"
PRIMARY = "Geometry_PairedContext"
METHODS = ("Geometry", "BroadVIP", "MeanProb_VIP", "MeanLogit_VIP", "Anchored_VIP",
           "MeanLogit_Context", "ShuffledContextWriteback", PRIMARY)
CONFIG = {
    "support": "local Geometry conditional probability strictly above valid-row uniform; footprint overlap transport",
    "reference": "same wide backbone tokens; intersect VIP donors with transported support in both head blocks",
    "unmapped_queries": "unchanged full donor set",
    "empty_rows": "self Value fallback",
    "text_profile": "full-read salience reused for reference",
    "update": "min_delta .5||delta||^2+.5||A(delta-(full-reference))||^2",
    "solver": "unchanged unit-weight anchored solver",
    "all_aliases": 20,
}


def patch_boxes(rows, cols, top, left, step_y, step_x, device):
    y, x = torch.meshgrid(torch.arange(rows, device=device), torch.arange(cols, device=device), indexing="ij")
    return torch.stack((top+y*step_y, left+x*step_x,
                        top+(y+1)*step_y, left+(x+1)*step_x), -1).reshape(-1, 4).float()


def footprint_overlap(query_boxes, donor_boxes, image_height, image_width):
    limit = query_boxes.new_tensor((image_height, image_width))
    q0, q1 = query_boxes[:, :2].clamp_min(0), torch.minimum(query_boxes[:, 2:], limit)
    d0, d1 = donor_boxes[:, :2].clamp_min(0), torch.minimum(donor_boxes[:, 2:], limit)
    extent = (torch.minimum(q1[:, None], d1[None])-torch.maximum(q0[:, None], d0[None])).clamp_min(0)
    return extent.prod(-1)


def transported_support(relation, valid, overlap):
    """Transport fine support using real patch footprints, without text/labels."""
    if relation.shape != (1, valid.numel(), valid.numel()) or overlap.shape[1] != valid.numel():
        raise ValueError("Mismatched local relation, validity and footprint overlap.")
    donor_valid = valid.reshape(-1)
    weights = relation[0].float().masked_fill(~donor_valid[None], 0.)
    mass = weights.sum(-1, keepdim=True)
    weights = weights/mass.clamp_min(1e-12)
    supported = (weights > 1./donor_valid.sum().clamp_min(1)).float()
    supported *= donor_valid[:, None] & donor_valid[None]
    mapping = overlap.float()*donor_valid[None]
    mapping = mapping/mapping.sum(-1, keepdim=True).clamp_min(1e-12)
    restricted = (mapping @ supported @ mapping.T) > 0
    mapped = mapping.sum(-1) > 0
    restricted[~mapped] = True
    restricted |= torch.eye(len(mapping), device=mapping.device, dtype=torch.bool)
    return restricted, mapped


def restricted_proxy_attention(head, attention, tokens, raw, support):
    batch, _, channels = tokens.shape
    if batch != 1 or support.shape != (raw.shape[1], raw.shape[1]):
        raise ValueError("Expected one crop and a square query/donor support.")
    prefix, patches = tokens[:, :5].clone(), tokens[:, 5:]
    qkv = attention.qkv(patches).reshape(batch, patches.shape[1], 3,
                                         attention.num_heads, channels//attention.num_heads)
    query, key, value = [part.transpose(1, 2).squeeze(0).contiguous() for part in qkv.unbind(2)]
    normalized = F.normalize(raw.permute(0, 2, 1), dim=1)
    similarity = torch.einsum("b c m, b c n -> b m n", normalized, normalized)
    similarity = (similarity-similarity.mean()*1.5)*.5
    similarity[(similarity <= 0.) | ~support[None]] = -torch.inf
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
    return torch.cat((prefix, attention.proj_drop(attention.proj(read))), dim=1), int(empty.sum())


@dataclass
class WideCache:
    tokens: torch.Tensor
    raw: torch.Tensor
    full: torch.Tensor


class PairedContextObserver(FiniteVIPObserver):
    @torch.inference_mode()
    def prepare_crop(self, rgb):
        if rgb.shape != (3, 336, 336):
            raise ValueError("Expected a 336x336 crop.")
        image = ((rgb[None].to(self.device)-self.backbone._imagenet_mean)/self.backbone._imagenet_std).half()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cls, raw, registers = self.backbone.model.visual_model.get_backbone_features(image)
            if registers.shape[1] != 4:
                raise ValueError("Pinned VIP requires four registers.")
            tokens = torch.cat((cls[:, None], registers, raw), 1)
            full = self.read_tokens(tokens, raw, None)[0]
        return WideCache(tokens, raw, full)

    @torch.inference_mode()
    def read_tokens(self, initial, raw, support):
        head, tokens, empty_total = self.backbone.model.visual_model.head, initial, 0
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for block in head.blocks:
                if support is None:
                    attention, empty, rows = finite_proxy_attention(head, block.attn, block.norm1(tokens), raw)
                    self.empty_rows += empty
                    self.observed_rows += rows
                else:
                    attention, empty = restricted_proxy_attention(head, block.attn, block.norm1(tokens), raw, support)
                empty_total += empty
                tokens = tokens+block.ls1(attention)
                tokens = tokens+block.ls2(block.mlp(block.norm2(tokens)))
            projected = head.linear_projection(head.ln_final(tokens))
            result = F.normalize(projected[:, 5:].float(), dim=-1)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Nonfinite paired observer features.")
        return result, empty_total


@torch.inference_mode()
def fixed_profile_logits(features, profile_features, queries, settings):
    """Reference logits use exactly the full-read salience profile."""
    with torch.autocast(device_type=features.device.type, dtype=torch.bfloat16, enabled=features.is_cuda):
        similarities = torch.einsum("bnd,mtd->bnmt", features, queries.features.float()).mean(-1)[0]
        patch_mean = F.normalize(profile_features.mean(1), dim=-1)
        text_mean = F.normalize(queries.features.float().mean(1), dim=-1)
        salience = ((patch_mean @ text_mean.T)[0]/settings.tem).float()
        alias_logits = (similarities*settings.logit_scale).T.reshape(-1, 21, 21)
        classes = []
        for index in range(len(queries.class_names)):
            members = queries.parents == index
            weights = salience[members].softmax(0)
            scaled = alias_logits[members]*(weights/weights.mean())[:, None, None]
            classes.append(torch.logsumexp(settings.tau*scaled, 0)/settings.tau)
        return F.interpolate(torch.stack(classes)[None], size=(336, 336), mode="bilinear", align_corners=False)[0].float()
