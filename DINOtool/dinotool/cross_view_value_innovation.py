"""Matched fine/context Value reads inside the original frozen Geometry head."""
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .matched_readout_controls import qkv
from .tcpr import _finish_attention_block, _finish_head, _geometry_attended


IMPLEMENTATION = "geometry-cross-view-value-innovation-v1-20261002"
PRIMARY = "Geometry_ValueInnovation"
METHODS = ("Geometry", "SCLIP_Two", "VIPProxy_Two", "ContextGeometry",
           "MeanLogit_ContextGeometry", "Shuffled_ValueInnovation", PRIMARY)


@dataclass(frozen=True)
class ValueInnovationConfig:
    context_extent: int = 1024
    model_extent: int = 512
    query_chunk: int = 256

    def signature(self):
        return {"implementation": IMPLEMENTATION, "cross_view": asdict(self),
            "source": "same frozen DINOv3/DINO.text; context donors advance along original Geometry",
            "relation": "original fine G; cross-view raw cosine/.10 and same physical sigma128px",
            "reads": "same current fine Query; softmax(log(normalized valid G)+native QK*scale) for each view",
            "view_mass": "sigmoid(logZ_context-logZ_fine), normalized geometric priors avoid token-count bias",
            "writeback": "original Geo attended + native_patch_mass * view_mass * (context Value read-fine Value read)",
            "head": "both frozen head blocks; actual evolving Query/Key/Value, prefix/residual/MLP/LN retained",
            "identities": "no observed context or duplicate view gives original Geometry; common Values cancel",
            "text": "same20 aliases/six RS templates/normalized LME .07; no alias selection",
            "output": "single coupled fine descriptor, not final-logit fusion",
            "limits": "native QK and cross-view correspondences are not semantic correctness certificates; "
                      "signed Value displacement is not a nonnegative attention replacement",
            "novelty": "cross-attention, geometric priors and differential readouts have precedent; "
                       "this specific coupled operator is an unverified candidate, not a priority claim"}


def normalize_prior(prior, valid):
    if (prior.ndim != 3 or valid.shape != (prior.shape[0], prior.shape[2])
            or valid.dtype != torch.bool or not torch.isfinite(prior).all() or bool((prior < 0).any())):
        raise ValueError("Expected finite nonnegative [B,N,M] relation and valid donor mask.")
    masked = prior.float().masked_fill(~valid[:, None], 0.)
    return masked/masked.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)


def cross_view_prior(fine_raw, context_raw, fine_xy, context_xy, context_valid,
                     *, temperature=.1, spatial_sigma=.25):
    if (fine_raw.ndim != 3 or context_raw.ndim != 3 or fine_raw.shape[0] != context_raw.shape[0]
            or fine_raw.shape[-1] != context_raw.shape[-1]
            or fine_xy.shape != (fine_raw.shape[1], 2) or context_xy.shape != (context_raw.shape[1], 2)
            or context_valid.shape != context_raw.shape[:2] or context_valid.dtype != torch.bool
            or temperature <= 0 or spatial_sigma <= 0
            or not all(bool(torch.isfinite(x).all()) for x in (fine_raw, context_raw, fine_xy, context_xy))):
        raise ValueError("Invalid real-coordinate cross-view geometry.")
    with torch.autocast(device_type=fine_raw.device.type, enabled=False):
        similarity = F.normalize(fine_raw.float(), dim=-1) @ F.normalize(context_raw.float(), dim=-1).transpose(-1, -2)
        distance = (fine_xy[:, None].float()-context_xy[None].float()).square().sum(-1)
        logits = (similarity/temperature-distance/(2*spatial_sigma**2)).masked_fill(~context_valid[:, None], -torch.inf)
        available = context_valid.any(-1)[:, None, None]
        return torch.where(available, logits.masked_fill(~available, 0.).softmax(-1), 0.)


def conditioned_value_read(query, key, value, prior, scale):
    if (query.ndim != 4 or key.ndim != 4 or value.shape != key.shape
            or query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]
            or prior.shape != (query.shape[0], query.shape[2], key.shape[2])
            or not all(bool(torch.isfinite(x).all()) for x in (query, key, value, prior))
            or bool((prior < 0).any()) or scale <= 0):
        raise ValueError("Mismatched finite frozen Query/Key/Value or prior.")
    with torch.autocast(device_type=query.device.type, enabled=False):
        joint = (query.float() @ key.float().transpose(-1, -2))*scale + prior.float().log()[:, None]
        available = (prior.sum(-1) > 0)[:, None, :, None]
        safe = joint.masked_fill(~available, 0.)
        weights = safe.softmax(-1).masked_fill(~available, 0.)
        log_z = joint.logsumexp(-1, keepdim=True)
        return weights @ value.float(), log_z, available


def matched_value_displacement(query, fine_key, fine_value, fine_prior,
                               context_key, context_value, context_prior, scale):
    fine, fine_log_z, fine_available = conditioned_value_read(query, fine_key, fine_value, fine_prior, scale)
    context, context_log_z, context_available = conditioned_value_read(query, context_key, context_value, context_prior, scale)
    available = fine_available & context_available
    difference = torch.where(available, context_log_z-fine_log_z, 0.)
    fraction = difference.sigmoid().masked_fill(~available, 0.)
    return fraction*(context-fine), fraction


@torch.inference_mode()
def read_value_innovation(head, fine, context, fine_valid, context_valid, prior,
                          config=ValueInnovationConfig(), *, context_permutation=None):
    if (fine.backbone_tokens is None or context.backbone_tokens is None
            or fine.prefix_tokens != context.prefix_tokens or fine.block_index != context.block_index
            or fine_valid.shape != fine.raw_patch_tokens.shape[:2]
            or context_valid.shape != context.raw_patch_tokens.shape[:2]
            or fine_valid.dtype != torch.bool or context_valid.dtype != torch.bool
            or prior.shape != (fine_valid.shape[0], fine_valid.shape[1], context_valid.shape[1])
            or fine.block_index != 1 or config.query_chunk < 1):
        raise ValueError("Expected aligned original two-block Geometry preparations and image-only supports.")
    if context_permutation is not None:
        expected = torch.arange(context_valid.shape[1], device=context_valid.device)
        if (context_permutation.shape != expected.shape or context_permutation.dtype != torch.long
                or not torch.equal(context_permutation.sort().values, expected)
                or not torch.equal(context_valid[:, context_permutation], context_valid)):
            raise ValueError("Shuffle must preserve the original valid/padded donor membership.")
    current, donors, prefix = fine.backbone_tokens, context.backbone_tokens, fine.prefix_tokens
    fine_prior = normalize_prior(fine.geometry_patch_conditional, fine_valid)
    context_prior = normalize_prior(prior, context_valid)
    totals = {"mean_context_fraction": 0., "mean_absolute_value_displacement": 0.,
              "prefix_displacement_max_error": 0., "invalid_query_displacement_max_error": 0.}
    for block in head.blocks[:fine.block_index+1]:
        query, key, value = qkv(block, current)
        context_query, context_key, context_value = qkv(block, donors)
        reading_key, reading_value = context_key[..., prefix:, :], context_value[..., prefix:, :]
        if context_permutation is not None:
            reading_key = reading_key[:, :, context_permutation]
            reading_value = reading_value[:, :, context_permutation]
        native = ((query.float() @ key.float().transpose(-1, -2))*block.attn.scale).softmax(-1).to(value.dtype)
        attended = _geometry_attended(native, fine.geometry_patch_conditional, value, prefix)
        displacement = torch.zeros_like(attended[..., prefix:, :], dtype=torch.float32)
        fraction_sum, active_rows = 0., 0
        for start in range(0, fine_valid.shape[1], config.query_chunk):
            stop = start+config.query_chunk
            delta, fraction = matched_value_displacement(query[..., prefix+start:prefix+stop, :],
                key[..., prefix:, :], value[..., prefix:, :], fine_prior[:, start:stop],
                reading_key, reading_value, context_prior[:, start:stop], block.attn.scale)
            valid = fine_valid[:, None, start:stop, None]
            displacement[..., start:stop, :] = delta.masked_fill(~valid, 0.)
            fraction_sum += float(fraction.expand(query.shape[0], query.shape[1], -1, -1).masked_fill(~valid, 0.).sum())
            active_rows += int(valid.sum())*query.shape[1]
        mass = native[..., prefix:, prefix:].sum(-1, keepdim=True).float()
        changed = attended.clone()
        changed[..., prefix:, :] += (mass*displacement).to(value.dtype)
        totals["mean_context_fraction"] += fraction_sum/max(active_rows, 1)/(fine.block_index+1)
        totals["mean_absolute_value_displacement"] += float(displacement.abs().mean())/(fine.block_index+1)
        totals["prefix_displacement_max_error"] = max(totals["prefix_displacement_max_error"],
            float((changed[..., :prefix, :]-attended[..., :prefix, :]).abs().max()))
        totals["invalid_query_displacement_max_error"] = max(totals["invalid_query_displacement_max_error"],
            float(displacement.masked_fill(fine_valid[:, None, :, None], 0.).abs().max()))
        # Context donors evolve along original Geometry, not through coupled fine queries.
        cnative = ((context_query.float() @ context_key.float().transpose(-1, -2))*block.attn.scale).softmax(-1).to(context_value.dtype)
        donors = _finish_attention_block(block, donors,
            _geometry_attended(cnative, context.geometry_patch_conditional, context_value, prefix))
        current = _finish_attention_block(block, current, changed)
    projected = _finish_head(head, current, fine.block_index)
    donor_projected = _finish_head(head, donors, context.block_index)
    features = F.normalize(projected[:, prefix:].float(), dim=-1)
    donor_features = F.normalize(donor_projected[:, prefix:].float(), dim=-1)
    totals["context_geometry_replay_max_error"] = float((donor_features-context.geometry_projected).abs().max())
    if not torch.isfinite(features).all() or totals["context_geometry_replay_max_error"] != 0.:
        raise RuntimeError("Nonfinite coupled read or changed original context Geometry donor path.")
    return features, totals
