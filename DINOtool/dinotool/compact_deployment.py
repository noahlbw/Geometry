"""Execution-only reductions for the frozen Geometry/wide model.

Keep both visual weight dtypes: Geometry float32 and VIP float16. Sharing them
would introduce an extra rounding change. Text towers can leave CUDA after the
vocabulary has been encoded, and unused native/head diagnostics need not run.
"""
from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .tcpr import structural_logits, geometry_relation, _intervened_head, _resolve_block_index
from .vip_official_adapter import VIPSettings


def offload_text_towers(geometry, vip, *, release=False):
    """Retain the ability to re-encode later, with no online GPU text weights."""
    records = []
    for name, backbone in (('geometry', geometry.backbone), ('wide', vip.backbone)):
        text = backbone.model.text_model
        size = 0 if text is None else sum(p.numel() * p.element_size() for p in text.parameters())
        if release:
            backbone.model.text_model = None
        elif text is not None:
            text.cpu()
        records.append(dict(branch=name, text_weight_bytes=size, released=release))
    return records


class CompactGeometry:
    """Compute the same relation and only the requested Geometry head path."""
    def __init__(self, geometry):
        self.backbone, self.config = geometry.backbone, geometry.config
        self.device, self.patch_size = geometry.device, geometry.patch_size

    @torch.inference_mode()
    def prepare_image(self, rgb):
        backbone, config = self.backbone, self.config
        backbone._validate_rgb(rgb)
        normalized = (rgb.to(self.device, non_blocking=True) - backbone._imagenet_mean) / backbone._imagenet_std
        with backbone._autocast():
            visual = backbone.model.visual_model
            cls, raw, registers = visual.get_backbone_features(normalized)
            tokens = torch.cat((cls[:, None], registers, raw), dim=1)
            relation = geometry_relation(structural_logits(raw, rgb.shape[-2] // 16, rgb.shape[-1] // 16,
                temperature=config.geometry_temperature, spatial_sigma=config.spatial_sigma),
                raw, config.relation_policy).to(torch.get_autocast_dtype(self.device.type)
                    if torch.is_autocast_enabled(self.device.type) else tokens.dtype)
        return SimpleNamespace(backbone_tokens=tokens, geometry_patch_conditional=relation,
            prefix_tokens=registers.shape[1] + 1, block_index=_resolve_block_index(visual.head, config.head_block))

    @torch.inference_mode()
    def project(self, prepared, strength):
        if strength == 'original':
            relation, depth, prefix = prepared.geometry_patch_conditional, self.config.geometry_depth, self.config.prefix_policy
        else:
            relation, depth, prefix = prepared.geometry_patch_conditional * strength, 2, 'block'
        projected = _intervened_head(self.backbone.model.visual_model.head, prepared.backbone_tokens,
            relation, prepared.prefix_tokens, prepared.block_index, depth, prefix)
        return F.normalize(projected[:, prepared.prefix_tokens:].float(), dim=-1)


@torch.inference_mode()
def compact_observations(image, geometry, vip, strengths, *, wide_policy='long448'):
    import eval_development_readout as base
    from .bounded_physical_coupling import geometry_windows, resize_geometry
    from .gear_ov import _crop_at
    from .alias_action_capacity import reconstruction_operator
    from eval_rival_fine_full import tile_coordinates
    from .natural_wide_resolution import replace_wide
    # Use the original RGB resizing, crop order and reconstruction solver.
    local, resized = [], resize_geometry(image)
    h, w = resized.shape[-2:]
    for top, left in geometry_windows(h, w):
        prepared = geometry.prepare_image(_crop_at(resized, top, left, 512).to(geometry.device))
        coordinates = tile_coordinates(top, left, geometry.device)
        valid = (coordinates[:, 0] < h) & (coordinates[:, 1] < w)
        operator, _ = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
        with geometry.backbone._autocast():
            features = {s: geometry.project(prepared, s)[0] for s in dict.fromkeys(strengths)}
        local.append(dict(top=top, left=left, features=features, operator=operator))
        del prepared
    source = dict(local=local, size=(h, w), output_size=tuple(image.shape[-2:]))
    if wide_policy == 'natural_short336_cap672':
        return replace_wide(source, image, vip)
    if wide_policy != 'long448':
        raise ValueError('Unknown frozen wide policy.')
    from eval_matched_head_fov import resize_rgb
    from .inference import tile_starts
    rgb = resize_rgb(image, 448)
    wh, ww = rgb.shape[-2:]
    crops = []
    for top in tile_starts(wh, 336, 224):
        for left in tile_starts(ww, 336, 224):
            ah, aw = min(336, wh-top), min(336, ww-left)
            crop = F.pad(rgb[:, top:top+ah, left:left+aw], (0, 336-aw, 0, 336-ah))
            crops.append(dict(top=top, left=left, ah=ah, aw=aw, features=vip.crop_patch_features(crop)))
    if not 1 <= len(local) <= 4 or not 1 <= len(crops) <= 4:
        raise RuntimeError('Frozen visual budget changed.')
    return dict(source, wide=crops, wide_size=(wh, ww))


class CachedWideScorer:
    """Same template-level bfloat16 arithmetic, cached integer class groups."""
    def __init__(self):
        self.cache = {}

    @torch.inference_mode()
    def logits(self, features, query, settings):
        key = id(query)
        if key not in self.cache:
            parents = query.parents.detach().cpu().tolist()
            groups = tuple(torch.tensor([i for i, p in enumerate(parents) if p == c],
                device=query.features.device) for c in range(len(query.class_names)))
            text = query.features.float()
            self.cache[key] = query, text, F.normalize(text.mean(1), dim=-1), groups
        _, text, mean, groups = self.cache[key]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            similarities = torch.einsum('bnd,mtd->bnmt', features, text).mean(-1)[0]
            patch_mean = F.normalize(features.mean(1), dim=-1)
            salience = ((patch_mean @ mean.T)[0] / settings.tem).float()
            aliases = (similarities * settings.logit_scale).T.reshape(-1, 21, 21)
            logits = []
            for indices in groups:
                weights = salience[indices].softmax(0)
                scaled = aliases[indices] * (weights / weights.mean())[:, None, None]
                logits.append(torch.logsumexp(settings.tau * scaled, dim=0) / settings.tau)
            return F.interpolate(torch.stack(logits)[None], (336, 336), mode='bilinear', align_corners=False)[0].float()

    @torch.inference_mode()
    def __call__(self, source, query, tau, tem):
        h, w = source['wide_size']
        output = torch.zeros(len(query.class_names), h, w, device=query.features.device)
        count = torch.zeros(h, w, device=query.features.device)
        settings = VIPSettings(tau=tau, tem=tem)
        for crop in source['wide']:
            top, left, ah, aw = (crop[k] for k in ('top', 'left', 'ah', 'aw'))
            output[:, top:top+ah, left:left+aw] += self.logits(crop['features'], query, settings)[:, :ah, :aw]
            count[top:top+ah, left:left+aw] += 1
        return output / count[None]


@contextmanager
def cached_wide(scoring):
    """Scoped compatibility with the frozen reader; no permanent monkeypatch."""
    import eval_development_readout as base
    original = base.wide_scores
    base.wide_scores = scoring
    try:
        yield
    finally:
        base.wide_scores = original
