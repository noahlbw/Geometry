"""Frozen cross-scale relation grounding: coarse relations, fine Values.

The complete model replaces Multiscale's coarse score branch with a fine
head reread. There is no alias selection, iterative fitting or learned gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .gear_ov import _crop_at, _four_to_grid, class_scores
from .tcpr import TCPRConfig, TCPRSegmenter, TCPRTextBank, _finish_attention_block


IMPLEMENTATION = "geometry-cross-scale-relative-relations-v1-20261001"
METHOD = "CS_Grounded"
DIAGNOSTICS = (
    "available_context_mass", "relation_increment_p05", "relation_increment_p95",
    "relation_increment_abs_mean", "relation_positive_fraction", "relation_kl",
    "maximum_donor_weight", "effective_donors", "cross_detail_crop_mass",
    "fine_valid_fraction", "backbone_seconds", "grounding_seconds",
)


@dataclass(frozen=True)
class CrossScaleConfig:
    tile_size: int = 512
    geometry_temperature: float = 0.1
    spatial_sigma: float = 0.25   # coordinates normalized by enclosing 1024 field
    alias_temperature: float = 0.07
    relation_epsilon: float = 1e-6
    query_chunk: int = 256
    geometry_depth: int = 2

    def validate(self, patch_size: int) -> None:
        if self.tile_size != 512 or patch_size != 16 or self.geometry_depth != 2:
            raise ValueError("The complete model fixes 512 tiles, patch16 and two head blocks.")
        if min(self.geometry_temperature, self.spatial_sigma,
               self.alias_temperature, self.relation_epsilon) <= 0 or self.query_chunk < 1:
            raise ValueError("Invalid relation parameters.")


@dataclass(frozen=True)
class CrossScaleViews:
    local_aligned: Tensor
    detail_aligned: Tensor
    grounded_aligned: Tensor
    actual_height: int
    actual_width: int
    diagnostics: dict[str, float]


def grid_to_four(grid: Tensor) -> Tensor:
    """[2h,2w,D] -> [4,h*w,D], preserving each crop's token order."""
    h, w = grid.shape[:2]
    if h % 2 or w % 2:
        raise ValueError("Detail grid dimensions must be even.")
    return torch.stack([grid[y:y + h // 2, x:x + w // 2].reshape(-1, grid.shape[-1])
                        for y in (0, h // 2) for x in (0, w // 2)])


def four_to_grid(packed: Tensor) -> Tensor:
    if packed.shape[0] != 4:
        raise ValueError("Expected four crop states.")
    side = math.isqrt(packed.shape[1])
    if side * side != packed.shape[1]:
        raise ValueError("Detail crop patch grid must be square.")
    return _four_to_grid([value.reshape(side, side, -1) for value in packed])


def relation_logits(query: Tensor, donor: Tensor, query_xy: Tensor,
                    donor_xy: Tensor, valid_donor: Tensor,
                    config: CrossScaleConfig) -> Tensor:
    # Explicit FP32, including matmuls under the surrounding AMP context.
    with torch.autocast(device_type=query.device.type, enabled=False):
        score = query.float() @ donor.float().T / config.geometry_temperature
        distance = (query_xy[:, None].float() - donor_xy[None].float()).square().sum(-1)
        return (score - distance / (2 * config.spatial_sigma**2)).masked_fill(
            ~valid_donor[None], -torch.inf)


def aggregate_fine_reference(raw: Tensor, xy: Tensor, footprint: Tensor,
                             valid: Tensor, groups: int,
                             config: CrossScaleConfig) -> Tensor:
    """Exactly average valid fine query relations over known coarse footprints."""
    if not bool(valid.any()):
        raise ValueError("No real fine donor is available.")
    reference = torch.zeros(groups, groups, device=raw.device, dtype=torch.float32)
    counts = torch.zeros(groups, device=raw.device, dtype=torch.float32)
    counts.index_add_(0, footprint[valid], torch.ones_like(footprint[valid], dtype=torch.float32))
    for start in range(0, raw.shape[0], config.query_chunk):
        end = min(start + config.query_chunk, raw.shape[0])
        base = relation_logits(raw[start:end], raw, xy[start:end], xy, valid, config).softmax(-1)
        macro = torch.zeros(end - start, groups, device=raw.device, dtype=torch.float32)
        macro.scatter_add_(1, footprint[None].expand(end - start, -1), base)
        active = valid[start:end]
        reference.index_add_(0, footprint[start:end][active], macro[active])
    return reference / counts.clamp_min(1)[:, None]


def lift_relative_relations(raw: Tensor, xy: Tensor, footprint: Tensor,
                            owners: Tensor, valid: Tensor, increment: Tensor,
                            available: Tensor, config: CrossScaleConfig
                            ) -> tuple[Tensor, dict[str, float]]:
    """Lift a coarse relation ratio without changing within-footprint conditionals."""
    count = raw.shape[0]
    relation = torch.empty(count, count, device=raw.device, dtype=torch.float32)
    totals = dict.fromkeys(("relation_kl", "maximum_donor_weight", "effective_donors",
                           "cross_detail_crop_mass"), 0.0)
    active_count = int(valid.sum())
    for start in range(0, count, config.query_chunk):
        end = min(start + config.query_chunk, count)
        score = relation_logits(raw[start:end], raw, xy[start:end], xy, valid, config)
        log_base = score.log_softmax(-1)
        base = log_base.exp()
        added = increment[footprint[start:end]][:, footprint]
        shifted = (log_base + added).softmax(-1)
        mass = available[footprint[start:end], None]
        changed = (1 - mass) * base + mass * shifted
        relation[start:end] = changed
        active = valid[start:end]
        selected = changed[active]
        # Zero-probability padded donors contribute zero to KL.
        divergence = torch.where(valid[None],
                                 changed.clamp_min(1e-30).log()
                                 - log_base.masked_fill(~valid[None], 0), 0)
        totals["relation_kl"] += float((changed * divergence).sum(-1)[active].sum())
        totals["maximum_donor_weight"] += float(selected.max(-1).values.sum())
        totals["effective_donors"] += float(selected.square().sum(-1).reciprocal().sum())
        different = owners[start:end, None] != owners[None]
        totals["cross_detail_crop_mass"] += float((changed * different).sum(-1)[active].sum())
    if not bool(torch.isfinite(relation).all()):
        raise ValueError("Non-finite grounded relations.")
    return relation, {key: value / active_count for key, value in totals.items()}


def build_relations(fine_raw: Tensor, context_raw: Tensor, actual_height: int,
                    actual_width: int, image_height: int, image_width: int,
                    top: int, left: int, config: CrossScaleConfig
                    ) -> tuple[Tensor, dict[str, float]]:
    device = fine_raw.device
    fy, fx = torch.meshgrid(torch.arange(64, device=device),
                            torch.arange(64, device=device), indexing="ij")
    cy, cx = torch.meshgrid(torch.arange(32, device=device),
                            torch.arange(32, device=device), indexing="ij")
    # Both coordinates describe the same enclosing 1024-pixel field.
    fine_xy = torch.stack((256 + (fy.flatten() + .5) * 8,
                           256 + (fx.flatten() + .5) * 8), -1) / 1024
    coarse_xy = torch.stack(((cy.flatten() + .5) * 32,
                             (cx.flatten() + .5) * 32), -1) / 1024
    footprint = ((fy // 4) * 16 + fx // 4).flatten().long()
    owners = ((fy // 32) * 2 + fx // 32).flatten()
    valid_fine = ((fy * 8 < actual_height) & (fx * 8 < actual_width)).flatten()
    # Include partially real footprints; exclude purely replicated padding.
    y0, x0 = top - 256 + cy * 32, left - 256 + cx * 32
    valid_coarse = ((y0 < image_height) & (y0 + 32 > 0)
                    & (x0 < image_width) & (x0 + 32 > 0)).flatten()
    center = ((cy >= 8) & (cy < 24) & (cx >= 8) & (cx < 24)).flatten()
    fine = F.normalize(fine_raw.reshape(4096, -1).float(), dim=-1)
    coarse = F.normalize(context_raw.reshape(1024, -1).float(), dim=-1)
    reference = aggregate_fine_reference(fine, fine_xy, footprint, valid_fine, 256, config)
    context = relation_logits(coarse[center], coarse, coarse_xy[center], coarse_xy,
                              valid_coarse, config).softmax(-1)
    available = context[:, center].sum(-1)
    conditional = context[:, center] / available[:, None].clamp_min(1e-30)
    active_groups = reference.sum(-1) > 0
    available = available * active_groups
    increment = (conditional + config.relation_epsilon).log() \
        - (reference + config.relation_epsilon).log()
    increment[~active_groups] = 0
    increment[:, ~active_groups] = 0
    relation, diagnostics = lift_relative_relations(
        fine, fine_xy, footprint, owners, valid_fine, increment, available, config)
    values = increment[active_groups][:, active_groups].flatten()
    diagnostics.update(
        available_context_mass=float(available[active_groups].mean()),
        relation_increment_p05=float(torch.quantile(values, .05)),
        relation_increment_p95=float(torch.quantile(values, .95)),
        relation_increment_abs_mean=float(values.abs().mean()),
        relation_positive_fraction=float((values > 0).float().mean()),
        fine_valid_fraction=float(valid_fine.float().mean()),
    )
    return relation, diagnostics


def reread_fine_head(head, crop_tokens: Tensor, relation: Tensor,
                     depth: int, chunk: int) -> Tensor:
    """Preserve origin-crop prefix/native mass, but read global fine Values.

    Native QK stays within each origin crop. Only the structural patch
    conditional connects crops, so no invented global CLS is introduced.
    Each block derives Q/K/V from the actual preceding grounded state.
    """
    if crop_tokens.shape[0] != 4:
        raise ValueError("The head requires the four original detail crop states.")
    patches = relation.shape[0] // 4
    prefix = crop_tokens.shape[1] - patches
    if relation.shape != (4 * patches, 4 * patches) or prefix < 1:
        raise ValueError("Token/relation shape mismatch.")
    current = crop_tokens
    first = len(head.blocks) - depth
    if first < 0:
        raise ValueError("Geometry depth exceeds the frozen head.")
    for index, block in enumerate(head.blocks):
        if index < first:
            current = block(current)
            continue
        attention = block.attn
        channels = attention.qkv.in_features
        qkv = attention.qkv(block.norm1(current)).reshape(
            4, current.shape[1], 3, attention.num_heads, channels // attention.num_heads)
        query, key, value = [part.transpose(1, 2) for part in qkv.unbind(2)]
        donor = four_to_grid(value[:, :, prefix:].transpose(1, 2).reshape(
            4, patches, channels)).reshape(4 * patches, channels)
        with torch.autocast(device_type=relation.device.type, enabled=False):
            read = relation @ donor.float()
        read = grid_to_four(read.reshape(2 * math.isqrt(patches),
                                          2 * math.isqrt(patches), channels))
        read = read.reshape(4, patches, attention.num_heads,
                            channels // attention.num_heads).transpose(1, 2).to(value.dtype)
        attended = torch.empty_like(value)
        # Prefix queries and patch-to-prefix probabilities use the original
        # crop's native attention, computed in chunks rather than global QK.
        for start in range(0, current.shape[1], chunk):
            end = min(start + chunk, current.shape[1])
            logits = (query[:, :, start:end].float() @ key.float().transpose(-1, -2)) \
                * attention.scale
            native = logits.softmax(-1).to(value.dtype)
            prefix_rows = max(min(prefix, end) - start, 0)
            if prefix_rows:
                attended[:, :, start:start + prefix_rows] = native[:, :, :prefix_rows] @ value
            patch_start = max(start, prefix)
            if patch_start < end:
                rows = native[:, :, patch_start - start:]
                special = rows[..., :prefix] @ value[..., :prefix, :]
                mass = rows[..., prefix:].sum(-1, keepdim=True)
                attended[:, :, patch_start:end] = special + mass * read[:, :, patch_start - prefix:end - prefix]
        current = _finish_attention_block(block, current, attended)
    projected = head.linear_projection(head.ln_final(current))[:, prefix:]
    return F.normalize(four_to_grid(projected).float(), dim=-1)


class CrossScaleGeometrySegmenter:
    def __init__(self, backbone, config: CrossScaleConfig = CrossScaleConfig()) -> None:
        config.validate(backbone.patch_size)
        self.backbone = backbone
        self.config = config
        self.base = TCPRSegmenter(backbone, TCPRConfig(
            maximum_aliases_per_class=20, geometry_depth=2,
            prefix_policy="preserve", relation_policy="dense"))

    @property
    def device(self):
        return self.backbone.device

    def encode_text(self, classes) -> TCPRTextBank:
        bank = self.base.encode_text(classes)
        if any(int((bank.parent_indices == c).sum()) != 20 for c in range(bank.class_count)):
            raise ValueError("The complete model requires exactly 20 aliases per class.")
        return bank

    @torch.inference_mode()
    def prepare_views(self, image: Tensor, top: int, left: int) -> CrossScaleViews:
        import time

        started = time.perf_counter()
        tile = self.config.tile_size
        h, w = min(tile, image.shape[-2] - top), min(tile, image.shape[-1] - left)
        local_rgb = _crop_at(image, top, left, tile).to(self.device)
        local = self.base.prepare_image(local_rgb)
        local_aligned = local.geometry_projected[0].reshape(32, 32, -1)
        del local
        enlarged = F.interpolate(local_rgb, scale_factor=2, mode="bilinear", align_corners=False)
        aligned, raw, inputs = [], [], []
        for dy in (0, tile):
            for dx in (0, tile):
                prepared = self.base.prepare_image(enlarged[:, :, dy:dy + tile, dx:dx + tile])
                aligned.append(prepared.geometry_projected[0].reshape(32, 32, -1))
                raw.append(prepared.raw_patch_tokens[0].reshape(32, 32, -1))
                if prepared.backbone_tokens is None:
                    raise ValueError("Missing backbone head-input token cache.")
                inputs.append(prepared.backbone_tokens[0])
                del prepared
        context_rgb = _crop_at(image, top - 256, left - 256, 1024).to(self.device)
        context_rgb = F.interpolate(context_rgb, size=(512, 512), mode="bilinear", align_corners=False)
        context = self.base.prepare_image(context_rgb)
        context_raw = context.raw_patch_tokens[0]
        del context
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        encoded = time.perf_counter()
        relation, diagnostics = build_relations(
            _four_to_grid(raw), context_raw, h, w, *image.shape[-2:], top, left, self.config)
        with self.backbone._autocast():
            grounded = reread_fine_head(self.backbone.model.visual_model.head,
                                        torch.stack(inputs), relation,
                                        self.config.geometry_depth, self.config.query_chunk)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        diagnostics.update(backbone_seconds=encoded - started,
                           grounding_seconds=time.perf_counter() - encoded)
        return CrossScaleViews(local_aligned, _four_to_grid(aligned), grounded, h, w, diagnostics)

    @torch.inference_mode()
    def read_views(self, views: CrossScaleViews, bank: TCPRTextBank) -> Tensor:
        text = F.normalize(bank.features.float(), dim=-1)
        temperature = self.config.alias_temperature
        logits = []
        for features in (views.local_aligned, views.detail_aligned, views.grounded_aligned):
            scores = class_scores(features.float() @ text.T / temperature,
                                  bank.parent_indices, bank.class_count) * temperature
            logits.append(F.interpolate(scores.permute(2, 0, 1)[None], size=(512, 512),
                                        mode="bilinear", align_corners=False))
        return sum(logits) / 3

    def config_dict(self) -> dict[str, object]:
        return {**asdict(self.config), "prefix_policy": "preserve_origin_crop",
                "relation_policy": "dense", "templates": "REMOTE_SENSING_TEMPLATES",
                "output": "mean(local,detail,grounded)", "alias_selection": False}
