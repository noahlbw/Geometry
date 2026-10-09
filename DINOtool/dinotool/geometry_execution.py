"""Opt-in execution of the unchanged frozen Geometry preparation path."""
from dataclasses import fields, replace
import time

import torch
import torch.nn.functional as F

from .tcpr import (TCPRPreparedImage, _finish_attention_block, _finish_head,
                   _intervened_head, _resolve_block_index, geometry_relation, structural_logits)


@torch.inference_mode()
def prepare_image_pruned(segmenter, rgb):
    backbone, config = segmenter.backbone, segmenter.config
    backbone._validate_rgb(rgb)
    visual = backbone.model.visual_model
    head = visual.head
    if head.training:
        raise ValueError('Pruned Geometry execution requires the frozen evaluation head.')
    if config.geometry_depth == 1 and config.prefix_policy == 'preserve':
        return segmenter.prepare_image(rgb)
    normalized = (rgb.to(segmenter.device, non_blocking=True) - backbone._imagenet_mean) / backbone._imagenet_std
    with backbone._autocast():
        cls, raw, registers = visual.get_backbone_features(normalized)
        tokens = torch.cat((cls.unsqueeze(1), registers, raw), dim=1)
        block_index = _resolve_block_index(head, config.head_block)
        prefix = tokens.shape[1] - raw.shape[1]
        shared = tokens
        for earlier in head.blocks[:block_index]:
            shared = earlier(shared)
        block, attention = head.blocks[block_index], head.blocks[block_index].attn
        qkv = attention.qkv(block.norm1(shared))
        batch, token_count, _ = qkv.shape
        channels = attention.qkv.in_features
        qkv = qkv.reshape(batch, token_count, 3, attention.num_heads, channels // attention.num_heads)
        query, key, value = (part.transpose(1, 2) for part in torch.unbind(qkv, dim=2))
        native_logits = (query.float() @ key.float().transpose(-1, -2)) * attention.scale
        native_attention = torch.softmax(native_logits, dim=-1).to(value.dtype)
        native_patch = torch.softmax(native_logits[..., prefix:, prefix:], dim=-1).to(value.dtype)
        geometry_logits = structural_logits(raw, rgb.shape[-2] // segmenter.patch_size,
            rgb.shape[-1] // segmenter.patch_size, temperature=config.geometry_temperature,
            spatial_sigma=config.spatial_sigma)
        geometry_patch = geometry_relation(geometry_logits, raw, config.relation_policy).to(value.dtype)
        native_block = _finish_attention_block(block, shared, native_attention @ value)
        native_projected = _finish_head(head, native_block, block_index)
        # The legacy depth-one Geometry result is overwritten by this route.
        geometry_projected = _intervened_head(head, tokens, geometry_patch, prefix, block_index,
                                              config.geometry_depth, config.prefix_policy)
    return TCPRPreparedImage(shared_tokens=shared, raw_patch_tokens=F.normalize(raw.float(), dim=-1),
        value_tokens=value, native_attention=native_attention, native_patch_conditional=native_patch,
        geometry_patch_conditional=geometry_patch,
        native_projected=F.normalize(native_projected[:, prefix:].float(), dim=-1),
        geometry_projected=F.normalize(geometry_projected[:, prefix:].float(), dim=-1),
        prefix_tokens=prefix, grid_height=rgb.shape[-2] // segmenter.patch_size,
        grid_width=rgb.shape[-1] // segmenter.patch_size, block_index=block_index,
        native_global=F.normalize(torch.cat((native_projected[:, 0].float(),
                                           native_projected[:, prefix:].float().mean(dim=1)), dim=-1), dim=-1),
        backbone_tokens=tokens)


class GeometryExecution:
    """Frozen pruned preparation, optionally replayed as one fixed-shape CUDA graph."""

    def __init__(self, segmenter, *, graph=False):
        self.segmenter = segmenter
        self.device, self.patch_size = segmenter.device, segmenter.patch_size
        self.backbone, self.config = segmenter.backbone, segmenter.config
        self.use_graph = graph
        self.graph = self.static_rgb = self.static_prepared = None
        self.setup_seconds = None
        self.replays = 0

    @torch.inference_mode()
    def prepare_image(self, rgb):
        if self.segmenter.config != self.config:
            raise ValueError('Geometry execution belongs to the unchanged frozen configuration.')
        if not self.use_graph:
            return prepare_image_pruned(self.segmenter, rgb)
        self.backbone._validate_rgb(rgb)
        device = torch.device(self.device)
        if device.type != 'cuda' or rgb.shape[0] != 1 or not rgb.is_floating_point():
            raise ValueError('Geometry graph requires one floating-point BCHW RGB image and CUDA.')
        with torch.cuda.device(device):
            if self.graph is None:
                started = time.perf_counter()
                self.static_rgb = rgb.to(device).clone()
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        prepare_image_pruned(self.segmenter, self.static_rgb)
                torch.cuda.current_stream(device).wait_stream(stream)
                torch.cuda.synchronize(device)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.static_prepared = prepare_image_pruned(self.segmenter, self.static_rgb)
                torch.cuda.synchronize(device)
                self.setup_seconds = time.perf_counter() - started
            if rgb.shape != self.static_rgb.shape or rgb.dtype != self.static_rgb.dtype:
                raise ValueError('Fixed Geometry graph RGB shape/dtype changed.')
            self.static_rgb.copy_(rgb)
            self.graph.replay()
            self.replays += 1
            # Preserve prepare_image's ownership: earlier results survive later replays.
            snapshots = {field.name: getattr(self.static_prepared, field.name).clone()
                         for field in fields(self.static_prepared)
                         if isinstance(getattr(self.static_prepared, field.name), torch.Tensor)}
            return replace(self.static_prepared, **snapshots)
