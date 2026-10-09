"""Read frozen native backbone donors with Geometry-conditioned real queries."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .matched_readout_controls import run_head


IMPLEMENTATION = "geometry-native-donor-reacquisition-v1-last4-20261002"
PRIMARY = "Reacquired_Geometry"
METHODS = ("Geometry", "SCLIP_Two", "VIPProxy_Two", "Reacquired_Native",
           "MeanLogit_Reacquired", PRIMARY)


@dataclass(frozen=True)
class ReacquisitionConfig:
    last_blocks: int = 4

    def validate(self, count: int) -> None:
        if not 1 <= self.last_blocks <= count:
            raise ValueError("Replay depth must be within the backbone block count.")

    def signature(self) -> dict:
        return {**asdict(self), "implementation": IMPLEMENTATION,
                "donor_stream": "cached unmodified native backbone",
                "query_stream": "real patch queries with original residual/MLP/RoPE",
                "attention": "native output + native patch mass * (conditioned patch read - native patch read)",
                "relation": "same original Geometry G in replay and final Geometry head",
                "special_path": "fixed native attention contribution during backbone replay",
                "text": "fixed20 aliases/six RS templates/normalized LME .07",
                "view": "native512/128 overlap/Hann probability blending"}


@dataclass
class NativeBlockRecord:
    state: Tensor
    rope: tuple[Tensor, Tensor] | None


def split_qkv(attention, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    encoded = attention.qkv(state)
    batch, count, channels3 = encoded.shape
    heads = attention.num_heads
    parts = encoded.reshape(batch, count, 3, heads, channels3 // (3 * heads)).unbind(2)
    return tuple(part.transpose(1, 2) for part in parts)


def relation_bias(relation: Tensor, patches: int) -> Tensor | None:
    if (relation.ndim != 3 or relation.shape[-2:] != (patches, patches)
            or not bool(torch.isfinite(relation).all()) or bool((relation < 0).any())
            or bool((relation.sum(-1) <= 0).any())):
        raise ValueError("Replay requires finite nonnegative relations with nonempty rows.")
    # Row constants cancel exactly. Omit their mask to replay the native kernel.
    if bool((relation == relation[..., :1]).all()):
        return None
    return relation.float().log()[:, None]


def replay_block(block, query_state: Tensor, record: NativeBlockRecord,
                 bias: Tensor | None, prefix: int) -> tuple[Tensor, dict[str, float]]:
    native = record.state
    if (block.training or native.ndim != 3 or not 0 <= prefix < native.shape[1]
            or query_state.shape != native[:, prefix:].shape):
        raise ValueError("Frozen replay requires matching real patch queries and native donors.")
    attention = block.attn
    native_query, key, value = split_qkv(attention, block.norm1(native))
    current = torch.cat((native[:, :prefix], query_state), dim=1)
    query, _, _ = split_qkv(attention, block.norm1(current))
    if record.rope is not None:
        unrotated_key = key
        native_query, key = attention.apply_rope(native_query, key, record.rope)
        # apply_rope expects query/key grids of equal length including prefixes.
        query, _ = attention.apply_rope(query, unrotated_key, record.rope)
    scale = attention.scale
    native_read = F.scaled_dot_product_attention(native_query, key, value, scale=scale)
    patch_query = native_query[..., prefix:, :]
    patch_key, patch_value = key[..., prefix:, :], value[..., prefix:, :]
    native_patch_read = F.scaled_dot_product_attention(patch_query, patch_key, patch_value, scale=scale)
    conditioned_read = F.scaled_dot_product_attention(query[..., prefix:, :], patch_key, patch_value,
                                                     attn_mask=bias, scale=scale)
    weights = ((patch_query.float() @ key.float().transpose(-1, -2)) * scale).softmax(-1)
    mass = weights[..., prefix:].sum(-1, keepdim=True)
    # This form preserves the fused native output and gives exact zero displacement
    # when the query and conditional patch distribution have not changed.
    displacement = (mass * (conditioned_read.float() - native_patch_read.float())).to(native_read.dtype)
    output = torch.cat((native_read[..., :prefix, :], native_read[..., prefix:, :] + displacement), dim=-2)
    merged = output.transpose(1, 2).reshape_as(native)
    after_attention = current + block.ls1(attention.proj_drop(attention.proj(merged)))
    after_mlp = after_attention + block.ls2(block.mlp(block.norm2(after_attention)))
    return after_mlp[:, prefix:], {
        "native_patch_mass": float(mass.mean()),
        "mean_abs_attention_displacement": float(displacement.float().abs().mean()),
    }


class GeometryReacquisition:
    """Capture an unchanged native forward, then reread its last frozen blocks."""

    def __init__(self, backbone, config: ReacquisitionConfig = ReacquisitionConfig()):
        self.backbone = backbone
        self.visual = backbone.model.visual_model
        self.config = config
        config.validate(len(self.visual.backbone.blocks))
        if self.visual.patch_token_layer != 1 or self.visual.backbone.training:
            raise ValueError("This fixed candidate expects a frozen final-layer native patch stream.")
        self.blocks = tuple(self.visual.backbone.blocks[-config.last_blocks:])
        self.records: list[NativeBlockRecord] = []
        self.hooks = [block.register_forward_pre_hook(self._capture(index), with_kwargs=True)
                      for index, block in enumerate(self.blocks)]

    def _capture(self, index):
        def capture(module, args, kwargs):
            if index == 0:
                self.records.clear()
            state = args[0]
            rope = args[1] if len(args) > 1 else kwargs.get("rope_or_rope_list")
            if not isinstance(state, Tensor):
                raise ValueError("Expected the single-image native backbone path.")
            if rope is not None:
                rope = tuple(part.detach().clone() for part in rope)
            self.records.append(NativeBlockRecord(state.detach().clone(), rope))
        return capture

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.records.clear()

    @torch.inference_mode()
    def replay(self, prepared, relation: Tensor | None = None):
        if len(self.records) != len(self.blocks):
            raise RuntimeError("An unchanged native prepare_image forward must precede replay.")
        prefix = prepared.prefix_tokens
        state = self.records[0].state[:, prefix:]
        original_relation = prepared.geometry_patch_conditional
        relation = original_relation if relation is None else relation
        if relation.shape[0] != state.shape[0]:
            raise ValueError("Relation batch differs from real queries.")
        bias = relation_bias(relation, state.shape[1])
        diagnostics = {"native_patch_mass": 0., "mean_abs_attention_displacement": 0.}
        for block, record in zip(self.blocks, self.records):
            state, observed = replay_block(block, state, record, bias, prefix)
            for field, value in observed.items():
                diagnostics[field] += value / len(self.blocks)
        patches = self.visual.backbone.norm(state)
        tokens = torch.cat((prepared.backbone_tokens[:, :prefix], patches), dim=1)
        diagnostics.update(
            mean_backbone_cosine=float(F.cosine_similarity(patches.float(),
                                     prepared.backbone_tokens[:, prefix:].float(), dim=-1).mean()),
            mean_abs_backbone_change=float((patches.float()-prepared.backbone_tokens[:, prefix:].float()).abs().mean()),
            backbone_prefix_change=float((tokens[:, :prefix]-prepared.backbone_tokens[:, :prefix]).abs().max()) if prefix else 0.,
        )
        return tokens, diagnostics

    @torch.inference_mode()
    def read(self, prepared, *, controls=True):
        with self.backbone._autocast():
            tokens, diagnostics = self.replay(prepared)
            args = (self.visual.head, tokens, tokens[:, prepared.prefix_tokens:],
                    prepared.geometry_patch_conditional, prepared.prefix_tokens)
            coupled, _ = run_head(*args, "Geometry", prepared.block_index)
            features = {PRIMARY: coupled}
            if controls:
                features["Reacquired_Native"], _ = run_head(*args, "Native", prepared.block_index)
                features["Geometry"] = prepared.geometry_projected
                for method in ("SCLIP_Two", "VIPProxy_Two"):
                    features[method], observed = run_head(
                        self.visual.head, prepared.backbone_tokens,
                        prepared.backbone_tokens[:, prepared.prefix_tokens:],
                        prepared.geometry_patch_conditional, prepared.prefix_tokens, method, prepared.block_index)
                    diagnostics.update(observed)
        if any(not bool(torch.isfinite(feature).all()) for feature in features.values()):
            raise ValueError("Nonfinite reacquired semantic features.")
        return features, diagnostics
