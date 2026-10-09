"""Observe the existing Geometry head without modifying its prediction path."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .tcpr import _finish_head, _geometry_attended


@torch.inference_mode()
def trace_geometry_head(head, prepared, config):
    first = prepared.block_index - config.geometry_depth + 1
    current = prepared.backbone_tokens
    features = {"backbone_tokens": current}
    probes = {"backbone": current}
    components = {}
    for index, block in enumerate(head.blocks[:prepared.block_index + 1]):
        if index < first:
            current = block(current)
            continue
        features[f"block{index}_input"] = current
        attention = block.attn
        qkv = attention.qkv(block.norm1(current))
        batch, count, channels3 = qkv.shape
        channels = channels3 // 3
        qkv = qkv.reshape(batch, count, 3, attention.num_heads, channels // attention.num_heads)
        query, key, value = (part.transpose(1, 2) for part in torch.unbind(qkv, dim=2))
        logits = (query.float() @ key.float().transpose(-1, -2)) * attention.scale
        native = logits.softmax(-1).to(value.dtype)
        attended = _geometry_attended(native, prepared.geometry_patch_conditional, value,
                                     prepared.prefix_tokens, config.prefix_policy)
        merged = attended.transpose(1, 2).reshape(batch, count, channels)
        increment = block.ls1(attention.proj_drop(attention.proj(merged)))
        after_attention = current + increment
        mlp_increment = block.ls2(block.mlp(block.norm2(after_attention)))
        after_mlp = after_attention + mlp_increment
        features.update({f"block{index}_values": value,
                         f"block{index}_attention_increment": increment,
                         f"block{index}_after_attention": after_attention,
                         f"block{index}_mlp_increment": mlp_increment,
                         f"block{index}_after_mlp": after_mlp})
        probes[f"block{index}_after_attention"] = after_attention
        probes[f"block{index}_after_mlp"] = after_mlp
        if index == prepared.block_index:
            prefix = prepared.prefix_tokens
            prefix_read = native[..., prefix:, :prefix] @ value[..., :prefix, :]
            patch_mass = native[..., prefix:, prefix:].sum(-1, keepdim=True)
            patch_read = prepared.geometry_patch_conditional[:, None].to(value.dtype) @ value[..., prefix:, :]
            if config.prefix_policy == "preserve":
                patch_read = patch_mass * patch_read
            else:
                prefix_read = torch.zeros_like(prefix_read)
            def project_component(part):
                part = part.transpose(1, 2).reshape(batch, count - prefix, channels)
                projected = F.linear(part.float(), attention.proj.weight.float(), None)
                return block.ls1(projected).float()
            prefix_component = project_component(prefix_read)
            patch_component = project_component(patch_read)
            if attention.proj.bias is None:
                bias_component = torch.zeros_like(prefix_component)
            else:
                bias_component = block.ls1(attention.proj.bias.float())[None, None].expand_as(prefix_component)
            components = {"residual": current[:, prefix:].float(),
                          "prefix": prefix_component, "geometry_patch": patch_component,
                          "attention_bias": bias_component, "mlp": mlp_increment[:, prefix:].float()}
            components["rounding"] = after_mlp[:, prefix:].float() - sum(components.values())
            features["final_patch_mass"] = patch_mass
        current = after_mlp
    projected = _finish_head(head, current, prepared.block_index)
    features["final_pre_norm"] = current
    features["final_projected"] = projected
    return {"features": features, "probes": probes, "components": components,
            "projected": projected, "pre_norm": current}


def alias_class_scores(alias_scores, parents, classes, temperature=.07):
    return torch.stack([
        temperature * (torch.logsumexp(alias_scores[..., parents == c] / temperature, -1)
                       - math.log(int((parents == c).sum())))
        for c in range(classes)], dim=-1)


@torch.inference_mode()
def probe_scores(head, trace, bank, prefix):
    scores = {}
    text = F.normalize(bank.features.float(), dim=-1)
    for name, state in trace["probes"].items():
        projected = head.linear_projection(head.ln_final(state))[:, prefix:]
        with torch.autocast(device_type=text.device.type, enabled=False):
            aliases = F.normalize(projected.float(), dim=-1) @ text.T
            scores[name] = alias_class_scores(aliases, bank.parent_indices, bank.class_count)
    before_norm = head.linear_projection(trace["pre_norm"])[:, prefix:]
    final = trace["projected"][:, prefix:]
    with torch.autocast(device_type=text.device.type, enabled=False):
        scores["final_without_norm_probe"] = alias_class_scores(
            F.normalize(before_norm.float(), dim=-1) @ text.T, bank.parent_indices, bank.class_count)
        aliases = F.normalize(final.float(), dim=-1) @ text.T
        scores["final"] = alias_class_scores(aliases, bank.parent_indices, bank.class_count)
    return scores, aliases


@torch.inference_mode()
def exact_alias_attribution(head, trace, bank, prefix):
    """Conditional additive contributions, with observed LN and L2 denominators."""
    state = trace["pre_norm"][:, prefix:].float()
    norm = head.ln_final
    mean = state.mean(-1, keepdim=True)
    scale = ((state - mean).square().mean(-1, keepdim=True) + norm.eps).rsqrt()
    weight = norm.weight.float()
    projection = head.linear_projection

    def project(value, include_bias=False):
        if isinstance(projection, torch.nn.Identity):
            return value
        return F.linear(value, projection.weight.float(),
                        projection.bias if include_bias else None)

    projected_components = {}
    for name, component in trace["components"].items():
        normalized = (component.float() - component.float().mean(-1, keepdim=True)) * scale * weight
        projected_components[name] = project(normalized)
    bias = project(norm.bias.float(), include_bias=True)[None, None].expand_as(trace["projected"][:, prefix:])
    projected_components["norm_bias"] = bias
    final = trace["projected"][:, prefix:].float()
    projected_components["projection_rounding"] = final - sum(projected_components.values())
    denominator = final.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    text = F.normalize(bank.features.float(), dim=-1)
    return {name: (component / denominator) @ text.T
            for name, component in projected_components.items()}


def class_attribution(alias_scores, alias_components, parents, classes, temperature=.07):
    result = {name: [] for name in alias_components}
    entropy = []
    for c in range(classes):
        active = parents == c
        scaled = alias_scores[..., active] / temperature
        responsibility = scaled.softmax(-1)
        for name, component in alias_components.items():
            result[name].append((responsibility * component[..., active]).sum(-1))
        log_responsibility = scaled.log_softmax(-1)
        entropy.append(temperature * (-(responsibility * log_responsibility).sum(-1)
                                       - math.log(int(active.sum()))))
    output = {name: torch.stack(values, -1) for name, values in result.items()}
    output["alias_entropy"] = torch.stack(entropy, -1)
    return output
