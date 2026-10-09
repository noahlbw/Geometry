"""Read Geometry supports with a frozen, independently pretrained semantic head."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .gear_ov import _crop_at
from .geometry_readout_trace import alias_class_scores
from .prompts import REMOTE_SENSING_TEMPLATES
from .support_conditioned_geometry import SupportConfig, partition_supports


IMPLEMENTATION = "geometry-regional-siglip2-trained-pool-v2-native-scale-20261001"
METHODS = ("Geometry", "CropGlobalFusion", "RegionSemantic", "RegionFusion", "RegionJoint")


@dataclass(frozen=True)
class RegionReadoutConfig:
    fine_segments: int = 64
    coarse_segments: int = 16
    crop_minimum: int = 64
    crop_margin: int = 16
    encoder_batch: int = 8
    alias_temperature: float = .07
    solver_iterations: int = 32
    solver_tolerance: float = 1e-5


def restricted_pool(head, hidden, support):
    """Reuse the trained probe, K/V projections, residual and MLP on support tokens.

    Input tokens are contextualized by the unmasked real image. Restricting the
    pool does not claim that their backbone features have no context influence.
    """
    if hidden.ndim != 3 or support.shape != hidden.shape[:2]:
        raise ValueError("Expected [R,N,D] features and [R,N] support weights.")
    if not bool(torch.isfinite(support).all()) or bool((support < 0).any()):
        raise ValueError("Support weights must be finite and nonnegative.")
    if not bool((support.sum(-1) > 0).all()):
        raise ValueError("An empty support cannot enter the semantic pool.")
    weights = support.float() / support.sum(-1, keepdim=True)
    bias = weights.clamp_min(1e-30).log().masked_fill(weights == 0, -torch.inf)
    bias = bias-bias.amax(-1, keepdim=True)
    bias = bias[:, None].repeat_interleave(head.attention.num_heads, dim=0)
    probe = head.probe.expand(len(hidden), -1, -1)
    value = head.attention(probe, hidden, hidden, attn_mask=bias.to(hidden.dtype))[0]
    return (value + head.mlp(head.layernorm(value)))[:, 0]


def support_weights(raw, relation, valid, config):
    """Keep every connected fine/coarse support, including one-token supports."""
    height, width = valid.shape
    masks, rows, levels = [], [], []
    for level, segments in enumerate((config.fine_segments, config.coarse_segments)):
        labels = partition_supports(raw.reshape(height, width, -1), valid,
                                    SupportConfig(segments=segments, maximum_regions=1))
        for label in torch.unique(labels[labels >= 0]).tolist():
            mask = labels == label
            selected = mask.flatten()
            row = relation[selected].float().mean(0).masked_fill(~selected, 0.)
            if float(row.sum()) == 0:
                row = selected.float()
            masks.append(mask)
            rows.append(row / row.sum())
            levels.append(level)
    if not rows:
        raise ValueError("No valid Geometry support.")
    h = torch.stack(rows)
    masks = torch.stack(masks)
    weights = h * masks.flatten(1).sum(-1, keepdim=True)
    weights = weights / weights.sum(0, keepdim=True).clamp_min(1e-12)
    return h, weights, masks, levels


def region_crops(rgb, operator, masks, config, grid=16):
    """Preserve real crop content; project the same support into crop coordinates."""
    crops, crop_supports = [], []
    support_grid = operator.reshape(-1, 1, *masks.shape[-2:])
    stride = rgb.shape[-1] / masks.shape[-1]
    for index, mask in enumerate(masks):
        coords = mask.nonzero()
        lower, upper = coords.amin(0), coords.amax(0) + 1
        extent = max(config.crop_minimum,
                     math.ceil(float(((upper-lower)*stride).max()) + 2*config.crop_margin))
        extent = math.ceil(extent / 16) * 16
        center = (lower+upper).float() * stride / 2
        top, left = math.floor(float(center[0])-extent/2), math.floor(float(center[1])-extent/2)
        crop = _crop_at(rgb[0], top, left, extent)
        crops.append(F.interpolate(crop, (256, 256), mode="bilinear",
                                   align_corners=False, antialias=True)[0])
        yy = top + (torch.arange(grid, device=rgb.device)+.5)*extent/grid
        xx = left + (torch.arange(grid, device=rgb.device)+.5)*extent/grid
        y, x = torch.meshgrid(yy, xx, indexing="ij")
        coordinates = torch.stack((2*x/rgb.shape[-1]-1, 2*y/rgb.shape[-2]-1), -1)
        weights = F.grid_sample(support_grid[index:index+1].float(), coordinates[None],
                                mode="bilinear", align_corners=False)[0, 0].flatten()
        crop_supports.append(weights)
    return torch.stack(crops), torch.stack(crop_supports)


def semantic_probability(context, detail, available, temperature):
    pc, pd = (context / temperature).softmax(-1), (detail / temperature).softmax(-1)
    pd = torch.where(available[:, None], pd, pc)
    q = (pc+pd)/2
    entropy = -(q*q.clamp_min(1e-30).log()).sum(-1)
    js = .5*((pc*(pc.clamp_min(1e-30).log()-q.clamp_min(1e-30).log())).sum(-1)
             +(pd*(pd.clamp_min(1e-30).log()-q.clamp_min(1e-30).log())).sum(-1))
    confidence = (1-entropy/math.log(q.shape[-1])).clamp(0, 1)
    agreement = (1-js/math.log(2)).clamp(0, 1)
    # Confidence and view agreement control influence, not correctness probability.
    return q, confidence*agreement, js


def joint_energy(p, z, geometry_probability, semantic_probability, weights):
    log = lambda x: x.clamp_min(1e-30).log()
    n = weights.sum(-1)
    local = (p*(log(p)-log(geometry_probability))).sum()
    semantic = (n[:, None]*semantic_probability*(log(semantic_probability)-log(z))).sum()
    coupling = (weights @ (p*log(p)).sum(-1)).sum() - ((weights @ p)*log(z)).sum()
    return local+semantic+coupling


def joint_readout(local, q, weights, config):
    """Alternating exact block minimizers of a jointly convex KL energy.

    z is a latent regional class prior, not a predicted pixel area fraction.
    min KL(p||g)+sum_r n_r KL(q_r||z_r)+sum_ri W_ri KL(p_i||z_r).
    """
    if local.ndim != 2 or q.shape[-1] != local.shape[-1] or weights.shape != (len(q), len(local)):
        raise ValueError("Invalid local, semantic or support dimensions.")
    if not all(bool(torch.isfinite(x).all()) for x in (local, q, weights)):
        raise ValueError("Readout inputs must be finite.")
    if bool((weights < 0).any()) or bool((q < 0).any()):
        raise ValueError("Probabilities and support weights must be nonnegative.")
    temperature = config.alias_temperature
    g = (local.float()/temperature).softmax(-1)
    if float(weights.sum()) == 0:
        return local.clone(), q.clone(), {"solver_iterations": 0., "solver_max_change": 0.,
                                        "energy_before": 0., "energy_after": 0.}
    q = q.float()/q.sum(-1, keepdim=True).clamp_min(1e-30)
    w = weights.float()
    n, mass = w.sum(-1, keepdim=True), w.sum(0)[:, None]
    p, z = g.clone(), q.clone()
    log_g = g.clamp_min(1e-30).log()
    before = float(joint_energy(p, z, g, q, w))
    for step in range(config.solver_iterations):
        updated_p = ((log_g+w.T @ z.clamp_min(1e-30).log())/(1+mass)).softmax(-1)
        updated_z = (n*q+w @ updated_p)/(2*n).clamp_min(1e-30)
        updated_z = torch.where(n > 0, updated_z, q)
        change = max(float((updated_p-p).abs().max()), float((updated_z-z).abs().max()))
        p, z = updated_p, updated_z
        if change <= config.solver_tolerance:
            break
    after = float(joint_energy(p, z, g, q, w))
    if after > before + 1e-4*max(1., abs(before)):
        raise ValueError("Regional inference energy increased.")
    result = temperature*p.clamp_min(1e-30).log()
    result = torch.where(mass > 0, result, local)
    return result, z, {"solver_iterations": float(step+1), "solver_max_change": change,
                       "energy_before": before, "energy_after": after}


class FrozenRegionObserver:
    encoder_size = 256

    def __init__(self, source, banks, device, config=RegionReadoutConfig()):
        from transformers import AutoModel, AutoTokenizer

        self.config, self.device = config, device
        self.manifest = json.loads((Path(source)/"region_source_manifest.json").read_text())
        self.model = AutoModel.from_pretrained(source, local_files_only=True,
                                               attn_implementation="eager").to(device).eval()
        self.model.requires_grad_(False)
        self.semantic_temperature = float(self.model.logit_scale.detach().float().neg().exp())
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        self.texts = {}
        for key, bank in banks.items():
            prompts = [template.format(label=alias) for alias in bank.alias_names
                       for template in REMOTE_SENSING_TEMPLATES]
            parts = []
            for start in range(0, len(prompts), 64):
                inputs = self.tokenizer(prompts[start:start+64], padding="max_length",
                                        truncation=True, max_length=64, return_tensors="pt")
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    text = self.model.text_model(input_ids=inputs.input_ids.to(device)).pooler_output
                parts.append(F.normalize(text.float(), dim=-1))
            self.texts[key] = F.normalize(torch.cat(parts).reshape(len(bank.alias_names),
                len(REMOTE_SENSING_TEMPLATES), -1).mean(1), dim=-1)
        self.last_audit = None

    @torch.inference_mode()
    def visual(self, images):
        pixels = 2*F.interpolate(images.float(), (self.encoder_size, self.encoder_size), mode="bilinear",
                                align_corners=False, antialias=True)-1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model.vision_model(pixel_values=pixels)
        return output.last_hidden_state, F.normalize(output.pooler_output.float(), dim=-1)

    def context_supports(self, operator, grid_valid):
        return F.interpolate(operator.reshape(-1, 1, *grid_valid.shape),
                             (16, 16), mode="area").flatten(1)

    def detail_crops(self, rgb, operator, masks):
        return region_crops(rgb, operator, masks, self.config)

    @torch.inference_mode()
    def __call__(self, geometry, prepared, banks, texts, valid, rgb):
        started = time.perf_counter()
        grid_valid = valid[0].reshape(prepared.grid_height, prepared.grid_width)
        operator, base_weights, masks, levels = support_weights(prepared.raw_patch_tokens[0],
            prepared.geometry_patch_conditional[0], grid_valid, self.config)
        partition_seconds = time.perf_counter()-started
        hidden, global_context = self.visual(rgb)
        context_weights = self.context_supports(operator, grid_valid)
        head = self.model.vision_model.head
        with torch.autocast("cuda", dtype=torch.bfloat16):
            context = restricted_pool(head, hidden.expand(len(operator), -1, -1), context_weights)
        context = F.normalize(context.float(), dim=-1)
        crops, detail_weights = self.detail_crops(rgb, operator, masks)
        available = detail_weights.sum(-1) > 0
        detail, global_detail = [], []
        for start in range(0, len(crops), self.config.encoder_batch):
            end = start+self.config.encoder_batch
            features, descriptor = self.visual(crops[start:end])
            weights = detail_weights[start:end].clone()
            weights[~available[start:end]] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                descriptor_region = restricted_pool(head, features, weights)
            detail.append(F.normalize(descriptor_region.float(), dim=-1))
            global_detail.append(descriptor)
        detail, global_detail = torch.cat(detail), torch.cat(global_detail)
        scores, audit = {}, {}
        diagnostics = {"regions": float(len(operator)), "fine_regions": float(levels.count(0)),
                       "coarse_regions": float(levels.count(1)),
                       "one_token_regions": float((masks.flatten(1).sum(-1) == 1).sum()),
                       "detail_support_available": float(available.float().mean()),
                       "partition_seconds": partition_seconds}
        for key, bank in banks.items():
            text = self.texts[key]
            local = alias_class_scores(prepared.geometry_projected.float() @ texts[key].T,
                                       bank.parent_indices, bank.class_count)[0]
            def classes(value):
                return alias_class_scores(value @ text.T, bank.parent_indices, bank.class_count,
                                          self.semantic_temperature)
            contextual, detailed = classes(context), classes(detail)
            q, trust, js = semantic_probability(contextual, detailed, available, self.semantic_temperature)
            global_q, global_trust, _ = semantic_probability(classes(global_context).expand_as(contextual),
                classes(global_detail), torch.ones_like(available), self.semantic_temperature)
            weights, global_weights = base_weights*trust[:, None], base_weights*global_trust[:, None]
            def fuse(probability, influence, semantic_only=False):
                mass = influence.sum(0)[:, None]
                pooled = influence.T @ probability.clamp_min(1e-30).log()
                g = (local/self.config.alias_temperature).log_softmax(-1)
                logits = pooled/mass.clamp_min(1e-30) if semantic_only else (g+pooled)/(1+mass)
                return torch.where(mass > 0, self.config.alias_temperature*logits, local)
            joint, posterior, solver = joint_readout(local, q, weights, self.config)
            output = {"Geometry": local, "CropGlobalFusion": fuse(global_q, global_weights),
                      "RegionSemantic": fuse(q, weights, True), "RegionFusion": fuse(q, weights),
                      "RegionJoint": joint}
            scores[key] = {method: value[None] for method, value in output.items()}
            audit[key] = {"Geometry": operator @ local, "CropGlobal": global_q,
                          "RegionSemantic": q, "RegionPosterior": posterior}
            for field, value in {**solver, "semantic_influence": float(trust.mean()),
                    "view_js_divergence": float(js.mean()),
                    "view_class_agreement": float((contextual.argmax(-1) == detailed.argmax(-1)).float().mean()),
                    "changed_patch_fraction": float(((joint.argmax(-1) != local.argmax(-1)) & valid[0]).sum()/valid.sum()),
                    "mean_abs_class_delta": float(((joint-joint.mean(-1, keepdim=True))
                        -(local-local.mean(-1, keepdim=True))).abs().mean())}.items():
                diagnostics[field] = diagnostics.get(field, 0.)+value/len(banks)
        self.last_audit = {"operator": operator.cpu(), "support_sizes": masks.flatten(1).sum(-1).cpu(),
                           "levels": levels, "observations": {key: {method: values.cpu() for method, values in group.items()}
                                                               for key, group in audit.items()}}
        diagnostics["observer_readout_seconds"] = time.perf_counter()-started
        return scores, diagnostics
