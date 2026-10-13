"""Geov2 v0.2: one frozen encoder, native wide observation, gated writeback.

All parameters are global. Text vectors are prepared offline; image inference
uses a single visual model. Dense structure is restricted to <=1024 patches.
"""
from dataclasses import dataclass, asdict
import math

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Geov2Config:
    wide_long: int = 448
    local_long: int = 896
    window: int = 512
    feature_temperature: float = .10
    spatial_sigma: float = .25
    alias_temperature: float = .07
    correction_lambda: float = 1.
    # The solver first computes the structure-anchored target Z*.  The
    # second stage controls how far the prediction moves from L towards Z*.
    # ``adaptive`` is the released setting; ``fixed`` is kept for matched
    # ablations and reproduces the old scalar-gain writer.
    gate_mode: str = 'adaptive'
    writeback_gain: float = 1.
    gate_min: float = .5
    gate_max: float = 2.
    gate_temperature: float = .25
    gate_adaptation: float = .75
    solve_chunk: int = 32
    output_rows: int = 64
    output_classes: int = 16


def structure(raw, height, width, config=Geov2Config()):
    """Complete symmetric Gaussian kernel, row read A, normalized S, degree."""
    with torch.autocast(device_type=raw.device.type, enabled=False):
        f = F.normalize(raw[0].float(), dim=-1)
        norm = f.square().sum(-1)
        feature_distance = (norm[:, None]+norm[None, :]-2*(f @ f.T)).clamp_min(0)
        y, x = torch.meshgrid(torch.arange(height, device=f.device),
                              torch.arange(width, device=f.device), indexing='ij')
        u = torch.stack((y, x), -1).reshape(-1, 2).float()*16/512
        distance = (u[:, None]-u[None, :]).square().sum(-1)
        kernel = torch.exp(-feature_distance/(2*config.feature_temperature)
                           -distance/(2*config.spatial_sigma**2))
        kernel.fill_diagonal_(1.)
        degree = kernel.sum(-1)
        root = degree.sqrt()
        return kernel/degree[:, None], kernel/root[:, None]/root[None, :], degree


def anchored_correction(local, wide, symmetric, degree, strength=1., chunk=32):
    """Solve (I+lambda S^2)Y=lambda S^2 sqrt(D)(W-L), Z=L+D^-1/2 Y."""
    with torch.autocast(device_type=local.device.type, enabled=False):
        local, wide = local.float(), wide.float()
        if strength == 0:
            return local.clone()
        square = symmetric @ symmetric
        system = strength*square
        system.diagonal().add_(1.)
        factor = torch.linalg.cholesky(system)
        root = degree.sqrt()[:, None]
        out = torch.empty_like(local)
        for start in range(0, local.shape[-1], chunk):
            end = start+chunk
            difference = root*(wide[:, start:end]-local[:, start:end])
            rhs = strength*(square @ difference)
            out[:, start:end] = local[:, start:end]+torch.cholesky_solve(rhs, factor)/root
        return out


def _normalized_margin(logits, eps=1e-6):
    """A label-free confidence proxy from the top two class responses."""
    if logits.shape[-1] < 2:
        return torch.ones(logits.shape[:-1], device=logits.device, dtype=logits.dtype)
    top = logits.topk(2, dim=-1).values
    margin = (top[..., 0] - top[..., 1]) / (top[..., 0].abs() + top[..., 1].abs() + eps)
    return margin.clamp(0., 1.)


def adaptive_writeback_gain(local, wide, target, *, g_min=.5, g_max=2.,
                            temperature=.25, adaptation=.75, eps=1e-6):
    """Compute an image-only step size for ``Z=L+g(Z*-L)``.

    ``target`` is the solution of the structure-anchored quadratic objective.
    The gate uses only quantities available at inference: how much the target
    correction agrees with the raw wide-minus-local displacement, and whether
    the wide branch is more decisive than the local branch.  A zero
    disagreement is an exact no-op and receives gain one.  The result is one
    scalar per local patch, so it adds O(N*C) arithmetic and no dense tensor.
    """
    if g_min < 0 or g_max < g_min:
        raise ValueError('Require 0 <= g_min <= g_max.')
    if temperature <= 0:
        raise ValueError('Gate temperature must be positive.')
    displacement = wide.float() - local.float()
    correction = target.float() - local.float()
    energy = displacement.square().sum(-1)
    # Projection of the structure-supported correction onto the observed
    # wide-minus-local direction.  Clamp it because a negative projection is
    # evidence that the two views disagree in this patch.
    support = (correction * displacement).sum(-1) / (energy + eps)
    support = support.clamp(0., 1.)
    local_margin = _normalized_margin(local.float(), eps)
    wide_margin = _normalized_margin(wide.float(), eps)
    wide_advantage = 2. * torch.sigmoid((wide_margin - local_margin) / temperature) - 1.
    if adaptation < 0:
        raise ValueError('Gate adaptation must be non-negative.')
    # Center the gate at the exact fixed-target writer (g=1).  This makes the
    # adaptive branch a bounded correction to a proven operating point instead
    # of an unconditional attenuation of every wide-view update.
    gain = 1. + adaptation * support * wide_advantage
    gain = gain.clamp(g_min, g_max)
    # When the views already agree the correction is zero.  Returning one in
    # that case makes the identity property explicit and keeps diagnostics
    # interpretable.
    return torch.where(energy <= eps, torch.ones_like(gain), gain)


def structure_writeback(local, wide, symmetric, degree, config=Geov2Config()):
    """Return ``(Z, Z*, g)`` for the released two-stage writer.

    First solve for ``Z*`` under the quadratic structure objective.  Then take
    a fixed or adaptive step towards that target.  ``gate_mode='target'`` is a
    useful diagnostic that exposes the un-damped optimum itself.
    """
    target = anchored_correction(local, wide, symmetric, degree,
                                 config.correction_lambda, config.solve_chunk)
    mode = config.gate_mode
    if mode == 'target':
        gain = torch.ones(local.shape[0], device=local.device, dtype=local.dtype)
    elif mode == 'fixed':
        gain = torch.full((local.shape[0],), float(config.writeback_gain),
                          device=local.device, dtype=local.dtype)
    elif mode == 'adaptive':
        gain = adaptive_writeback_gain(local, wide, target,
                                       g_min=config.gate_min,
                                       g_max=config.gate_max,
                                       temperature=config.gate_temperature,
                                       adaptation=config.gate_adaptation)
    else:
        raise ValueError("gate_mode must be 'adaptive', 'fixed', or 'target'.")
    result = local.float() + gain[:, None].float() * (target.float() - local.float())
    return result.to(local.dtype), target, gain


def value_head(head, raw, relation):
    """Frozen patch-only two-block head; no Q/K or prefix computations."""
    current = raw
    for block in head.blocks:
        linear = block.attn.qkv
        width = current.shape[-1]
        bias = None if linear.bias is None else linear.bias[2*width:]
        value = F.linear(block.norm1(current), linear.weight[2*width:], bias)
        with torch.autocast(device_type=raw.device.type, enabled=False):
            read = (relation @ value[0].float())[None].to(value.dtype)
        current = current+block.ls1(block.attn.proj_drop(block.attn.proj(read)))
        current = current+block.ls2(block.mlp(block.norm2(current)))
    return F.normalize(head.linear_projection(head.ln_final(current)).float(), dim=-1)


class AliasScorer:
    def __init__(self, features, parents, class_names, union_classes=(), temperature=.07):
        self.features = F.normalize(features.float(), dim=-1)
        self.names = tuple(class_names)
        self.parents = parents.long()
        self.indices = [torch.where(self.parents == c)[0] for c in range(len(self.names))]
        if any(len(i) == 0 for i in self.indices):
            raise ValueError('Every class requires at least one alias.')
        self.unions = frozenset(union_classes)
        self.temperature = temperature

    def __call__(self, features):
        with torch.autocast(device_type=features.device.type, enabled=False):
            aliases = features.float() @ self.features.T
            scores = []
            for c, index in enumerate(self.indices):
                values = aliases[..., index]
                if c in self.unions:
                    scores.append(values.max(-1).values)
                else:
                    scores.append(self.temperature*(torch.logsumexp(values/self.temperature, -1)
                                                     -math.log(len(index))))
            return torch.stack(scores, -1)


def resize_pad(rgb, long_edge):
    # Match the historical CPU bilinear resize boundary. Import lazily for CPU
    # algebra tests, which do not require OpenCV or model checkpoints.
    import cv2
    h, w = rgb.shape[-2:]
    rh = max(1, int(h*long_edge/max(h, w)+.5))
    rw = max(1, int(w*long_edge/max(h, w)+.5))
    data = rgb.detach().cpu().permute(1, 2, 0).contiguous().numpy()
    resized = cv2.resize(data, (rw, rh), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous()
    return F.pad(tensor, (0, (-rw) % 16, 0, (-rh) % 16), mode='replicate'), (rh, rw)


def window_starts(extent, window=512):
    return (0,) if extent <= window else (0, extent-window)


def sample_wide(scores, top, left, grid_hw, local_hw, wide_hw, wide_padded_hw):
    h, w = grid_hw
    device = scores.device
    y = (top+(torch.arange(h, device=device)+.5)*16)*wide_hw[0]/local_hw[0]
    x = (left+(torch.arange(w, device=device)+.5)*16)*wide_hw[1]/local_hw[1]
    yy, xx = torch.meshgrid(2*y/wide_padded_hw[0]-1, 2*x/wide_padded_hw[1]-1, indexing='ij')
    grid = torch.stack((xx, yy), -1)[None]
    return F.grid_sample(scores[None], grid, mode='bilinear', padding_mode='border',
                         align_corners=False)[0].flatten(1).T


def restore_ids(scores, output_hw, resized_hw, padded_hw, rows=64, classes=16):
    """Bilinear patch scores -> original CPU IDs, without C*H*W GPU planes."""
    oh, ow = output_hw
    count = scores.shape[0]
    result = np.empty((oh, ow), np.uint8 if count <= 256 else np.uint16)
    x = (torch.arange(ow, device=scores.device)+.5)*resized_hw[1]/ow
    x = 2*x/padded_hw[1]-1
    for top in range(0, oh, rows):
        bottom = min(top+rows, oh)
        y = (torch.arange(top, bottom, device=scores.device)+.5)*resized_hw[0]/oh
        yy, xx = torch.meshgrid(2*y/padded_hw[0]-1, x, indexing='ij')
        grid = torch.stack((xx, yy), -1)[None]
        best = torch.full((bottom-top, ow), -torch.inf, device=scores.device)
        ids = torch.zeros_like(best, dtype=torch.long)
        for start in range(0, count, classes):
            values = F.grid_sample(scores[None, start:start+classes], grid,
                                   mode='bilinear', padding_mode='border', align_corners=False)[0]
            value, label = values.max(0)
            replace = value > best
            ids = torch.where(replace, label+start, ids)
            best = torch.maximum(best, value)
        result[top:bottom] = ids.cpu().numpy()
    return result


class Geov2:
    def __init__(self, backbone, scorer, config=Geov2Config()):
        self.backbone, self.scorer, self.config = backbone, scorer, config
        self.device = backbone.device

    def raw_features(self, image):
        normalized = (image[None].to(self.device)-self.backbone._imagenet_mean)/self.backbone._imagenet_std
        with self.backbone._autocast():
            return self.backbone.model.visual_model.get_backbone_features(normalized)

    @torch.inference_mode()
    def predict(self, rgb, diagnostics=False):
        cfg = self.config
        wide_rgb, wide_hw = resize_pad(rgb, cfg.wide_long)
        cls, raw, registers = self.raw_features(wide_rgb)
        head = self.backbone.model.visual_model.head
        prefix = 1+registers.shape[1]
        with self.backbone._autocast():
            # The frozen DINOv3 head exposes the transformer state through
            # ``ln_final`` and ``linear_projection``; calling the wrapper
            # directly would require the upstream dino_feat side input.
            tokens = torch.cat((cls[:, None], registers, raw), dim=1)
            encoded = head.linear_projection(head.ln_final(tokens))
        wide_features = F.normalize(encoded[0, prefix:].float(), dim=-1)
        wh, ww = wide_rgb.shape[-2:]
        wide = self.scorer(wide_features).T.reshape(len(self.scorer.names), wh//16, ww//16)
        del cls, raw, registers, encoded, wide_features
        local_rgb, local_hw = resize_pad(rgb, cfg.local_long)
        lh, lw = local_rgb.shape[-2:]
        arms = ('Geov2', 'Local', 'Wide', 'Mean') if diagnostics else ('Geov2',)
        canvases = {k:torch.zeros((len(self.scorer.names), lh//16, lw//16), device=self.device) for k in arms}
        mass = torch.zeros((lh//16, lw//16), device=self.device)
        for top in window_starts(lh, cfg.window):
            for left in window_starts(lw, cfg.window):
                crop = local_rgb[:, top:top+cfg.window, left:left+cfg.window]
                h, w = crop.shape[-2]//16, crop.shape[-1]//16
                _, raw, _ = self.raw_features(crop)
                relation, symmetric, degree = structure(raw, h, w, cfg)
                with self.backbone._autocast():
                    local_features = value_head(head, raw, relation)[0]
                local = self.scorer(local_features)
                broad = sample_wide(wide, top, left, (h,w), local_hw, wide_hw, (wh,ww))
                corrected, target, gain = structure_writeback(
                    local, broad, symmetric, degree, cfg)
                values = {'Geov2':corrected}
                if diagnostics:
                    values.update(Local=local, Wide=broad, Mean=(local+broad)*.5)
                wy = torch.sin(math.pi*(torch.arange(h, device=self.device)+.5)/h).square()
                wx = torch.sin(math.pi*(torch.arange(w, device=self.device)+.5)/w).square()
                weight = wy[:, None]*wx[None, :]
                iy, ix = top//16, left//16
                for k in arms:
                    canvases[k][:, iy:iy+h, ix:ix+w] += values[k].T.reshape(-1,h,w)*weight
                mass[iy:iy+h, ix:ix+w] += weight
                del raw, relation, symmetric, degree, local_features, local, broad
                del corrected, target, gain, values
        out = {k:restore_ids(v/mass[None], rgb.shape[-2:], local_hw, (lh,lw),
                             cfg.output_rows, cfg.output_classes) for k,v in canvases.items()}
        return out if diagnostics else out['Geov2']

    def specification(self):
        return asdict(self.config)
