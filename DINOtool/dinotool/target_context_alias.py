"""Matched-view target/context interventions for bounded contextual alias admission."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .calibrated_competitive_alias import profiled_logits
from .excess_alias_rejection import excess_delta
from .stratified_soft_alias import crop_stencil


IMPLEMENTATION = 'geometry-target-context-alias-source-v1-20261003'
PRIMARY = 'TargetContext_Exact'
SHUFFLED = tuple('ShuffledAlias'+str(i)+'_Exact' for i in range(3))
METHODS = ('Geometry', 'Anchored_Exact', 'ClassRelativeReject_CG', PRIMARY,
           'ContextOnly_Exact', 'ShuffledSupport_Exact', *SHUFFLED,
           'MeanLogit_Original', 'MeanLogit_TargetContext')


@dataclass(frozen=True)
class TargetContextConfig:
    cell_pixels: int = 128
    minimum_effective_support: float = 4.
    canonical_temperature: float = .07
    random_seed: int = 20261003
    epsilon: float = 1e-6


def geometry_masks(relation, coordinates, valid, config=TargetContextConfig()):
    n = len(valid)
    if relation.shape != (n, n) or coordinates.shape != (n, 2) or n != 1024:
        raise ValueError('One fixed512 window with32x32 Geometry coordinates required.')
    if bool((relation < 0).any()) or not bool(torch.isfinite(relation).all()):
        raise ValueError('Finite nonnegative Geometry required.')
    if config.cell_pixels != 128:
        raise ValueError('Frozen4x4 shared support protocol required.')
    cells = (coordinates/128).floor().long().clamp(0, 3)
    assignments = cells[:, 0]*4+cells[:, 1]
    masks, supported = [], []
    for cell in range(16):
        rows = valid & (assignments == cell)
        if not bool(rows.any()) or not bool(valid.any()):
            masks.append(relation.new_zeros(n))
            supported.append(False)
            continue
        weights = relation[rows].float().mean(0)
        low, high = weights[valid].min(), weights[valid].max()
        mask = ((weights-low)/(high-low).clamp_min(config.epsilon)).clamp(0, 1).masked_fill(~valid, 0.)
        complement = (1-mask).masked_fill(~valid, 0.)
        effective = [value.sum().square()/value.square().sum().clamp_min(config.epsilon**2)
                     for value in (mask, complement)]
        known = bool(high-low > config.epsilon and all(value >= config.minimum_effective_support for value in effective))
        masks.append(mask if known else torch.zeros_like(mask))
        supported.append(known)
    return torch.stack(masks).reshape(16, 32, 32), assignments, torch.tensor(supported, device=relation.device)


def crop_masks(masks, crop, resized_size, image_size):
    """Map original local512 support to the exact unchanged336 wide crop."""
    device = masks.device
    yy = (crop.top+torch.arange(336, device=device).float()+.5)*image_size[0]/resized_size[0]-.5
    xx = (crop.left+torch.arange(336, device=device).float()+.5)*image_size[1]/resized_size[1]-.5
    y, x = torch.meshgrid(yy, xx, indexing='ij')
    grid = torch.stack((2*(x+.5)/512-1, 2*(y+.5)/512-1), -1)
    enlarged = F.interpolate(masks[:, None], (512, 512), mode='bilinear', align_corners=False)
    values = F.grid_sample(enlarged, grid[None].expand(len(masks), -1, -1, -1),
                           mode='bilinear', padding_mode='zeros', align_corners=False)[:, 0]
    valid = ((y >= -.5) & (y < 511.5) & (x >= -.5) & (x < 511.5))
    valid[crop.actual_height:] = False
    valid[:, crop.actual_width:] = False
    return values.masked_fill(~valid, 0.)


def shuffled_crop_masks(masks, actual_height, actual_width, seed):
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(actual_height*actual_width, generator=generator).to(masks.device)
    result = torch.zeros_like(masks)
    result[:, :actual_height, :actual_width] = masks[:, :actual_height, :actual_width].flatten(1)[:, permutation].reshape(
        len(masks), actual_height, actual_width)
    return result


def intervention_retention(full, kept, removed, parents, canonical, supported, config=TargetContextConfig()):
    if (kept.shape != removed.shape or kept.ndim != 3 or kept.shape[0] != 2
            or kept.shape[1:] != full.shape or supported.shape != full.shape[:1]
            or len(parents) != full.shape[-1]):
        raise ValueError('Two fills with matched raw alias responses and query validity required.')
    contextual = (full[None]-kept).clamp_min(0)/((full[None]-kept).abs()+(full[None]-removed).abs()).clamp_min(config.epsilon)
    p = (kept[:, :, canonical]/config.canonical_temperature).softmax(-1)
    own = p[:, :, parents, None]
    rivals = p[:, :, None]
    leakage = (rivals-own).clamp_min(0)/(rivals+own).clamp_min(config.epsilon)
    risk = (contextual[..., None]*leakage).amin(0).amax(-1)
    context_only = contextual.amin(0)
    for value in (risk, context_only):
        value[:, canonical] = 0.
        value.masked_fill_(~supported[:, None], 0.)
    return 1-risk.clamp(0, 1), 1-context_only.clamp(0, 1)


def alias_permutations(members, canonical, seed=20261003):
    result = []
    for offset in range(3):
        generator = torch.Generator().manual_seed(seed+offset)
        permutation = torch.arange(members.shape[-1], device=members.device).expand_as(members).clone()
        for c, row in enumerate(members):
            slots = (row != canonical[c]).nonzero().flatten()
            permutation[c, slots] = slots[torch.randperm(len(slots), generator=generator).to(slots.device)]
        result.append(permutation)
    return result


def observation_delta(crops, count, coordinates, image_size, members, retention, valid, beta=1.):
    output = coordinates.new_zeros((len(coordinates), len(members)))
    for start in range(0, len(coordinates), 128):
        sl = slice(start, start+128)
        grouped = retention[sl, members]
        for crop in crops:
            indices, coefficients = crop_stencil(crop, count, coordinates[sl], image_size)
            evidence = profiled_logits(crop, members)[indices]
            output[sl] += (excess_delta(evidence, grouped[:, None].expand_as(evidence), beta)*coefficients[..., None]).sum(1)
    return output.masked_fill(~valid[:, None], 0.)
