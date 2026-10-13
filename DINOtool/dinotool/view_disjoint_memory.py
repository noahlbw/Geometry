"""Read frozen Values only from outside a specified local image view.

This controls donor provenance in the semantic head. Backbone tokens already
contain image context, so this is not a causal removal of local RGB evidence.
"""
import torch

from .context_memory_observer import read_context_head, memory_read


def view_donor_weights(grid_hw, wide_hw, local_hw, tile, *, device):
    """Fractional visible patch area inside/outside a local512 rectangle."""
    gh, gw = grid_hw
    wh, ww = wide_hw
    lh, lw = local_hw
    if min(gh, gw, wh, ww, lh, lw) <= 0 or gh*16 < wh or gw*16 < ww:
        raise ValueError('Positive compatible grids and image dimensions required.')
    y0 = torch.arange(gh, device=device, dtype=torch.float32)*16
    x0 = torch.arange(gw, device=device, dtype=torch.float32)*16
    y1, x1 = (y0+16).clamp_max(wh), (x0+16).clamp_max(ww)
    visible = ((y1-y0).clamp_min(0)[:, None]*(x1-x0).clamp_min(0)[None])/256
    top, left = tile['top']*wh/lh, tile['left']*ww/lw
    bottom, right = min(tile['top']+512, lh)*wh/lh, min(tile['left']+512, lw)*ww/lw
    iy = (torch.minimum(y1, y1.new_tensor(bottom))-torch.maximum(y0, y0.new_tensor(top))).clamp_min(0)
    ix = (torch.minimum(x1, x1.new_tensor(right))-torch.maximum(x0, x0.new_tensor(left))).clamp_min(0)
    inside = (iy[:, None]*ix[None])/256
    outside = (visible-inside).clamp_min(0)
    return visible.flatten(), inside.flatten(), outside.flatten()


def weighted_memory_factors(assignment, donor_weights):
    if assignment.ndim != 3 or assignment.shape[0] != 1 or donor_weights.shape != assignment.shape[1:2]:
        raise ValueError('Batch-one [1,N,K] assignment and [N] donor weights required.')
    p = assignment.float()
    collect = p*donor_weights[None, :, None].float()
    mass = collect.sum(1)
    if not bool((mass > 0).all()):
        raise ValueError('Every soft memory slot must have observed donors.')
    return p, collect, mass


def weighted_memory_read(factors, values):
    with torch.autocast(device_type=values.device.type, enabled=False):
        distribute, collect, mass = factors
        slots = (collect.transpose(1, 2) @ values.float())/mass[..., None]
        return (distribute @ slots).to(values.dtype)


@torch.inference_mode()
def batched_view_head(head, raw, assignment, donor_weights):
    """Share the first frozen Value projection across independent donor views."""
    if not donor_weights:
        raise ValueError('At least one donor view required.')
    factors = [weighted_memory_factors(assignment, weights) for weights in donor_weights]
    batched = tuple(torch.cat([row[index] for row in factors], dim=0) for index in range(3))
    # Block1 receives batch-one raw/Values, and independent reads broadcast
    # them into T views. Block2 then acts independently on each view's state.
    return read_context_head(head, raw, batched, weighted_memory_read)


def retained_query_factors(assignment, outside_weights, *, source_weights=None, uniform=False):
    """Keep original slot-specific outside probability; return removed mass to self.

    Alternative source weights retain exactly the same slot/query self mass,
    isolating actual donor provenance from a generic self-Value mixture.
    """
    p = assignment.float()
    full_mass = p.sum(1)
    outside_mass = (p*outside_weights[None,:,None]).sum(1)
    distribute = p*(outside_mass/full_mass.clamp_min(1e-20))[:,None]
    self_mass = 1-distribute.sum(-1,keepdim=True)
    weights = outside_weights if source_weights is None else source_weights
    collect = (torch.ones_like(p) if uniform else p)*weights[None,:,None]
    mass = collect.sum(1).clamp_min(1e-20)
    return distribute, collect, mass, self_mass


def retained_query_read(factors, values):
    with torch.autocast(device_type=values.device.type, enabled=False):
        distribute, collect, mass, self_mass = factors
        v = values.float()
        slots = (collect.transpose(1,2) @ v)/mass[...,None]
        return (distribute @ slots+self_mass*v).to(values.dtype)


@torch.inference_mode()
def retained_context_head(head, raw, assignment, factors, *, final_only=False):
    if not final_only:
        return read_context_head(head,raw,factors,retained_query_read)
    if len(head.blocks) != 2:
        raise ValueError('The final-only comparison requires the frozen two-block head.')
    count = 0
    def read(factors, values):
        nonlocal count
        count += 1
        return memory_read(assignment,values) if count==1 else retained_query_read(factors,values)
    return read_context_head(head,raw,factors,read)
