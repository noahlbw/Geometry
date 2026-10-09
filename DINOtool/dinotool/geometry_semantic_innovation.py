"""Replace supported semantic content while retaining Geometry's local residual."""
from __future__ import annotations

import torch


IMPLEMENTATION = "geometry-semantic-innovation-v3-finite-observer-20261001"


@torch.inference_mode()
def semantic_innovation(local, observation, relation, valid):
    """z = g + A(b-g), with the historical Geometry relation A.

    g-A g retains the local evidence not explained by the support readout.
    A b supplies genuinely different semantic content on that same support.
    Inputs are class logits in their declared softmax units, not probabilities.
    """
    if (local.ndim != 3 or observation.shape != local.shape
            or relation.shape != (*local.shape[:2], local.shape[1])
            or valid.shape != local.shape[:2]):
        raise ValueError("Expected matching [B,N,C] scores, [B,N,N] relation and [B,N] validity.")
    if not bool(torch.isfinite(local).all() and torch.isfinite(observation).all()
                and torch.isfinite(relation).all()) or bool((relation < 0).any()):
        raise ValueError("Scores and nonnegative relations must be finite.")
    with torch.autocast(device_type=local.device.type, enabled=False):
        weights = relation.float().masked_fill(~valid[:, None], 0.)
        mass = weights.sum(-1, keepdim=True)
        weights = weights / mass.clamp_min(1e-12)
        increment = weights @ (observation.float() - local.float())
        increment = torch.where(valid[..., None] & (mass > 0), increment, 0.)
        output = local.float() + increment
    return output, {
        "mean_absolute_innovation": float(increment.abs().mean()),
        "supported_donor_mass": float(mass.mean()),
        "changed_patch_fraction": float(((output.argmax(-1) != local.argmax(-1)) & valid).sum()
                                         / valid.sum().clamp_min(1)),
    }


@torch.inference_mode()
def anchored_innovation(local, observation, relation, valid):
    """Solve min_z .5||z-g||^2 + .5||A(z-b)||^2 without fitted weights.

    The local fidelity term prevents wholesale replacement. Conjugate gradient
    solves (I + A^T A) delta = A^T A (b-g); stopping is numerical, not semantic.
    """
    semantic_innovation(local, observation, relation, valid)
    with torch.autocast(device_type=local.device.type, enabled=False):
        weights = relation.float().masked_fill(~valid[:, None], 0.)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        weights = weights.masked_fill(~valid[..., None], 0.)
        transpose = weights.transpose(-1, -2)
        difference = observation.float()-local.float()
        right = transpose @ (weights @ difference)
        delta = torch.zeros_like(right)
        residual = right.clone()
        direction = residual.clone()
        norm = residual.square().sum(1, keepdim=True)
        initial = norm.clone()
        iterations = 0
        for iterations in range(1, 33):
            active = norm > initial*1e-12
            if not bool(active.any()):
                break
            product = direction + transpose @ (weights @ direction)
            alpha = torch.where(active, norm/(direction*product).sum(1, keepdim=True).clamp_min(1e-30), 0.)
            delta = delta + alpha*direction
            residual = residual-alpha*product
            new_norm = residual.square().sum(1, keepdim=True)
            direction = residual + torch.where(active, new_norm/norm.clamp_min(1e-30), 0.)*direction
            norm = new_norm
        delta = delta.masked_fill(~valid[..., None], 0.)
        energy_before = .5*(weights @ difference).square().sum()
        energy_after = .5*(delta.square().sum() + (weights @ (delta-difference)).square().sum())
    return local.float()+delta, {
        "mean_absolute_innovation": float(delta.abs().mean()),
        "changed_patch_fraction": float(((local+delta).argmax(-1) != local.argmax(-1))[valid].float().mean())
                                  if bool(valid.any()) else 0.,
        "solver_relative_residual": float((norm/initial.clamp_min(1e-30)).sqrt().max()),
        "solver_iterations": iterations,
        "energy_before": float(energy_before), "energy_after": float(energy_after),
    }
