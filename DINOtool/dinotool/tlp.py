from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import TLPConfig


@dataclass(frozen=True)
class TLPDiagnostics:
    iterations: int
    relative_residual: float
    mean_horizontal_weight: float
    mean_vertical_weight: float


@dataclass(frozen=True)
class TLPState:
    """Frozen coefficients for replaying one spatial propagation action."""

    fidelity: Tensor
    horizontal: Tensor
    vertical: Tensor
    smoothing_strength: float
    cg_max_iterations: int
    cg_tolerance: float


def semantic_correlation(text_features: Tensor, config: TLPConfig) -> Tensor:
    """Build the symmetric text prior from DinoSplat-OV equation (2)."""
    normalized = F.normalize(text_features.float(), dim=-1)
    scores = normalized @ normalized.transpose(0, 1)
    matrix = torch.softmax(scores / config.semantic_temperature, dim=-1)
    if config.diagonal_boost:
        matrix = matrix + torch.eye(matrix.shape[0], device=matrix.device) * config.diagonal_boost
        matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return 0.5 * (matrix + matrix.transpose(0, 1))


def text_aware_laplacian_propagation(
    logits: Tensor,
    rgb: Tensor,
    text_features: Tensor,
    config: TLPConfig,
    structure_features: Tensor | None = None,
) -> tuple[Tensor, TLPDiagnostics]:
    """Solve DinoSplat-OV equation (7) on a patch grid with batched CG."""
    state = build_tlp_state(logits, rgb, text_features, config, structure_features)
    return apply_tlp_state(logits, state)


def build_tlp_state(
    logits: Tensor,
    rgb: Tensor,
    text_features: Tensor,
    config: TLPConfig,
    structure_features: Tensor | None = None,
) -> TLPState:
    """Construct and freeze TLP coefficients from an anchor observation."""
    config.validate()
    if logits.ndim != 4 or rgb.ndim != 4:
        raise ValueError("logits and rgb must be BCHW tensors.")
    if logits.shape[0] != rgb.shape[0] or logits.shape[-2:] != rgb.shape[-2:]:
        raise ValueError("rgb must be resized to the logits grid before TLP.")
    if text_features.shape[0] != logits.shape[1]:
        raise ValueError("text feature count must match the logit class dimension.")

    work_logits = logits.float()
    probabilities = torch.softmax(work_logits / config.probability_temperature, dim=1)
    semantic = semantic_correlation(text_features, config)
    semantic_projection = torch.einsum("ij,bjhw->bihw", semantic, probabilities)
    consistency = (probabilities * semantic_projection).sum(dim=1, keepdim=True)
    confidence = probabilities.amax(dim=1, keepdim=True)
    fidelity = confidence.clamp_min(config.min_confidence).square() * (1.0 + consistency)

    horizontal, vertical = _edge_weights(probabilities, rgb.float(), semantic, config, structure_features)

    return TLPState(
        fidelity=fidelity,
        horizontal=horizontal,
        vertical=vertical,
        smoothing_strength=config.smoothing_strength,
        cg_max_iterations=config.cg_max_iterations,
        cg_tolerance=config.cg_tolerance,
    )


def apply_tlp_state(logits: Tensor, state: TLPState) -> tuple[Tensor, TLPDiagnostics]:
    """Apply an anchor-derived TLP action without recomputing its coefficients."""
    if logits.ndim != 4:
        raise ValueError("logits must be a BCHW tensor.")
    if logits.shape[0] != state.fidelity.shape[0] or logits.shape[-2:] != state.fidelity.shape[-2:]:
        raise ValueError("Replay logits must share the anchor batch and grid dimensions.")
    work_logits = logits.float()

    def system(value: Tensor) -> Tensor:
        return state.fidelity * value + state.smoothing_strength * _laplacian(
            value,
            state.horizontal,
            state.vertical,
        )

    rhs = state.fidelity * work_logits
    solution, iterations, residual = _conjugate_gradient(
        system,
        rhs,
        initial=work_logits,
        max_iterations=state.cg_max_iterations,
        tolerance=state.cg_tolerance,
    )
    diagnostics = TLPDiagnostics(
        iterations=iterations,
        relative_residual=residual,
        mean_horizontal_weight=float(state.horizontal.mean().item()) if state.horizontal.numel() else 0.0,
        mean_vertical_weight=float(state.vertical.mean().item()) if state.vertical.numel() else 0.0,
    )
    return solution.to(logits.dtype), diagnostics


def _edge_weights(
    probabilities: Tensor,
    rgb: Tensor,
    semantic: Tensor,
    config: TLPConfig,
    structure_features: Tensor | None,
) -> tuple[Tensor, Tensor]:
    grayscale = (rgb[:, :3] * rgb.new_tensor((0.2989, 0.5870, 0.1140)).view(1, 3, 1, 1)).sum(
        dim=1, keepdim=True
    )
    grad_h = (grayscale[..., :, 1:] - grayscale[..., :, :-1]).abs()
    grad_v = (grayscale[..., 1:, :] - grayscale[..., :-1, :]).abs()
    gradient_values = [item.flatten(1) for item in (grad_h, grad_v) if item.numel()]
    if gradient_values:
        mean_gradient = torch.cat(gradient_values, dim=1).mean(dim=1, keepdim=True).view(-1, 1, 1, 1)
        mean_gradient = mean_gradient.clamp_min(1e-6)
    else:
        mean_gradient = torch.ones((rgb.shape[0], 1, 1, 1), device=rgb.device, dtype=rgb.dtype)

    image_h = torch.exp(-config.image_edge_scale * grad_h / mean_gradient)
    image_v = torch.exp(-config.image_edge_scale * grad_v / mean_gradient)
    projected = torch.einsum("ij,bjhw->bihw", semantic, probabilities)
    semantic_h = 1.0 + (projected[..., :, :-1] * probabilities[..., :, 1:]).sum(dim=1, keepdim=True)
    semantic_v = 1.0 + (projected[..., :-1, :] * probabilities[..., 1:, :]).sum(dim=1, keepdim=True)
    horizontal = image_h * semantic_h
    vertical = image_v * semantic_v

    if structure_features is not None:
        if structure_features.shape[0] != probabilities.shape[0] or structure_features.shape[-2:] != probabilities.shape[-2:]:
            raise ValueError("structure_features must share batch and grid dimensions with logits.")
        structure = F.normalize(structure_features.float(), dim=1)
        similarity_h = (structure[..., :, :-1] * structure[..., :, 1:]).sum(dim=1, keepdim=True)
        similarity_v = (structure[..., :-1, :] * structure[..., 1:, :]).sum(dim=1, keepdim=True)
        horizontal = horizontal * torch.exp((similarity_h - 1.0) / config.structure_temperature)
        vertical = vertical * torch.exp((similarity_v - 1.0) / config.structure_temperature)

    return horizontal.clamp_(0.0, 1.0), vertical.clamp_(0.0, 1.0)


def _laplacian(value: Tensor, horizontal: Tensor, vertical: Tensor) -> Tensor:
    result = torch.zeros_like(value)
    if horizontal.numel():
        difference_h = value[..., :, :-1] - value[..., :, 1:]
        weighted_h = horizontal * difference_h
        result[..., :, :-1] += weighted_h
        result[..., :, 1:] -= weighted_h
    if vertical.numel():
        difference_v = value[..., :-1, :] - value[..., 1:, :]
        weighted_v = vertical * difference_v
        result[..., :-1, :] += weighted_v
        result[..., 1:, :] -= weighted_v
    return result


def _conjugate_gradient(
    operator,
    rhs: Tensor,
    initial: Tensor,
    max_iterations: int,
    tolerance: float,
) -> tuple[Tensor, int, float]:
    solution = initial.clone()
    residual = rhs - operator(solution)
    direction = residual.clone()
    squared = _spatial_dot(residual, residual)
    initial_norm = squared.sqrt().clamp_min(1e-12)
    relative = float((squared.sqrt() / initial_norm).amax().item())
    completed = 0
    for iteration in range(1, max_iterations + 1):
        applied = operator(direction)
        denominator = _spatial_dot(direction, applied).clamp_min(1e-20)
        alpha = squared / denominator
        solution = solution + alpha * direction
        residual = residual - alpha * applied
        next_squared = _spatial_dot(residual, residual)
        relative = float((next_squared.sqrt() / initial_norm).amax().item())
        completed = iteration
        if relative <= tolerance:
            break
        beta = next_squared / squared.clamp_min(1e-20)
        direction = residual + beta * direction
        squared = next_squared
    return solution, completed, relative


def _spatial_dot(left: Tensor, right: Tensor) -> Tensor:
    return (left * right).sum(dim=(-2, -1), keepdim=True)
