"""Projected primal-dual solver for class-contrast interface capacities."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .interface_budget import InterfaceBudget
from .structure_interfaces import StructurePartition
from .tlp import TLPState


@dataclass(frozen=True)
class ContrastFlowConfig:
    primal_step: float = 0.3
    dual_step: float = 0.3
    max_iterations: int = 100
    tolerance: float = 1e-4
    projection_iterations: int = 48

    def validate(self) -> None:
        if self.primal_step <= 0 or self.dual_step <= 0:
            raise ValueError("Primal and dual steps must be positive.")
        if self.primal_step * self.dual_step * 8.0 >= 1.0:
            raise ValueError("Four-neighbor stability requires primal_step * dual_step * 8 < 1.")
        if self.max_iterations < 1 or self.projection_iterations < 1 or self.tolerance <= 0:
            raise ValueError("Iteration counts and tolerance must be positive.")


@dataclass(frozen=True)
class ContrastFlowDiagnostics:
    iterations: int
    converged: bool
    relative_update: float
    stationarity_residual: float
    maximum_capacity_violation: float
    constrained_edges: int


def solve_contrast_flow(
    initial_logits: Tensor,
    proposal_logits: Tensor,
    state: TLPState,
    partition: StructurePartition,
    budgets: tuple[InterfaceBudget, ...],
    config: ContrastFlowConfig = ContrastFlowConfig(),
) -> tuple[Tensor, ContrastFlowDiagnostics]:
    """Solve the fixed structure-capacity objective for one image."""
    config.validate()
    initial, layout = _flatten_logits(initial_logits, partition)
    proposal, _ = _flatten_logits(proposal_logits, partition)
    if not budgets:
        return proposal_logits.clone(), ContrastFlowDiagnostics(
            iterations=0,
            converged=True,
            relative_update=0.0,
            stationarity_residual=_stationarity_without_constraints(
                initial,
                proposal,
                state,
                partition,
            ),
            maximum_capacity_violation=0.0,
            constrained_edges=0,
        )
    _validate_budgets(budgets, partition, initial.shape[1])
    sources, targets = partition.edge_index
    edge_weight = _edge_weights(state, partition)
    conductance = state.smoothing_strength * edge_weight
    fidelity = state.fidelity[0, 0].float().reshape(-1, 1)
    z = proposal.clone()
    z_bar = z.clone()
    dual = conductance[:, None] * (z[sources] - z[targets])
    positive = conductance > 0
    h = torch.ones_like(conductance)
    h[positive] = 1.0 + config.dual_step / conductance[positive]
    relative = float("inf")
    converged = False
    completed = 0

    for iteration in range(1, config.max_iterations + 1):
        gradient = z_bar[sources] - z_bar[targets]
        candidate_dual = torch.zeros_like(dual)
        candidate_dual[positive] = (
            dual[positive] + config.dual_step * gradient[positive]
        ) / h[positive, None]
        for budget in budgets:
            _project_budget(candidate_dual, h, budget, config.projection_iterations)

        divergence = _divergence(candidate_dual, sources, targets, len(z))
        next_z = (
            z - config.primal_step * divergence + config.primal_step * fidelity * initial
        ) / (1.0 + config.primal_step * fidelity)
        primal_relative = (next_z - z).norm() / next_z.norm().clamp_min(1e-12)
        dual_relative = (candidate_dual - dual).norm() / candidate_dual.norm().clamp_min(1e-12)
        relative = float(torch.maximum(primal_relative, dual_relative).item())
        z_bar = 2.0 * next_z - z
        z = next_z
        dual = candidate_dual
        completed = iteration
        if relative <= config.tolerance:
            converged = True
            break

    stationarity = fidelity * (z - initial) + _divergence(dual, sources, targets, len(z))
    stationarity_scale = (fidelity * initial).norm().clamp_min(1e-12)
    residual = float((stationarity.norm() / stationarity_scale).item())
    violation = max((_capacity_violation(dual, budget) for budget in budgets), default=0.0)
    output = z.reshape(partition.height, partition.width, -1).permute(2, 0, 1)
    if layout == "batched":
        output = output.unsqueeze(0)
    return output.to(initial_logits.dtype), ContrastFlowDiagnostics(
        iterations=completed,
        converged=converged,
        relative_update=relative,
        stationarity_residual=residual,
        maximum_capacity_violation=violation,
        constrained_edges=sum(int(budget.edge_indices.numel()) for budget in budgets),
    )


def _flatten_logits(logits: Tensor, partition: StructurePartition) -> tuple[Tensor, str]:
    layout = "plain"
    if logits.ndim == 4:
        if logits.shape[0] != 1:
            raise ValueError("Contrast flow accepts one image at a time.")
        logits = logits[0]
        layout = "batched"
    if logits.ndim != 3 or logits.shape[-2:] != (partition.height, partition.width):
        raise ValueError("Logits must share the structure grid.")
    return logits.float().permute(1, 2, 0).reshape(-1, logits.shape[0]), layout


def _edge_weights(state: TLPState, partition: StructurePartition) -> Tensor:
    if state.fidelity.shape[0] != 1 or state.fidelity.shape[-2:] != (partition.height, partition.width):
        raise ValueError("TLP state must contain the same single-image grid.")
    full = torch.cat((state.horizontal[0, 0].reshape(-1), state.vertical[0, 0].reshape(-1)))
    return full[partition.tlp_edge_indices].float()


def _divergence(values: Tensor, sources: Tensor, targets: Tensor, nodes: int) -> Tensor:
    result = values.new_zeros((nodes, values.shape[1]))
    result.index_add_(0, sources, values)
    result.index_add_(0, targets, -values)
    return result


def _project_budget(dual: Tensor, metric: Tensor, budget: InterfaceBudget, iterations: int) -> None:
    edges = budget.edge_indices
    values = budget.signs * (dual[edges] @ budget.contrast)
    positive_sum = float(values.clamp_min(0).sum().item())
    if positive_sum <= budget.capacity:
        return
    local_metric = metric[edges]
    lower = values.new_tensor(0.0)
    upper = (values.clamp_min(0) * local_metric).amax()
    for _ in range(iterations):
        middle = (lower + upper) * 0.5
        remaining = (values - middle / local_metric).clamp_min(0).sum()
        if float(remaining.item()) > budget.capacity:
            lower = middle
        else:
            upper = middle
    adjusted = values - torch.minimum(values.clamp_min(0), upper / local_metric)
    delta = budget.signs * (adjusted - values)
    dual[edges] += delta[:, None] * budget.contrast[None]


def _capacity_violation(dual: Tensor, budget: InterfaceBudget) -> float:
    values = budget.signs * (dual[budget.edge_indices] @ budget.contrast)
    return max(0.0, float(values.clamp_min(0).sum().item()) - budget.capacity)


def _validate_budgets(
    budgets: tuple[InterfaceBudget, ...],
    partition: StructurePartition,
    classes: int,
) -> None:
    seen: set[int] = set()
    for budget in budgets:
        if budget.capacity < 0 or budget.contrast.shape != (classes,):
            raise ValueError("Invalid interface budget capacity or contrast shape.")
        if budget.edge_indices.shape != budget.signs.shape:
            raise ValueError("Budget edge indices and signs must have equal shape.")
        for edge in budget.edge_indices.tolist():
            if edge < 0 or edge >= partition.edge_index.shape[1]:
                raise ValueError("Budget references an edge outside the partition.")
            if edge in seen:
                raise ValueError("V1 contrast-flow budgets must use disjoint edge sets.")
            seen.add(edge)


def _stationarity_without_constraints(
    initial: Tensor,
    proposal: Tensor,
    state: TLPState,
    partition: StructurePartition,
) -> float:
    fidelity = state.fidelity[0, 0].float().reshape(-1, 1)
    sources, targets = partition.edge_index
    conductance = state.smoothing_strength * _edge_weights(state, partition)
    flux = conductance[:, None] * (proposal[sources] - proposal[targets])
    stationarity = fidelity * (proposal - initial) + _divergence(
        flux,
        sources,
        targets,
        len(proposal),
    )
    scale = (fidelity * initial).norm().clamp_min(1e-12)
    return float(stationarity.norm().div(scale).item())
