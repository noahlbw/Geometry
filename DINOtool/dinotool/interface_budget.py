"""Build class-contrast flow capacities from structure and TLP proposals."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .structure_interfaces import StructureInterface, StructurePartition
from .tlp import TLPState


@dataclass(frozen=True)
class InterfaceBudgetConfig:
    witness_radius: int = 2
    minimum_witnesses: int = 3
    witness_quantile: float = 0.25
    capacity_ratio: float = 0.5
    activation_tolerance: float = 1e-8

    def validate(self) -> None:
        if self.witness_radius < 1 or self.minimum_witnesses < 1:
            raise ValueError("Witness radius and minimum count must be positive.")
        if not 0.0 <= self.witness_quantile <= 1.0:
            raise ValueError("witness_quantile must be in [0, 1].")
        if self.capacity_ratio < 0 or self.activation_tolerance < 0:
            raise ValueError("Capacity ratio and tolerance must be non-negative.")


@dataclass(frozen=True)
class InterfaceBudget:
    interface_id: int
    recipient_atom: int
    old_class: int
    new_class: int
    edge_indices: Tensor
    signs: Tensor
    recipient_nodes: Tensor
    contrast: Tensor
    capacity: float
    proposal_flux: float
    evidence_reserve: float


@dataclass(frozen=True)
class BudgetDiagnostics:
    interfaces: int
    eligible_interfaces: int
    candidate_interfaces: int
    active_budgets: int
    witnessed_nodes: int
    proposal_flux: float
    capacity: float


@dataclass
class _Candidate:
    interface: StructureInterface
    recipient_atom: int
    old_class: int
    new_class: int
    edge_indices: list[int]
    signs: list[float]
    node_evidence: dict[int, float]
    contrast: Tensor
    proposal_flux: float = 0.0
    reserve: float = 0.0
    score: float = 0.0


def build_interface_budgets(
    initial_logits: Tensor,
    proposal_logits: Tensor,
    state: TLPState,
    partition: StructurePartition,
    config: InterfaceBudgetConfig = InterfaceBudgetConfig(),
) -> tuple[tuple[InterfaceBudget, ...], BudgetDiagnostics]:
    """Select one supported, propagation-active semantic contrast per interface."""
    config.validate()
    initial = _single_logits(initial_logits, partition)
    proposal = _single_logits(proposal_logits, partition)
    classes = initial.shape[1]
    fidelity = _single_fidelity(state, partition).reshape(-1)
    edge_weight = _partition_edge_weights(state, partition)
    edge_flux = state.smoothing_strength * edge_weight[:, None] * (
        proposal[partition.edge_index[0]] - proposal[partition.edge_index[1]]
    )
    old = initial.argmax(dim=1)
    new = proposal.argmax(dim=1)
    atom_ids = partition.atom_ids.reshape(-1)
    candidate_interfaces = 0
    witnessed_nodes: set[int] = set()
    selected: list[_Candidate] = []

    for interface in partition.interfaces:
        if not interface.eligible:
            continue
        candidates: dict[tuple[int, int, int], _Candidate] = {}
        for atom, nodes in (
            (interface.atom_a, interface.atom_a_nodes),
            (interface.atom_b, interface.atom_b_nodes),
        ):
            for node_tensor in nodes:
                node = int(node_tensor.item())
                old_class = int(old[node].item())
                new_class = int(new[node].item())
                if old_class == new_class:
                    continue
                evidence = _witness_evidence(
                    node,
                    atom,
                    old_class,
                    new_class,
                    initial,
                    atom_ids,
                    partition.height,
                    partition.width,
                    config,
                )
                if evidence is None:
                    continue
                incident = _incident_interface_edges(node, interface.edge_indices, partition.edge_index)
                if not incident:
                    continue
                witnessed_nodes.add(node)
                key = (atom, old_class, new_class)
                candidate = candidates.get(key)
                if candidate is None:
                    contrast = initial.new_zeros(classes)
                    contrast[old_class] = 1.0 / math.sqrt(2.0)
                    contrast[new_class] = -1.0 / math.sqrt(2.0)
                    candidate = _Candidate(
                        interface=interface,
                        recipient_atom=atom,
                        old_class=old_class,
                        new_class=new_class,
                        edge_indices=[],
                        signs=[],
                        node_evidence={},
                        contrast=contrast,
                    )
                    candidates[key] = candidate
                candidate.node_evidence[node] = max(candidate.node_evidence.get(node, 0.0), evidence)
                for edge in incident:
                    if edge not in candidate.edge_indices:
                        candidate.edge_indices.append(edge)
                        source = int(partition.edge_index[0, edge].item())
                        candidate.signs.append(1.0 if source == node else -1.0)

        viable: list[_Candidate] = []
        for candidate in candidates.values():
            edges = torch.tensor(candidate.edge_indices, device=initial.device, dtype=torch.long)
            signs = initial.new_tensor(candidate.signs)
            directed = signs * (edge_flux[edges] @ candidate.contrast)
            candidate.proposal_flux = float(directed.clamp_min(0).sum().item())
            candidate.reserve = sum(
                float(fidelity[node].item()) * evidence
                for node, evidence in candidate.node_evidence.items()
            )
            candidate.score = candidate.proposal_flux / max(candidate.reserve, 1e-12)
            if candidate.proposal_flux > 0 and candidate.reserve > 0:
                viable.append(candidate)
        if not viable:
            continue
        candidate_interfaces += 1
        viable.sort(key=lambda item: item.score, reverse=True)
        if len(viable) > 1 and math.isclose(viable[0].score, viable[1].score, rel_tol=1e-12, abs_tol=1e-12):
            continue
        selected.append(viable[0])

    node_use: dict[int, int] = {}
    for candidate in selected:
        for node in candidate.node_evidence:
            node_use[node] = node_use.get(node, 0) + 1

    budgets: list[InterfaceBudget] = []
    for candidate in selected:
        reserve = sum(
            float(fidelity[node].item()) * evidence / node_use[node]
            for node, evidence in candidate.node_evidence.items()
        )
        capacity = config.capacity_ratio * reserve
        if candidate.proposal_flux <= capacity + config.activation_tolerance:
            continue
        budgets.append(
            InterfaceBudget(
                interface_id=candidate.interface.interface_id,
                recipient_atom=candidate.recipient_atom,
                old_class=candidate.old_class,
                new_class=candidate.new_class,
                edge_indices=torch.tensor(candidate.edge_indices, device=initial.device, dtype=torch.long),
                signs=initial.new_tensor(candidate.signs),
                recipient_nodes=torch.tensor(
                    sorted(candidate.node_evidence), device=initial.device, dtype=torch.long
                ),
                contrast=candidate.contrast,
                capacity=capacity,
                proposal_flux=candidate.proposal_flux,
                evidence_reserve=reserve,
            )
        )
    diagnostics = BudgetDiagnostics(
        interfaces=len(partition.interfaces),
        eligible_interfaces=sum(interface.eligible for interface in partition.interfaces),
        candidate_interfaces=candidate_interfaces,
        active_budgets=len(budgets),
        witnessed_nodes=len(witnessed_nodes),
        proposal_flux=sum(item.proposal_flux for item in budgets),
        capacity=sum(item.capacity for item in budgets),
    )
    return tuple(budgets), diagnostics


def _single_logits(logits: Tensor, partition: StructurePartition) -> Tensor:
    if logits.ndim == 4:
        if logits.shape[0] != 1:
            raise ValueError("Budget construction accepts one image at a time.")
        logits = logits[0]
    if logits.ndim != 3 or logits.shape[-2:] != (partition.height, partition.width):
        raise ValueError("Logits must share the structure grid.")
    return logits.float().permute(1, 2, 0).reshape(-1, logits.shape[0])


def _single_fidelity(state: TLPState, partition: StructurePartition) -> Tensor:
    if state.fidelity.shape[0] != 1 or state.fidelity.shape[-2:] != (partition.height, partition.width):
        raise ValueError("TLP state must contain the same single-image grid.")
    return state.fidelity[0, 0].float()


def _partition_edge_weights(state: TLPState, partition: StructurePartition) -> Tensor:
    full = torch.cat((state.horizontal[0, 0].reshape(-1), state.vertical[0, 0].reshape(-1)))
    return full[partition.tlp_edge_indices].float()


def _incident_interface_edges(node: int, edges: Tensor, edge_index: Tensor) -> list[int]:
    result = []
    for edge_tensor in edges:
        edge = int(edge_tensor.item())
        if node in (int(edge_index[0, edge].item()), int(edge_index[1, edge].item())):
            result.append(edge)
    return result


def _witness_evidence(
    node: int,
    atom: int,
    old_class: int,
    new_class: int,
    initial: Tensor,
    atom_ids: Tensor,
    height: int,
    width: int,
    config: InterfaceBudgetConfig,
) -> float | None:
    row, column = divmod(node, width)
    members = []
    for delta_row in range(-config.witness_radius, config.witness_radius + 1):
        remaining = config.witness_radius - abs(delta_row)
        for delta_column in range(-remaining, remaining + 1):
            other_row, other_column = row + delta_row, column + delta_column
            if not (0 <= other_row < height and 0 <= other_column < width):
                continue
            other = other_row * width + other_column
            if other == node or int(atom_ids[other].item()) != atom:
                continue
            members.append(other)
    if len(members) < config.minimum_witnesses:
        return None
    member_index = torch.tensor(members, device=initial.device, dtype=torch.long)
    member_margin = (
        initial[member_index, old_class] - initial[member_index, new_class]
    ) / math.sqrt(2.0)
    local = float(torch.quantile(member_margin, config.witness_quantile).item())
    recipient = float(
        ((initial[node, old_class] - initial[node, new_class]) / math.sqrt(2.0)).item()
    )
    if local <= 0 or recipient <= 0:
        return None
    return min(local, recipient)

