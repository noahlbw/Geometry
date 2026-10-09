"""Connected DINO structure atoms and their patch-grid interfaces."""
from __future__ import annotations

from dataclasses import dataclass
import heapq

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class StructureInterfaceConfig:
    atom_similarity_threshold: float = 0.85
    internal_distance_quantile: float = 0.75

    def validate(self) -> None:
        if not -1.0 <= self.atom_similarity_threshold <= 1.0:
            raise ValueError("atom_similarity_threshold must be in [-1, 1].")
        if not 0.0 <= self.internal_distance_quantile <= 1.0:
            raise ValueError("internal_distance_quantile must be in [0, 1].")


@dataclass(frozen=True)
class StructureInterface:
    interface_id: int
    atom_a: int
    atom_b: int
    edge_indices: Tensor
    atom_a_nodes: Tensor
    atom_b_nodes: Tensor
    external_distance: float
    internal_reference: float
    eligible: bool


@dataclass(frozen=True)
class StructurePartition:
    height: int
    width: int
    atom_ids: Tensor
    edge_index: Tensor
    tlp_edge_indices: Tensor
    interface_ids: Tensor
    valid_mask: Tensor
    interfaces: tuple[StructureInterface, ...]
    atom_count: int


def grid_edge_index(height: int, width: int, device: torch.device | str | None = None) -> Tensor:
    """Return right/down oriented four-neighbor edges in TLP storage order."""
    nodes = torch.arange(height * width, device=device).reshape(height, width)
    sources = []
    targets = []
    if width > 1:
        sources.append(nodes[:, :-1].reshape(-1))
        targets.append(nodes[:, 1:].reshape(-1))
    if height > 1:
        sources.append(nodes[:-1, :].reshape(-1))
        targets.append(nodes[1:, :].reshape(-1))
    if not sources:
        return torch.empty((2, 0), device=device, dtype=torch.long)
    return torch.stack((torch.cat(sources), torch.cat(targets)))


def build_structure_interfaces(
    raw_features: Tensor,
    config: StructureInterfaceConfig = StructureInterfaceConfig(),
    valid_mask: Tensor | None = None,
) -> StructurePartition:
    """Agglomerate spatially adjacent patches and expose connected interfaces.

    This v1 implementation intentionally handles one image at a time. The
    caller can loop over a batch without coupling structures across images.
    """
    config.validate()
    if raw_features.ndim == 4:
        if raw_features.shape[0] != 1:
            raise ValueError("Structure discovery accepts one image at a time.")
        raw_features = raw_features[0]
    if raw_features.ndim != 3:
        raise ValueError("raw_features must have shape [D,H,W] or [1,D,H,W].")
    _, height, width = raw_features.shape
    device = raw_features.device
    if valid_mask is None:
        valid_mask = torch.ones((height, width), device=device, dtype=torch.bool)
    else:
        if valid_mask.ndim == 3 and valid_mask.shape[0] == 1:
            valid_mask = valid_mask[0]
        if valid_mask.shape != (height, width):
            raise ValueError("valid_mask must match the feature grid.")
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)

    edge_index = grid_edge_index(height, width, device)
    valid_flat = valid_mask.reshape(-1)
    edge_valid = valid_flat[edge_index[0]] & valid_flat[edge_index[1]]
    tlp_edge_indices = edge_valid.nonzero(as_tuple=False).flatten()
    edge_index = edge_index[:, edge_valid]
    features = F.normalize(raw_features.float(), dim=0).permute(1, 2, 0).reshape(-1, raw_features.shape[0])
    atom_ids_cpu, atom_count = _agglomerate(
        features.detach().cpu(),
        edge_index.detach().cpu(),
        valid_flat.detach().cpu(),
        config.atom_similarity_threshold,
    )
    atom_ids = atom_ids_cpu.to(device).reshape(height, width)

    interface_ids = torch.full((edge_index.shape[1],), -1, device=device, dtype=torch.long)
    edge_distances = 1.0 - (features[edge_index[0]] * features[edge_index[1]]).sum(dim=1)
    flat_atoms = atom_ids.reshape(-1)
    source_atoms = flat_atoms[edge_index[0]]
    target_atoms = flat_atoms[edge_index[1]]
    internal_distances: dict[int, list[Tensor]] = {atom: [] for atom in range(atom_count)}
    grouped: dict[tuple[int, int], list[int]] = {}
    for edge in range(edge_index.shape[1]):
        left = int(source_atoms[edge].item())
        right = int(target_atoms[edge].item())
        if left == right:
            internal_distances[left].append(edge_distances[edge])
            continue
        pair = (min(left, right), max(left, right))
        grouped.setdefault(pair, []).append(edge)

    interfaces: list[StructureInterface] = []
    for pair in sorted(grouped):
        for component in _connected_interface_components(
            grouped[pair],
            edge_index,
            flat_atoms,
            pair,
            width,
        ):
            interface_id = len(interfaces)
            component_tensor = torch.tensor(component, device=device, dtype=torch.long)
            interface_ids[component_tensor] = interface_id
            atom_a, atom_b = pair
            side_a: set[int] = set()
            side_b: set[int] = set()
            for edge in component:
                source = int(edge_index[0, edge].item())
                target = int(edge_index[1, edge].item())
                (side_a if int(flat_atoms[source].item()) == atom_a else side_b).add(source)
                (side_a if int(flat_atoms[target].item()) == atom_a else side_b).add(target)
            external = float(edge_distances[component_tensor].median().item())
            reference_values = []
            for atom in pair:
                values = internal_distances[atom]
                if values:
                    stacked = torch.stack(values)
                    reference_values.append(
                        float(torch.quantile(stacked, config.internal_distance_quantile).item())
                    )
            reference = max(reference_values) if len(reference_values) == 2 else float("inf")
            interfaces.append(
                StructureInterface(
                    interface_id=interface_id,
                    atom_a=atom_a,
                    atom_b=atom_b,
                    edge_indices=component_tensor,
                    atom_a_nodes=torch.tensor(sorted(side_a), device=device, dtype=torch.long),
                    atom_b_nodes=torch.tensor(sorted(side_b), device=device, dtype=torch.long),
                    external_distance=external,
                    internal_reference=reference,
                    eligible=external > reference,
                )
            )
    return StructurePartition(
        height=height,
        width=width,
        atom_ids=atom_ids,
        edge_index=edge_index,
        tlp_edge_indices=tlp_edge_indices,
        interface_ids=interface_ids,
        valid_mask=valid_mask,
        interfaces=tuple(interfaces),
        atom_count=atom_count,
    )


def _agglomerate(
    features: Tensor,
    edge_index: Tensor,
    valid: Tensor,
    threshold: float,
) -> tuple[Tensor, int]:
    node_count = features.shape[0]
    parent = list(range(node_count))
    sums = [features[index].clone() for index in range(node_count)]
    neighbors = [set() for _ in range(node_count)]

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def similarity(left: int, right: int) -> float:
        return float(F.cosine_similarity(sums[left][None], sums[right][None]).item())

    heap: list[tuple[float, int, int]] = []
    for source, target in edge_index.t().tolist():
        neighbors[source].add(target)
        neighbors[target].add(source)
        score = similarity(source, target)
        heapq.heappush(heap, (-score, min(source, target), max(source, target)))

    while heap:
        _, first, second = heapq.heappop(heap)
        left, right = find(first), find(second)
        if left == right or right not in neighbors[left]:
            continue
        score = similarity(left, right)
        if score < threshold:
            continue
        root, absorbed = (left, right) if left < right else (right, left)
        parent[absorbed] = root
        sums[root] = sums[root] + sums[absorbed]
        updated = (neighbors[root] | neighbors[absorbed]) - {root, absorbed}
        for neighbor in tuple(updated):
            neighbor = find(neighbor)
            if neighbor == root:
                continue
            neighbors[neighbor].discard(absorbed)
            neighbors[neighbor].discard(root)
            neighbors[neighbor].add(root)
        neighbors[root] = {find(item) for item in updated if find(item) != root}
        neighbors[absorbed].clear()
        for neighbor in sorted(neighbors[root]):
            score = similarity(root, neighbor)
            heapq.heappush(heap, (-score, min(root, neighbor), max(root, neighbor)))

    roots = sorted({find(index) for index in range(node_count) if bool(valid[index])})
    remap = {root: index for index, root in enumerate(roots)}
    atom_ids = torch.full((node_count,), -1, dtype=torch.long)
    for index in range(node_count):
        if bool(valid[index]):
            atom_ids[index] = remap[find(index)]
    return atom_ids, len(roots)


def _connected_interface_components(
    edges: list[int],
    edge_index: Tensor,
    atom_ids: Tensor,
    pair: tuple[int, int],
    width: int,
) -> list[list[int]]:
    """Split repeated contacts of one atom pair into spatially connected segments.

    Consecutive parallel grid edges do not share a patch endpoint, so ordinary
    edge-graph connectivity would incorrectly split a straight boundary into
    one interface per row or column. Two boundary edges are connected when
    they share either side endpoint, or when the endpoints on both atom sides
    are four-neighbors. Contacts separated elsewhere in the image stay split.
    """
    remaining = set(edges)
    sides: dict[int, tuple[int, int]] = {}
    for edge in edges:
        source, target = (int(item) for item in edge_index[:, edge].tolist())
        if int(atom_ids[source].item()) == pair[0]:
            sides[edge] = (source, target)
        else:
            sides[edge] = (target, source)

    def adjacent(first: int, second: int) -> bool:
        if first == second:
            return True
        first_row, first_column = divmod(first, width)
        second_row, second_column = divmod(second, width)
        return abs(first_row - second_row) + abs(first_column - second_column) == 1

    linked: dict[int, set[int]] = {edge: set() for edge in edges}
    for position, first in enumerate(edges):
        first_a, first_b = sides[first]
        for second in edges[position + 1 :]:
            second_a, second_b = sides[second]
            connected = (
                first_a == second_a
                or first_b == second_b
                or (adjacent(first_a, second_a) and adjacent(first_b, second_b))
            )
            if connected:
                linked[first].add(second)
                linked[second].add(first)
    components: list[list[int]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component = []
        while stack:
            edge = stack.pop()
            component.append(edge)
            neighbors = linked[edge] & remaining
            remaining.difference_update(neighbors)
            stack.extend(sorted(neighbors, reverse=True))
        components.append(sorted(component))
    return components
