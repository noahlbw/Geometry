import math

import torch

from dinotool.config import TLPConfig
from dinotool.contrast_flow import ContrastFlowConfig, solve_contrast_flow
from dinotool.interface_budget import (
    InterfaceBudget,
    InterfaceBudgetConfig,
    build_interface_budgets,
)
from dinotool.model import DINOTextSegmenter
from dinotool.stride_ov import StrideOVConfig, stride_ov_from_features
from dinotool.structure_interfaces import (
    StructureInterfaceConfig,
    StructurePartition,
    build_structure_interfaces,
)
from dinotool.tlp import TLPState, apply_tlp_state


def _two_region_features(height: int = 3, width: int = 4) -> torch.Tensor:
    features = torch.zeros((2, height, width))
    features[0, :, : width // 2] = 1.0
    features[1, :, width // 2 :] = 1.0
    return features


def _two_region_partition() -> StructurePartition:
    return build_structure_interfaces(
        _two_region_features(),
        StructureInterfaceConfig(atom_similarity_threshold=0.99),
    )


def _grid_state(height: int = 3, width: int = 4) -> TLPState:
    return TLPState(
        fidelity=torch.ones((1, 1, height, width)),
        horizontal=torch.ones((1, 1, height, max(width - 1, 0))),
        vertical=torch.ones((1, 1, max(height - 1, 0), width)),
        smoothing_strength=1.0,
        cg_max_iterations=200,
        cg_tolerance=1e-8,
    )


def test_encode_image_remains_compatible_when_structure_is_exposed() -> None:
    class FakeModel:
        def encode_image_with_patch_tokens(self, rgb, normalize=False):
            del normalize
            batch = rgb.shape[0]
            patch = torch.arange(12, dtype=rgb.dtype).reshape(1, 4, 3).repeat(batch, 1, 1)
            raw = torch.flip(patch, dims=(2,)) + 1.0
            image = torch.arange(6, dtype=rgb.dtype).reshape(1, 6).repeat(batch, 1)
            return image, patch, raw

    segmenter = object.__new__(DINOTextSegmenter)
    segmenter.device = torch.device("cpu")
    segmenter.use_amp = False
    segmenter.model = FakeModel()
    segmenter._imagenet_mean = torch.zeros((1, 3, 1, 1))
    segmenter._imagenet_std = torch.ones((1, 3, 1, 1))
    rgb = torch.rand((1, 3, 32, 32))

    legacy_patches, legacy_anchor = segmenter.encode_image(rgb)
    patches, structure, anchor = segmenter.encode_image_with_structure(rgb)

    assert torch.equal(legacy_patches, patches)
    assert torch.equal(legacy_anchor, anchor)
    assert structure.shape == patches.shape
    assert not torch.equal(structure, patches)


def test_straight_boundary_is_one_connected_interface() -> None:
    partition = _two_region_partition()

    assert partition.atom_count == 2
    assert len(partition.interfaces) == 1
    assert partition.interfaces[0].edge_indices.numel() == 3
    assert partition.interfaces[0].eligible


def test_invalid_patches_are_removed_from_atoms_and_edges() -> None:
    valid = torch.ones((3, 4), dtype=torch.bool)
    valid[-1, -1] = False
    partition = build_structure_interfaces(
        _two_region_features(),
        StructureInterfaceConfig(atom_similarity_threshold=0.99),
        valid,
    )

    assert partition.atom_ids[-1, -1].item() == -1
    assert partition.edge_index.shape[1] == 15
    invalid_node = 11
    assert not partition.edge_index.eq(invalid_node).any()


def test_isolated_semantic_change_has_no_member_witness_budget() -> None:
    partition = _two_region_partition()
    state = _grid_state()
    initial = torch.zeros((1, 2, 3, 4))
    initial[:, 1] = 2.0
    initial[0, 0, 1, 1] = 3.0
    initial[0, 1, 1, 1] = 0.0
    initial[:, 1, :, 2:] = 4.0
    proposal = initial.clone()
    proposal[0, 0, 1, 1] = 0.0
    proposal[0, 1, 1, 1] = 2.0

    budgets, diagnostics = build_interface_budgets(
        initial,
        proposal,
        state,
        partition,
        InterfaceBudgetConfig(capacity_ratio=0.0),
    )

    assert budgets == ()
    assert diagnostics.witnessed_nodes == 0


def test_supported_interface_change_creates_one_shared_budget() -> None:
    partition = _two_region_partition()
    state = _grid_state()
    initial = torch.zeros((1, 2, 3, 4))
    initial[:, 0, :, :2] = 2.0
    initial[:, 1, :, 2:] = 4.0
    proposal = initial.clone()
    proposal[:, 0, :, 1] = 0.0
    proposal[:, 1, :, 1] = 2.0

    budgets, diagnostics = build_interface_budgets(
        initial,
        proposal,
        state,
        partition,
        InterfaceBudgetConfig(capacity_ratio=0.0),
    )

    assert len(budgets) == 1
    assert diagnostics.active_budgets == 1
    assert budgets[0].old_class == 0
    assert budgets[0].new_class == 1
    assert budgets[0].edge_indices.numel() == 3
    assert budgets[0].proposal_flux > budgets[0].capacity


def test_no_budget_returns_tlp_proposal_exactly() -> None:
    partition = _two_region_partition()
    state = _grid_state()
    initial = torch.randn((1, 3, 3, 4), generator=torch.Generator().manual_seed(4))
    proposal, _ = apply_tlp_state(initial, state)

    output, diagnostics = solve_contrast_flow(initial, proposal, state, partition, ())

    assert torch.equal(output, proposal)
    assert diagnostics.iterations == 0
    assert diagnostics.constrained_edges == 0
    assert math.isfinite(diagnostics.stationarity_residual)


def test_constrained_solver_satisfies_capacity_without_changing_orthogonal_directions() -> None:
    partition = StructurePartition(
        height=1,
        width=2,
        atom_ids=torch.tensor([[0, 1]]),
        edge_index=torch.tensor([[0], [1]]),
        tlp_edge_indices=torch.tensor([0]),
        interface_ids=torch.tensor([0]),
        valid_mask=torch.ones((1, 2), dtype=torch.bool),
        interfaces=(),
        atom_count=2,
    )
    state = _grid_state(height=1, width=2)
    initial = torch.tensor([[[[4.0, 0.0]], [[0.0, 2.0]], [[2.0, -1.0]]]])
    proposal, _ = apply_tlp_state(initial, state)
    contrast = torch.tensor([1.0, -1.0, 0.0]) / math.sqrt(2.0)
    budget = InterfaceBudget(
        interface_id=0,
        recipient_atom=0,
        old_class=0,
        new_class=1,
        edge_indices=torch.tensor([0]),
        signs=torch.tensor([1.0]),
        recipient_nodes=torch.tensor([0]),
        contrast=contrast,
        capacity=0.05,
        proposal_flux=1.0,
        evidence_reserve=1.0,
    )

    output, diagnostics = solve_contrast_flow(
        initial,
        proposal,
        state,
        partition,
        (budget,),
        ContrastFlowConfig(max_iterations=1000, tolerance=1e-7),
    )
    proposal_nodes = proposal[0].permute(1, 2, 0).reshape(2, 3)
    output_nodes = output[0].permute(1, 2, 0).reshape(2, 3)
    orthogonal = torch.tensor([1.0, 1.0, 0.0]) / math.sqrt(2.0)

    assert diagnostics.maximum_capacity_violation <= 1e-5
    assert diagnostics.stationarity_residual <= 2e-4
    assert torch.allclose(output_nodes @ orthogonal, proposal_nodes @ orthogonal, atol=2e-4)
    assert torch.allclose(output_nodes[:, 2], proposal_nodes[:, 2], atol=2e-4)


def test_end_to_end_stride_ov_is_finite_and_keeps_batch_graphs_separate() -> None:
    generator = torch.Generator().manual_seed(19)
    text_patches = torch.randn((2, 4, 3, 4), generator=generator)
    raw_patches = torch.randn((2, 5, 3, 4), generator=generator)
    rgb = torch.rand((2, 3, 48, 64), generator=generator)
    text = torch.randn((3, 4), generator=generator)
    anchors = torch.randn((2, 4), generator=generator)
    valid = torch.ones((2, 3, 4), dtype=torch.bool)
    valid[1, :, -1] = False
    config = StrideOVConfig(
        tlp=TLPConfig(cg_max_iterations=80, cg_tolerance=1e-6),
        flow=ContrastFlowConfig(max_iterations=200, tolerance=1e-5),
    )

    result = stride_ov_from_features(
        text_patches,
        raw_patches,
        rgb,
        text,
        config,
        valid_mask=valid,
        anchors=anchors,
    )

    assert result.logits.shape == (2, 3, 3, 4)
    assert torch.isfinite(result.logits).all()
    assert torch.equal(result.anchors, anchors)
    assert len(result.diagnostics) == 2
    assert torch.allclose(
        result.proposal_logits[1, :, :, -1],
        result.initial_logits[1, :, :, -1],
        atol=1e-6,
    )
