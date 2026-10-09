import numpy as np
import pytest
import torch

from dinotool.depth_diagnosis import (
    AlignmentStatistics,
    binary_spatial_metrics,
    combine_edge_affinities,
    cosine_logits,
    grid_edge_affinity,
    gt_seed_labels,
    oracle_pair_matrix,
    seeded_grid_diffusion,
    seeded_grid_diffusion_batched,
    semantic_seed_labels,
)


def test_procrustes_recovers_rotation() -> None:
    torch.manual_seed(3)
    source = torch.randn(300, 8)
    q, _ = torch.linalg.qr(torch.randn(8, 8))
    target = source @ q
    statistics = AlignmentStatistics(8)
    statistics.update(source[:170], target[:170])
    statistics.update(source[170:], target[170:])
    matrix, report = statistics.orthogonal_procrustes()
    assert torch.allclose(source @ matrix, target, atol=1e-4)
    assert report.fitted_patch_pairs == 300
    assert report.effective_degrees_of_freedom == 28


def test_ridge_and_cosine_logits() -> None:
    source = torch.eye(4).repeat(20, 1)
    target = source * 2.0
    statistics = AlignmentStatistics(4)
    statistics.update(source, target)
    matrix, report = statistics.ridge(1e-5)
    logits = cosine_logits(source[:2], torch.eye(4), matrix)
    assert logits.argmax(dim=1).tolist() == [0, 1]
    assert report.stored_parameters == 16


def test_ridge_many_matches_independent_solves() -> None:
    torch.manual_seed(17)
    source = torch.randn(40, 6)
    target = torch.randn(40, 6)
    statistics = AlignmentStatistics(6)
    statistics.update(source, target)
    values = (1e-3, 0.1, 1.0)
    batched = statistics.ridge_many(values)
    for value in values:
        expected, _ = statistics.ridge(value)
        actual, report = batched[value]
        assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
        assert report.regularization == value


def test_grid_affinity_and_seeded_diffusion_respect_barrier() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0],
            [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0],
        ]
    )
    horizontal, vertical = grid_edge_affinity(features, 2, 4, temperature=0.1)
    combined = combine_edge_affinities(((horizontal, vertical), (horizontal, vertical)))
    initial = torch.zeros(2, 2, 4)
    seeds = torch.full((2, 4), -1, dtype=torch.long)
    seeds[0, 0] = 0
    seeds[0, 3] = 1
    result = seeded_grid_diffusion(initial, *combined, seed_labels=seeds, iterations=40, restart=0.05)
    prediction = result.argmax(dim=0)
    assert prediction[:, :2].eq(0).all()
    assert prediction[:, 2:].eq(1).all()


def test_batched_diffusion_matches_independent_graphs() -> None:
    torch.manual_seed(11)
    initial = torch.randn(3, 4, 3, 5)
    horizontal = torch.rand(3, 3, 4)
    vertical = torch.rand(3, 2, 5)
    seeds = torch.full((3, 3, 5), -1, dtype=torch.long)
    seeds[:, 0, 0] = torch.tensor([0, 1, 2])
    batched = seeded_grid_diffusion_batched(
        initial,
        horizontal,
        vertical,
        seed_labels=seeds,
        iterations=7,
        restart=0.2,
    )
    independent = torch.stack(
        [
            seeded_grid_diffusion(
                initial[index],
                horizontal[index],
                vertical[index],
                seed_labels=seeds[index],
                iterations=7,
                restart=0.2,
            )
            for index in range(initial.shape[0])
        ]
    )
    assert torch.allclose(batched, independent, atol=1e-6)


def test_head_affinity_has_expected_shapes() -> None:
    features = torch.randn(12, 3, 5)
    horizontal, vertical = grid_edge_affinity(features, 3, 4)
    assert horizontal.shape == (3, 3)
    assert vertical.shape == (2, 4)


def test_semantic_and_gt_seed_selection() -> None:
    logits = torch.tensor([[[3.0, -1.0]], [[-1.0, 3.0]]])
    semantic = semantic_seed_labels(logits, confidence_threshold=0.9)
    assert semantic.tolist() == [[0, 1]]
    target = torch.tensor([[0, 0, 1], [0, 1, 1]])
    gt = gt_seed_labels(target, fraction=0.1)
    assert (gt == 0).any() and (gt == 1).any()


def test_spatial_metrics_detect_leakage_and_fragmentation() -> None:
    target = np.array([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]], dtype=bool)
    perfect = binary_spatial_metrics(target, target)
    assert perfect.region_iou == 1.0
    assert perfect.boundary_f1 == 1.0
    assert perfect.connectivity_recall == 1.0
    fragmented = target.copy()
    fragmented[:, 2] = False
    fragmented[:, 4] = True
    metrics = binary_spatial_metrics(fragmented, target)
    assert metrics.region_iou < 1.0
    assert metrics.connectivity_recall < 1.0
    assert metrics.leakage_ratio > 0.0


def test_oracle_pair_matrix_reports_same_depth_gap() -> None:
    layers = (4, 8)
    scores = {(4, 4): 0.4, (4, 8): 0.7, (8, 4): 0.5, (8, 8): 0.6}
    result = oracle_pair_matrix(scores, layers)
    assert result["best_pair"]["semantic_layer"] == 4
    assert result["best_pair"]["spatial_layer"] == 8
    assert result["dual_depth_oracle_gap"] == pytest.approx(0.1)
