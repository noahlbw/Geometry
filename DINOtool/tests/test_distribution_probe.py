import numpy as np

from dinotool.distribution_probe import (
    descriptor,
    match_classifier_dimension,
    spatial_organization_descriptor,
    stable_token_shuffle,
)


def test_distribution_descriptors_ignore_token_order() -> None:
    grid = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
    shuffled = stable_token_shuffle(grid, "region-a", seed=7)
    for condition in ("mean", "mean_variance", "mean_covariance"):
        original = descriptor(condition, grid, covariance_dimension=3, radial_bins=2)
        reordered = descriptor(condition, shuffled, covariance_dimension=3, radial_bins=2)
        assert np.allclose(original, reordered)


def test_spatial_descriptor_changes_when_token_positions_are_shuffled() -> None:
    grid = np.zeros((8, 8, 2), dtype=np.float32)
    grid[:, ::2, 0] = 1.0
    grid[::2, :, 1] = 1.0
    shuffled = stable_token_shuffle(grid, "region-b", seed=13)
    original = spatial_organization_descriptor(grid, radial_bins=3)
    reordered = spatial_organization_descriptor(shuffled, radial_bins=3)
    assert original.shape == reordered.shape
    assert not np.allclose(original, reordered)


def test_classifier_dimension_is_fixed_without_label_information() -> None:
    narrow = np.ones((3, 2), dtype=np.float32)
    wide = np.arange(30, dtype=np.float32).reshape(3, 10)
    assert match_classifier_dimension(narrow, output_dimension=6, seed=3).shape == (3, 6)
    first = match_classifier_dimension(wide, output_dimension=6, seed=3)
    second = match_classifier_dimension(wide, output_dimension=6, seed=3)
    assert np.allclose(first, second)
