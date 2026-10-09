import numpy as np

from dinotool.handcrafted_probe import lbp_histogram, rgb_lbp_hog, rgb_moments_histogram, spatial_hog


def test_handcrafted_descriptors_have_fixed_shapes_and_are_finite() -> None:
    rgb = np.zeros((16, 20, 3), dtype=np.uint8)
    rgb[:, ::2, 0] = 255
    rgb[::2, :, 1] = 128
    assert rgb_moments_histogram(rgb).shape == (54,)
    assert lbp_histogram(rgb).shape == (256,)
    assert spatial_hog(rgb).shape == (4 * 4 * 9,)
    assert rgb_lbp_hog(rgb).shape == (54 + 256 + 4 * 4 * 9,)
    assert np.isfinite(rgb_lbp_hog(rgb)).all()


def test_lbp_is_invariant_to_global_luminance_shift_without_clipping() -> None:
    base = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
    rgb = np.repeat((base + 40)[..., None], 3, axis=2)
    shifted = np.repeat((base + 80)[..., None], 3, axis=2)
    assert np.allclose(lbp_histogram(rgb), lbp_histogram(shifted))
