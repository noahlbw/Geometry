"""Fixed RGB and texture controls for region-distribution diagnostics.

The functions here are intentionally conventional: colour moments/histograms,
8-neighbour local-binary-pattern histograms, and spatial HOG. They provide
non-DINO baselines for testing whether a token-distribution result is merely a
proxy for low-level remote-sensing appearance.
"""

from __future__ import annotations

import numpy as np


def rgb_moments_histogram(rgb: np.ndarray, *, bins: int = 16) -> np.ndarray:
    """RGB mean/std plus per-channel fixed-range colour histograms."""
    values = _as_rgb_float(rgb)
    if bins < 2:
        raise ValueError("bins must be at least two.")
    parts = [values.mean(axis=(0, 1)), values.std(axis=(0, 1))]
    for channel in range(3):
        histogram, _ = np.histogram(values[..., channel], bins=bins, range=(0.0, 1.0))
        parts.append((histogram / max(histogram.sum(), 1)).astype(np.float32))
    return np.concatenate(parts).astype(np.float32)


def lbp_histogram(rgb: np.ndarray) -> np.ndarray:
    """Classic 8-neighbour LBP histogram on luminance, normalized to one."""
    gray = _as_rgb_float(rgb) @ np.asarray((0.299, 0.587, 0.114), dtype=np.float32)
    if min(gray.shape) < 3:
        raise ValueError("LBP requires an image at least 3x3.")
    center = gray[1:-1, 1:-1]
    code = np.zeros(center.shape, dtype=np.uint8)
    neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    height, width = gray.shape
    for bit, (dy, dx) in enumerate(neighbours):
        neighbour = gray[1 + dy : height - 1 + dy, 1 + dx : width - 1 + dx]
        code |= ((neighbour >= center).astype(np.uint8) << bit)
    histogram = np.bincount(code.ravel(), minlength=256).astype(np.float32)
    return histogram / max(float(histogram.sum()), 1.0)


def spatial_hog(rgb: np.ndarray, *, cells: int = 4, orientations: int = 9) -> np.ndarray:
    """A compact, fixed spatial HOG descriptor with no learned parameters."""
    if cells < 1 or orientations < 2:
        raise ValueError("cells and orientations are invalid.")
    gray = _as_rgb_float(rgb) @ np.asarray((0.299, 0.587, 0.114), dtype=np.float32)
    if min(gray.shape) < cells:
        raise ValueError("Image is too small for the requested HOG grid.")
    gradient_y, gradient_x = np.gradient(gray)
    magnitude = np.hypot(gradient_x, gradient_y)
    angle = np.mod(np.arctan2(gradient_y, gradient_x), np.pi)
    assignments = np.minimum((angle * orientations / np.pi).astype(np.int32), orientations - 1)
    rows = np.array_split(np.arange(gray.shape[0]), cells)
    columns = np.array_split(np.arange(gray.shape[1]), cells)
    descriptors: list[np.ndarray] = []
    for row_indices in rows:
        for column_indices in columns:
            selected_bins = assignments[np.ix_(row_indices, column_indices)].ravel()
            selected_magnitudes = magnitude[np.ix_(row_indices, column_indices)].ravel()
            histogram = np.bincount(selected_bins, weights=selected_magnitudes, minlength=orientations).astype(np.float32)
            histogram /= np.sqrt(float(np.square(histogram).sum()) + 1e-6)
            descriptors.append(histogram)
    return np.concatenate(descriptors).astype(np.float32)


def rgb_lbp_hog(rgb: np.ndarray) -> np.ndarray:
    """Concatenate the three fixed handcrafted controls."""
    return np.concatenate((rgb_moments_histogram(rgb), lbp_histogram(rgb), spatial_hog(rgb))).astype(np.float32)


def _as_rgb_float(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb)
    if values.ndim != 3 or values.shape[-1] != 3 or min(values.shape[:2]) < 1:
        raise ValueError("rgb must have shape [height, width, 3].")
    if np.issubdtype(values.dtype, np.integer):
        values = values.astype(np.float32) / float(np.iinfo(values.dtype).max)
    else:
        values = values.astype(np.float32, copy=False)
    return np.clip(values, 0.0, 1.0)
