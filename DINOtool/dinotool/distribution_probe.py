"""Region-level DINO token descriptors for the distributional-field diagnosis.

The module deliberately contains no segmentation decoder.  It exposes fixed,
label-free statistics of a token grid so that the experiment can ask whether
they add discriminative evidence beyond a first-order token mean.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TokenProjection:
    """A source-fitted, category-agnostic PCA projection."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    @property
    def dimension(self) -> int:
        return int(self.components.shape[0])

    def transform(self, token_grid: np.ndarray) -> np.ndarray:
        if token_grid.ndim != 3:
            raise ValueError("token_grid must have shape [height, width, channels].")
        if token_grid.shape[-1] != self.mean.shape[0]:
            raise ValueError("token_grid channel count does not match the projection.")
        flat = token_grid.reshape(-1, token_grid.shape[-1]).astype(np.float32, copy=False)
        projected = (flat - self.mean) @ self.components.T
        return projected.reshape(*token_grid.shape[:2], self.dimension).astype(np.float32, copy=False)


def fit_token_projection(
    token_grids: Sequence[np.ndarray],
    *,
    dimension: int,
    maximum_tokens: int,
    seed: int,
) -> TokenProjection:
    """Fit PCA from source-train tokens only, independent of class labels."""
    if not token_grids:
        raise ValueError("At least one token grid is required to fit the projection.")
    if dimension < 1 or maximum_tokens < dimension:
        raise ValueError("Projection dimensions are invalid.")
    flat = np.concatenate(
        [grid.reshape(-1, grid.shape[-1]).astype(np.float32, copy=False) for grid in token_grids],
        axis=0,
    )
    if flat.shape[0] < dimension:
        raise ValueError("Not enough source tokens for the requested PCA dimension.")
    if flat.shape[0] > maximum_tokens:
        generator = np.random.default_rng(seed)
        indices = generator.choice(flat.shape[0], size=maximum_tokens, replace=False)
        flat = flat[indices]
    from sklearn.decomposition import PCA

    pca = PCA(n_components=dimension, svd_solver="randomized", random_state=seed)
    pca.fit(flat)
    return TokenProjection(
        mean=pca.mean_.astype(np.float32, copy=False),
        components=pca.components_.astype(np.float32, copy=False),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32, copy=False),
    )


def stable_token_shuffle(token_grid: np.ndarray, key: str, *, seed: int) -> np.ndarray:
    """Shuffle positions but preserve the token multiset for a region."""
    if token_grid.ndim != 3:
        raise ValueError("token_grid must have shape [height, width, channels].")
    digest = hashlib.blake2b(f"{seed}:{key}".encode("utf-8"), digest_size=8).digest()
    local_seed = int.from_bytes(digest, byteorder="little", signed=False)
    indices = np.random.default_rng(local_seed).permutation(token_grid.shape[0] * token_grid.shape[1])
    return token_grid.reshape(-1, token_grid.shape[-1])[indices].reshape(token_grid.shape)


def mean_descriptor(projected_grid: np.ndarray) -> np.ndarray:
    _validate_grid(projected_grid)
    return projected_grid.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)


def mean_variance_descriptor(projected_grid: np.ndarray) -> np.ndarray:
    mean = mean_descriptor(projected_grid)
    variance = projected_grid.var(axis=(0, 1), dtype=np.float64).astype(np.float32)
    return np.concatenate((mean, variance), axis=0)


def mean_covariance_descriptor(projected_grid: np.ndarray, *, covariance_dimension: int) -> np.ndarray:
    _validate_grid(projected_grid)
    if covariance_dimension < 1 or covariance_dimension > projected_grid.shape[-1]:
        raise ValueError("covariance_dimension is outside the projected feature range.")
    mean = mean_descriptor(projected_grid)
    values = projected_grid[..., :covariance_dimension].reshape(-1, covariance_dimension).astype(np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
    upper = covariance[np.triu_indices(covariance_dimension)].astype(np.float32)
    return np.concatenate((mean, upper), axis=0)


def spatial_organization_descriptor(projected_grid: np.ndarray, *, radial_bins: int) -> np.ndarray:
    """Rotation-tolerant radial spectrum plus structure-tensor anisotropy."""
    _validate_grid(projected_grid)
    if radial_bins < 1:
        raise ValueError("radial_bins must be positive.")
    height, width, channels = projected_grid.shape
    centered = projected_grid - projected_grid.mean(axis=(0, 1), keepdims=True)
    spectrum = np.fft.rfft2(centered, axes=(0, 1), norm="ortho")
    power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag).astype(np.float64)
    y_frequency = np.fft.fftfreq(height)[:, None]
    x_frequency = np.fft.rfftfreq(width)[None, :]
    radius = np.sqrt(y_frequency * y_frequency + x_frequency * x_frequency)
    positive = radius > 0
    edges = np.linspace(0.0, float(radius.max()) + 1e-12, radial_bins + 1)
    radial: list[np.ndarray] = []
    for index in range(radial_bins):
        in_bin = positive & (radius >= edges[index]) & (radius < edges[index + 1])
        if not bool(in_bin.any()):
            radial.append(np.zeros(channels, dtype=np.float32))
        else:
            radial.append(np.log1p(power[in_bin].mean(axis=0)).astype(np.float32))
    if height < 3 or width < 3:
        anisotropy = np.zeros(channels, dtype=np.float32)
    else:
        grad_x = centered[1:-1, 2:, :] - centered[1:-1, :-2, :]
        grad_y = centered[2:, 1:-1, :] - centered[:-2, 1:-1, :]
        jxx = np.mean(grad_x * grad_x, axis=(0, 1))
        jyy = np.mean(grad_y * grad_y, axis=(0, 1))
        jxy = np.mean(grad_x * grad_y, axis=(0, 1))
        anisotropy = (
            np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-8)
        ).astype(np.float32)
    return np.concatenate((*radial, anisotropy), axis=0)


def descriptor(
    condition: str,
    projected_grid: np.ndarray,
    *,
    covariance_dimension: int,
    radial_bins: int,
    shuffle_key: str | None = None,
    shuffle_seed: int = 0,
) -> np.ndarray:
    """Build one pre-registered descriptor condition from a token grid."""
    if condition == "mean":
        return mean_descriptor(projected_grid)
    if condition == "mean_variance":
        return mean_variance_descriptor(projected_grid)
    covariance = mean_covariance_descriptor(projected_grid, covariance_dimension=covariance_dimension)
    if condition == "mean_covariance":
        return covariance
    if condition == "mean_covariance_spatial":
        return np.concatenate((covariance, spatial_organization_descriptor(projected_grid, radial_bins=radial_bins)))
    if condition == "mean_covariance_spatial_shuffled":
        if shuffle_key is None:
            raise ValueError("shuffle_key is required for the shuffled spatial control.")
        shuffled = stable_token_shuffle(projected_grid, shuffle_key, seed=shuffle_seed)
        return np.concatenate((covariance, spatial_organization_descriptor(shuffled, radial_bins=radial_bins)))
    raise ValueError(f"Unknown descriptor condition: {condition}")


def match_classifier_dimension(values: np.ndarray, *, output_dimension: int, seed: int) -> np.ndarray:
    """Use a label-free projection so every probe has the same classifier width."""
    if values.ndim != 2:
        raise ValueError("values must be a [samples, features] matrix.")
    if output_dimension < 1:
        raise ValueError("output_dimension must be positive.")
    if values.shape[1] == output_dimension:
        return values.astype(np.float32, copy=False)
    if values.shape[1] < output_dimension:
        padded = np.zeros((values.shape[0], output_dimension), dtype=np.float32)
        padded[:, : values.shape[1]] = values
        return padded
    generator = np.random.default_rng(seed)
    matrix = generator.normal(
        0.0,
        1.0 / np.sqrt(output_dimension),
        size=(values.shape[1], output_dimension),
    ).astype(np.float32)
    return (values.astype(np.float32, copy=False) @ matrix).astype(np.float32, copy=False)


def _validate_grid(token_grid: np.ndarray) -> None:
    if token_grid.ndim != 3 or min(token_grid.shape[:2]) < 1 or token_grid.shape[-1] < 1:
        raise ValueError("token_grid must have shape [positive_height, positive_width, positive_channels].")
