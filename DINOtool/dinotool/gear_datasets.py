"""Official FLAIR-1 main taxonomy and LandCover.ai v1 validation adapters."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch


CLASSES = {
    "flair1": ("building", "pervious surface", "impervious surface", "bare soil", "water",
               "coniferous", "deciduous", "brushwood", "vineyard", "herbaceous vegetation",
               "agricultural land", "plowed land"),
    "landcoverai": ("background", "building", "woodland", "water", "road"),
}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path
    key: str


def class_names(dataset: str) -> tuple[str, ...]:
    return CLASSES[dataset]


def discover_samples(dataset: str, root: Path) -> list[Sample]:
    if dataset == "flair1":
        images = root / "extracted" / "flair_1_aerial_test"
        masks = root / "extracted" / "flair_1_labels_test"
        pairs = []
        for image in sorted(images.glob("*/*/img/IMG_*.tif")):
            relative = image.relative_to(images)
            mask = masks / relative.parent.parent / "msk" / image.name.replace("IMG_", "MSK_", 1)
            pairs.append(Sample(image, mask, relative.as_posix()))
    elif dataset == "landcoverai":
        split = root / "val.txt"
        keys = [line.strip() for line in split.read_text().splitlines() if line.strip()]
        pairs = [Sample(root / "val" / "images" / f"{key}.jpg",
                        root / "val" / "labels" / f"{key}_m.png", key) for key in keys]
    else:
        raise ValueError(f"Unsupported GEAR dataset: {dataset}")
    if not pairs:
        raise FileNotFoundError(f"No samples for {dataset} under {root}.")
    if len({sample.key for sample in pairs}) != len(pairs):
        raise ValueError("Duplicate sample keys.")
    for sample in pairs:
        if not sample.image_path.is_file() or not sample.mask_path.is_file():
            raise FileNotFoundError(f"Missing image/label for {sample.key}.")
    return pairs


def load_rgb(sample: Sample, dataset: str) -> torch.Tensor:
    if dataset == "flair1":
        import rasterio

        with rasterio.open(sample.image_path) as source:
            array = source.read((1, 2, 3))
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[0] != 3:
            raise ValueError("FLAIR RGB must be uint8 bands 1,2,3; no spectral normalization.")
        array = np.moveaxis(array, 0, -1)
    else:
        with Image.open(sample.image_path) as source:
            array = np.asarray(source.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(np.array(array, copy=True, order="C")).permute(2, 0, 1).float().div_(255)


def load_target(sample: Sample, dataset: str, shape: tuple[int, int]) -> np.ndarray:
    if dataset == "flair1":
        import rasterio

        with rasterio.open(sample.mask_path) as source:
            raw = source.read(1)
        unexpected = set(np.unique(raw).tolist()) - set(range(20)) - {255}
        if unexpected:
            raise ValueError(f"Unexpected FLAIR labels: {sorted(unexpected)}")
        # Official main evaluation uses IDs 1..12 and ignores the residual group.
        target = np.full(raw.shape, 255, dtype=np.int64)
        valid = (raw >= 1) & (raw <= 12)
        target[valid] = raw[valid].astype(np.int64) - 1
    else:
        with Image.open(sample.mask_path) as source:
            target = np.asarray(source, dtype=np.int64)
        unexpected = set(np.unique(target).tolist()) - set(range(5))
        if unexpected:
            raise ValueError(f"Unexpected LandCover.ai v1 labels: {sorted(unexpected)}")
    if target.shape != shape:
        raise ValueError(f"Image/mask shape mismatch for {sample.key}: {shape} vs {target.shape}")
    return target
