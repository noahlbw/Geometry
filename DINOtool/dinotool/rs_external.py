"""Dataset adapters for frozen Geometry/BoundedUnion external checks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch


@dataclass(frozen=True)
class RSample:
    image_path: Path
    mask_path: Path
    key: str


DATASET_CLASSES: dict[str, tuple[str, ...]] = {
    "vdd": ("background", "building", "road", "vegetation", "vehicle", "roof", "water"),
    "potsdam": ("impervious surface", "building", "low vegetation", "tree", "car", "clutter"),
    "vaihingen": ("impervious surface", "building", "low vegetation", "tree", "car"),
}


def class_names(dataset: str) -> tuple[str, ...]:
    try:
        return DATASET_CLASSES[dataset]
    except KeyError as error:
        raise ValueError(f"Unsupported external dataset: {dataset}") from error


def residual_names(dataset: str) -> tuple[str, ...]:
    # Clutter is a valid Potsdam class. Vaihingen preprocessing maps it to 255.
    return ()


def discover_samples(dataset: str, root: str | Path) -> list[RSample]:
    root = Path(root).expanduser().resolve()
    if dataset == "vdd":
        image_dir, mask_dir = root / "val" / "src", root / "val" / "gt"
        pairs = [
            RSample(image, mask_dir / f"{image.stem}.png", image.name)
            for image in sorted(image_dir.glob("*.JPG"))
        ]
    elif dataset in ("potsdam", "vaihingen"):
        image_dir, mask_dir = root / "val" / "images", root / "val" / "labels"
        pairs = [
            RSample(
                image,
                mask_dir / (image.name.replace("_RGB_", "_label_") if dataset == "potsdam" else image.name),
                image.name,
            )
            for image in sorted(image_dir.glob("*.tif"))
        ]
    else:
        raise ValueError(f"Unsupported external dataset: {dataset}")
    if not pairs:
        raise FileNotFoundError(
            f"No validation image/label pairs found for {dataset} under {root}. "
            "For Potsdam/Vaihingen, run the fixed 1000-pixel preprocessing first."
        )
    missing = [sample for sample in pairs if not sample.mask_path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing labels for {missing[0].key}: {missing[0].mask_path}")
    return pairs


def load_rgb(sample: RSample, dataset: str) -> torch.Tensor:
    if dataset == "vdd":
        array = np.asarray(Image.open(sample.image_path).convert("RGB"), dtype=np.uint8)
    elif dataset == "potsdam":
        # Pillow/libtiff handles the LZW-compressed validation tiles on A800.
        array = np.asarray(Image.open(sample.image_path).convert("RGB"), dtype=np.uint8)
    else:
        import tifffile

        array = np.asarray(tifffile.imread(sample.image_path))
        if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise ValueError(f"Expected uint8 three-band Vaihingen input, got "
                             f"{array.shape}/{array.dtype}: {sample.image_path}")
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float().div_(255.0)


def load_target(sample: RSample, dataset: str, shape: tuple[int, int]) -> np.ndarray:
    if dataset == "vdd":
        target = np.asarray(Image.open(sample.mask_path), dtype=np.uint8)
        allowed = set(range(len(DATASET_CLASSES[dataset])))
        unexpected = sorted(set(np.unique(target).tolist()) - allowed)
        if unexpected:
            raise ValueError(f"Unexpected VDD labels in {sample.key}: {unexpected}")
    else:
        if dataset == "potsdam":
            target = np.asarray(Image.open(sample.mask_path), dtype=np.uint8)
        else:
            import tifffile

            target = np.asarray(tifffile.imread(sample.mask_path), dtype=np.uint8)
        if dataset == "vaihingen":
            # The provided preprocessing maps clutter to the ignore value 255.
            unexpected = sorted(set(np.unique(target).tolist()) - set(range(5)) - {255})
        else:
            unexpected = sorted(set(np.unique(target).tolist()) - set(range(6)) - {255})
        if unexpected:
            raise ValueError(f"Unexpected {dataset} labels in {sample.key}: {unexpected}")
    if target.shape != shape:
        raise ValueError(f"Image/mask mismatch for {sample.key}: {shape} != {target.shape}")
    return target.astype(np.int64, copy=False)


def add_non_residual_metric(summary: dict[str, object], names: tuple[str, ...], residual: tuple[str, ...]) -> None:
    if not residual:
        return
    excluded = {name.casefold() for name in residual}
    values = [
        item["iou_percent"] for item in summary["per_class"]
        if item["name"].casefold() not in excluded and item["iou_percent"] is not None
    ]
    summary["non_residual_mean_iou_percent"] = round(float(np.mean(values)), 4)
