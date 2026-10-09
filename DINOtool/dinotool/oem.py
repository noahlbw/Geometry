from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .prompts import ClassSpec


OEM_CLASSES: tuple[ClassSpec, ...] = (
    ClassSpec("bareland", ("bareland", "bare soil", "barren land", "exposed earth")),
    ClassSpec("rangeland", ("rangeland", "grassland", "open grass")),
    ClassSpec("developed space", ("developed space", "built-up land", "urban surface")),
    ClassSpec("road", ("road", "street", "paved roadway")),
    ClassSpec("tree", ("tree", "trees", "woodland", "forest")),
    ClassSpec("water", ("water", "river", "lake", "pond")),
    ClassSpec("agriculture land", ("agriculture land", "agricultural land", "farmland", "cropland")),
    ClassSpec("building", ("building", "house", "rooftop")),
)

# These concepts deliberately exclude the supervised OpenEarthMap vocabulary.
# They make the adapter retain dino.txt scores for adjacent and unseen remote-
# sensing concepts while the source-class loss teaches the domain correction.
OPEN_VOCABULARY_PRESERVATION_CLASSES: tuple[ClassSpec, ...] = (
    ClassSpec("background", ("background", "other terrain", "unclassified land")),
    ClassSpec("impervious surface", ("impervious surface", "pavement", "hardscape")),
    ClassSpec("low vegetation", ("low vegetation", "short grass", "ground cover")),
    ClassSpec("shrubland", ("shrubland", "shrubs", "bushes")),
    ClassSpec("parking area", ("parking area", "parking lot", "car park")),
    ClassSpec("railway", ("railway", "railroad track", "train track")),
    ClassSpec("bridge", ("bridge", "overpass", "viaduct")),
    ClassSpec("vehicle", ("vehicle", "car", "truck")),
    ClassSpec("solar panel", ("solar panel", "photovoltaic array", "solar farm")),
)

OEM_RAW_TO_CLASS_INDEX: dict[int, int] = {raw_id: raw_id - 1 for raw_id in range(1, len(OEM_CLASSES) + 1)}
OEM_IGNORED_RAW_IDS = frozenset((0, 255))
OEM_SOURCE = {
    "name": "OpenEarthMap",
    "doi": "10.5281/zenodo.7223446",
    "record": "https://zenodo.org/records/7223446",
    "archive": "OpenEarthMap.zip",
    "archive_md5": "64155d1dc9d3b68536063f79878e1a67",
    "license_note": (
        "OpenEarthMap label licensing follows the underlying RGB imagery and may vary by source; "
        "the project states CC BY-NC-SA 4.0 for labels where an image license is absent or public domain."
    ),
    "classes": [spec.name for spec in OEM_CLASSES],
    "raw_label_mapping": {str(raw_id): OEM_CLASSES[class_index].name for raw_id, class_index in OEM_RAW_TO_CLASS_INDEX.items()},
    "ignored_raw_label_ids": sorted(OEM_IGNORED_RAW_IDS),
}


@dataclass(frozen=True)
class OemSample:
    image_path: Path
    mask_path: Path
    key: str


def discover_oem_samples(
    data_root: str | Path,
    split: str,
    *,
    max_images: int | None = None,
    allow_missing_images: bool = False,
) -> list[OemSample]:
    """Discover the official OpenEarthMap image/label pairs for a named split.

    ``allow_missing_images`` is intended only for the official
    ``OpenEarthMap_wo_xBD`` release, whose archive omits xBD RGB files. Callers
    must opt in so an incomplete arbitrary dataset remains an error.
    """
    if not split or Path(split).name != split:
        raise ValueError("split must be a simple split name such as 'train' or 'val'.")
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    split_path = root / f"{split}.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"OpenEarthMap split file is missing: {split_path}")
    requested = _read_split_file(split_path)
    images = _find_oem_images(root)
    by_name: dict[str, list[Path]] = {}
    by_relative: dict[str, Path] = {}
    for image_path in images:
        by_name.setdefault(image_path.name, []).append(image_path)
        by_relative[image_path.relative_to(root).as_posix()] = image_path

    samples: list[OemSample] = []
    missing: list[str] = []
    for entry in requested:
        image_path = _match_split_entry(entry, root, by_name, by_relative)
        if image_path is None:
            missing.append(entry)
            continue
        mask_path = _mask_path_for_image(root, image_path)
        if not mask_path.is_file():
            raise FileNotFoundError(f"OpenEarthMap label is missing for {image_path}: {mask_path}")
        samples.append(OemSample(image_path=image_path, mask_path=mask_path, key=image_path.relative_to(root).as_posix()))
    if missing and not allow_missing_images:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} OpenEarthMap split entries had no matching image, for example: {preview}")
    if not samples:
        raise ValueError(f"No OpenEarthMap samples were found for split '{split}'.")
    samples = sorted(samples, key=lambda sample: sample.key)
    if max_images is not None:
        if max_images < 1:
            raise ValueError("max_images must be positive when provided.")
        samples = samples[:max_images]
    return samples


class OpenEarthMapDataset(Dataset[tuple[Tensor, Tensor]]):
    """OpenEarthMap RGB crops with strict raw-label validation.

    Labels are converted to contiguous source-vocabulary IDs 0..7. Pixels whose
    source label is 0 or 255 are ignored, never reassigned to a LoveDA class.
    """

    def __init__(
        self,
        samples: Iterable[OemSample],
        *,
        crop_size: int,
        training: bool,
        augment_scale_min: float = 0.75,
        augment_scale_max: float = 1.25,
    ) -> None:
        self.samples = list(samples)
        if not self.samples:
            raise ValueError("OpenEarthMapDataset requires at least one sample.")
        if crop_size < 16 or crop_size % 16:
            raise ValueError("crop_size must be a multiple of 16 and at least 16.")
        if not 0 < augment_scale_min <= augment_scale_max:
            raise ValueError("OpenEarthMap augmentation scale bounds are invalid.")
        self.crop_size = crop_size
        self.training = training
        self.augment_scale_min = augment_scale_min
        self.augment_scale_max = augment_scale_max
        self._lut = np.full(256, 255, dtype=np.uint8)
        for raw_id, class_index in OEM_RAW_TO_CLASS_INDEX.items():
            self._lut[raw_id] = class_index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self.samples[index]
        image = _read_rgb(sample.image_path)
        raw_mask = _read_mask(sample.mask_path)
        if image.shape[:2] != raw_mask.shape:
            raise ValueError(
                f"OpenEarthMap image/mask size mismatch for {sample.key}: {image.shape[:2]} != {raw_mask.shape}."
            )
        _validate_raw_labels(raw_mask, sample.key)
        image, raw_mask = self._transform(image, raw_mask)
        target = self._lut[raw_mask]
        image_tensor = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1).float().div_(255.0)
        target_tensor = torch.from_numpy(np.array(target, dtype=np.int64, copy=True))
        return image_tensor, target_tensor

    def _transform(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.training:
            scale = float(torch.empty(()).uniform_(self.augment_scale_min, self.augment_scale_max).item())
            image, mask = _resize_pair(image, mask, scale)
            image, mask = _pad_to_minimum(image, mask, self.crop_size)
            max_top = image.shape[0] - self.crop_size
            max_left = image.shape[1] - self.crop_size
            top = int(torch.randint(max_top + 1, ()).item())
            left = int(torch.randint(max_left + 1, ()).item())
            image = image[top : top + self.crop_size, left : left + self.crop_size]
            mask = mask[top : top + self.crop_size, left : left + self.crop_size]
            if bool(torch.randint(2, ()).item()):
                image = image[:, ::-1]
                mask = mask[:, ::-1]
            if bool(torch.randint(2, ()).item()):
                image = image[::-1]
                mask = mask[::-1]
            return image, mask
        return _resize_to_square(image, mask, self.crop_size)


def oem_source_manifest(
    data_root: str | Path,
    split: str,
    samples: Iterable[OemSample],
    *,
    allow_missing_images: bool = False,
) -> dict[str, object]:
    root = Path(data_root).expanduser().resolve()
    selected = list(samples)
    digest = hashlib.sha256("\n".join(sample.key for sample in selected).encode("utf-8")).hexdigest()
    provenance = _load_provenance(root)
    declared_entries = _read_split_file(root / f"{split}.txt")
    available_samples = discover_oem_samples(root, split, allow_missing_images=True)
    missing_image_count = len(declared_entries) - len(available_samples)
    return {
        **OEM_SOURCE,
        "data_root": str(root),
        "split": split,
        "sample_count": len(selected),
        "declared_sample_count": len(declared_entries),
        "available_sample_count": len(available_samples),
        "missing_image_count": missing_image_count,
        "allow_missing_images": allow_missing_images,
        "sample_keys_sha256": digest,
        "provenance": provenance,
    }


def oem_class_histogram(samples: Iterable[OemSample]) -> np.ndarray:
    """Return source-label pixel counts for reproducible class-balanced loss."""

    counts = np.zeros(len(OEM_CLASSES), dtype=np.int64)
    for sample in samples:
        raw_mask = _read_mask(sample.mask_path)
        _validate_raw_labels(raw_mask, sample.key)
        raw_counts = np.bincount(raw_mask.ravel(), minlength=256)
        for raw_id, class_index in OEM_RAW_TO_CLASS_INDEX.items():
            counts[class_index] += int(raw_counts[raw_id])
    if not bool(counts.any()):
        raise ValueError("OpenEarthMap samples contain no labeled source pixels.")
    return counts


def _read_split_file(path: Path) -> list[str]:
    entries = [line.strip().replace("\\", "/") for line in path.read_text(encoding="utf-8-sig").splitlines()]
    entries = [entry for entry in entries if entry]
    if not entries:
        raise ValueError(f"OpenEarthMap split file is empty: {path}")
    if len(entries) != len(set(entries)):
        raise ValueError(f"OpenEarthMap split file contains duplicate entries: {path}")
    return entries


def _find_oem_images(root: Path) -> list[Path]:
    extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    images = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions and path.parent.name == "images"
    ]
    if not images:
        raise ValueError(f"No OpenEarthMap images were found under {root}. Expected nested images/ directories.")
    return images


def _match_split_entry(
    entry: str,
    root: Path,
    by_name: dict[str, list[Path]],
    by_relative: dict[str, Path],
) -> Path | None:
    normalized = entry.lstrip("./")
    direct = root / normalized
    if direct.is_file() and direct.parent.name == "images":
        return direct
    if normalized in by_relative:
        return by_relative[normalized]
    filename = Path(normalized).name
    matches = by_name.get(filename, [])
    if len(matches) > 1:
        raise ValueError(
            f"OpenEarthMap split entry '{entry}' is ambiguous because {len(matches)} image paths share its filename."
        )
    return matches[0] if matches else None


def _mask_path_for_image(root: Path, image_path: Path) -> Path:
    relative = image_path.relative_to(root)
    parts = list(relative.parts)
    image_index = len(parts) - 2
    if parts[image_index] != "images":
        raise ValueError(f"Expected an images/ parent for {image_path}")
    parts[image_index] = "labels"
    return root.joinpath(*parts)


def _read_rgb(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio

            with rasterio.open(path) as source:
                if source.count < 3:
                    raise ValueError(f"OpenEarthMap RGB image has fewer than three bands: {path}")
                image = np.moveaxis(source.read((1, 2, 3)), 0, -1)
        except Exception as error:
            raise ValueError(f"Could not read OpenEarthMap raster {path}: {error}") from error
    else:
        image = np.asarray(Image.open(path).convert("RGB"))
    if image.dtype != np.uint8:
        image = _to_uint8(image)
    return np.asarray(image, dtype=np.uint8)


def _read_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio

            with rasterio.open(path) as source:
                mask = source.read(1)
        except Exception as error:
            raise ValueError(f"Could not read OpenEarthMap label raster {path}: {error}") from error
    else:
        mask = np.asarray(Image.open(path))
    if mask.ndim != 2:
        raise ValueError(f"OpenEarthMap labels must be single-band IDs, got shape {mask.shape} at {path}")
    if mask.dtype != np.uint8:
        if np.nanmin(mask) < 0 or np.nanmax(mask) > 255:
            raise ValueError(f"OpenEarthMap labels must be 8-bit IDs at {path}")
        mask = mask.astype(np.uint8)
    return mask


def _validate_raw_labels(mask: np.ndarray, key: str) -> None:
    observed = set(np.unique(mask).tolist())
    allowed = set(OEM_RAW_TO_CLASS_INDEX) | set(OEM_IGNORED_RAW_IDS)
    unexpected = sorted(observed - allowed)
    if unexpected:
        raise ValueError(f"OpenEarthMap label IDs {unexpected} are not supported for {key}")


def _to_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    result = np.empty_like(image, dtype=np.uint8)
    for channel in range(image.shape[-1]):
        values = image[..., channel]
        low, high = np.percentile(values[np.isfinite(values)], (2, 98)) if np.isfinite(values).any() else (0.0, 1.0)
        if high <= low:
            high = low + 1.0
        result[..., channel] = np.clip((values - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    return result


def _resize_pair(image: np.ndarray, mask: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    height = max(1, round(image.shape[0] * scale))
    width = max(1, round(image.shape[1] * scale))
    return _resize_pair_to(image, mask, width, height)


def _resize_to_square(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    return _resize_pair_to(image, mask, size, size)


def _resize_pair_to(image: np.ndarray, mask: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.fromarray(image, mode="RGB").resize((width, height), Image.Resampling.BICUBIC))
    target = np.asarray(Image.fromarray(mask, mode="L").resize((width, height), Image.Resampling.NEAREST))
    return rgb, target


def _pad_to_minimum(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    pad_height = max(0, size - image.shape[0])
    pad_width = max(0, size - image.shape[1])
    if not pad_height and not pad_width:
        return image, mask
    image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")
    mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode="constant", constant_values=0)
    return image, mask


def _load_provenance(root: Path) -> dict[str, object] | None:
    for path in (root / "provenance.json", root.parent / "provenance.json"):
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            payload["local_path"] = str(path)
            payload["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            return payload
    return None
