"""COCO-Stuff 2017 subset used by the source-only CAFe experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset


# These are PNG values in the official stuffthingmaps archive, not labels.txt
# IDs. In particular, 1 is bicycle and 0 (person) is deliberately unselected.
COCO_CAFE41_RAW_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8,
    94, 95, 96, 110, 112, 123, 124, 125, 126, 127, 128, 131,
    134, 135, 139, 141, 143, 144, 145, 146, 147, 148, 149, 150,
    153, 154, 157, 158, 161, 163, 168, 177, 181,
)
COCO_CAFE41_NAMES = (
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "bridge", "building", "bush", "dirt", "fence", "grass", "gravel", "ground",
    "hill", "house", "leaves", "metal", "mountain", "mud", "pavement", "plant",
    "platform", "playing field", "railing", "railroad", "river", "road", "rock", "roof",
    "sand", "sea", "skyscraper", "snow", "stone", "structural", "tree", "water", "wood",
)

if len(COCO_CAFE41_RAW_IDS) != 41 or len(COCO_CAFE41_NAMES) != 41:
    raise RuntimeError("The locked CAFe COCO vocabulary must contain 41 classes.")


def cafe41_remap() -> np.ndarray:
    remap = np.full(256, 255, dtype=np.uint8)
    for subset_id, raw_id in enumerate(COCO_CAFE41_RAW_IDS):
        remap[raw_id] = subset_id
    return remap


COCO_CAFE41_REMAP = cafe41_remap()


def validate_cafe41_mapping() -> None:
    expected = {1: 0, 148: 29, 168: 38, 255: 255}
    actual = {raw: int(COCO_CAFE41_REMAP[raw]) for raw in expected}
    if actual != expected:
        raise RuntimeError(f"Unexpected COCO-Stuff subset mapping: {actual}")
    if COCO_CAFE41_REMAP[0] != 255:
        raise RuntimeError("COCO-Stuff PNG value 0 (person) must not become background.")


@dataclass(frozen=True)
class CocoStuffSample:
    key: str
    image_path: Path
    label_path: Path

    def record(self) -> dict[str, str]:
        return {
            "key": self.key,
            "image_path": str(self.image_path),
            "label_path": str(self.label_path),
        }


def samples_from_manifest(path: str | Path, *, relocate_root: str | Path | None = None,
                          verify_paths: bool = False) -> list[CocoStuffSample]:
    """Read a locked manifest, optionally relocating its immutable root.

    A copied COCO mirror may retain absolute paths from the machine where its
    manifest was prepared. ``relocate_root`` is deliberately restricted to a
    prefix substitution: every recorded image/label must lie beneath the
    manifest's declared root and keeps exactly the same relative path, key,
    ordering, and label.  With ``verify_paths`` enabled, fail before training
    if even one such record is unavailable rather than failing mid-epoch.
    """
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError(f"COCO manifest has no samples: {path}")
    source_root = Path(payload.get("root", "")) if isinstance(payload, dict) else Path()
    target_root = Path(relocate_root).resolve() if relocate_root is not None else None
    if target_root is not None and (not source_root.is_absolute() or not str(source_root)):
        raise ValueError(f"COCO manifest has no absolute root for safe relocation: {manifest_path}")
    samples = []
    missing: list[Path] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("COCO manifest sample must be an object.")
        def resolve(record_path: str) -> Path:
            original = Path(record_path)
            if target_root is None:
                return original
            try:
                return target_root / original.relative_to(source_root)
            except ValueError as error:
                raise ValueError(
                    f"Manifest path escapes its declared root and cannot be safely relocated: {original}"
                ) from error
        sample = CocoStuffSample(
            key=str(record["key"]),
            image_path=resolve(record["image_path"]),
            label_path=resolve(record["label_path"]),
        )
        if sample.image_path.suffix.lower() != ".jpg" or sample.label_path.suffix.lower() != ".png":
            raise ValueError(f"Unexpected COCO file types for {sample.key}")
        if verify_paths:
            missing.extend(candidate for candidate in (sample.image_path, sample.label_path) if not candidate.is_file())
        samples.append(sample)
    if len({sample.key for sample in samples}) != len(samples):
        raise ValueError(f"COCO manifest repeats image IDs: {path}")
    if missing:
        examples = ", ".join(str(item) for item in missing[:3])
        raise FileNotFoundError(f"COCO manifest relocation found {len(missing)} unavailable files; examples: {examples}")
    return samples


class CocoStuffCafe41Dataset(Dataset[tuple[Tensor, Tensor]]):
    """A fixed 224-pixel source loader with paired RGB/mask augmentation."""

    def __init__(self, samples: Iterable[CocoStuffSample], *, crop_size: int, training: bool) -> None:
        self.samples = tuple(samples)
        self.crop_size = crop_size
        self.training = training
        if not self.samples:
            raise ValueError("COCO dataset needs at least one paired sample.")
        if crop_size < 16 or crop_size % 16:
            raise ValueError("crop_size must be divisible by 16 and at least 16.")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _resize(image: np.ndarray, target: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        rgb = Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR)
        mask = Image.fromarray(target).resize((width, height), Image.Resampling.NEAREST)
        return np.asarray(rgb), np.asarray(mask, dtype=np.uint8)

    def _augment(self, image: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = target.shape
        scale = float(torch.empty(()).uniform_(0.5, 2.0))
        image, target = self._resize(image, target, max(1, round(height * scale)), max(1, round(width * scale)))
        height, width = target.shape
        padding_h, padding_w = max(self.crop_size - height, 0), max(self.crop_size - width, 0)
        if padding_h or padding_w:
            image = np.pad(image, ((0, padding_h), (0, padding_w), (0, 0)), mode="edge")
            target = np.pad(target, ((0, padding_h), (0, padding_w)), constant_values=255)
            height, width = target.shape
        top = int(torch.randint(height - self.crop_size + 1, ()).item())
        left = int(torch.randint(width - self.crop_size + 1, ()).item())
        image = image[top:top + self.crop_size, left:left + self.crop_size]
        target = target[top:top + self.crop_size, left:left + self.crop_size]
        if bool(torch.randint(2, ()).item()):
            image, target = image[:, ::-1].copy(), target[:, ::-1].copy()
        return image, target

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as handle:
            image = np.asarray(handle.convert("RGB"))
        with Image.open(sample.label_path) as handle:
            target = np.asarray(handle, dtype=np.uint8)
        if image.shape[:2] != target.shape:
            raise ValueError(f"COCO RGB/label size mismatch for {sample.key}")
        if self.training:
            image, target = self._augment(image, target)
        else:
            image, target = self._resize(image, target, self.crop_size, self.crop_size)
        return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().div_(255), torch.from_numpy(np.array(target, copy=True)).long()
