"""Official iSAID validation discovery and semantic-mask decoding.

The dataset stores semantic labels in ``*_instance_color_RGB.png`` and instance
identifiers in ``*_instance_id_RGB.png``.  iSAID's supplied devkit loads those
PNGs in BGR channel order and turns a pixel into
``B + 256 * G + 256**2 * R``.  This module uses the same mapping while loading
with PIL (which returns RGB), so it reverses RGB before building the integer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


# Raw semantic IDs and BGR-packed colours copied from the official iSAID
# devkit's cityscapesscripts/helpers/labels.py.  The list order is exactly raw
# ID 1 through 15; ID 0 is official ``unlabeled`` and remains ignored.
ISAID_CLASS_NAMES: tuple[str, ...] = (
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "swimming pool",
    "roundabout",
    "soccer ball field",
    "airplane",
    "harbor",
)
ISAID_COLOR_CODES: tuple[int, ...] = (
    4_128_768, 4_144_896, 16_128, 8_339_200, 12_533_504,
    16_727_808, 4_161_280, 8_355_584, 8_323_072, 12_517_376,
    16_711_680, 8_371_968, 12_549_888, 16_744_192, 10_183_680,
)
ISAID_IGNORE_ID = 255
ISAID_INSTANCE_AREA_BIN_NAMES: tuple[str, ...] = ("small", "medium", "large")
ISAID_INSTANCE_AREA_EDGES: tuple[int, int] = (1_024, 9_216)

if len(ISAID_CLASS_NAMES) != 15 or len(ISAID_COLOR_CODES) != 15:
    raise RuntimeError("iSAID constants must contain the 15 official foreground labels.")
if len(set(ISAID_COLOR_CODES)) != len(ISAID_COLOR_CODES):
    raise RuntimeError("iSAID official foreground colors must be unique.")


@dataclass(frozen=True)
class ISaidSample:
    key: str
    image_path: Path
    semantic_mask_path: Path
    instance_id_path: Path


class ISaidForegroundConfusionMatrix:
    """Foreground-only iSAID mIoU with an explicit background prediction column.

    Official iSAID marks non-instance pixels as ``unlabeled``. They are ignored
    for the foreground mIoU, but a model may legitimately emit the fixed
    background prompt on a labeled object pixel.  Keeping that prediction
    column makes a missed object count as a false negative instead of silently
    dropping it from the metric.
    """

    def __init__(self) -> None:
        self.matrix = np.zeros((len(ISAID_CLASS_NAMES), len(ISAID_CLASS_NAMES) + 1), dtype=np.int64)
        self.ignored_pixels = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction shape {prediction.shape} does not match target shape {target.shape}.")
        valid = (target >= 0) & (target < len(ISAID_CLASS_NAMES))
        self.ignored_pixels += int((~valid).sum())
        if not bool(valid.any()):
            return
        # Model predictions use index 0 for the fixed background prompt and
        # 1..15 for the ordered official foreground vocabulary.
        selected_prediction = prediction[valid]
        if np.any((selected_prediction < 0) | (selected_prediction > len(ISAID_CLASS_NAMES))):
            raise ValueError("Prediction contains an invalid iSAID vocabulary index on a labeled pixel.")
        bins = target[valid].astype(np.int64) * self.matrix.shape[1] + selected_prediction.astype(np.int64)
        self.matrix += np.bincount(bins, minlength=self.matrix.size).reshape(self.matrix.shape)

    def summary(self) -> dict[str, object]:
        intersection = np.diag(self.matrix[:, 1:])
        target_pixels = self.matrix.sum(axis=1)
        predicted_pixels = self.matrix[:, 1:].sum(axis=0)
        union = target_pixels + predicted_pixels - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.full(len(ISAID_CLASS_NAMES), np.nan, dtype=np.float64),
            where=union > 0,
        )
        precision = np.divide(
            intersection,
            predicted_pixels,
            out=np.full(len(ISAID_CLASS_NAMES), np.nan, dtype=np.float64),
            where=predicted_pixels > 0,
        )
        recall = np.divide(
            intersection,
            target_pixels,
            out=np.full(len(ISAID_CLASS_NAMES), np.nan, dtype=np.float64),
            where=target_pixels > 0,
        )
        labeled_pixels = int(target_pixels.sum())
        return {
            "mean_iou": float(np.nanmean(iou)),
            "mean_iou_percent": round(float(np.nanmean(iou)) * 100.0, 4),
            "labeled_pixels": labeled_pixels,
            "ignored_pixels": self.ignored_pixels,
            "foreground_to_background_miss_pixels": int(self.matrix[:, 0].sum()),
            "foreground_to_background_miss_percent": (
                None if not labeled_pixels else round(float(self.matrix[:, 0].sum() / labeled_pixels) * 100.0, 4)
            ),
            "class_names": list(ISAID_CLASS_NAMES),
            "prediction_column_names": ["background", *ISAID_CLASS_NAMES],
            "per_class": [
                {
                    "id": index + 1,
                    "name": name,
                    "iou": None if np.isnan(iou[index]) else float(iou[index]),
                    "iou_percent": None if np.isnan(iou[index]) else round(float(iou[index]) * 100.0, 4),
                    "precision": None if np.isnan(precision[index]) else float(precision[index]),
                    "recall": None if np.isnan(recall[index]) else float(recall[index]),
                    "target_pixels": int(target_pixels[index]),
                    "predicted_pixels": int(predicted_pixels[index]),
                    "intersection_pixels": int(intersection[index]),
                    "union_pixels": int(union[index]),
                    "background_miss_pixels": int(self.matrix[index, 0]),
                }
                for index, name in enumerate(ISAID_CLASS_NAMES)
            ],
            "confusion_matrix": self.matrix.tolist(),
        }


def _bgr_packed_png(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint32)
    # PIL is RGB; the upstream devkit's lycon loader is BGR.
    bgr = rgb[..., ::-1]
    return bgr[..., 0] + 256 * bgr[..., 1] + 65_536 * bgr[..., 2]


def _validation_image_dirs(data_root: Path) -> list[Path]:
    roots = [data_root, data_root / "iSAID"]
    candidates: list[Path] = []
    for root in roots:
        candidate = root / "val" / "images"
        if candidate.is_dir():
            candidates.append(candidate.resolve())
    # Google Drive releases can add one archive-name directory.  Permit exactly
    # one nested official val/images directory but never guess between multiples.
    for candidate in data_root.rglob("images"):
        if candidate.parent.name == "val" and candidate.is_dir():
            resolved = candidate.resolve()
            if resolved not in candidates:
                candidates.append(resolved)
    return sorted(candidates)


def discover_isaid_validation(data_root: str | Path) -> list[ISaidSample]:
    """Discover complete original validation triples without touching training data."""
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    candidates = _validation_image_dirs(root)
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one official iSAID val/images directory below "
            f"{root}; found {[str(path) for path in candidates]}."
        )
    image_dir = candidates[0]
    rgb_paths = sorted(
        path for path in image_dir.glob("*.png")
        if not path.name.endswith("_instance_color_RGB.png")
        and not path.name.endswith("_instance_id_RGB.png")
    )
    if not rgb_paths:
        raise ValueError(f"No original RGB validation PNGs found in {image_dir}.")
    samples: list[ISaidSample] = []
    for image_path in rgb_paths:
        stem = image_path.stem
        semantic = image_dir / f"{stem}_instance_color_RGB.png"
        instance = image_dir / f"{stem}_instance_id_RGB.png"
        if not semantic.is_file() or not instance.is_file():
            raise ValueError(f"Missing official iSAID masks for {image_path.name}.")
        samples.append(ISaidSample(stem, image_path, semantic, instance))
    return samples


def isaid_target_ids(mask_path: str | Path) -> np.ndarray:
    """Return foreground target IDs in ``[0, 14]`` and 255 for unlabeled pixels."""
    packed = _bgr_packed_png(Path(mask_path))
    result = np.full(packed.shape, ISAID_IGNORE_ID, dtype=np.uint8)
    valid_codes = np.array((0, *ISAID_COLOR_CODES), dtype=np.uint32)
    unknown = ~np.isin(packed, valid_codes)
    if bool(unknown.any()):
        values, counts = np.unique(packed[unknown], return_counts=True)
        observed = list(zip(values[:8].astype(int).tolist(), counts[:8].astype(int).tolist()))
        raise ValueError(f"Unknown iSAID semantic color code(s) in {mask_path}: {observed}")
    for class_index, color_code in enumerate(ISAID_COLOR_CODES):
        result[packed == color_code] = class_index
    return result


def isaid_instance_ids(instance_path: str | Path) -> np.ndarray:
    """Decode the official BGR-packed instance identifier PNG verbatim."""
    return _bgr_packed_png(Path(instance_path)).astype(np.int64, copy=False)


def instance_area_bin(area: int) -> int:
    if area < 1:
        raise ValueError(f"Instance area must be positive, got {area}.")
    if area < ISAID_INSTANCE_AREA_EDGES[0]:
        return 0
    if area < ISAID_INSTANCE_AREA_EDGES[1]:
        return 1
    return 2


def instance_size_recall_totals(
    prediction: np.ndarray,
    target: np.ndarray,
    instance_path: str | Path,
) -> np.ndarray:
    """Aggregate exact-instance target pixels/correct pixels in the fixed bins.

    The returned rows are ``(instance_count, target_pixels, correct_pixels)``
    for small, medium, and large original-image instances respectively.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction shape {prediction.shape} does not match target shape {target.shape}.")
    instances = isaid_instance_ids(instance_path)
    if instances.shape != target.shape:
        raise ValueError(f"Instance-mask shape {instances.shape} does not match target shape {target.shape}.")
    totals = np.zeros((len(ISAID_INSTANCE_AREA_BIN_NAMES), 3), dtype=np.int64)
    for instance_id in np.unique(instances):
        # This follows the devkit's definition: IDs below 1000 are not an
        # individual object instance and must not enter the size diagnostic.
        if int(instance_id) < 1_000:
            continue
        pixels = instances == instance_id
        target_values = np.unique(target[pixels])
        target_values = target_values[(target_values >= 0) & (target_values < len(ISAID_CLASS_NAMES))]
        if len(target_values) != 1:
            raise ValueError(
                f"Official iSAID instance {int(instance_id)} in {instance_path} does not have one foreground class."
            )
        class_index = int(target_values[0])
        area = int(pixels.sum())
        bin_index = instance_area_bin(area)
        totals[bin_index, 0] += 1
        totals[bin_index, 1] += area
        totals[bin_index, 2] += int((prediction[pixels] == class_index + 1).sum())
    return totals
