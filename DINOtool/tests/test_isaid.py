from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from dinotool.isaid import (
    ISAID_IGNORE_ID,
    ISaidForegroundConfusionMatrix,
    discover_isaid_validation,
    instance_area_bin,
    isaid_instance_ids,
    isaid_target_ids,
)


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def test_decodes_official_bgr_color_codes_from_pil_rgb(tmp_path: Path) -> None:
    # PIL RGB pixels are the reverse of the BGR triples in the devkit.
    mask = np.array([[[63, 0, 0], [63, 63, 0], [0, 0, 0]]], dtype=np.uint8)
    path = tmp_path / "P0003_instance_color_RGB.png"
    _write_rgb(path, mask)
    target = isaid_target_ids(path)
    assert target.tolist() == [[0, 1, ISAID_IGNORE_ID]]


def test_decodes_packed_instance_ids_in_the_official_channel_order(tmp_path: Path) -> None:
    # BGR=(7, 0, 0) encodes ID 7, and BGR=(7, 1, 0) encodes 263.
    instance = np.array([[[0, 0, 7], [0, 1, 7]]], dtype=np.uint8)
    path = tmp_path / "P0003_instance_id_RGB.png"
    _write_rgb(path, instance)
    assert isaid_instance_ids(path).tolist() == [[7, 263]]


def test_discovers_only_complete_original_validation_triples(tmp_path: Path) -> None:
    image_dir = tmp_path / "iSAID" / "val" / "images"
    image_dir.mkdir(parents=True)
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    _write_rgb(image_dir / "P0003.png", rgb)
    _write_rgb(image_dir / "P0003_instance_color_RGB.png", rgb)
    _write_rgb(image_dir / "P0003_instance_id_RGB.png", rgb)
    samples = discover_isaid_validation(tmp_path)
    assert [(sample.key, sample.image_path.name) for sample in samples] == [("P0003", "P0003.png")]


def test_discovery_rejects_missing_auxiliary_mask(tmp_path: Path) -> None:
    image_dir = tmp_path / "val" / "images"
    image_dir.mkdir(parents=True)
    _write_rgb(image_dir / "P0003.png", np.zeros((1, 1, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="Missing official iSAID masks"):
        discover_isaid_validation(tmp_path)


@pytest.mark.parametrize(
    ("area", "expected"),
    [(1, 0), (1023, 0), (1024, 1), (9215, 1), (9216, 2)],
)
def test_instance_area_bins_are_fixed(area: int, expected: int) -> None:
    assert instance_area_bin(area) == expected


def test_foreground_metric_keeps_background_misses_as_false_negatives() -> None:
    metric = ISaidForegroundConfusionMatrix()
    metric.update(
        np.array([[1, 0, 2, 0]], dtype=np.uint8),
        np.array([[0, 0, 1, ISAID_IGNORE_ID]], dtype=np.uint8),
    )
    summary = metric.summary()
    assert summary["per_class"][0]["iou"] == 0.5
    assert summary["per_class"][0]["background_miss_pixels"] == 1
    assert summary["per_class"][1]["iou"] == 1.0
