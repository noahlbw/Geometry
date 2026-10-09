import numpy as np
import pytest
from PIL import Image

from dinotool.loveda import LoveDAConfusionMatrix, discover_loveda_samples, loveda_labeled_split


def _save_rgb(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8), mode="RGB").save(path)


def _save_mask(path, value: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 5), value, dtype=np.uint8), mode="L").save(path)


def test_discover_official_loveda_layout(tmp_path) -> None:
    for domain, name in (("Urban", "12.png"), ("Rural", "2.png")):
        _save_rgb(tmp_path / "Val" / domain / "images_png" / name)
        _save_mask(tmp_path / "Val" / domain / "masks_png" / name)
    samples = discover_loveda_samples(tmp_path)
    assert [sample.key for sample in samples] == ["Val/Rural/2.png", "Val/Urban/12.png"]
    assert loveda_labeled_split(samples) == "validation"


def test_discover_flat_mirror_layout_and_require_pairs(tmp_path) -> None:
    _save_rgb(tmp_path / "images" / "1.png")
    _save_mask(tmp_path / "masks" / "1.png")
    assert discover_loveda_samples(tmp_path)[0].key == "1.png"
    _save_rgb(tmp_path / "images" / "2.png")
    with pytest.raises(ValueError, match="without masks"):
        discover_loveda_samples(tmp_path)


def test_loveda_metric_mapping_and_ignore_ids() -> None:
    target = np.array([[0, 1, 2], [7, 255, 3]], dtype=np.uint8)
    prediction = np.array([[6, 0, 1], [6, 0, 1]], dtype=np.uint8)
    metric = LoveDAConfusionMatrix()
    metric.update(prediction, target)
    result = metric.summary()
    assert result["ignored_pixels"] == 2
    assert result["labeled_pixels"] == 4
    assert result["pixel_accuracy"] == pytest.approx(0.75)
    assert result["mean_iou"] == pytest.approx(0.625)


def test_loveda_metric_rejects_unknown_ground_truth_id() -> None:
    metric = LoveDAConfusionMatrix()
    with pytest.raises(ValueError, match="ground-truth IDs"):
        metric.update(np.zeros((1, 1), dtype=np.uint8), np.array([[8]], dtype=np.uint8))
