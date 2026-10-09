import numpy as np
from PIL import Image

from dinotool.oem import OEM_CLASSES, OpenEarthMapDataset, discover_oem_samples, oem_source_manifest


def _save_pair(root, region: str, name: str, values: np.ndarray) -> None:
    image_path = root / region / "images" / name
    label_path = root / region / "labels" / name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((*values.shape, 3), 127, dtype=np.uint8), mode="RGB").save(image_path)
    Image.fromarray(values.astype(np.uint8), mode="L").save(label_path)


def test_discover_oem_samples_and_map_official_raw_ids(tmp_path) -> None:
    values = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.uint8)
    _save_pair(tmp_path, "region-a", "a.png", values)
    _save_pair(tmp_path, "region-b", "b.png", values)
    (tmp_path / "train.txt").write_text("a.png\n", encoding="utf-8")
    (tmp_path / "val.txt").write_text("b.png\n", encoding="utf-8")

    train = discover_oem_samples(tmp_path, "train")
    val = discover_oem_samples(tmp_path, "val")
    assert train[0].key == "region-a/images/a.png"
    assert val[0].key == "region-b/images/b.png"

    dataset = OpenEarthMapDataset(train, crop_size=16, training=False)
    image, target = dataset[0]
    assert image.shape == (3, 16, 16)
    assert set(target.unique().tolist()) == {0, 1, 2, 3, 4, 5, 6, 7, 255}
    manifest = oem_source_manifest(tmp_path, "train", train)
    assert manifest["sample_count"] == 1
    assert manifest["classes"] == [spec.name for spec in OEM_CLASSES]


def test_oem_rejects_ambiguous_split_filenames(tmp_path) -> None:
    values = np.full((10, 10), 9, dtype=np.uint8)
    _save_pair(tmp_path, "region-a", "same.png", values)
    _save_pair(tmp_path, "region-b", "same.png", values)
    (tmp_path / "train.txt").write_text("same.png\n", encoding="utf-8")
    (tmp_path / "val.txt").write_text("same.png\n", encoding="utf-8")

    try:
        discover_oem_samples(tmp_path, "train")
    except ValueError as error:
        assert "ambiguous" in str(error)
    else:
        raise AssertionError("Expected an ambiguity error")


def test_oem_requires_explicit_opt_in_for_official_missing_xbd_images(tmp_path) -> None:
    values = np.ones((10, 10), dtype=np.uint8)
    _save_pair(tmp_path, "region-a", "present.png", values)
    (tmp_path / "train.txt").write_text("present.png\nomitted-xbd.png\n", encoding="utf-8")
    (tmp_path / "val.txt").write_text("present.png\n", encoding="utf-8")
    (tmp_path / "xbd_files.csv").write_text("source.png,omitted-xbd.png\n", encoding="utf-8")

    try:
        discover_oem_samples(tmp_path, "train")
    except ValueError as error:
        assert "no matching image" in str(error)
    else:
        raise AssertionError("Expected missing images to be rejected without opt-in")

    train = discover_oem_samples(tmp_path, "train", allow_missing_images=True)
    assert [sample.key for sample in train] == ["region-a/images/present.png"]
    manifest = oem_source_manifest(tmp_path, "train", train, allow_missing_images=True)
    assert manifest["declared_sample_count"] == 2
    assert manifest["available_sample_count"] == 1
    assert manifest["missing_image_count"] == 1
    assert manifest["allow_missing_images"] is True


def test_oem_rejects_unknown_raw_label_ids(tmp_path) -> None:
    values = np.full((10, 10), 9, dtype=np.uint8)
    _save_pair(tmp_path, "region-a", "a.png", values)
    _save_pair(tmp_path, "region-b", "b.png", np.ones((10, 10), dtype=np.uint8))
    (tmp_path / "train.txt").write_text("a.png\n", encoding="utf-8")
    (tmp_path / "val.txt").write_text("b.png\n", encoding="utf-8")
    dataset = OpenEarthMapDataset(discover_oem_samples(tmp_path, "train"), crop_size=16, training=False)
    try:
        dataset[0]
    except ValueError as error:
        assert "not supported" in str(error)
    else:
        raise AssertionError("Expected an unknown-label error")
