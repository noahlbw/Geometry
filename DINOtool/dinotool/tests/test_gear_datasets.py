"""Validate the two newly added official dataset adapters without real target data."""
from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image
import rasterio

from dinotool.gear_datasets import discover_samples, load_rgb, load_target
from scripts.prepare_landcoverai_val import prepare


def test_flair_rgb_and_main12_mapping() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        image_dir = root / "extracted/flair_1_aerial_test/D015_2020/Z1/img"
        mask_dir = root / "extracted/flair_1_labels_test/D015_2020/Z1/msk"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        bands = np.stack([np.full((2, 3), value, np.uint8) for value in (10, 20, 30, 90, 100)])
        mask = np.array([[1, 12, 13], [19, 0, 255]], dtype=np.uint8)
        with rasterio.open(image_dir / "IMG_000001.tif", "w", driver="GTiff", width=3,
                           height=2, count=5, dtype="uint8") as output:
            output.write(bands)
        with rasterio.open(mask_dir / "MSK_000001.tif", "w", driver="GTiff", width=3,
                           height=2, count=1, dtype="uint8") as output:
            output.write(mask, 1)
        sample = discover_samples("flair1", root)[0]
        rgb = load_rgb(sample, "flair1")
        np.testing.assert_allclose(rgb[:, 0, 0].numpy(), np.array([10, 20, 30]) / 255)
        np.testing.assert_array_equal(load_target(sample, "flair1", (2, 3)),
                                      [[0, 11, 255], [255, 255, 255]])


def test_landcoverai_keeps_original_split_indexing() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root, output = Path(directory) / "raw", Path(directory) / "prepared"
        (root / "images").mkdir(parents=True)
        (root / "masks").mkdir()
        # split.py increments k for partial columns too, so the next row starts at 3.
        rgb = np.zeros((1100, 1200, 3), np.uint8)
        mask = np.zeros((1100, 1200), np.uint8)
        rgb[512:1024, :512] = [5, 40, 180]
        mask[512:1024, :512] = 4
        cv2.imwrite(str(root / "images/scene.tif"), rgb)
        cv2.imwrite(str(root / "masks/scene.tif"), mask)
        (root / "val.txt").write_text("scene_0\nscene_3\n")
        result = prepare(root, output)
        assert result["images"] == 2 and result["original_images"] == 1
        samples = discover_samples("landcoverai", output)
        assert [sample.key for sample in samples] == ["scene_0", "scene_3"]
        assert bool((load_target(samples[1], "landcoverai", (512, 512)) == 4).all())
        with Image.open(samples[1].image_path) as image:
            np.testing.assert_allclose(np.asarray(image)[100, 100], [180, 40, 5], atol=2)


if __name__ == "__main__":
    checks = [value for name, value in list(globals().items())
              if name.startswith("test_") and callable(value)]
    for check in checks:
        check()
    print(f"{len(checks)} GEAR dataset checks passed.")
