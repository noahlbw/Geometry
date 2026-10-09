from pathlib import Path

import numpy as np
from PIL import Image

from dinotool.coco_stuff import (
    COCO_CAFE41_REMAP,
    CocoStuffCafe41Dataset,
    CocoStuffSample,
    validate_cafe41_mapping,
)


def test_locked_coco_mapping_preserves_void_and_never_invents_background():
    validate_cafe41_mapping()
    assert COCO_CAFE41_REMAP[1] == 0
    assert COCO_CAFE41_REMAP[148] == 29
    assert COCO_CAFE41_REMAP[168] == 38
    assert COCO_CAFE41_REMAP[0] == 255
    assert COCO_CAFE41_REMAP[255] == 255


def test_coco_dataset_applies_paired_crop_and_nearest_mask_resize(tmp_path: Path):
    image = np.zeros((11, 17, 3), dtype=np.uint8)
    image[..., 1] = 255
    label = np.full((11, 17), 255, dtype=np.uint8)
    label[2:9, 3:14] = 29
    image_path, label_path = tmp_path / "x.jpg", tmp_path / "x.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(label).save(label_path)
    sample = CocoStuffSample("x", image_path, label_path)
    train_image, train_label = CocoStuffCafe41Dataset([sample], crop_size=16, training=True)[0]
    eval_image, eval_label = CocoStuffCafe41Dataset([sample], crop_size=16, training=False)[0]
    assert train_image.shape == (3, 16, 16) and train_label.shape == (16, 16)
    assert eval_image.shape == (3, 16, 16) and eval_label.shape == (16, 16)
    assert set(eval_label.unique().tolist()) <= {29, 255}
