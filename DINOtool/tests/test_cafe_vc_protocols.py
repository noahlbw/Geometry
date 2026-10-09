"""P/D protocol and checkpoint isolation, with no target dataset required."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_cafe_vc_protocols import checkpoint_config, protocol_settings, PublishedReadout
from cafedino_locked_loveda import target_ids, ConfusionMatrix
from train_cafe_rc import sliding_logits


def test_locked_tracks():
    p, d = protocol_settings("P"), protocol_settings("D")
    assert (p["evaluation_size"], p["window_size"], p["stride"]) == (512, 224, 112)
    assert p["classes"] == ["building", "road", "water", "barren", "tree", "farm"]
    assert not p["include_background"]
    assert (d["evaluation_size"], d["window_size"], d["stride"]) == (0, 448, 224)
    assert d["classes"] == ["background", "building", "road", "water", "barren", "forest", "agricultural"]
    assert d["include_background"]


def test_foreground_ignores_gt_background_not_just_mean(tmp_path):
    path = tmp_path / "mask.png"
    Image.fromarray(np.array([[0, 1, 2, 3, 4, 5, 6, 7, 255]], dtype=np.uint8)).save(path)
    p = target_ids(path, 0, include_background=False)
    d = target_ids(path, 0, include_background=True)
    assert p.tolist() == [[255, 255, 0, 1, 2, 3, 4, 5, 255]]
    assert d.tolist() == [[255, 0, 1, 2, 3, 4, 5, 6, 255]]
    metric = ConfusionMatrix(tuple(protocol_settings("P")["classes"]))
    metric.update(np.array([[0, 5, 0, 1, 2, 3, 4, 5, 0]]), p)
    assert metric.summary()["mean_iou"] == 1.0
    assert metric.ignored_pixels == 3


def test_old_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="v1"):
        checkpoint_config({"format": "cafe_vc_coco_v1"})


def test_sliding_aggregation_preserves_coordinate_scores():
    class CoordinateModel(torch.nn.Module):
        def forward(self, image, text, pre_text_emb):
            assert pre_text_emb
            return image[:, :len(text)]
    image = torch.arange(3 * 35 * 49, dtype=torch.float32).reshape(1, 3, 35, 49)
    actual = sliding_logits(PublishedReadout(CoordinateModel()), image, torch.ones(2, 4), 32, 16, "fp32")
    torch.testing.assert_close(actual, image[0, :2], rtol=0, atol=0)
