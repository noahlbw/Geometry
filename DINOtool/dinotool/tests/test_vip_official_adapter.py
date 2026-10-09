from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dinotool.vip_official_adapter import VIPOfficialAdapter, VIPQueries, VIPSettings, upstream_aliases, upstream_settings
from eval_vip_official_eight import Confusion


class VIPAdapterTests(unittest.TestCase):
    def test_official_settings_and_comma_space_alias_parser(self):
        self.assertEqual(upstream_settings("potsdam").prob_thd, 0.25)
        self.assertEqual(upstream_settings("vdd").bg_idx, 0)
        self.assertEqual(upstream_settings("vaihingen").bg_idx, 5)
        self.assertEqual(upstream_settings("oem").tau, 4.0)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "classes.txt"
            source.write_text("grass, grassland, farmland,forest area\nbuilding\n", encoding="utf-8")
            self.assertEqual(upstream_aliases(source),
                             (("grass", "grassland", "farmland,forest area"), ("building",)))

    def test_unscored_vaihingen_query_remains_a_false_negative(self):
        metric = Confusion(("road", "building"), predicted_classes=3)
        metric.update(np.array([[0, 2]], dtype=np.uint8), np.array([[0, 1]], dtype=np.uint8))
        result = metric.summary()
        self.assertEqual(result["confusion_matrix"], [[1, 0, 0], [0, 0, 1]])
        self.assertEqual(result["mean_iou_percent"], 50.0)
        self.assertEqual(result["unscored_prediction_classes"], 1)

    def test_other_exclusion_does_not_assume_first_class(self):
        metric = Confusion(("road", "building", "other"), predicted_classes=3)
        metric.update(np.array([[0, 1, 1]], dtype=np.uint8),
                      np.array([[0, 1, 2]], dtype=np.uint8))
        result = metric.summary()
        self.assertEqual(result["non_residual_mean_iou_percent"], 75.0)

    def test_short_edge_is_padded_without_changing_original_prediction_shape(self):
        class StubVIP(VIPOfficialAdapter):
            def __init__(self):
                self.backbone = type("Backbone", (), {"device": torch.device("cpu")})()

            def crop_logits(self, rgb, queries, settings):
                assert rgb.shape == (3, 336, 336)
                return torch.stack((torch.ones((336, 336)), torch.zeros((336, 336))))

        queries = VIPQueries(torch.empty((2, 1, 1)), torch.tensor([0, 1]), ("a", "b"), ("a", "b"))
        prediction, padded = StubVIP().predict(torch.zeros((3, 100, 500)), queries, VIPSettings())
        self.assertEqual(prediction.shape, (100, 500))
        self.assertFalse(prediction.any())
        self.assertGreater(padded, 0)


if __name__ == "__main__":
    unittest.main()
