from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from eval_frozen_semantic_path import predict_image


class RGBReaderTests(unittest.TestCase):
    def test_rgb_callback_preserves_original_geometry_predictions(self):
        image = torch.rand(3, 49, 73)
        bank = SimpleNamespace(class_count=2, features=torch.eye(2))
        geometry = SimpleNamespace(device=torch.device("cpu"), prepare_image=lambda rgb: rgb.clone())
        patch_scores = torch.randn(1, 1024, 2)
        observed = []

        def original_reader(geometry, prepared, banks, texts, valid):
            return {"test": {"Geometry": patch_scores}}, {"regions": 0.}

        def rgb_reader(geometry, prepared, banks, texts, valid, rgb):
            torch.testing.assert_close(prepared, rgb, rtol=0, atol=0)
            torch.testing.assert_close(rgb[0, :, :49, :73], image, rtol=0, atol=0)
            self.assertEqual(int(valid.sum()), 15)
            observed.append(rgb.shape)
            return original_reader(geometry, prepared, banks, texts, valid)

        with TemporaryDirectory() as directory:
            first, first_diag = predict_image(image, geometry, {"test": bank}, Path(directory),
                                               ("Geometry",), original_reader)
            second, second_diag = predict_image(image, geometry, {"test": bank}, Path(directory),
                                                 ("Geometry",), rgb_reader, reader_uses_rgb=True)
        np.testing.assert_array_equal(first["test"]["Geometry"], second["test"]["Geometry"])
        self.assertEqual(first_diag, second_diag)
        self.assertEqual(observed, [torch.Size([1, 3, 512, 512])])


if __name__ == "__main__":
    unittest.main()
