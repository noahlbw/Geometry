import unittest

import numpy as np

from eval_cached_localized_likelihood import dense_predictions
from dinotool.geometry_localized_likelihood import METHODS


class CachedLocalLikelihoodTests(unittest.TestCase):
    def test_scores_argmax_preserves_tiny_margin_before_softmax_rounding(self):
        value = np.zeros((1024, 7), np.float32)
        value[:, 6] = 1e-10
        row = {method: value for method in METHODS if method != "MeanProb_BoxLikelihood"}
        predictions = dense_predictions(row, 7)
        for method in row:
            self.assertTrue(np.array_equal(predictions[method], np.full((512, 512), 6)))
        self.assertEqual(predictions["MeanProb_BoxLikelihood"].shape, (512, 512))


if __name__ == "__main__":
    unittest.main()
