import unittest

import numpy as np

from dinotool.grounded_mask_observer import METHODS
from eval_grounded_mask_likelihood import predictions


class MaskLikelihoodReplayTests(unittest.TestCase):
    def test_near_ties_keep_original_direct_argmax(self):
        scores = np.zeros((1024, 7), np.float32)
        scores[:, 6] = 1e-10
        raw = {method: scores for method in METHODS if method != "MeanProb_MaskLikelihood"}
        output = predictions(raw, 7)
        for method in raw:
            self.assertTrue(np.array_equal(output[method], np.full((512, 512), 6)))

    def test_probability_blend_preserves_equal_strong_prediction(self):
        scores = np.zeros((1024, 3), np.float32)
        scores[:, 2] = .1
        raw = {method: scores for method in METHODS if method != "MeanProb_MaskLikelihood"}
        output = predictions(raw, 3)
        self.assertTrue(np.array_equal(output["MeanProb_MaskLikelihood"], output["Geometry"]))


if __name__ == "__main__":
    unittest.main()
