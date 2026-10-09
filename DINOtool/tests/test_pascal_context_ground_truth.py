import unittest

import numpy as np

from dinotool.natural_evaluation import map_target
from dinotool.pascal_context_ground_truth import CLASSES, SOURCE_IDS, map_full_category_mask


class PascalContextGroundTruthTest(unittest.TestCase):
    def test_selected_raw_ids_keep_standard_order(self):
        raw = np.asarray(SOURCE_IDS, dtype=np.uint16)[None]
        np.testing.assert_array_equal(map_full_category_mask(raw), np.arange(60)[None])
        self.assertEqual(CLASSES[SOURCE_IDS.index(260)], 'mouse')
        self.assertEqual(CLASSES[SOURCE_IDS.index(284)], 'person')

    def test_nonselected_concepts_are_residual_background(self):
        np.testing.assert_array_equal(map_full_category_mask(np.array([[1, 3, 255, 459]], dtype=np.uint16)), [[0, 0, 0, 0]])

    def test_one_png_supports_both_context_protocols(self):
        raw = map_full_category_mask(np.array([[0, 2, 284, 458]], dtype=np.uint16))
        np.testing.assert_array_equal(map_target(raw, 'context60'), [[0, 1, 37, 59]])
        np.testing.assert_array_equal(map_target(raw, 'context59'), [[-1, 0, 36, 58]])

    def test_non_original_masks_are_rejected(self):
        for raw in (np.array([2]), np.array([[2.]]), np.array([[-1]]), np.array([[460]])):
            with self.assertRaises(ValueError):
                map_full_category_mask(raw)


if __name__ == '__main__':
    unittest.main()
