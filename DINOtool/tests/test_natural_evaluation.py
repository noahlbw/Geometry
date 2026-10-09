import unittest

import numpy as np

from dinotool.natural_evaluation import expand_aliases, map_target


class NaturalEvaluationTest(unittest.TestCase):
    def test_expansion_is_fixed_unique_and_canonical_first(self):
        for name in ('background', 'aeroplane', 'chair', 'wall-brick', 'playingfield', 'road'):
            values = expand_aliases(name)
            self.assertEqual(values[0], name)
            self.assertEqual(len(values), 20)
            self.assertEqual(len({v.casefold() for v in values}), 20)
            self.assertFalse(any('satellite' in v or 'aerial' in v for v in values))

    def test_foreground_and_background_voc_protocols(self):
        raw = np.array([[0, 1, 20, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(map_target(raw, 'voc20'), [[-1, 0, 19, -1]])
        np.testing.assert_array_equal(map_target(raw, 'voc21'), [[0, 1, 20, -1]])

    def test_ade_zero_is_ignore_and_one_based_ids_shift(self):
        np.testing.assert_array_equal(map_target(np.array([[0, 1, 150, 255]]), 'ade150'), [[-1, 0, 149, -1]])

    def test_context_pair_protocols(self):
        raw = np.array([[0, 1, 59, 255]])
        np.testing.assert_array_equal(map_target(raw, 'context59'), [[-1, 0, 58, -1]])
        np.testing.assert_array_equal(map_target(raw, 'context60'), [[0, 1, 59, -1]])

    def test_coco_ignore_does_not_become_background(self):
        np.testing.assert_array_equal(map_target(np.array([[0, 80, 255]]), 'coco_object81'), [[0, 80, -1]])
        np.testing.assert_array_equal(map_target(np.array([[0, 170, 255]]), 'coco_stuff171'), [[0, 170, -1]])

    def test_city_raw_ids(self):
        np.testing.assert_array_equal(map_target(np.array([[0, 7, 8, 33, 255]]), 'cityscapes19', True), [[-1, 0, 1, 18, -1]])

    def test_unexpected_ids_fail(self):
        with self.assertRaises(ValueError):
            map_target(np.array([[151]]), 'ade150')


if __name__ == '__main__':
    unittest.main()
