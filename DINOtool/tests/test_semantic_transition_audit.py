import importlib.util
from pathlib import Path
import unittest

import numpy as np


path = Path(__file__).resolve().parents[1] / "dinotool/semantic_correction_audit.py"
spec = importlib.util.spec_from_file_location("transition_audit", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TransitionAuditTest(unittest.TestCase):
    def test_three_change_types_and_tie(self):
        counts = module.transition_counts(np.array([0, 0, 0, 1]),
                                          np.array([1, 1, 1, 1]),
                                          np.array([1, 0, 2, 1]), 3)
        summary = module.transition_summary(counts)
        self.assertEqual(summary["valid"], 4)
        self.assertEqual(summary["changed"], 3)
        self.assertEqual(summary["beneficial"], 1)
        self.assertEqual(summary["harmful"], 1)
        self.assertEqual(summary["wrong_to_wrong"], 1)
        self.assertEqual(summary["base_confusion"], [[1, 0, 0], [1, 1, 0], [1, 0, 0]])
        self.assertEqual(summary["proposal_confusion"], [[0, 1, 0], [0, 2, 0], [0, 1, 0]])

    def test_chunk_independence(self):
        old, new, target = (np.array(values) for values in ([0, 1, 0, 1],
                                                           [1, 0, 0, 1],
                                                           [0, 0, 1, 1]))
        np.testing.assert_array_equal(module.transition_counts(old, new, target, 2, 1),
                                      module.transition_counts(old, new, target, 2, 100))

    def test_ignored_mask_not_prediction(self):
        counts = module.transition_counts(np.array([-1, 0]), np.array([-1, 1]),
                                          np.array([255, 1]), 2)
        self.assertEqual(counts.sum(), 1)

    def test_prediction_invalid_on_scored_pixel(self):
        with self.assertRaises(ValueError):
            module.transition_counts(np.array([-1]), np.array([0]), np.array([0]), 2)

    def test_shard_sum_equals_full(self):
        old, new, target = (np.array(values) for values in ([0, 0, 1, 1],
                                                           [1, 0, 0, 1],
                                                           [1, 1, 0, 0]))
        parts = sum((module.transition_counts(old[i:i+2], new[i:i+2], target[i:i+2], 2)
                     for i in (0, 2)), np.zeros((2, 2, 2), dtype=np.int64))
        np.testing.assert_array_equal(parts, module.transition_counts(old, new, target, 2))

    def test_float_indices_rejected(self):
        with self.assertRaises(ValueError):
            module.transition_counts(np.array([0.5]), np.array([0]), np.array([0]), 2)


if __name__ == "__main__":
    unittest.main()
