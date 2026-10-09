import unittest

import numpy as np

from diagnose_geometry_language_observation import query_metric


class QueryMetricTests(unittest.TestCase):
    def test_abstentions_are_false_negatives_not_dropped_or_an_extra_class(self):
        metrics = query_metric(np.array([[1, 0, 1], [0, 1, 0]]), ["a", "b"])
        self.assertEqual(metrics["mean_iou_percent"], 75.)
        self.assertEqual(metrics["abstained_queries"], 1)
        self.assertEqual(metrics["per_class"][0]["fn"], 1)

    def test_absent_class_false_activation_is_still_penalized(self):
        metrics = query_metric(np.array([[1, 1, 0], [0, 0, 0]]), ["a", "b"])
        self.assertEqual(metrics["mean_iou_percent"], 25.)
        self.assertEqual(metrics["per_class"][1]["fp"], 1)


if __name__ == "__main__":
    unittest.main()
