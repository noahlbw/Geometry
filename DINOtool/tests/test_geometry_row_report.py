import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"tools"))
from report_geometry_row_composition import paired_interval


class PairedReportTests(unittest.TestCase):
    def test_identical_arms_have_identical_paired_samples(self):
        keys = ["a/images/a1.tif", "a/images/a2.tif", "b/images/b1.tif"]
        cm = np.asarray([[[8, 2], [1, 9]], [[6, 4], [2, 8]], [[9, 1], [3, 7]]])
        groups, values = paired_interval(keys, {"a": cm, "b": cm.copy()}, "oem", replicates=100)
        self.assertEqual(groups, 2)
        np.testing.assert_array_equal(values["a"]["bootstrap"], values["b"]["bootstrap"])

    def test_deterministic_and_actual_aggregate_point(self):
        keys = ["a", "b"]
        cm = np.asarray([[[9, 1], [2, 8]], [[8, 2], [1, 9]]])
        _, a = paired_interval(keys, {"a": cm}, "vdd", replicates=100)
        _, b = paired_interval(keys, {"a": cm}, "vdd", replicates=100)
        np.testing.assert_array_equal(a["a"]["bootstrap"], b["a"]["bootstrap"])
        self.assertAlmostEqual(a["a"]["point"], 100*17/23)

    def test_duplicate_keys_or_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):
            paired_interval(["a", "a"], {"a": np.ones((2, 2, 2))}, "vdd")
        with self.assertRaises(ValueError):
            paired_interval(["a", "b"], {"a": np.ones((1, 2, 2))}, "vdd")


if __name__ == "__main__":
    unittest.main()
