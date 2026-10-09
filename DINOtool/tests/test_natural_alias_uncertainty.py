import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from report_natural_alias_uncertainty import paired_bootstrap, scores, sufficient_counts


class NaturalAliasUncertaintyTests(unittest.TestCase):
    def test_rectangular_rejection_counts_as_false_negative(self):
        matrix = np.asarray([[[2, 1, 4], [3, 5, 2]]])
        counts = sufficient_counts(matrix)
        np.testing.assert_array_equal(counts, [[[2, 5], [10, 11]]])
        self.assertAlmostEqual(scores(counts.sum(0), np.ones(2, bool)),
                               50 * (2 / 10 + 5 / 11))

    def test_sufficient_counts_commute_with_image_resampling(self):
        matrices = np.asarray([[[2, 1], [3, 5]], [[7, 6], [4, 3]]])
        weights = np.asarray([3, 2])
        compressed = np.einsum("n,nkc->kc", weights, sufficient_counts(matrices))
        direct = sufficient_counts(np.einsum("n,nij->ij", weights, matrices)[None])[0]
        np.testing.assert_array_equal(compressed, direct)

    def test_identical_arms_have_zero_paired_interval(self):
        one = sufficient_counts(np.asarray([[[2, 1], [3, 5]], [[7, 6], [4, 3]]]))
        result = paired_bootstrap(np.stack([one, one, one]), np.ones(2, bool), 100)
        for row in result.values():
            self.assertEqual(row["delta_pp"], 0)
            self.assertEqual(row["paired_95_interval_pp"], [0, 0])

    def test_fixed_support_does_not_hide_missing_class(self):
        self.assertEqual(scores(np.asarray([[1, 0], [1, 0]]), np.ones(2, bool)), 50)


if __name__ == "__main__":
    unittest.main()
