from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_geometry_support_specificity import rank_auc, support_signals


class SupportSpecificityAuditTest(unittest.TestCase):
    def test_rank_auc_ties_and_missing_group(self):
        self.assertEqual(rank_auc([1., 2., 2., 3.], [False, False, True, True]), (.875, 4))
        self.assertEqual(rank_auc([1.], [True]), (None, 0))

    def test_uniform_support_matches_control(self):
        margin = np.array([1., 2., 4.])
        fields = support_signals(np.ones((3, 3)), margin)
        np.testing.assert_allclose(fields["support_margin"], [3., 2.5, 1.5])
        np.testing.assert_allclose(fields["support_minus_window_control"], 0.)

    def test_diagonal_does_not_leak_query(self):
        relation = np.array([[100., 1., 0.], [1., 100., 0.], [0., 1., 100.]])
        fields = support_signals(relation, [5., 2., -1.])
        np.testing.assert_allclose(fields["support_margin"], [2., 5., 2.])
        np.testing.assert_allclose(fields["query_minus_support"], [3., -3., -3.])

    def test_rejects_missing_nonself_support(self):
        with self.assertRaises(ValueError):
            support_signals(np.eye(3), [1., 2., 3.])


if __name__ == "__main__":
    unittest.main()
