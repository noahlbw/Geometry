import math
import unittest

import torch

from dinotool.bounded_alias_path_audit import (
    PATHS, POOLS, removal_changes, routed_change, privileged_donor_changes)


class AliasPathAuditTests(unittest.TestCase):
    def test_all_removals_match_explicit_reduction(self):
        values = torch.randn(3, 2, 20, dtype=torch.float64)
        fixed, normalized = removal_changes(values)
        for k in range(20):
            retained = values[..., torch.arange(20) != k]
            expected = retained.logsumexp(-1)-values.logsumexp(-1)
            torch.testing.assert_close(fixed[..., k], expected)
            torch.testing.assert_close(normalized[..., k], expected+math.log(20/19))

    def test_dominant_alias_is_finite_without_probability_clamp(self):
        values = torch.tensor([[[1000., -1000., -999.]]])
        fixed, _ = removal_changes(values)
        self.assertTrue(bool(torch.isfinite(fixed).all()))
        self.assertLess(float(fixed[..., 0]), -1990.)

    def test_fixed_slots_never_increase(self):
        fixed, _ = removal_changes(torch.randn(4, 5, 20))
        self.assertLessEqual(float(fixed.max()), 1e-12)

    def test_equal_evidence_normalized_deletion_is_zero(self):
        _, normalized = removal_changes(torch.ones(2, 3, 20))
        torch.testing.assert_close(normalized, torch.zeros_like(normalized), atol=1e-12, rtol=0)

    def test_removing_low_alias_can_raise_normalized_score(self):
        _, normalized = removal_changes(torch.tensor([[[5., 5., -10.]]]))
        self.assertGreater(float(normalized[..., -1]), 0.)

    def test_local_and_wide_actions_match_recomputed_coupling(self):
        local, wide = torch.randn(4, 3).double(), torch.randn(4, 3).double()
        dl, db = torch.randn_like(local), torch.randn_like(wide)
        h = torch.randn(4, 4).double()
        before = local+h@(wide-local)
        for path, after in (('Local', local+dl+h@(wide-local-dl)),
                            ('Wide', local+h@(wide+db-local)),
                            ('Joint', local+dl+h@(wide+db-local-dl)),
                            ('DirectWide', before+db)):
            torch.testing.assert_close(before+routed_change(dl, db, h, path), after)

    def test_signed_operator_can_raise_a_different_query(self):
        h = torch.tensor([[.4, -.1], [-.1, .4]])
        delta = routed_change(torch.zeros(2, 1), torch.tensor([[-1.], [0.]]), h, 'Wide')
        self.assertGreater(float(delta[1, 0]), 0.)

    def test_privileged_donor_preserves_target_unknown_and_canonical(self):
        local = torch.tensor([[[-9., -2., -1.], [-9., -1., -3.]],
                              [[-9., -2., -1.], [-9., -1., -3.]],
                              [[-9., -2., -1.], [-9., -1., -3.]]])
        wide = local*2
        dl, db = privileged_donor_changes(local, wide, torch.tensor([0, 0]), torch.tensor([0, 1, -1]), 'Joint')
        torch.testing.assert_close(dl, torch.tensor([[0., -3.], [-2., 0.], [0., 0.]]))
        torch.testing.assert_close(db, dl*2)

    def test_invalid_inputs_and_path_rejected(self):
        with self.assertRaises(ValueError):
            removal_changes(torch.zeros(2, 3, 1))
        with self.assertRaises(ValueError):
            removal_changes(torch.full((2, 3, 20), torch.nan))
        with self.assertRaises(ValueError):
            routed_change(torch.zeros(2, 1), torch.zeros(2, 1), torch.eye(2), 'Undeclared')
        self.assertEqual(len(PATHS)*len(POOLS), 8)


if __name__ == '__main__':
    unittest.main()
