import unittest

import torch

from dinotool.family_dependency_audit import alias_margins, dependency_fields, summarize


class FamilyDependencyAuditTest(unittest.TestCase):
    def test_useful_native_dependency_increases_the_old_risk(self):
        full, held = torch.tensor([[[2., 3.]]]), torch.tensor([[[1., 3.]]])
        args = (torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.bool))
        fields, guard = dependency_fields(held, args[0], full, args[0], *args[1:])
        self.assertEqual(float(fields['margin_loss'][0, 0]), 1.)
        self.assertGreater(float(fields['added_risk'][0, 0]), 0.)
        row = summarize(fields, torch.ones(1, 1), guard=guard)
        self.assertEqual(row['increased_risk_with_lost_margin'], 1)

    def test_unknown_and_invalid_have_no_comparison(self):
        field = torch.tensor([[[1., 3.]], [[1., 3.]]])
        margin, known = alias_margins(field, torch.tensor([[True, False]]),
            torch.zeros(2, dtype=torch.long), torch.tensor([0, 1]), torch.tensor([True, False]))
        self.assertFalse(bool(known.any()))
        self.assertEqual(float(margin.abs().max()), 0.)

    def test_class_common_offsets_do_not_change_dependency(self):
        full, held = torch.tensor([[[2., 3.]]], dtype=torch.float64), torch.tensor([[[1., 3.]]], dtype=torch.float64)
        args = (torch.ones(1, 2, dtype=torch.bool), torch.zeros(2, dtype=torch.long),
            torch.tensor([0, 1]), torch.ones(1, dtype=torch.bool))
        a, _ = dependency_fields(held, args[0], full, args[0], *args[1:])
        b, _ = dependency_fields(held+17., args[0], full+17., args[0], *args[1:])
        for name in a:
            torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)

    def test_all_risk_increases_follow_margin_loss(self):
        generator = torch.Generator().manual_seed(20261003)
        held = torch.randn(31, 4, 3, generator=generator, dtype=torch.float64)
        full = torch.randn(31, 1, 3, generator=generator, dtype=torch.float64)
        a, p, valid = torch.arange(4), torch.tensor([0, 1, 2, 0]), torch.ones(31, dtype=torch.bool)
        fields, guard = dependency_fields(held, torch.ones(4, 3, dtype=torch.bool),
            full, torch.ones(1, 3, dtype=torch.bool), a, p, valid)
        row = summarize(fields, torch.ones(31, 4), guard=guard)
        self.assertEqual(row['increased_risk_comparisons'], row['increased_risk_with_lost_margin'])


if __name__ == '__main__':
    unittest.main()
