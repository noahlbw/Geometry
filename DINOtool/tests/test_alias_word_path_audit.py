import unittest

import torch

from dinotool.alias_word_path_audit import decomposition, ratio


class WordPathAuditTests(unittest.TestCase):
    def setUp(self):
        self.members = torch.arange(12).reshape(3, 4)
        self.canonical = self.members[:, 0]
        self.valid = torch.tensor([True, True, False])
        self.risk = torch.zeros(3, 12, 3, dtype=torch.float64)
        self.zero = torch.zeros(3, 3, 3, dtype=torch.float64)
        self.operator = torch.diag(self.valid.double())
        self.positive = torch.tensor([[2., 1., 0.], [0., 2., 1.], [0., 0., 0.]], dtype=torch.float64)

    def run_audit(self, directed=None, common=None):
        return decomposition(self.risk, self.members, self.canonical, self.valid,
            self.zero if directed is None else directed, self.zero if common is None else common,
            self.operator, self.positive)

    def test_zero_action_has_finite_zero_ratios(self):
        row = self.run_audit()
        for key in ("within_class_risk_power_fraction", "word_pair_gradient_power_fraction",
                    "word_H_power_retention", "word_write_relative_norm", "soft_vs_class_patch_argmax_fraction"):
            self.assertEqual(row[key], 0.)
        self.assertEqual(row["valid_queries"], 2)

    def test_equal_noncanonical_risk_has_zero_within_class_power(self):
        self.risk[0, 1:4, 1] = .25
        self.assertEqual(self.run_audit()["within_class_risk_power_fraction"], 0.)

    def test_word_difference_is_not_hidden_by_canonical_zero_slots(self):
        self.risk[0, 1, 1] = .75
        self.assertAlmostEqual(self.run_audit()["within_class_risk_power_fraction"], 2/3)

    def test_gradient_and_cycle_power_are_orthogonal(self):
        directed = self.zero.clone()
        directed[0, 0, 1], directed[0, 1, 2], directed[0, 2, 0] = -.3, -.2, -.1
        row = self.run_audit(directed)
        self.assertAlmostEqual(row["word_pair_gradient_power_fraction"] + row["word_pair_cycle_power_fraction"], 1.)
        self.assertLess(row["projection_orthogonality_absolute_error"], 1e-12)
        self.assertAlmostEqual(row["word_H_power_retention"], 1.)

    def test_pure_cycle_cannot_enter_class_logits(self):
        directed = self.zero.clone()
        directed[0, 0, 1] = directed[0, 1, 2] = directed[0, 2, 0] = -.2
        row = self.run_audit(directed)
        self.assertAlmostEqual(row["word_pair_cycle_power_fraction"], 1.)
        self.assertEqual(row["word_write_relative_norm"], 0.)

    def test_common_action_has_no_word_difference(self):
        directed = self.zero.clone()
        directed[0, 0, 1] = -.2
        self.assertEqual(self.run_audit(directed, directed)["word_write_relative_norm"], 0.)

    def test_rejects_nonfinite_or_zero_reference_mismatch(self):
        with self.assertRaisesRegex(ValueError, "zero reference"):
            ratio(1, 0)
        self.risk[0, 1, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "Matching finite"):
            self.run_audit()


if __name__ == "__main__":
    unittest.main()
