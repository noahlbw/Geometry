import unittest

import torch

from dinotool.context_memory_observer import ContextMemoryConfig, context_assignments
from dinotool.geov2_wide_optimized import (
    WIDE_READOUT_CHOICES,
    factor_correction,
    factor_reconstruction,
    wide_assignments,
    wide_readout_config,
)


class WideOptimizationTests(unittest.TestCase):
    def test_fast_slot_assignment_matches_incumbent(self):
        torch.manual_seed(7)
        raw = torch.randn(1, 96, 12)
        config = ContextMemoryConfig(capacity=12, temperature=.1, iterations=3,
                                     center=False, hard=False)
        self.assertTrue(torch.equal(wide_assignments(raw, config),
                                    context_assignments(raw, config)))

    def test_factorized_writer_matches_dense_operator(self):
        torch.manual_seed(11)
        relation = torch.softmax(torch.randn(25, 25), dim=-1)
        for partial in (False, True):
            with self.subTest(partial=partial):
                valid = torch.arange(25) % 3 != 0 if partial else torch.ones(25, dtype=torch.bool)
                factors, _ = factor_reconstruction(relation, valid)
                weights = relation.double().masked_fill(~valid[None], 0.)
                weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-12)
                weights = weights.masked_fill(~valid[:, None], 0.)
                gram = weights.T @ weights
                dense = torch.cholesky_solve(gram, torch.linalg.cholesky(
                    torch.eye(25, dtype=torch.float64) + gram))
                difference = torch.randn(25, 6, dtype=torch.float64)
                actual = factor_correction(factors, difference)
                torch.testing.assert_close(dense @ difference, actual, atol=1e-12, rtol=1e-12)
                self.assertEqual(torch.count_nonzero(actual[~valid]), 0)

    def test_readout_menu_keeps_incumbent_explicit(self):
        self.assertEqual(WIDE_READOUT_CHOICES, ('incumbent', 'shared_canvas', 'optimized48'))
        self.assertIsNone(wide_readout_config('incumbent'))
        self.assertFalse(wide_readout_config('shared_canvas').direct)
        optimized = wide_readout_config('optimized48')
        self.assertTrue(optimized.capacity == 48 and optimized.shared_visual and optimized.fast_assignment)


if __name__ == '__main__':
    unittest.main()
