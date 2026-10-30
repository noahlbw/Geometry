import unittest

import torch

from dinotool.context_memory_observer import ContextMemoryConfig, context_assignments
from dinotool.geov2_wide_optimized import (
    WIDE_READOUT_CHOICES,
    _sample_global,
    _sample_global_batch,
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
        self.assertEqual(WIDE_READOUT_CHOICES, ('incumbent', 'shared_canvas', 'anchored_infra', 'anchored_infra_fast', 'anchored_infra_bf16', 'anchored_infra_bf16_compile', 'anchored_infra_shared_visual', 'optimized48'))
        self.assertIsNone(wide_readout_config('incumbent'))
        self.assertFalse(wide_readout_config('shared_canvas').direct)
        infra = wide_readout_config('anchored_infra')
        self.assertTrue(infra.capacity == 24 and infra.direct and infra.fast_assignment)
        self.assertFalse(infra.shared_visual)
        self.assertEqual(infra.solve_dtype, 'float32')
        fast = wide_readout_config('anchored_infra_fast')
        self.assertEqual(fast.local_batch_size, 4)
        self.assertTrue(fast.batch_anchored and fast.compile_visual and fast.compile_project)
        bf16 = wide_readout_config('anchored_infra_bf16')
        self.assertEqual(bf16.weight_dtype, 'bfloat16')
        self.assertTrue(bf16.batch_anchored and bf16.capacity == 24)
        compiled = wide_readout_config('anchored_infra_bf16_compile')
        self.assertEqual(compiled.weight_dtype, 'bfloat16')
        self.assertTrue(compiled.compile_visual and compiled.compile_project)
        self.assertEqual(compiled.local_batch_size, 1)
        shared = wide_readout_config('anchored_infra_shared_visual')
        self.assertTrue(shared.shared_visual and shared.readout == 'anchored')

    def test_batched_anchor_sampling_matches_per_tile_sampling(self):
        torch.manual_seed(23)
        features = torch.randn(1, 16, 5)
        positions = torch.tensor([[0., 0.], [16., 8.]])
        batched = _sample_global_batch(features, (4, 4), (64, 64), positions)
        individual = torch.cat([
            _sample_global(features, (4, 4), (64, 64), int(top), int(left))
            for top, left in positions
        ], dim=0)
        torch.testing.assert_close(batched, individual, atol=1e-6, rtol=1e-6)
        optimized = wide_readout_config('optimized48')
        self.assertTrue(optimized.capacity == 48 and optimized.shared_visual and optimized.fast_assignment)


if __name__ == '__main__':
    unittest.main()
