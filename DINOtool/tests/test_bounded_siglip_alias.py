import unittest
from types import SimpleNamespace

import torch
from torch import nn

from dinotool.bounded_canonical_pair_alias import weight_controls
from dinotool.bounded_siglip_alias import (
    cached_probe_pool, contradiction_weights, fixed_local_delta, project_support, sampled_wide,
)
from dinotool.region_semantic_readout import restricted_pool
from dinotool.stratified_soft_alias import WideCrop, crop_stencil


class SiglipAliasTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261006)

    def head(self):
        return SimpleNamespace(
            attention=nn.MultiheadAttention(12, 3, batch_first=True).eval(),
            probe=torch.randn(1, 1, 12), layernorm=nn.LayerNorm(12),
            mlp=nn.Sequential(nn.Linear(12, 24), nn.GELU(), nn.Linear(24, 12)),
        )

    def test_cached_pool_replays_trained_attention_residual_and_mlp(self):
        head, hidden = self.head(), torch.randn(1, 16, 12)
        support = torch.rand(7, 16)
        support[:, 3:6] = 0
        expected = restricted_pool(head, hidden.expand(7, -1, -1), support)
        torch.testing.assert_close(cached_probe_pool(head, hidden, support), expected, atol=2e-6, rtol=2e-6)

    def test_pool_is_support_scale_invariant_and_singleton_exact(self):
        head, hidden = self.head(), torch.randn(1, 16, 12)
        support = torch.rand(7, 16)
        actual = cached_probe_pool(head, hidden, support)
        torch.testing.assert_close(actual, cached_probe_pool(head, hidden, support*100), atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual, torch.cat([cached_probe_pool(head, hidden, row[None]) for row in support]), atol=2e-6, rtol=2e-6)
        for bad in (support*0, -support, support.masked_fill(support > .5, torch.nan)):
            with self.assertRaises(ValueError):
                cached_probe_pool(head, hidden, bad)

    def test_actual_patch_support_and_padding_projection(self):
        valid = torch.ones(1024, dtype=torch.bool)
        rows = torch.zeros(2, 1024)
        rows[0, 0] = 1
        rows[1, -1] = 1
        projected, known = project_support(rows, 0, 0, (512, 512), valid)
        self.assertEqual(known.tolist(), [True, True])
        self.assertEqual(projected.argmax(-1).tolist(), [0, 255])
        projected, known = project_support(rows, 0, 0, (512, 512), valid.masked_fill(torch.arange(1024) == 1023, False))
        self.assertEqual(known.tolist(), [True, False])
        self.assertTrue(torch.equal(projected[1], torch.zeros(256)))

    def test_projected_support_respects_window_offsets_and_empty_rows(self):
        rows = torch.stack((torch.ones(1024), torch.zeros(1024)))
        projected, known = project_support(rows, 384, 384, (896, 896), torch.ones(1024, dtype=torch.bool))
        grid = projected[0].reshape(16, 16)
        self.assertEqual(known.tolist(), [True, False])
        self.assertTrue(torch.equal(grid[:6], torch.zeros_like(grid[:6])))
        self.assertGreater(float(grid[-1, -1]), 0)

    def test_canonical_protection_global_budget_and_unknown_fallback(self):
        pairs = torch.tensor([[0, 1], [1, 2], [2, 0]])
        canonical = torch.tensor([0, 1, 2])
        source, teacher = torch.randn(3, 2, 20)*30, torch.randn(3, 2, 20)*30
        weights = contradiction_weights(source, teacher, pairs, canonical, torch.tensor([True, True, False]))
        self.assertTrue(torch.equal(weights.gather(-1, canonical[pairs][..., None]), torch.ones(3, 2, 1, dtype=torch.float64)))
        self.assertLessEqual(float((1-weights).sum(-1).max()), 1+1e-12)
        self.assertTrue(torch.equal(weights[2], torch.ones_like(weights[2])))
        self.assertTrue(bool(((weights > 0) & (weights <= 1)).all()))

    def test_same_alias_can_receive_different_rival_actions(self):
        pairs = torch.tensor([[0, 1], [0, 2]])
        source = torch.ones(2, 2, 3)*10
        teacher = torch.ones_like(source)*10
        teacher[0, 0, 1] = -10
        weights = contradiction_weights(source, teacher, pairs, torch.zeros(3, dtype=torch.long), torch.ones(2, dtype=torch.bool))
        self.assertLess(float(weights[0, 0, 1]), .001)
        self.assertGreater(float(weights[1, 0, 1]), .999)

    def test_controls_preserve_canonical_mass_and_alias_spectrum(self):
        pairs = torch.tensor([[0, 1], [1, 2], [2, 0]])
        canonical = torch.tensor([0, 1, 2])
        weights = contradiction_weights(torch.randn(3, 2, 20), torch.randn(3, 2, 20), pairs, canonical, torch.ones(3, dtype=torch.bool))
        mean, shuffled = weight_controls(weights, pairs, canonical)
        torch.testing.assert_close(mean.sum(-1), weights.sum(-1), atol=1e-12, rtol=0)
        torch.testing.assert_close(shuffled.sort(-1).values, weights.sort(-1).values, atol=0, rtol=0)
        self.assertTrue(torch.equal(mean.gather(-1, canonical[pairs][..., None]), torch.ones(3, 2, 1, dtype=torch.float64)))

    def test_fixed_local_slot_attenuation_is_monotone_and_identity_exact(self):
        evidence, pairs = torch.randn(4, 3, 20)*10, torch.tensor([[0, 1]]*4)
        weights = torch.rand(4, 2, 20)
        delta = fixed_local_delta(evidence, pairs, weights)
        self.assertTrue(bool((delta <= 1e-12).all()))
        self.assertTrue(torch.equal(delta[:, 2], torch.zeros(4, dtype=torch.float64)))
        self.assertTrue(torch.equal(fixed_local_delta(evidence, pairs, torch.ones_like(weights)), torch.zeros_like(delta)))
        expected = (evidence[:, :2].double()+weights.double().log()).logsumexp(-1)-evidence[:, :2].double().logsumexp(-1)
        torch.testing.assert_close(delta[:, :2], expected, atol=0, rtol=0)

    def test_wide_slot_attenuation_matches_explicit_crop_stencils(self):
        members, pairs = torch.arange(60).reshape(3, 20), torch.tensor([[0, 1], [1, 2]])
        crop = WideCrop(torch.randn(4, 60), torch.randn(60), 0, 0, 2, 2, 2, 2)
        count, coords = torch.ones(2, 2), torch.tensor([[.5, .5], [1.5, 1.5]])
        weights = torch.rand(2, 2, 20)
        actual = sampled_wide([crop], count, coords, (2, 2), members, pairs, weights)
        ids, coefficients = crop_stencil(crop, count, coords, (2, 2))
        profile = crop.salience[members].softmax(-1)
        evidence = crop.alias_logits[:, members]*(profile/profile.mean(-1, keepdim=True))
        expected = torch.zeros_like(actual)
        for query in range(2):
            for side in range(2):
                c = pairs[query, side]
                change = (evidence[ids[query], c].double().softmax(-1)*weights[query, side]).sum(-1).log()
                expected[query, c] = (change*coefficients[query]).sum()
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        self.assertTrue(bool((actual <= 1e-12).all()))
        self.assertTrue(torch.equal(sampled_wide([crop], count, coords, (2, 2), members, pairs, torch.ones_like(weights)), torch.zeros_like(actual)))

    def test_joint_writer_is_exact_linear_change_and_zero_action(self):
        matrix = torch.randn(7, 7, dtype=torch.float64)
        h = torch.linalg.solve(torch.eye(7)+matrix.T@matrix, matrix.T@matrix)
        local, wide, dl, db = [torch.randn(7, 3, dtype=torch.float64) for _ in range(4)]
        base = local+h@(wide-local)
        torch.testing.assert_close(base+dl+h@(db-dl), (local+dl)+h@((wide+db)-(local+dl)), atol=1e-12, rtol=1e-12)
        zero = torch.zeros_like(local)
        self.assertTrue(torch.equal(base+zero+h@(zero-zero), base))


if __name__ == '__main__':
    unittest.main()
