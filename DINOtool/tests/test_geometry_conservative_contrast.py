import unittest
from types import SimpleNamespace

import torch

from dinotool.geometry_conservative_contrast import BASELINES, FACTORS, METHODS, REFERENCES, contrast_correction, read_head
from dinotool.geometry_csa_allocation import read_head as allocation_head
from dinotool.matched_readout_controls import run_head
from dinotool.tcpr import _geometry_attended
from test_frozen_semantic_path import Block


class ConservativeContrastTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(305)
        self.a = torch.randn(1, 2, 6, 6).softmax(-1)
        self.g = torch.randn(1, 4, 4).softmax(-1)
        self.v = torch.randn(1, 2, 6, 2)
        self.tokens = torch.randn(1, 6, 4)
        self.head = SimpleNamespace(blocks=[Block(), Block()], ln_final=torch.nn.LayerNorm(4),
                                    linear_projection=torch.nn.Identity())
        self.prepared = SimpleNamespace(backbone_tokens=self.tokens, prefix_tokens=2,
                                        geometry_patch_conditional=self.g, block_index=1)

    def test_explicit_formula_and_preserved_mean(self):
        v = self.v[..., 2:, :]
        p = self.g[:, None] @ v
        e0 = (v-v.mean(-2, keepdim=True)).square().sum(-1).mean(-1, keepdim=True)[..., None]
        e1 = (p-p.mean(-2, keepdim=True)).square().sum(-1).mean(-1, keepdim=True)[..., None]
        gain = (1-e1/e0).clamp(0, 1)
        d = gain*self.a[..., 2:, 2:].sum(-1, keepdim=True)*(p-v.mean(-2, keepdim=True))
        d -= d.mean(-2, keepdim=True)
        original = _geometry_attended(self.a, self.g, self.v, 2)
        actual, stats = contrast_correction(self.a, self.g, self.v, 2, "conservative", trace=True)
        torch.testing.assert_close(actual[..., 2:, :], original[..., 2:, :]+d, atol=2e-7, rtol=2e-7)
        torch.testing.assert_close(actual[..., 2:, :].mean(-2), original[..., 2:, :].mean(-2), atol=1e-7, rtol=1e-7)
        self.assertLess(stats["correction_mean_max_error"], 1e-7)

    def test_identity_geometry_replays_original_exactly(self):
        g = torch.eye(4)[None]
        actual, stats = contrast_correction(self.a, g, self.v, 2, "conservative", trace=True)
        expected = _geometry_attended(self.a, g, self.v, 2)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(stats["gain_max"], 0)

    def test_uniform_geometry_cannot_invent_collapsed_contrast(self):
        g = torch.ones_like(self.g)/4
        actual, _ = contrast_correction(self.a, g, self.v, 2, "conservative")
        torch.testing.assert_close(actual, _geometry_attended(self.a, g, self.v, 2), atol=1e-7, rtol=1e-7)

    def test_constant_patch_values_have_no_adaptive_change(self):
        v = self.v.clone()
        v[..., 2:, :] = v[..., 2:3, :]
        actual, stats = contrast_correction(self.a, self.g, v, 2, "conservative", trace=True)
        torch.testing.assert_close(actual, _geometry_attended(self.a, self.g, v, 2), atol=0, rtol=0)
        self.assertEqual(stats["gain_max"], 0)

    def test_patch_common_shift_does_not_change_correction(self):
        shifted = self.v.clone()
        shifted[..., 2:, :] += torch.tensor([.25, -.75])
        before = contrast_correction(self.a, self.g, self.v, 2, "conservative")[0]-_geometry_attended(self.a, self.g, self.v, 2)
        after = contrast_correction(self.a, self.g, shifted, 2, "conservative")[0]-_geometry_attended(self.a, self.g, shifted, 2)
        torch.testing.assert_close(before, after, atol=1e-7, rtol=2e-7)

    def test_prefix_query_reads_are_unchanged_in_every_mode(self):
        expected = self.a[..., :2, :] @ self.v
        for mode in FACTORS.values():
            actual, _ = contrast_correction(self.a, self.g, self.v, 2, mode)
            torch.testing.assert_close(actual[..., :2, :], expected, atol=0, rtol=0)

    def test_gains_bounded_and_shuffle_preserves_multiset(self):
        _, primary = contrast_correction(self.a, self.g, self.v, 2, "conservative", trace=True)
        _, shuffled = contrast_correction(self.a, self.g, self.v, 2, "head_shuffle", trace=True)
        self.assertGreaterEqual(primary["gain_min"], 0)
        self.assertLessEqual(primary["gain_max"], 1)
        self.assertEqual(primary["gain_mean"], shuffled["applied_gain_mean"])

    def test_uncentered_control_keeps_same_input_gain(self):
        _, primary = contrast_correction(self.a, self.g, self.v, 2, "conservative", trace=True)
        _, control = contrast_correction(self.a, self.g, self.v, 2, "uncentered", trace=True)
        self.assertEqual(primary["gain_mean"], control["gain_mean"])
        self.assertGreater(control["correction_mean_norm"], primary["correction_mean_norm"])

    def test_gain_only_is_whole_read_amplification(self):
        actual, stats = contrast_correction(self.a, self.g, self.v, 2, "gain_only", trace=True)
        self.assertGreater(stats["correction_mean_norm"], 0)
        self.assertTrue(torch.isfinite(actual).all())

    def test_exact_baseline_and_old_factor_replays(self):
        for method in BASELINES:
            expected = run_head(self.head, self.tokens, self.tokens[:, 2:], self.g, 2, method, 1)[0]
            torch.testing.assert_close(read_head(self.head, self.prepared, method)[0], expected, atol=0, rtol=0)
        for method in REFERENCES:
            torch.testing.assert_close(read_head(self.head, self.prepared, method)[0], allocation_head(self.head, self.prepared, method)[0], atol=0, rtol=0)

    def test_no_input_mutation_and_finite_two_block_outputs(self):
        tokens, g, v = self.tokens.clone(), self.g.clone(), self.v.clone()
        for method in METHODS:
            self.assertTrue(torch.isfinite(read_head(self.head, self.prepared, method, trace=True)[0]).all())
        self.assertTrue(torch.equal(tokens, self.tokens) and torch.equal(g, self.g) and torch.equal(v, self.v))

    def test_invalid_mode_or_geometry_rejected(self):
        with self.assertRaises(ValueError):
            contrast_correction(self.a, self.g, self.v, 2, "unknown")
        with self.assertRaises(ValueError):
            contrast_correction(self.a, torch.zeros_like(self.g), self.v, 2, "conservative")


if __name__ == "__main__":
    unittest.main()
