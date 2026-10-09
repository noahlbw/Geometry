import unittest
from types import SimpleNamespace

import torch

from dinotool.geometry_csa_allocation import BASELINES, FACTORS, METHODS, factor_attended, read_head
from dinotool.matched_readout_controls import run_head
from dinotool.tcpr import _geometry_attended
from test_frozen_semantic_path import Block


class CSAAllocationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(205)
        self.a = torch.randn(1, 2, 6, 6).softmax(-1)
        self.c = torch.randn(1, 2, 6, 6).softmax(-1)+torch.randn(1, 2, 6, 6).softmax(-1)
        self.g = torch.randn(1, 4, 4).softmax(-1)
        self.v = torch.randn(1, 2, 6, 2)
        self.head = SimpleNamespace(blocks=[Block(), Block()], ln_final=torch.nn.LayerNorm(4),
                                    linear_projection=torch.nn.Identity())
        self.tokens = torch.randn(1, 6, 4)
        self.prepared = SimpleNamespace(backbone_tokens=self.tokens, prefix_tokens=2,
                                        geometry_patch_conditional=self.g, block_index=1)

    def test_native_factor_is_exact_geometry(self):
        actual = factor_attended(self.a, None, self.g, self.v, 2, "native", 1.)
        expected = _geometry_attended(self.a, self.g, self.v, 2)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_explicit_primary_attention_and_row_mass(self):
        special = self.c[..., 2:, :2]
        mass = self.c[..., 2:, 2:].sum(-1, keepdim=True)
        weights = torch.cat((special, mass*self.g[:, None]), -1)
        torch.testing.assert_close(weights.sum(-1), torch.ones(1, 2, 4)*2, atol=3e-7, rtol=0)
        actual = factor_attended(self.a, self.c, self.g, self.v, 2, "allocation", 1.)
        torch.testing.assert_close(actual[..., 2:, :], weights @ self.v, atol=2e-7, rtol=2e-7)

    def test_fixed_source_conditional_recovers_full_csa_patch_query_read(self):
        weights = self.c[..., 2:, 2:]
        conditional = weights/weights.sum(-1, keepdim=True)
        for multiplier in (.5, 1.):
            actual = factor_attended(self.a, self.c, conditional, self.v, 2, "allocation", multiplier)
            expected = factor_attended(self.a, self.c, self.g, self.v, 2, "csa", multiplier)
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)

    def test_every_new_factor_has_native_prefix_query_read(self):
        expected = self.a[..., :2, :] @ self.v
        for mode, multiplier in FACTORS.values():
            actual = factor_attended(self.a, self.c, self.g, self.v, 2, mode, multiplier)
            torch.testing.assert_close(actual[..., :2, :], expected, atol=0, rtol=0)

    def test_sum_mean_and_double_controls_change_only_patch_read_strength(self):
        for mode in ("csa", "allocation", "mass", "native"):
            a = factor_attended(self.a, self.c, self.g, self.v, 2, mode, 1.)
            b = factor_attended(self.a, self.c, self.g, self.v, 2, mode, .5)
            torch.testing.assert_close(a[..., 2:, :], b[..., 2:, :]*2, atol=0, rtol=0)
        a = factor_attended(self.a, self.c, self.g, self.v, 2, "native", 1.)
        b = factor_attended(self.a, self.c, self.g, self.v, 2, "native", 2.)
        torch.testing.assert_close(b[..., 2:, :], a[..., 2:, :]*2, atol=0, rtol=0)

    def test_allocation_and_mass_factors_agree_when_special_conditionals_agree(self):
        c = self.c.clone()
        special_mass = c[..., 2:, :2].sum(-1, keepdim=True)
        c[..., 2:, :2] = special_mass*self.a[..., 2:, :2]/self.a[..., 2:, :2].sum(-1, keepdim=True)
        a = factor_attended(self.a, c, self.g, self.v, 2, "allocation", 1.)
        b = factor_attended(self.a, c, self.g, self.v, 2, "mass", 1.)
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-7)

    def test_constant_values_expose_declared_row_masses(self):
        for mode, multiplier in FACTORS.values():
            actual = factor_attended(self.a, self.c, self.g, torch.ones_like(self.v), 2, mode, multiplier)
            mass = multiplier*(2 if mode in ("csa", "allocation", "mass") else 1)
            torch.testing.assert_close(actual[..., 2:, :], torch.ones_like(actual[..., 2:, :])*mass, atol=3e-7, rtol=0)

    def test_zero_group_conditionals_remain_finite(self):
        a, c = self.a.clone(), self.c.clone()
        a[..., 2:, :2] = 0
        c[..., 2:, 2:] = 0
        for mode in ("mass", "conditional"):
            actual = factor_attended(a, c, self.g, self.v, 2, mode, 1.)
            self.assertTrue(torch.isfinite(actual).all())

    def test_original_four_controls_replay_exactly(self):
        for method in BASELINES:
            expected = run_head(self.head, self.tokens, self.tokens[:, 2:], self.g, 2, method, 1)[0]
            actual = read_head(self.head, self.prepared, method)[0]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_no_mutation_and_finite_outputs(self):
        tokens, g = self.tokens.clone(), self.g.clone()
        for method in METHODS:
            self.assertTrue(torch.isfinite(read_head(self.head, self.prepared, method, trace=True)[0]).all())
        self.assertTrue(torch.equal(tokens, self.tokens) and torch.equal(g, self.g))

    def test_invalid_factor_or_method_rejected(self):
        with self.assertRaises(ValueError):
            factor_attended(self.a, self.c, self.g, self.v, 0, "allocation", 1.)
        with self.assertRaises(ValueError):
            read_head(self.head, self.prepared, "unknown")


if __name__ == "__main__":
    unittest.main()
