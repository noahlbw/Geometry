import unittest
from types import SimpleNamespace

import torch

from dinotool.geometry_row_composition import (BASELINES, FACTORS, METHODS, PRIMARY,
                                             composed_attended, normalize_rows, read_head)
from dinotool.geometry_semantic_response import read_head as prior_head
from dinotool.matched_readout_controls import run_head
from test_frozen_semantic_path import Block


class RowCompositionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(105)
        self.a = torch.randn(1, 2, 6, 6).softmax(-1)
        self.g = torch.randn(1, 4, 4).softmax(-1)
        self.v = torch.randn(1, 2, 6, 2)
        self.tokens = torch.randn(1, 6, 4)
        self.head = SimpleNamespace(blocks=[Block(), Block()], ln_final=torch.nn.LayerNorm(4),
                                    linear_projection=torch.nn.Identity())
        self.prepared = SimpleNamespace(backbone_tokens=self.tokens, prefix_tokens=2,
                                        geometry_patch_conditional=self.g, block_index=1)

    def test_explicit_composed_row_and_mass(self):
        a, g = normalize_rows(self.a), normalize_rows(self.g)[:, None]
        m = a[..., 2:, 2:].sum(-1, keepdim=True)
        coefficients = torch.cat((g @ a[..., 2:, :2], g*m.transpose(-1, -2)), dim=-1)
        torch.testing.assert_close(coefficients.sum(-1), torch.ones(1, 2, 4), atol=2e-7, rtol=0)
        actual, diagnostics = composed_attended(self.a, self.g, self.v, 2, "full", trace=True)
        torch.testing.assert_close(actual[..., 2:, :], coefficients @ self.v, atol=2e-7, rtol=2e-7)
        self.assertLess(diagnostics["row_mass_max_error"], 2e-7)

    def test_identity_recovers_fp_read_all_local_factors(self):
        eye = torch.eye(4)[None]
        expected = composed_attended(self.a, eye, self.v, 2, "original")[0]
        for mode in ("full", "budget", "special", "group", "conditional"):
            actual = composed_attended(self.a, eye, self.v, 2, mode)[0]
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)

    def test_prefix_is_native_for_every_factor(self):
        expected = normalize_rows(self.a)[..., :2, :] @ self.v
        for mode in FACTORS.values():
            result = composed_attended(self.a, self.g, self.v, 2, mode)[0]
            torch.testing.assert_close(result[..., :2, :], expected, atol=0, rtol=0)

    def test_all_factors_read_constant_value_without_changing_mass(self):
        v = torch.ones_like(self.v)*3
        for mode in FACTORS.values():
            result = composed_attended(self.a, self.g, v, 2, mode)[0]
            torch.testing.assert_close(result, v, atol=7e-7, rtol=0)

    def test_constant_patch_mass_removes_budget_and_conditional_effect(self):
        a = self.a.clone()
        a[..., 2:, :2] = a[..., 2:, :2]/a[..., 2:, :2].sum(-1, keepdim=True)*.3
        a[..., 2:, 2:] = a[..., 2:, 2:]/a[..., 2:, 2:].sum(-1, keepdim=True)*.7
        original = composed_attended(a, self.g, self.v, 2, "original")[0]
        for mode in ("budget", "conditional", "global"):
            torch.testing.assert_close(composed_attended(a, self.g, self.v, 2, mode)[0],
                                       original, atol=2e-7, rtol=2e-7)
        group = composed_attended(a, self.g, self.v, 2, "group")[0]
        full = composed_attended(a, self.g, self.v, 2, "full")[0]
        torch.testing.assert_close(group, full, atol=2e-7, rtol=2e-7)

    def test_group_and_full_differ_with_variable_donor_budget(self):
        group = composed_attended(self.a, self.g, self.v, 2, "group")[0]
        full = composed_attended(self.a, self.g, self.v, 2, "full")[0]
        self.assertFalse(torch.allclose(group[..., 2:, :], full[..., 2:, :]))

    def test_common_all_value_shift_cancels_but_patch_shift_can_change_displacement(self):
        def delta(values):
            return (composed_attended(self.a, self.g, values, 2, "full")[0]
                    -composed_attended(self.a, self.g, values, 2, "original")[0])
        before = delta(self.v)
        shift = torch.tensor([.2, -.5])
        torch.testing.assert_close(delta(self.v+shift), before, atol=3e-7, rtol=3e-7)
        changed = self.v.clone()
        changed[..., 2:, :] += shift
        actual = delta(changed)-before
        m = normalize_rows(self.a)[..., 2:, 2:].sum(-1, keepdim=True)
        expected = (normalize_rows(self.g)[:, None] @ m-m)*shift
        torch.testing.assert_close(actual[..., 2:, :], expected, atol=3e-7, rtol=3e-7)
        self.assertGreater(float(actual[..., 2:, :].abs().max()), 1e-4)

    def test_zero_group_fallbacks_are_finite_and_keep_mass(self):
        for patch_only in (False, True):
            a = self.a.clone()
            a[..., 2:, :2 if patch_only else 0] = 0
            if not patch_only:
                a[..., 2:, 2:] = 0
            for mode in FACTORS.values():
                result = composed_attended(a, self.g, torch.ones_like(self.v), 2, mode)[0]
                self.assertTrue(torch.isfinite(result).all())
                torch.testing.assert_close(result, torch.ones_like(result), atol=2e-7, rtol=0)

    def test_original_four_controls_are_exact(self):
        for method in BASELINES:
            actual = read_head(self.head, self.prepared, method)[0]
            expected = run_head(self.head, self.tokens, self.tokens[:, 2:], self.g, 2, method, 1)[0]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_prior_donor_before_is_exact(self):
        actual = read_head(self.head, self.prepared, "Geometry_DonorBefore")[0]
        expected = prior_head(self.head, self.prepared, "Geometry_DonorBefore")[0]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_no_input_or_parameter_mutation_and_finite_outputs(self):
        tokens, g, a, v = [x.clone() for x in (self.tokens, self.g, self.a, self.v)]
        weights = [block.attn.qkv.weight.clone() for block in self.head.blocks]
        for method in METHODS:
            self.assertTrue(torch.isfinite(read_head(self.head, self.prepared, method, trace=True)[0]).all())
        for before, after in zip((tokens, g, a, v), (self.tokens, self.g, self.a, self.v)):
            self.assertTrue(torch.equal(before, after))
        self.assertTrue(all(torch.equal(before, block.attn.qkv.weight) for before, block in zip(weights, self.head.blocks)))

    def test_bf16_inputs_keep_value_dtype_and_finite(self):
        actual = composed_attended(self.a.bfloat16(), self.g.bfloat16(), self.v.bfloat16(), 2, "full")[0]
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(actual).all())

    def test_invalid_requests_rejected(self):
        with self.assertRaises(ValueError):
            composed_attended(self.a, self.g, self.v, 0, "full")
        with self.assertRaises(ValueError):
            composed_attended(self.a, self.g[:, :2], self.v, 2, "full")
        with self.assertRaises(ValueError):
            read_head(self.head, self.prepared, "unknown")


if __name__ == "__main__":
    unittest.main()
