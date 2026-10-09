import math
import unittest

import torch

from dinotool.bounded_alias_readout import bounded_classes, bounded_score, capped_responsibilities


class BoundedAliasTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.scores = torch.randn(5, 20, dtype=torch.float64)

    def test_unrestricted_exactly_recovers_lme_and_lse(self):
        for normalized in (False, True):
            result, _ = bounded_score(self.scores, .07, 20, normalize_count=normalized)
            expected = .07 * (torch.logsumexp(self.scores / .07, -1)
                              - (math.log(20) if normalized else 0))
            self.assertTrue(torch.equal(result, expected))

    def test_uniform_limit(self):
        for normalized in (False, True):
            result, _ = bounded_score(self.scores, .07, 1, normalize_count=normalized)
            expected = self.scores.mean(-1) + (0 if normalized else .07 * math.log(20))
            self.assertTrue(torch.equal(result, expected))

    def test_inactive_cap_exact_identity_in_mixed_rows(self):
        for dtype in (torch.float32, torch.float64):
            scores = torch.linspace(-.01, .01, 20, dtype=dtype).repeat(4, 1)
            scores[1, 0] = 2
            scores[3, 3] = 3
            original_weights = (scores / .07).softmax(-1)
            inactive = original_weights.amax(-1) <= .2
            self.assertEqual(inactive.tolist(), [True, False, True, False])
            weights = capped_responsibilities(scores, .07, 4)
            self.assertTrue(torch.equal(weights[inactive], original_weights[inactive]))
            for normalized in (False, True):
                result, diagnostics = bounded_score(scores, .07, 4, normalize_count=normalized)
                expected = .07 * (torch.logsumexp(scores / .07, -1)
                                  - (math.log(20) if normalized else 0))
                self.assertTrue(torch.equal(result[inactive], expected[inactive]))
                self.assertTrue(bool((result[~inactive] < expected[~inactive]).all()))
                self.assertEqual(diagnostics["inactive_max_absolute_score_change"], 0.0)

    def test_simplex_and_cap(self):
        for count, multiplier in ((20, 4), (20, 2.3), (7, 1.1), (7, 3.5)):
            values = torch.randn(50, count, dtype=torch.float64) * 10
            weights = capped_responsibilities(values, .07, multiplier)
            self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(50, dtype=torch.float64), atol=1e-10))
            self.assertLessEqual(float(weights.max()), multiplier / count + 1e-12)

    def test_kkt_conditions(self):
        temperature, cap = .3, .2
        weights = capped_responsibilities(self.scores, temperature, 4)
        derivative = self.scores - temperature * (weights.log() + 1)
        for weight, row in zip(weights, derivative):
            free = weight < cap - 1e-9
            self.assertTrue(torch.allclose(row[free], row[free][0].expand_as(row[free]), atol=1e-9))
            self.assertTrue(bool((row[~free] >= row[free][0] - 1e-9).all()))

    def test_score_is_bounded_by_mean_and_original_lme(self):
        result, _ = bounded_score(self.scores, .07, 4, normalize_count=True)
        original, _ = bounded_score(self.scores, .07, 20, normalize_count=True)
        self.assertTrue(bool((result >= self.scores.mean(-1) - 1e-10).all()))
        self.assertTrue(bool((result <= original + 1e-10).all()))

    def test_finite_difference_matches_responsibility(self):
        result, _ = bounded_score(self.scores, .3, 4, normalize_count=True)
        weights = capped_responsibilities(self.scores, .3, 4)
        shifted = self.scores.clone()
        shifted[:, 3] += 1e-6
        changed, _ = bounded_score(shifted, .3, 4, normalize_count=True)
        self.assertTrue(torch.allclose((changed - result) / 1e-6, weights[:, 3], atol=1e-6))

    def test_permutation_and_common_shift(self):
        original, _ = bounded_score(self.scores, .07, 4, normalize_count=True)
        permuted, _ = bounded_score(self.scores.flip(-1), .07, 4, normalize_count=True)
        shifted, _ = bounded_score(self.scores + 7, .07, 4, normalize_count=True)
        self.assertTrue(torch.allclose(original, permuted, atol=1e-12))
        self.assertTrue(torch.allclose(original + 7, shifted, atol=1e-12))

    def test_equal_aliases_unchanged(self):
        scores = torch.ones(3, 20, dtype=torch.float64) * .43
        result, _ = bounded_score(scores, .07, 4, normalize_count=True)
        self.assertTrue(torch.allclose(result, scores[:, 0], atol=1e-12))

    def test_one_extreme_alias_has_bounded_marginal_influence(self):
        scores = torch.zeros(1, 20, dtype=torch.float64)
        scores[0, 0] = 100
        result, _ = bounded_score(scores, .07, 4, normalize_count=True)
        scores[0, 0] += 1
        changed, _ = bounded_score(scores, .07, 4, normalize_count=True)
        self.assertAlmostEqual(float(changed - result), .2, places=9)

    def test_single_alias_and_class_order(self):
        parents = torch.tensor([0] * 20 + [1] * 20)
        values = torch.cat((self.scores, self.scores + .2), -1)
        scores, _ = bounded_classes(values, parents, 2)
        swapped, _ = bounded_classes(values, 1 - parents, 2)
        self.assertTrue(torch.equal(scores.flip(-1), swapped))
        single, _ = bounded_score(self.scores[:, :1], .07, 4, normalize_count=True)
        self.assertTrue(torch.allclose(single, self.scores[:, 0], atol=1e-12))

    def test_float32_underflow_is_finite(self):
        scores = torch.randn(100, 20) * 40
        scores[:, 0] += 200
        result, diagnostics = bounded_score(scores, 1, 4, normalize_count=False)
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertLess(diagnostics["responsibility_mass_error"], 2e-5)

    def test_invalid_input(self):
        for multiplier in (.1, float("nan")):
            with self.assertRaises(ValueError):
                capped_responsibilities(self.scores, .07, multiplier)
        with self.assertRaises(ValueError):
            capped_responsibilities(self.scores * float("nan"), .07, 4)


if __name__ == "__main__":
    unittest.main()
