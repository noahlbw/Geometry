import math
import unittest
from dataclasses import replace

import torch

from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.soft_competitive_alias import (SoftAliasConfig, SoftCompetitiveAliases,
    apply_pair_margin, leave_one_out_scores, leave_out_witnesses)
from test_pair_conditional_alias import example_bank


class SoftCompetitiveAliasTest(unittest.TestCase):
    def setUp(self):
        self.bank = example_bank()
        self.config = SoftAliasConfig(relation_neighbors=16)
        self.reader = SoftCompetitiveAliases(self.bank, self.config)
        a = [.8, .65, .3, .2, .1, .1, .1, .1, -1., -1., -1., -1.]
        b = [.1, .65, .1, .1, .8, .7, .3, .2, -1., -1., -1., -1.]
        self.aliases = torch.tensor([[a] * 8 + [b] * 8])
        self.relation = torch.full((1, 16, 16), 1 / 16)
        self.valid = torch.ones(1, 16, dtype=torch.bool)
        self.raw = alias_class_scores(self.aliases, self.bank.parent_indices, 3)
        self.broad = self.raw / .07
        self.without = leave_one_out_scores(self.aliases, self.bank.parent_indices, 3) / .07

    def read(self, **changes):
        args = {"aliases": self.aliases, "relation": self.relation, "broad": self.broad,
                "broad_without_alias": self.without, "valid": self.valid}
        args.update(changes)
        return self.reader.read(**args)

    def test_leave_one_out_matches_fresh_aggregation(self):
        torch.manual_seed(5)
        aliases = torch.randn(2, 8, 12)
        scores = leave_one_out_scores(aliases, self.bank.parent_indices, 3)
        for a, parent in enumerate(self.bank.parent_indices):
            kept = (self.bank.parent_indices == parent) & (torch.arange(12) != a)
            expected = .07 * (torch.logsumexp(aliases[..., kept] / .07, -1) - math.log(3))
            self.assertTrue(torch.allclose(scores[..., a], expected))

    def test_weak_alias_deletion_increases_parent_score(self):
        aliases = self.aliases.clone()
        scores = leave_one_out_scores(aliases, self.bank.parent_indices, 3)
        self.assertGreater(float(scores[0, 0, 3]), float(self.raw[0, 0, 0]))

    def test_witnesses_do_not_use_removed_alias(self):
        scores = torch.tensor([[[10., 3., 1.]]])
        without = torch.tensor([[[0., 2., 0.]]])
        winners, quality = leave_out_witnesses(scores, without, torch.arange(3))
        self.assertEqual(winners.tolist(), [[[1, 0, 0]]])
        self.assertTrue(bool((quality > 0).all()))

    def test_pair_update_preserves_other_probabilities_and_partition(self):
        scores = torch.tensor([[[1., 2., 3.], [5., -2., 1.]]])
        pairs = torch.tensor([[[0, 1], [0, 2]]])
        changed = apply_pair_margin(scores, pairs, torch.tensor([[.4, -.3]]), torch.ones(1, 2, dtype=torch.bool))
        self.assertTrue(torch.allclose(changed.logsumexp(-1), scores.logsumexp(-1), atol=1e-6))
        self.assertAlmostEqual(float(changed.softmax(-1)[0, 0, 2]), float(scores.softmax(-1)[0, 0, 2]), places=6)
        self.assertEqual(float(changed[0, 1, 1]), float(scores[0, 1, 1]))

    def test_zero_update_exact_identity(self):
        pairs = self.broad.topk(2, -1).indices
        updated = apply_pair_margin(self.broad, pairs, torch.zeros(1, 16), self.valid)
        self.assertTrue(torch.equal(updated, self.broad))

    def test_no_reference_class_abstains_exactly(self):
        wide = self.broad.clone()
        wide[..., 2] = 100
        variants, diagnostics = self.read(broad=wide)
        self.assertEqual(diagnostics["reference_supported_alias_fraction"], 0)
        self.assertTrue(torch.equal(variants["soft"]["local"], self.broad))
        self.assertTrue(torch.equal(variants["soft"]["observation"], wide))

    def test_alias_weights_remain_positive_normalized_and_shuffled_matched(self):
        variants, diagnostics = self.read()
        weights = variants["soft"]["weights"]
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1))))
        self.assertGreaterEqual(float(weights.min()), .5 / 4)
        self.assertEqual(diagnostics["shuffled_weight_spectrum_error"], 0)
        self.assertGreater(diagnostics["soft_active_patch_fraction"], 0)
        self.assertLessEqual(diagnostics["maximum_absolute_soft_margin_correction"], .5)

    def test_visual_confounder_receives_less_weight_than_discriminative_alias(self):
        variants, _ = self.read()
        weights = variants["soft"]["weights"][0, 0, 0]
        self.assertGreater(float(weights[0]), float(weights[1]))

    def test_invalid_patches_do_not_change(self):
        valid = self.valid.clone()
        valid[:, :3] = False
        variants, _ = self.read(valid=valid)
        for row in variants.values():
            self.assertTrue(torch.equal(row["local"][~valid], self.broad[~valid]))
            self.assertTrue(torch.equal(row["observation"][~valid], self.broad[~valid]))

    def test_uniform_prior_one_is_exact_identity(self):
        self.reader = SoftCompetitiveAliases(self.bank, replace(self.config, uniform_prior=1))
        variants, _ = self.read()
        self.assertTrue(all(torch.equal(row["local"], self.broad) for row in variants.values()))

    def test_common_logit_shift_does_not_change_allocation_or_margin(self):
        left, _ = self.read()
        right, _ = self.read(aliases=self.aliases + .3, broad=self.broad + 2,
                             broad_without_alias=self.without + 2)
        self.assertTrue(torch.allclose(left["soft"]["weights"], right["soft"]["weights"], atol=1e-5))
        self.assertTrue(torch.allclose(left["soft"]["correction"], right["soft"]["correction"], atol=1e-5))

    def test_nonfinite_scores_rejected(self):
        with self.assertRaises(ValueError):
            self.read(aliases=torch.full_like(self.aliases, float("nan")))


if __name__ == "__main__":
    unittest.main()
