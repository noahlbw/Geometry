import unittest

import torch

from dinotool.matched_contribution_alias import contribution_margins
from dinotool.pool_isolated_alias import (CLASS_MEAN, METHODS, OBSERVATION_MEAN,
    PRIMARY, PROFILE_POOL, WIDE_HARD, IsolatedCrop, isolated_scores,
    reference_margins, sampled_reference)
from dinotool.reciprocal_alias_admission import (ONE_SIDED_SOFT as OLD_SOFT,
    OBSERVATION_MEAN as OLD_MEAN, ONE_SIDED_HARD as OLD_HARD, reciprocal_scores)


class PoolIsolationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1606)
        self.members = torch.arange(12).reshape(3, 4)
        self.canonical = self.members[:, 0]
        self.parents = torch.arange(3).repeat_interleave(4)
        self.valid = torch.tensor([True, True, True, False])
        self.coords = torch.randn(4, 2)
        self.operator = torch.diag(self.valid.double()*.5)
        self.relation = torch.eye(4)
        self.local, self.broad, self.field = (torch.randn(4, 3) for _ in range(3))
        ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
        weights = torch.tensor([[.3, .7]]*4)
        self.wide, self.fine = [], []
        for target in (self.wide, self.fine):
            raw = torch.randn(4, 3, 4)
            e = raw*(4*torch.randn(3, 4).softmax(-1))
            target.append(IsolatedCrop(e, contribution_margins(e), ids, weights, raw))

    def scores(self, methods=METHODS[3:]):
        return isolated_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coords, self.relation, self.valid,
            self.members, self.canonical, self.parents, methods=methods)

    def test_raw_canonical_reference_ignores_other_rival_aliases(self):
        raw = self.wide[0].raw
        original = reference_margins(raw, self.members, self.canonical, 'canonical')
        changed = raw.clone()
        changed[:, 1, 1:] += 100
        actual = reference_margins(changed, self.members, self.canonical, 'canonical')
        self.assertTrue(torch.equal(actual[:, 1, 1], original[:, 1, 1]))
        old_pool = reference_margins(raw, self.members, self.canonical, 'pool')
        new_pool = reference_margins(changed, self.members, self.canonical, 'pool')
        self.assertGreater(float((old_pool[:, 1, 1]-new_pool[:, 1, 1]).abs().max()), 90)

    def test_raw_canonical_ignores_vocabulary_salience(self):
        raw = self.wide[0].raw
        old = reference_margins(raw, self.members, self.canonical, 'canonical')
        profile = raw*(4*torch.tensor([[0., 0., 0., 0.], [100., 0., 0., 0.], [0., 0., 0., 0.]]).softmax(-1))
        new = reference_margins(raw, self.members, self.canonical, 'canonical')
        self.assertTrue(torch.equal(old, new))
        self.assertGreater(float((reference_margins(profile, self.members, self.canonical, 'canonical')-old).abs().max()), .1)

    def test_pool_margins_exactly_replay_existing_source(self):
        actual = reference_margins(self.wide[0].evidence, self.members, self.canonical, 'pool')
        self.assertTrue(torch.equal(actual, contribution_margins(self.wide[0].evidence)))

    def test_noncontiguous_alias_ids_and_canonical_positions(self):
        members = self.members[:, torch.tensor([2, 0, 3, 1])].T.reshape(3, 4)
        canonical = members[:, 2]
        raw = self.wide[0].raw
        actual = reference_margins(raw, members, canonical, 'canonical')
        for c in range(3):
            for k in range(4):
                for r in range(3):
                    torch.testing.assert_close(actual[:, members[c, k], r], raw[:, c, k]-raw[:, r, 2], atol=0, rtol=0)

    def test_raw_reference_uses_original_multitoken_crop_stencil(self):
        e = self.wide[0]
        margins = reference_margins(e.raw, self.members, self.canonical, 'canonical')
        actual = sampled_reference(self.wide, self.members, self.canonical, self.valid, raw=True, reference='canonical')
        expected = (margins[e.indices]*e.coefficients[..., None, None]).sum(1)
        expected[~self.valid] = 0
        self.assertTrue(torch.equal(actual, expected))

    def test_strong_soft_and_hard_and_positive_endpoints_replay_bitwise(self):
        actual, _ = self.scores()
        previous, _ = reciprocal_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coords, self.relation, self.valid,
            self.members, self.canonical, self.parents, methods=(OLD_SOFT, OLD_HARD, OLD_MEAN))
        for current, old in ((PROFILE_POOL, OLD_SOFT), (WIDE_HARD, OLD_HARD), (OBSERVATION_MEAN, OLD_MEAN)):
            self.assertTrue(torch.equal(actual[current], previous[old]))

    def test_primary_does_not_use_geometry_as_semantic_truth(self):
        actual, _ = self.scores((PRIMARY,))
        shuffled, _ = isolated_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coords, self.relation.flip(0), self.valid,
            self.members, self.canonical, self.parents, methods=(PRIMARY,))
        self.assertTrue(torch.equal(actual[PRIMARY], shuffled[PRIMARY]))

    def test_canonical_self_and_invalid_risks_are_zero(self):
        from dinotool.native_query_alias import native_risk

        wm = sampled_reference(self.wide, self.members, self.canonical, self.valid, raw=True, reference='canonical')
        fm = sampled_reference(self.fine, self.members, self.canonical, self.valid, raw=True, reference='canonical')
        risk = native_risk(wm, fm, fm, self.parents, self.canonical, self.valid)
        self.assertEqual(float(risk[:, self.canonical].abs().max()), 0)
        self.assertEqual(float(risk[~self.valid].abs().max()), 0)
        self.assertEqual(float(risk.gather(-1, self.parents[None, :, None].expand(4, -1, 1)).abs().max()), 0)

    def test_same_alias_risk_can_differ_between_rivals(self):
        from dinotool.native_query_alias import native_risk

        wm = torch.ones(4, 12, 3)
        fm = torch.ones_like(wm)
        fm[0, 1, 1] = -1
        risk = native_risk(wm, fm, fm, self.parents, self.canonical, self.valid)
        self.assertGreater(float(risk[0, 1, 1]), 0)
        self.assertEqual(float(risk[0, 1, 2]), 0)

    def test_raw_agreement_is_exact_same_information_identity(self):
        self.fine[0].raw.copy_(self.wide[0].raw)
        scores, _ = self.scores()
        self.assertTrue(torch.equal(scores[PRIMARY], scores[OBSERVATION_MEAN]))

    def test_singleton_predictions_match_simultaneous_controls(self):
        combined, _ = self.scores()
        for name in METHODS[3:]:
            single, _ = self.scores((name,))
            self.assertTrue(torch.equal(single[name], combined[name]), name)

    def test_class_mass_and_class_gauge_invariants(self):
        scores, diag = self.scores()
        self.assertLessEqual(diag['class_budget_max_error'], 1e-10)
        torch.testing.assert_close((scores[PRIMARY]-scores[OBSERVATION_MEAN]).sum(-1),
            torch.zeros(4, dtype=torch.float64), atol=1e-12, rtol=0)
        self.assertTrue(torch.equal(scores[PRIMARY][~self.valid], scores[OBSERVATION_MEAN][~self.valid]))

    def test_invalid_method_or_canonical_membership_rejected(self):
        with self.assertRaises(ValueError):
            self.scores(('Unfrozen',))
        with self.assertRaises(ValueError):
            reference_margins(self.wide[0].raw, self.members, torch.tensor([99, 4, 8]), 'canonical')


if __name__ == '__main__':
    unittest.main()
