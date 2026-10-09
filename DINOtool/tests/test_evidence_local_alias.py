import unittest

import torch

from dinotool.evidence_local_alias import (CLASS_MEAN, DIRECT, GLOBAL_HARD,
    GLOBAL_SOFT, METHODS, OBSERVATION_MEAN, PRIMARY, QUERY_ONLY, SHUFFLES,
    SHUFFLED_WRITE, local_scores, match_norm, pair_support, supported_write)
from dinotool.native_query_alias import native_risk
from dinotool.reciprocal_alias_admission import (OBSERVATION_MEAN as OLD_MEAN,
    ONE_SIDED_HARD as OLD_HARD, ONE_SIDED_SOFT as OLD_SOFT,
    continuous_controls, reciprocal_scores)
from dinotool.rival_alias_fast import CachedCrop


class EvidenceLocalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2606)
        self.members = torch.arange(12).reshape(3, 4)
        self.canonical = self.members[:, 0]
        self.parents = torch.arange(3).repeat_interleave(4)
        self.valid = torch.tensor([True, True, True, False])
        self.coords = torch.randn(4, 2)
        self.relation = torch.rand(4, 4)
        weights = self.relation.double().masked_fill(~self.valid[None], 0.)
        weights /= weights.sum(-1, keepdim=True)
        weights[~self.valid] = 0
        gram = weights.T@weights
        self.operator = torch.linalg.solve(torch.eye(4)+gram, gram)
        self.operator[~self.valid] = 0
        self.operator[:, ~self.valid] = 0
        self.local, self.broad, self.field = (torch.randn(4, 3) for _ in range(3))
        self.wm = torch.ones(4, 12, 3)
        self.fm = self.wm.clone()
        self.fm[0, 1, 1] = -2
        self.fm[1, 9, 1] = -1
        ids, coefficients = torch.arange(4)[:, None], torch.ones(4, 1)
        self.wide = [CachedCrop(torch.randn(4, 3, 4), self.wm, ids, coefficients)]
        self.fine = [CachedCrop(torch.randn(4, 3, 4), self.fm, ids, coefficients)]

    def scores(self, methods=METHODS[3:]):
        return local_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coords, self.relation, self.valid,
            self.members, self.canonical, self.parents, methods=methods)

    def risk(self):
        return native_risk(self.wm, self.fm, self.fm, self.parents,
            self.canonical, self.valid)

    def test_support_is_pair_specific_symmetric_and_padding_free(self):
        support = pair_support(self.risk(), self.members, self.valid)
        self.assertTrue(torch.equal(support, support.transpose(-1, -2)))
        self.assertTrue(support[0, 0, 1])
        self.assertFalse(support[0, 0, 2])
        self.assertFalse(bool(support[~self.valid].any()))
        self.assertFalse(bool(support.diagonal(dim1=-2, dim2=-1).any()))

    def test_support_tracks_actual_noncontiguous_member_ids(self):
        risk = self.risk()
        changed_members = self.members[:, torch.tensor([2, 0, 3, 1])]
        self.assertTrue(torch.equal(pair_support(risk, self.members, self.valid),
            pair_support(risk, changed_members, self.valid)))

    def test_canonical_self_and_padding_risks_are_zero(self):
        risk = self.risk()
        self.assertEqual(float(risk[:, self.canonical].abs().max()), 0)
        self.assertEqual(float(risk[~self.valid].abs().max()), 0)
        self.assertEqual(float(risk.gather(-1, self.parents[None, :, None].expand(4, -1, 1)).abs().max()), 0)

    def test_word_controls_preserve_risk_mass_and_pair_support(self):
        risk = self.risk()
        controls, error = continuous_controls(risk, self.members, self.canonical)
        self.assertLess(error, 1e-12)
        for value in controls.values():
            self.assertTrue(torch.equal(pair_support(value, self.members, self.valid),
                pair_support(risk, self.members, self.valid)))

    def test_transport_stops_at_unsupported_class_pair(self):
        margins = torch.zeros(4, 3, 3, dtype=torch.float64)
        margins[0, 0, 1], margins[0, 1, 0] = 2., -2.
        support = margins != 0
        output, stats = supported_write(margins, self.operator, support, self.valid)
        self.assertEqual(float(output[1:].abs().max()), 0)
        self.assertEqual(float(output[:, 2].abs().max()), 0)
        self.assertGreater(float((self.operator@margins.mean(-1))[1].abs().max()), 0)
        self.assertEqual(stats['inactive_class_displacement_max'], 0)

    def test_class_gauge_and_whole_field_norm_match(self):
        actual, stats = self.scores()
        reference = actual[GLOBAL_SOFT]-actual[OBSERVATION_MEAN]
        for name in (PRIMARY, QUERY_ONLY, DIRECT, CLASS_MEAN, *SHUFFLES, SHUFFLED_WRITE):
            correction = actual[name]-actual[OBSERVATION_MEAN]
            torch.testing.assert_close(correction.norm(), reference.norm(), atol=1e-12, rtol=0)
            torch.testing.assert_close(correction.sum(-1), torch.zeros(4, dtype=torch.float64), atol=1e-12, rtol=0)
        self.assertLess(stats['write_norm_relative_error'], 1e-12)

    def test_identity_geometry_recovers_direct_source_action(self):
        margins = torch.randn(4, 3, 3, dtype=torch.float64)
        margins -= margins.transpose(-1, -2).clone()
        margins[~self.valid] = 0
        support = (margins != 0)
        output, _ = supported_write(margins, torch.eye(4), support, self.valid)
        torch.testing.assert_close(output, margins.mean(-1), atol=1e-12, rtol=0)

    def test_full_pair_support_recovers_unrestricted_geometry(self):
        margins = torch.randn(4, 3, 3, dtype=torch.float64)
        margins -= margins.transpose(-1, -2).clone()
        margins[~self.valid] = 0
        support = self.valid[:, None, None] & (~torch.eye(3, dtype=torch.bool))[None]
        output, _ = supported_write(margins, self.operator, support, self.valid)
        torch.testing.assert_close(output, self.operator@margins.mean(-1), atol=1e-12, rtol=0)

    def test_exact_previous_positive_soft_and_hard_replay(self):
        actual, _ = self.scores()
        previous, _ = reciprocal_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coords, self.relation, self.valid, self.members,
            self.canonical, self.parents, methods=(OLD_MEAN, OLD_SOFT, OLD_HARD))
        for new, old in ((OBSERVATION_MEAN, OLD_MEAN), (GLOBAL_SOFT, OLD_SOFT), (GLOBAL_HARD, OLD_HARD)):
            self.assertTrue(torch.equal(actual[new], previous[old]))

    def test_zero_risk_is_exact_same_information_identity(self):
        self.fm.copy_(self.wm)
        actual, _ = self.scores()
        self.assertTrue(torch.equal(actual[PRIMARY], actual[OBSERVATION_MEAN]))

    def test_geometry_relation_is_not_a_semantic_truth_gate(self):
        original, _ = self.scores((PRIMARY,))
        self.relation = self.relation.flip(0)
        changed, _ = self.scores((PRIMARY,))
        self.assertTrue(torch.equal(original[PRIMARY], changed[PRIMARY]))

    def test_singleton_matches_combined_controls(self):
        combined, _ = self.scores()
        for name in METHODS[3:]:
            single, _ = self.scores((name,))
            self.assertTrue(torch.equal(single[name], combined[name]), name)

    def test_primary_changes_no_inactive_class_logit(self):
        actual, _ = self.scores()
        inactive = ~pair_support(self.risk(), self.members, self.valid).any(-1)
        self.assertTrue(torch.equal(actual[PRIMARY][inactive], actual[OBSERVATION_MEAN][inactive]))

    def test_zero_norm_and_missing_direction_are_explicit(self):
        zero = torch.zeros(4, 3, dtype=torch.float64)
        actual, stats = match_norm(torch.ones_like(zero), zero)
        self.assertTrue(torch.equal(actual, zero))
        self.assertEqual(stats['write_scale'], 0)
        with self.assertRaises(RuntimeError):
            match_norm(zero, torch.ones_like(zero))

    def test_invalid_support_or_undeclared_method_rejected(self):
        margins = torch.zeros(4, 3, 3, dtype=torch.float64)
        support = torch.zeros_like(margins, dtype=torch.bool)
        support[0, 0, 1] = True
        with self.assertRaises(ValueError):
            supported_write(margins, self.operator, support, self.valid)
        with self.assertRaises(ValueError):
            self.scores(('Unfrozen',))


if __name__ == '__main__':
    unittest.main()
