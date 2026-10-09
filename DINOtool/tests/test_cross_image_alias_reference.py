import unittest
from unittest.mock import patch

import torch

from dinotool.cross_image_alias_reference import (CLASS_MEAN, IN_IMAGE, MATCHED_OLD, METHODS,
    OBSERVATION_MEAN, OLD_HARD, OLD_SOFT, PRIMARY, REFERENCE_NULL, SHUFFLES, RankReference,
    RawCachedCrop, canonical_reference, joint_reference_risk, scores)
from dinotool.matched_contribution_alias import contribution_margins
from dinotool.one_sided_alias_audit import HARD as PREVIOUS_HARD, PRIMARY as PREVIOUS_SOFT
from dinotool.one_sided_alias_audit import OBSERVATION_MEAN as PREVIOUS_MEAN, scores as previous_scores


class CrossImageReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6106)
        self.members = torch.arange(12).reshape(3, 4)
        self.parents = torch.arange(3).repeat_interleave(4)
        self.canonical = self.members[:, 0]
        self.valid = torch.tensor([True, True, True, False])
        self.coordinates = torch.randn(4, 2)
        self.operator = torch.diag(self.valid.double()*.5)
        self.relation = torch.eye(4)
        self.local, self.broad, self.field = (torch.randn(4, 3) for _ in range(3))
        raw = torch.randn(80, 12)*3
        pseudo = canonical_reference(raw, self.canonical)
        self.reference = RankReference.build(raw, pseudo)
        self.shuffled = RankReference.build(raw, pseudo.flip(0))
        self.wide, self.fine = [], []
        for target in (self.wide, self.fine):
            evidence = torch.randn(4, 3, 4)
            ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
            target.append(RawCachedCrop(evidence, contribution_margins(evidence), ids,
                torch.tensor([[.3, .7]]*4), torch.randn(4, 12)))

    def score(self, methods=METHODS[3:]):
        return scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coordinates, self.relation, self.valid,
            self.members, self.canonical, self.parents, self.reference, self.shuffled, methods=methods)

    def test_rank_lookup_matches_bruteforce_and_expands_ties(self):
        raw = torch.tensor([[0., 1.], [0., 2.], [1., 3.], [2., 3.], [2., 4.], [3., 4.]])
        q = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.], [1., 0.], [0., 1.]])
        bank = RankReference.build(raw, q)
        actual, _ = bank.lookup(raw)
        for a in range(2):
            values = raw[:, a].sort().values
            for n, query in enumerate(raw[:, a]):
                first = int((values < query).sum())
                last = int((values <= query).sum())
                middle = (first+last-1)//2
                low = max(0, min(middle-3//2, 6-3))
                selected = (raw[:, a] >= values[low]) & (raw[:, a] <= values[low+2])
                torch.testing.assert_close(actual[n, a], q[selected].double().sum(0)/q.sum(0))

    def test_positive_affine_transform_preserves_rank_and_unknown(self):
        raw = torch.randn(100, 12)
        q = torch.softmax(torch.randn(100, 3), -1)
        query = torch.randn(9, 12)
        a, known = RankReference.build(raw, q).lookup(query)
        b, other = RankReference.build(raw*7+4, q).lookup(query*7+4)
        torch.testing.assert_close(a, b)
        self.assertTrue(torch.equal(known, other))

    def test_no_extrapolation_and_absent_class_are_unknown(self):
        raw = torch.arange(6.).reshape(3, 2)
        q = torch.tensor([[1., 0.], [1., 0.], [1., 0.]])
        _, known = RankReference.build(raw, q).lookup(torch.tensor([[-1., 7.], [2., 3.]]))
        self.assertFalse(bool(known[0].any()))
        self.assertFalse(bool(known[..., 1].any()))

    def test_all_ties_produce_equal_class_density(self):
        raw = torch.ones(12, 12)
        q = torch.arange(1., 37.).reshape(12, 3)
        density, known = RankReference.build(raw, q).lookup(torch.ones(2, 12))
        torch.testing.assert_close(density, torch.ones_like(density))
        self.assertTrue(bool(known.all()))

    def test_canonical_reference_is_soft_and_uniform_is_uninformative(self):
        q = canonical_reference(torch.zeros(8, 12), self.canonical)
        self.assertEqual(float(q.abs().max()), 0.)

    def test_shared_positive_wrong_response_can_receive_risk(self):
        density = torch.ones(4, 12, 3, dtype=torch.float64)
        density[:, 1, 0] = .1
        wide = fine = torch.ones_like(density)
        risk = joint_reference_risk(density, torch.ones_like(density, dtype=torch.bool),
            wide, fine, self.parents, self.canonical, self.valid)
        self.assertGreater(float(risk[0, 1, 1]), .8)
        self.assertEqual(float(risk[:, self.canonical].abs().max()), 0.)
        self.assertEqual(float(risk[~self.valid].abs().max()), 0.)
        self.assertEqual(float(risk.gather(-1, self.parents[None, :, None].expand(4, -1, 1)).abs().max()), 0.)

    def test_fine_negative_entries_receive_no_new_risk(self):
        shape = (4, 12, 3)
        risk = joint_reference_risk(torch.ones(shape), torch.ones(shape, dtype=torch.bool),
            torch.ones(shape), -torch.ones(shape), self.parents, self.canonical, self.valid)
        self.assertEqual(float(risk.abs().max()), 0.)

    def test_original_endpoints_are_bitwise_exact(self):
        actual, _ = self.score()
        old, _ = previous_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coordinates, self.relation, self.valid,
            self.members, self.canonical, self.parents, methods=(PREVIOUS_SOFT, PREVIOUS_HARD, PREVIOUS_MEAN))
        for new, previous in ((OLD_SOFT, PREVIOUS_SOFT), (OLD_HARD, PREVIOUS_HARD), (OBSERVATION_MEAN, PREVIOUS_MEAN)):
            self.assertTrue(torch.equal(actual[new], old[previous]), new)

    def test_primary_singleton_matches_and_does_not_compute_controls(self):
        combined, _ = self.score()
        with patch('dinotool.cross_image_alias_reference.continuous_controls', side_effect=AssertionError), \
             patch('dinotool.cross_image_alias_reference.rank_densities', side_effect=AssertionError):
            single, _ = self.score((PRIMARY,))
        self.assertTrue(torch.equal(combined[PRIMARY], single[PRIMARY]))

    def test_all_control_singletons_and_matched_norms(self):
        combined, diagnostics = self.score()
        original = combined[OBSERVATION_MEAN]
        norm = (combined[PRIMARY]-original)[self.valid].norm()
        for name in METHODS[3:]:
            single, _ = self.score((name,))
            self.assertTrue(torch.equal(single[name], combined[name]), name)
        for name in (CLASS_MEAN, MATCHED_OLD, *SHUFFLES, REFERENCE_NULL, IN_IMAGE):
            torch.testing.assert_close((combined[name]-original)[self.valid].norm(), norm, atol=1e-12, rtol=0)
        self.assertEqual(diagnostics['matched_unmatchable'], 0)

    def test_unknown_reference_restores_exact_previous_soft(self):
        self.reference = RankReference.build(torch.ones(10, 12)*1000, torch.ones(10, 3))
        values, _ = self.score((PRIMARY, OLD_SOFT))
        self.assertTrue(torch.equal(values[PRIMARY], values[OLD_SOFT]))

    def test_invalid_methods_and_reference_rows_rejected(self):
        with self.assertRaises(ValueError):
            self.score(('NotFrozen',))
        with self.assertRaises(ValueError):
            RankReference.build(torch.ones(1, 12), torch.ones(1, 3))


if __name__ == '__main__':
    unittest.main()
