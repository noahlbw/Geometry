import math
import unittest
from unittest.mock import patch

import torch

from dinotool import alias_budget_attribution as budget
from dinotool import one_sided_alias_audit as audit
from dinotool import own_peer_alias as current
from dinotool.matched_contribution_alias import contribution_margins
from dinotool.rival_alias_fast import CachedCrop


class OwnPeerAliasTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6326)
        self.members = torch.arange(12).reshape(3, 4)
        self.parents = torch.arange(3).repeat_interleave(4)
        self.canonical = self.members[:, 0]
        self.valid = torch.tensor([True, True, True, False])
        self.coordinates = torch.randn(4, 2)
        self.operator = torch.diag(self.valid.double()*.5)
        self.relation = torch.eye(4)
        self.local, self.broad, self.field = (torch.randn(4, 3) for _ in range(3))
        self.wide, self.fine = [], []
        for target in (self.wide, self.fine):
            evidence = torch.randn(4, 3, 4)
            ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
            target.append(CachedCrop(evidence, contribution_margins(evidence), ids,
                torch.tensor([[.3, .7]]*4)))

    def args(self):
        return (self.local, self.operator, self.broad, self.field, self.wide, self.fine,
            self.coordinates, self.relation, self.valid, self.members, self.canonical, self.parents)

    def test_omission_reference_matches_explicit_same_word_deletion(self):
        shifts, known, error = current.own_omission_shift(self.fine, self.members, self.valid)
        reference = torch.zeros(4, 3, 4, dtype=torch.float64)
        for crop in self.fine:
            raw = crop.evidence.double()
            full = raw.logsumexp(-1)-math.log(4)
            for a in range(4):
                keep = torch.arange(4) != a
                delta = raw[..., keep].logsumexp(-1)-math.log(3)-full
                reference[:, :, a] += (delta[crop.indices]*crop.coefficients.double()[..., None]).sum(1)
        reference[~self.valid] = 0.
        torch.testing.assert_close(shifts[:, self.members], reference, atol=1e-12, rtol=0)
        self.assertTrue(torch.equal(known, self.valid))
        self.assertLess(error, 1e-6)

    def test_excluded_word_cannot_vote_for_its_own_reference(self):
        def reference(value):
            raw = torch.tensor([[[0., value, 1.], [4., 4., 4.]]], dtype=torch.float64)
            crop = CachedCrop(raw, contribution_margins(raw), torch.tensor([[0]]), torch.ones(1, 1))
            members = torch.arange(6).reshape(2, 3)
            shift, _, _ = current.own_omission_shift([crop], members, torch.tensor([True]))
            full = raw.logsumexp(-1)-math.log(3)
            return full[0, 0]+shift[0, 1]
        torch.testing.assert_close(reference(2.), reference(1000.), atol=1e-12, rtol=0)

    def test_jointly_positive_single_word_can_be_unsupported_by_peers(self):
        raw = torch.tensor([[[0., 8., 0.], [4., 4., 4.]]], dtype=torch.float64)
        members, parents = torch.arange(6).reshape(2, 3), torch.tensor([0, 0, 0, 1, 1, 1])
        canonical, valid = torch.tensor([0, 3]), torch.tensor([True])
        crop = CachedCrop(raw, contribution_margins(raw), torch.tensor([[0]]), torch.ones(1, 1))
        full = raw.logsumexp(-1).reshape(1, 2)-math.log(3)
        broad = torch.tensor([[7., 4.]])
        shifts, known, _ = current.own_omission_shift([crop], members, valid)
        risk = current.own_peer_risk(broad, full, shifts, known, parents, canonical, valid)
        self.assertGreater(float(contribution_margins(raw)[0, 1, 1]), 0.)
        self.assertEqual(float(budget.pooled_class_risk(broad, full, valid).sum()), 0.)
        self.assertAlmostEqual(float(risk[0, 1, 1]), 4/7)
        self.assertEqual(float(risk[:, canonical].sum()), 0.)

    def test_uniform_aliases_reduce_to_pooled_class_risk(self):
        raw = torch.tensor([[[0., 0., 0.], [3., 3., 3.]]], dtype=torch.float64)
        members, parents = torch.arange(6).reshape(2, 3), torch.tensor([0, 0, 0, 1, 1, 1])
        canonical, valid = torch.tensor([0, 3]), torch.tensor([True])
        crop = CachedCrop(raw, contribution_margins(raw), torch.tensor([[0]]), torch.ones(1, 1))
        field, broad = torch.tensor([[0., 3.]]), torch.tensor([[2., 0.]])
        shifts, known, _ = current.own_omission_shift([crop], members, valid)
        actual = current.own_peer_risk(broad, field, shifts, known, parents, canonical, valid)
        expected = budget.pooled_class_risk(broad, field, valid)[:, parents].clone()
        expected[:, canonical] = 0.
        self.assertTrue(torch.equal(actual, expected))

    def test_unknown_and_invalid_coverage_never_votes(self):
        self.fine[0].coefficients.mul_(.5)
        shift, known, _ = current.own_omission_shift(self.fine, self.members, self.valid)
        self.assertFalse(bool(known.any()))
        risk = current.own_peer_risk(self.broad, self.field, shift, known,
            self.parents, self.canonical, self.valid)
        self.assertEqual(float(risk.sum()), 0.)

    def test_previous_soft_hard_and_pooled_replay_exactly(self):
        actual, _ = current.scores(*self.args())
        old, _ = audit.scores(*self.args(), methods=(audit.PRIMARY, audit.HARD, audit.OBSERVATION_MEAN))
        pooled, _ = budget.scores(*self.args(), methods=(budget.POOLED,))
        for new, previous in ((current.PREVIOUS_SOFT, audit.PRIMARY),
                              (current.PREVIOUS_HARD, audit.HARD),
                              (current.OBSERVATION_MEAN, audit.OBSERVATION_MEAN)):
            self.assertTrue(torch.equal(actual[new], old[previous]))
        self.assertTrue(torch.equal(actual[current.POOLED], pooled[budget.POOLED]))

    def test_every_singleton_matches_combined(self):
        actual, _ = current.scores(*self.args())
        for name in current.METHODS[3:]:
            single, _ = current.scores(*self.args(), methods=(name,))
            self.assertTrue(torch.equal(actual[name], single[name]), name)

    def test_primary_skips_controls_and_previous_source(self):
        with patch('dinotool.one_sided_alias_audit.native_risk', side_effect=AssertionError), \
             patch('dinotool.one_sided_alias_audit.continuous_controls', side_effect=AssertionError), \
             patch('dinotool.own_peer_alias.budget.scores', side_effect=AssertionError):
            current.scores(*self.args(), methods=(current.PRIMARY,))

    def test_pooled_control_skips_own_peer_reference(self):
        with patch('dinotool.own_peer_alias.own_omission_shift', side_effect=AssertionError):
            current.scores(*self.args(), methods=(current.POOLED,))

    def test_norm_matching_and_class_gauge_are_preserved(self):
        actual, diagnostics = current.scores(*self.args())
        positive = actual[current.OBSERVATION_MEAN]
        for name in (current.MATCHED_POOLED, current.OLD_NAMES[audit.MATCHED_CLASS_MEAN],
                     *(current.OLD_NAMES[m] for m in audit.MATCHED_SHUFFLES)):
            delta = actual[name]-positive
            torch.testing.assert_close(delta[self.valid].norm(),
                (actual[current.PRIMARY]-positive)[self.valid].norm(), atol=1e-12, rtol=0)
            torch.testing.assert_close(delta.sum(-1), torch.zeros(4, dtype=torch.float64), atol=1e-12, rtol=0)
            self.assertEqual(float(delta[~self.valid].abs().max()), 0.)
        self.assertEqual(diagnostics['canonical_risk_max'], 0.)
        self.assertEqual(diagnostics['peer_unknown_risk_max'], 0.)
        self.assertLessEqual(diagnostics['positive_directed_delta_max'], 1e-12)
        self.assertEqual(diagnostics['pooled_matched_unmatchable'], 0.)

    def test_no_observation_methods_skip_peer_work(self):
        with patch('dinotool.own_peer_alias.own_omission_shift', side_effect=AssertionError):
            current.scores(*self.args(), methods=(current.OBSERVATION_MEAN,))

    def test_invalid_methods_fail(self):
        with self.assertRaises(ValueError):
            current.scores(*self.args(), methods=('Unfrozen',))


if __name__ == '__main__':
    unittest.main()
