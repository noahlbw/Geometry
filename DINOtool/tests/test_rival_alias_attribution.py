import unittest

import torch

from dinotool.rival_alias_attribution import (METHODS, BASE, CLASS, ALIAS, SPATIAL, RIVAL,
    attribution_scores, alias_null, spatial_null, rival_degree_null)
from dinotool.rival_competition_admission import retained_competition_scores
from dinotool.rival_alias_fast import cache_observations, directed_cached
from test_rival_alias_fast import RivalAliasFastTest


class RivalAliasAttributionTest(unittest.TestCase):
    def fixture(self): return RivalAliasFastTest().fixture()

    def test_original_five_scores_and_risk_are_bitwise_unchanged(self):
        args = self.fixture()
        old, risk, stats = retained_competition_scores(*args, methods=BASE)
        new, new_risk, _ = attribution_scores(*args)
        self.assertTrue(torch.equal(risk, new_risk))
        for m in BASE: self.assertTrue(torch.equal(old[m], new[m]), m)

    def test_alias_null_keeps_per_pair_counts_spectrum_and_canonical(self):
        args = self.fixture()
        _, risk, _ = retained_competition_scores(*args, methods=BASE)
        members, canonical = args[10:12]
        changed = alias_null(risk, members, canonical, 9)
        self.assertTrue(torch.equal((risk[:, members] > 0).sum(2), (changed[:, members] > 0).sum(2)))
        self.assertTrue(torch.equal(risk[:, members].sort(2).values, changed[:, members].sort(2).values))
        self.assertEqual(float(changed[:, canonical].abs().max()), 0.)

    def test_spatial_null_only_moves_valid_queries(self):
        risk = torch.arange(18).reshape(3, 3, 2).double()
        valid = torch.tensor([True, False, True])
        changed = spatial_null(risk, valid, 6)
        self.assertTrue(torch.equal(changed[1], risk[1]))
        self.assertTrue(torch.equal(changed[valid].sort(0).values, risk[valid].sort(0).values))

    def test_rival_switch_preserves_both_degrees_and_protection(self):
        members = torch.arange(12).reshape(3, 4)
        canonical = members[:, 0]
        risk = torch.zeros(8, 12, 3)
        for c in range(3):
            rivals = [d for d in range(3) if d != c]
            risk[:, members[c, 1], rivals[0]] = .3
            risk[:, members[c, 2], rivals[1]] = .7
        changed, stats = rival_degree_null(risk, members, canonical, 4, attempts=64)
        self.assertGreater(stats['accepted_switches'], 0)
        self.assertGreater(stats['decision_changed_fraction'], 0)
        self.assertFalse(torch.equal(risk > 0, changed > 0))
        self.assertEqual(stats['alias_degree_max_error'], 0)
        self.assertEqual(stats['rival_degree_max_error'], 0)
        old, new = risk[:, members] > 0, changed[:, members] > 0
        self.assertTrue(torch.equal(old.sum(2), new.sum(2)))
        self.assertTrue(torch.equal(old.sum(3), new.sum(3)))
        self.assertEqual(float(changed[:, canonical].abs().max()), 0.)
        for c in range(3): self.assertEqual(float(changed[:, members[c], c].abs().max()), 0.)
        repeated, _ = rival_degree_null(risk, members, canonical, 4, attempts=64)
        self.assertTrue(torch.equal(changed, repeated))

    def test_binary_risk_replay_is_identical_for_hard_writer(self):
        args = self.fixture()
        _, risk, _ = retained_competition_scores(*args, methods=BASE)
        observed = cache_observations(args[3], args[4], args[7], args[13], args[10])
        old = directed_cached(observed, args[10], risk, args[9])
        new = directed_cached(observed, args[10], (risk > 0).to(risk.dtype), args[9])
        self.assertTrue(torch.equal(old, new))

    def test_class_strength_control_matches_candidate_norm(self):
        values, _, stats = attribution_scores(*self.fixture(), methods=(*BASE, *CLASS))
        baseline = values['NoAdmission_Exact']
        self.assertAlmostEqual(float((values[CLASS[1]]-baseline).norm()), float((values[BASE[3]]-baseline).norm()), places=10)
        self.assertLess(abs(stats['class_strength_norm_error']), 1e-10)

    def test_independent_controls_and_all_zero_risk_remain_finite(self):
        args = list(self.fixture())
        for crop in (*args[3], *args[5]):
            crop.alias_logits.zero_()
            crop.salience.zero_()
        for m in METHODS:
            values, risk, _ = attribution_scores(*args, methods=(m,))
            self.assertEqual(set(values), {m})
            self.assertTrue(torch.isfinite(values[m]).all())
            self.assertEqual(float(risk.abs().max()), 0.)
            if m not in ('Geometry', CLASS[0]):
                expected = args[0].double()+args[1].double()@(args[2].double()-args[0].double())
                self.assertTrue(torch.equal(values[m], expected), m)


if __name__ == '__main__': unittest.main()
