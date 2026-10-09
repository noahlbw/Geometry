import math
import unittest

import torch

from dinotool.rival_alias_fast import CachedCrop, cache_observations, directed_cached
from dinotool.rival_competition_admission import retained_competition_scores
from dinotool.rival_neutral_replacement import (BASE, METHODS, BENCHMARK,
    replacement_directed, neutral_replacement_scores)
from test_rival_alias_fast import RivalAliasFastTest


class RivalNeutralReplacementTest(unittest.TestCase):
    def fixture(self):
        crop = CachedCrop(torch.tensor([[[0., 4.], [1., 3.]]]), None, torch.tensor([[0]]), torch.ones(1, 1))
        members = torch.arange(4).reshape(2, 2)
        risk = torch.zeros(1, 4, 2)
        risk[0, 1, 1] = .3
        return crop, members, risk, torch.tensor([True])

    def test_neutral_caps_only_eligible_alias_against_original_rival_mean(self):
        crop, members, risk, valid = self.fixture()
        got = replacement_directed([crop], members, risk, valid)
        values = crop.evidence.double()
        cap = values[0, 1].logsumexp(0)-math.log(2)
        changed = torch.stack((values[0, 0, 0], cap))
        expected = changed.logsumexp(0)-values[0, 0].logsumexp(0)
        self.assertAlmostEqual(float(got[0, 0, 1]), float(expected), places=12)
        self.assertEqual(float(got[0, 1].abs().max()), 0.)
        self.assertEqual(float(got[0, 0, 0]), 0.)

    def test_fine_cap_is_no_larger_than_neutral_or_original(self):
        crop, members, risk, valid = self.fixture()
        fine = torch.zeros_like(risk)
        fine[0, 1, 1] = -1.
        neutral = replacement_directed([crop], members, risk, valid)
        changed = replacement_directed([crop], members, risk, valid, fine, 'fine')
        self.assertTrue(bool((changed <= neutral).all()))
        self.assertLess(float(changed[0, 0, 1]), float(neutral[0, 0, 1]))

    def test_already_lower_alias_is_not_boosted_to_the_rival_cap(self):
        crop, members, risk, valid = self.fixture()
        crop.evidence[0, 0, 1] = -2.
        got = replacement_directed([crop], members, risk, valid)
        self.assertEqual(float(got.abs().max()), 0.)

    def test_hard_fixed_mass_plus_count_normalizer_recovers_original_writer(self):
        args = RivalAliasFastTest().fixture()
        risk = retained_competition_scores(*args, methods=BASE)[1]
        for beta in (.3, 1., 2.):
            obs = cache_observations(args[3], args[4], args[7], args[-1], args[10], beta)
            for chunk in (7, 128):
                fixed = replacement_directed(obs, args[10], risk, args[9], kind='fixed_mass', beta=beta, chunk=chunk)
                remaining = (risk[:, args[10]] == 0).sum(2).double()
                coverage = sum(c.coefficients.double().sum(1) for c in obs)
                restored = fixed+(args[10].shape[1]/remaining).log()/beta*coverage[:, None, None]
                hard = directed_cached(obs, args[10], risk, args[9], beta=beta, chunk=chunk)
                self.assertTrue(torch.allclose(restored[args[9]], hard[args[9]], atol=1e-14, rtol=0))

    def test_no_method_boosts_source_classes_and_invalid_queries_stay_zero(self):
        args = RivalAliasFastTest().fixture()
        risk = retained_competition_scores(*args, methods=BASE)[1]
        obs = cache_observations(args[3], args[4], args[7], args[-1], args[10])
        fine = -torch.ones_like(risk)
        for kind in ('neutral', 'fine', 'fixed_mass'):
            directed = replacement_directed(obs, args[10], risk, args[9], fine, kind)
            self.assertLessEqual(float(directed.max()), 0.)
            self.assertEqual(float(directed[~args[9]].abs().max()), 0.)

    def test_rejects_missing_fine_or_fully_deleted_source(self):
        crop, members, risk, valid = self.fixture()
        with self.assertRaises(ValueError): replacement_directed([crop], members, risk, valid, kind='fine')
        with self.assertRaises(ValueError): replacement_directed([crop], members, torch.ones_like(risk), valid, kind='fixed_mass')

    def test_five_base_endpoints_and_standalone_scores_are_bitwise_equal(self):
        args = RivalAliasFastTest().fixture()
        old, old_risk, _ = retained_competition_scores(*args, methods=BASE)
        values, risk, _ = neutral_replacement_scores(*args)
        self.assertTrue(torch.equal(old_risk, risk))
        for m in BASE: self.assertTrue(torch.equal(values[m], old[m]), m)
        for m in BENCHMARK:
            one = neutral_replacement_scores(*args, methods=(m,))[0]
            self.assertTrue(torch.equal(values[m], one[m]), m)

    def test_zero_risk_is_exact_identity_for_every_new_endpoint(self):
        args = list(RivalAliasFastTest().fixture())
        for crop in (*args[3], *args[5]):
            crop.alias_logits.zero_()
            crop.salience.zero_()
        baseline = args[0].double()+args[1].double()@(args[2].double()-args[0].double())
        values, risk, _ = neutral_replacement_scores(*args)
        self.assertEqual(float(risk.abs().max()), 0.)
        for m in METHODS[len(BASE):]: self.assertTrue(torch.equal(values[m], baseline), m)


if __name__ == '__main__': unittest.main()
