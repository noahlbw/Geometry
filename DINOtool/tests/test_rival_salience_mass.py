import math
from types import SimpleNamespace
import unittest

import torch

from dinotool.rival_alias_fast import CachedCrop, cache_observations
from dinotool.rival_competition_admission import retained_competition_scores
from dinotool.rival_salience_mass import (BASE, CANDIDATES, METHODS, prior_normalizer,
    prior_adjustment, sampled_fine_log_prior, mean_noncanonical_prior, shuffled_prior, salience_mass_scores)
from test_rival_alias_fast import RivalAliasFastTest


class RivalSalienceMassTest(unittest.TestCase):
    def test_retained_prior_mass_is_not_retained_alias_count(self):
        prior = torch.tensor([[.2, .8], [.8, .2]], dtype=torch.float64).log()
        keep = torch.ones(1, 2, 2, 2, dtype=torch.bool)
        keep[:, :, 1] = False
        got = prior_normalizer(prior, keep)
        self.assertAlmostEqual(float(got[0, 0, 1]), -math.log(.2), places=12)
        self.assertAlmostEqual(float(got[0, 1, 0]), -math.log(.8), places=12)
        self.assertGreater(float(got[0, 0, 1]), math.log(2))
        self.assertLess(float(got[0, 1, 0]), math.log(2))

    def test_uniform_log_prior_recovers_count_rule_at_nonunit_beta(self):
        args = RivalAliasFastTest().fixture()
        risk = retained_competition_scores(*args, methods=BASE)[1]
        obs = cache_observations(args[3], args[4], args[7], args[-1], args[10])
        priors = [torch.full(args[10].shape, -math.log(args[10].shape[1]), dtype=torch.float64)]*len(obs)
        for beta in (.3, 1., 2.):
            got = prior_adjustment(obs, priors, args[10], risk, args[9], beta=beta, chunk=7)
            self.assertLess(float(got.abs().max()), 1e-14)

    def test_sampled_fine_prior_uses_actual_query_coverage_and_invalid_fallback(self):
        members = torch.arange(4).reshape(2, 2)
        salience = [torch.tensor([0., 2., 3., 0.]), torch.tensor([2., 0., 0., 3.])]
        crops = [SimpleNamespace(salience=v) for v in salience]
        cached = [CachedCrop(None, None, None, torch.tensor([[1.], [0.], [0.]])),
                  CachedCrop(None, None, None, torch.tensor([[0.], [1.], [0.]]))]
        valid = torch.tensor([True, True, False])
        got = sampled_fine_log_prior(crops, cached, members, valid)
        for i in range(2): self.assertTrue(torch.allclose(got[i], salience[i][members].double().log_softmax(-1), atol=1e-14, rtol=0))
        self.assertTrue(torch.equal(got[-1], torch.full_like(got[-1], -math.log(2))))
        with self.assertRaises(ValueError): sampled_fine_log_prior(crops, cached, members, torch.ones(3, dtype=torch.bool))

    def test_extreme_log_priors_keep_finite_canonical_survival(self):
        log_prior = torch.tensor([[-1000., 0.]], dtype=torch.float64).log_softmax(-1)
        keep = torch.tensor([[[[True, True], [False, False]]]])
        got = prior_normalizer(log_prior, keep)
        self.assertTrue(bool(torch.isfinite(got).all()))
        self.assertAlmostEqual(float(got[0, 0, 0]), 1000., places=12)
        with self.assertRaises(ValueError): prior_normalizer(log_prior, torch.zeros_like(keep))

    def test_mean_and_shuffled_controls_preserve_canonical_and_declared_mass(self):
        args = RivalAliasFastTest().fixture()
        torch.manual_seed(27)
        prior = torch.randn(7, *args[10].shape, dtype=torch.float64).log_softmax(-1)
        mean = mean_noncanonical_prior(prior, args[10], args[11])
        shuffled = shuffled_prior(prior, args[10], args[11], 20261003)
        self.assertTrue(torch.equal(mean[:, :, 0], prior[:, :, 0]))
        self.assertTrue(torch.allclose(mean.logsumexp(-1), prior.logsumexp(-1), atol=1e-14, rtol=0))
        self.assertTrue(torch.equal(shuffled[:, :, 0], prior[:, :, 0]))
        self.assertTrue(torch.equal(shuffled.sort(-1).values, prior.sort(-1).values))

    def test_normalizer_is_class_permutation_equivariant(self):
        torch.manual_seed(29)
        prior = torch.randn(5, 3, 7, dtype=torch.float64).log_softmax(-1)
        keep = torch.rand(5, 3, 7, 3) > .3
        keep[:, :, 0] = True
        order = torch.tensor([2, 0, 1])
        old = prior_normalizer(prior, keep)
        new = prior_normalizer(prior[:, order], keep[:, order][:, :, :, order])
        self.assertTrue(torch.equal(old[:, order][:, :, order], new))

    def test_five_base_and_standalone_candidate_scores_are_bitwise_equal(self):
        args = RivalAliasFastTest().fixture()
        old, old_risk, _ = retained_competition_scores(*args, methods=BASE)
        values, risk, _ = salience_mass_scores(*args)
        self.assertTrue(torch.equal(old_risk, risk))
        for m in BASE: self.assertTrue(torch.equal(old[m], values[m]), m)
        for m in CANDIDATES:
            singleton = salience_mass_scores(*args, methods=(m,))[0]
            self.assertTrue(torch.equal(values[m], singleton[m]), m)

    def test_zero_risk_exactly_restores_baseline_for_all_new_methods(self):
        args = list(RivalAliasFastTest().fixture())
        for crop in (*args[3], *args[5]):
            crop.alias_logits.zero_()
            crop.salience.zero_()
        baseline = args[0].double()+args[1].double()@(args[2].double()-args[0].double())
        values, risk, _ = salience_mass_scores(*args)
        self.assertEqual(float(risk.abs().max()), 0.)
        for m in METHODS[len(BASE):]: self.assertTrue(torch.equal(values[m], baseline), m)


if __name__ == '__main__': unittest.main()
