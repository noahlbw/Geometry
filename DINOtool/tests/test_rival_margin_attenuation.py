import math
import unittest

import torch

from dinotool.rival_alias_fast import CachedCrop, cache_observations, directed_cached
from dinotool.rival_alias_attribution import alias_null
from dinotool.rival_competition_admission import retained_competition_scores
from dinotool.rival_margin_attenuation import (BASE, METHODS, CANDIDATES, margin_log_weights,
    discount_directed, class_mean_discount, margin_attenuation_scores)
from test_rival_alias_fast import RivalAliasFastTest


class RivalMarginAttenuationTest(unittest.TestCase):
    def test_transfer_has_actual_margin_units_and_preserves_neutral_slots(self):
        b = torch.tensor([[[2., 0., 1.], [3., 2., 4.]]])
        f = torch.tensor([[[-1., -2., 2.], [-4., -1., -2.]]])
        risk = torch.tensor([[[1/3, 0., 0.], [0., 1/3, 1/3]]])
        for beta in (.3, 1., 2.):
            lw = margin_log_weights(b, f, risk, beta)
            self.assertTrue(torch.equal(lw[risk == 0], torch.zeros_like(lw[risk == 0])))
            self.assertTrue(torch.allclose((b.double()+lw/beta)[risk > 0], f.double()[risk > 0], atol=1e-14, rtol=0))
            self.assertTrue(torch.allclose((b.double()+margin_log_weights(b, f, risk, beta, 'neutral')/beta)[risk > 0], torch.zeros(3, dtype=torch.float64), atol=1e-14))

    def test_transfer_and_deficit_are_ordered_but_not_probability_weights(self):
        b, f = torch.tensor([[[2.]]]), torch.tensor([[[-1.]]])
        risk = torch.tensor([[[1/3]]])
        transfer = margin_log_weights(b, f, risk).exp()
        neutral = margin_log_weights(b, f, risk, kind='neutral').exp()
        deficit = margin_log_weights(b, f, risk, kind='deficit').exp()
        self.assertLess(float(transfer), float(neutral))
        self.assertLess(float(neutral), float(deficit))
        self.assertAlmostEqual(float(transfer), math.exp(-3), places=12)
        self.assertGreater(float(b.double()+torch.log1p(-risk.double())), 0.)

    def test_soft_writer_matches_direct_normalized_expression(self):
        crop = CachedCrop(torch.tensor([[[0., 2.], [1., -1.]]]), None, torch.tensor([[0]]), torch.ones(1, 1))
        members = torch.arange(4).reshape(2, 2)
        lw = torch.tensor([[[0., 0.], [0., -3.], [0., 0.], [-2., 0.]]], dtype=torch.float64)
        got = discount_directed([crop], members, lw, torch.tensor([True]))
        grouped = lw[:, members]
        full = crop.evidence.double().logsumexp(-1)
        expected = (crop.evidence.double()[..., None]+grouped).logsumexp(2)-full[..., None]+math.log(2)-grouped.logsumexp(2)
        self.assertTrue(torch.allclose(got, expected, atol=1e-14, rtol=0))

    def test_zero_and_binary_weights_match_baseline_and_hard_writer(self):
        args = RivalAliasFastTest().fixture()
        risk = retained_competition_scores(*args, methods=BASE)[1]
        obs = cache_observations(args[3], args[4], args[7], args[-1], args[10])
        zero = discount_directed(obs, args[10], torch.zeros_like(risk), args[9])
        self.assertEqual(float(zero.abs().max()), 0.)
        binary = torch.zeros_like(risk).masked_fill(risk > 0, -torch.inf)
        got = discount_directed(obs, args[10], binary, args[9])
        old = directed_cached(obs, args[10], risk, args[9])
        self.assertTrue(torch.allclose(got, old, atol=1e-14, rtol=0))

    def test_strong_discounts_are_finite_and_canonical_weight_survives(self):
        args = RivalAliasFastTest().fixture()
        obs = cache_observations(args[3], args[4], args[7], args[-1], args[10])
        lw = torch.full((len(args[9]), args[10].numel(), len(args[10])), -10000.)
        lw[:, args[11]] = 0
        self.assertTrue(bool(torch.isfinite(discount_directed(obs, args[10], lw, args[9])).all()))
        with self.assertRaises(ValueError): discount_directed(obs, args[10], torch.full_like(lw, -torch.inf), args[9])

    def test_mean_and_alias_controls_match_declared_budgets(self):
        args = RivalAliasFastTest().fixture()
        torch.manual_seed(19)
        lw = -torch.rand(len(args[9]), args[10].numel(), len(args[10]), dtype=torch.float64)
        lw[:, args[11]] = 0
        changed = class_mean_discount(lw, args[10], args[11])
        self.assertTrue(torch.allclose(changed[:, args[10]].sum(2), lw[:, args[10]].sum(2), atol=1e-14, rtol=0))
        shuffled = alias_null(lw, args[10], args[11], 20261003)
        self.assertTrue(torch.equal(shuffled[:, args[10]].sort(2).values, lw[:, args[10]].sort(2).values))
        self.assertEqual(float(shuffled[:, args[11]].abs().max()), 0.)

    def test_frozen_scores_and_standalone_candidates_are_bitwise_equal(self):
        args = RivalAliasFastTest().fixture()
        old, old_risk, _ = retained_competition_scores(*args, methods=BASE)
        values, risk, stats = margin_attenuation_scores(*args)
        self.assertTrue(torch.equal(risk, old_risk))
        for m in BASE: self.assertTrue(torch.equal(old[m], values[m]), m)
        for m in CANDIDATES:
            one = margin_attenuation_scores(*args, methods=(m,))[0]
            self.assertTrue(torch.equal(one[m], values[m]), m)
        self.assertEqual(stats['transfer_residual_margin_max_error'], 0.)

    def test_zero_risk_is_exact_identity_for_every_new_method(self):
        args = list(RivalAliasFastTest().fixture())
        for crop in (*args[3], *args[5]):
            crop.alias_logits.zero_()
            crop.salience.zero_()
        baseline = args[0].double()+args[1].double()@(args[2].double()-args[0].double())
        scores, risk, _ = margin_attenuation_scores(*args)
        self.assertEqual(float(risk.abs().max()), 0.)
        for m in METHODS[len(BASE):]: self.assertTrue(torch.equal(scores[m], baseline), m)


if __name__ == '__main__': unittest.main()
