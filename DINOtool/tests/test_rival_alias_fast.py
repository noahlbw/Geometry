import unittest

import torch

from dinotool.calibrated_competitive_alias import profiled_logits
from dinotool.matched_contribution_alias import stencil_margins
from dinotool.native_alias_noise import hard_pair_observation
from dinotool.pair_context_reader import pair_observation
from dinotool.rival_alias_fast import (cache_observations, sampled_cached, directed_cached,
    retained_scores_fast, soft_scores, class_mean_risk)
from dinotool.rival_fine_full import retained_scores
from dinotool.stratified_soft_alias import WideCrop


class RivalAliasFastTest(unittest.TestCase):
    def fixture(self, classes=3, k=20):
        torch.manual_seed(105)
        torch.set_num_threads(2)
        n = 19
        members = torch.arange(classes * k).reshape(classes, k)
        parents, canonical = torch.arange(classes).repeat_interleave(k), members[:, 0]
        crops = [WideCrop(torch.randn(441, classes*k), torch.randn(classes*k), 0, l, 336, 336) for l in (0, 112)]
        count = torch.ones(336, 448)
        count[:, 112:336] += 1
        fine = [WideCrop(torch.randn(1024, classes*k), torch.randn(classes*k), 0, 0, 512, 512, 32, 512)]
        coords = torch.rand(n, 2) * torch.tensor([511., 447.])
        valid = torch.ones(n, dtype=torch.bool)
        valid[-2:] = False
        return (torch.randn(n, classes), torch.eye(n)*.3, torch.randn(n, classes), crops, count,
                fine, torch.ones(512, 512), coords, coords, valid, members, canonical, parents, (512, 448))

    def test_complete_scores_risks_diagnostics_bitwise_equal(self):
        for classes in (2, 5, 12):
            args = self.fixture(classes)
            expected = retained_scores(*args)
            actual = retained_scores_fast(*args)
            self.assertTrue(torch.equal(expected[1], actual[1]))
            self.assertEqual(expected[2], actual[2])
            for key in expected[0]:
                self.assertTrue(torch.equal(expected[0][key], actual[0][key]), key)

    def test_cached_margins_and_hard_writer_at_nonunit_temperature(self):
        args = self.fixture()
        for beta in (.3, 1., 2.):
            prepared = cache_observations(args[3], args[4], args[7], args[-1], args[10], beta)
            expected = sum(stencil_margins(profiled_logits(c, args[10]), p.indices, p.coefficients, beta)
                           for c, p in zip(args[3], prepared))
            self.assertTrue(torch.equal(expected, sampled_cached(prepared, args[7])))
            risk = retained_scores(*args)[1]
            for chunk in (7, 128):
                expected = hard_pair_observation(args[3], args[4], args[7], args[-1], args[10], risk, args[9], beta, chunk)
                actual = directed_cached(prepared, args[10], risk, args[9], beta=beta, chunk=chunk)
                self.assertTrue(torch.equal(expected, actual))

    def test_zero_risk_identity_all_writers(self):
        args = self.fixture()
        prepared = cache_observations(args[3], args[4], args[7], args[-1], args[10])
        zeros = torch.zeros(len(args[9]), args[10].numel(), len(args[10]))
        for writer in ('hard', 'weighted', 'excess'):
            self.assertEqual(float(directed_cached(prepared, args[10], zeros, args[9], writer).abs().max()), 0.)

    def test_weighted_writer_becomes_hard_for_binary_risk(self):
        args = self.fixture()
        risk = (retained_scores(*args)[1] > 0).float()
        prepared = cache_observations(args[3], args[4], args[7], args[-1], args[10])
        hard = directed_cached(prepared, args[10], risk, args[9])
        soft = directed_cached(prepared, args[10], risk, args[9], 'weighted')
        self.assertTrue(torch.equal(hard, soft))

    def test_excess_matches_existing_soft_equation(self):
        args = self.fixture()
        risk = retained_scores(*args)[1]
        prepared = cache_observations(args[3], args[4], args[7], args[-1], args[10])
        expected = pair_observation(args[3], args[4], args[7], args[-1], args[10], risk, args[9])[2]
        actual = directed_cached(prepared, args[10], risk, args[9], 'excess')
        self.assertTrue(torch.allclose(actual, expected.double(), atol=1e-6, rtol=0))

    def test_reuse_methods_do_not_need_fine_and_preserve_canonical_and_gauge(self):
        args = self.fixture()
        local_aliases = torch.randn(len(args[9]), args[10].numel()) * .1
        values, stats = soft_scores(args[0], args[1], args[2], args[3], args[4], args[7], args[9],
            args[10], args[11], args[12], args[13], local_aliases,
            methods=('ReuseSoft_Weighted', 'ReuseSoft_Excess', 'ReuseClassMean_Weighted'))
        self.assertEqual(len(values), 3)
        for row in stats.values():
            self.assertEqual(row['canonical_risk_max'], 0.)
            self.assertLess(row['gauge_max_error'], 1e-10)

    def test_class_mean_control_preserves_noncanonical_pair_budget(self):
        args = self.fixture()
        risk = retained_scores(*args)[1]
        changed = class_mean_risk(risk, args[10], args[11])
        self.assertTrue(torch.allclose(changed[:, args[10]].sum(2), risk[:, args[10]].sum(2), atol=1e-6))
        self.assertEqual(float(changed[:, args[11]].abs().max()), 0.)

    def test_rejects_invalid_and_fully_removed_weights(self):
        args = self.fixture()
        prepared = cache_observations(args[3], args[4], args[7], args[-1], args[10])
        risk = torch.ones(len(args[9]), args[10].numel(), len(args[10]))
        for writer in ('hard', 'weighted', 'undefined'):
            with self.assertRaises(ValueError):
                directed_cached(prepared, args[10], risk, args[9], writer)


if __name__ == '__main__':
    unittest.main()
