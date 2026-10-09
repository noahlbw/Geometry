import unittest

import torch

from dinotool.rival_competition_admission import retained_competition_scores
from dinotool.rival_fine_support import PRIMARY as OLD_SUPPORT, support_scores
from dinotool.rival_matched_support import (PRIMARY, NEW_METHODS, RivalMatches,
    semantic_rivals, sampled_aliases, matched_weights, matched_scores)
from test_rival_alias_fast import RivalAliasFastTest


class MatchedSupportTests(unittest.TestCase):
    def fixture(self, k=20):
        inputs = RivalAliasFastTest().fixture(k=k)
        torch.manual_seed(22)
        matches = semantic_rivals(torch.randn(inputs[10].numel(), 2, 12), inputs[10])
        return inputs, matches

    def test_nearest_alias_and_semantic_duplicate(self):
        text = torch.tensor([[[1., 0.]], [[0., 1.]], [[1., 0.]], [[-1., 0.]]])
        matches = semantic_rivals(text, torch.arange(4).reshape(2, 2))
        self.assertEqual(int(matches.indices[0, 1]), 2)
        self.assertEqual(float(matches.affinity[0, 1]), 1.)
        scores = torch.tensor([[3., 0., 3., 0.]])
        risk = torch.zeros(1, 4, 2)
        weights = matched_weights(scores, matches, risk, torch.tensor([0, 0, 1, 1]),
                                  torch.tensor([1, 3]), torch.tensor([True]))
        self.assertEqual(float(weights[0, 0, 1]), .5)
        scores[0, 2] = -7
        changed = matched_weights(scores, matches, risk, torch.tensor([0, 0, 1, 1]),
                                  torch.tensor([1, 3]), torch.tensor([True]))
        self.assertGreater(float(changed[0, 0, 1]), .999)

    def test_unrelated_rival_retains_alias(self):
        matches = RivalMatches(torch.tensor([[0, 2]]*4), torch.zeros(4, 2))
        result = matched_weights(torch.tensor([[-100., 100., 100., -100.]]), matches, torch.zeros(1, 4, 2),
                                 torch.tensor([0, 0, 1, 1]), torch.tensor([1, 3]), torch.tensor([True]))
        self.assertTrue(torch.equal(result, torch.ones_like(result)))

    def test_same_alias_can_discriminate_a_different_rival(self):
        text = torch.tensor([[[1., 0.]], [[0., 1.]], [[1., 0.]], [[0., 1.]],
                             [[-1., 0.]], [[-1., -1.]]])
        matches = semantic_rivals(text, torch.arange(6).reshape(3, 2))
        weights = matched_weights(torch.full((1, 6), 3.), matches, torch.zeros(1, 6, 3),
                                  torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([1, 3, 5]), torch.tensor([True]))
        self.assertEqual(float(weights[0, 0, 1]), .5)
        self.assertEqual(float(weights[0, 0, 2]), 1.)

    def test_historical_and_singleton_scores_are_exact(self):
        for k in (20, 40):
            inputs, matches = self.fixture(k)
            values, diag = matched_scores(*inputs, matches=matches)
            old = retained_competition_scores(*inputs)[0]
            for method in ('NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact'):
                self.assertTrue(torch.equal(values[method], old[method]))
            self.assertTrue(torch.equal(values[OLD_SUPPORT], support_scores(*inputs, methods=(OLD_SUPPORT,))[0][OLD_SUPPORT]))
            solo = matched_scores(*inputs, matches=matches, methods=(PRIMARY,))[0][PRIMARY]
            self.assertTrue(torch.equal(solo, values[PRIMARY]))
            self.assertEqual(diag['canonical_weight_min'], 1.)
            self.assertEqual(diag['old_rejected_weights_max'], 0.)
            self.assertEqual(max(diag['control_spectrum_max_errors'].values()), 0.)
            self.assertLessEqual(diag['class_mean_weight_mass_max_error'], 1e-10)

    def test_invalid_queries_create_no_actions(self):
        inputs, matches = self.fixture()
        inputs = list(inputs)
        inputs[9].fill_(False)
        values, _ = matched_scores(*inputs, matches=matches)
        self.assertTrue(all(torch.equal(values[m], values['NoAdmission_Exact']) for m in NEW_METHODS))

    def test_sampled_aliases_preserve_stencil_sum(self):
        from dinotool.rival_alias_fast import CachedCrop
        evidence = torch.arange(12).reshape(2, 2, 3).float()
        crop = CachedCrop(evidence, None, torch.tensor([[0, 1]]), torch.tensor([[.25, .75]]))
        actual = sampled_aliases([crop], torch.zeros(1, 2))
        self.assertTrue(torch.equal(actual, (evidence[0]*.25+evidence[1]*.75).flatten()[None].double()))


if __name__ == '__main__':
    unittest.main()
