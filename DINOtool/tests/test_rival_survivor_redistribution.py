import unittest

import torch

from dinotool.rival_alias_fast import CachedCrop, directed_cached
from dinotool.rival_competition_admission import retained_competition_scores
from dinotool.rival_matched_support import PRIMARY as MATCHED, matched_scores, semantic_rivals
from dinotool.rival_survivor_redistribution import (PRIMARY, CLASS_MEAN, NEW_METHODS,
    redistribute, allocation_directed, redistribution_scores)
from test_rival_alias_fast import RivalAliasFastTest


class SurvivorRedistributionTests(unittest.TestCase):
    def fixture(self, k=20):
        inputs = RivalAliasFastTest().fixture(k=k)
        torch.manual_seed(35)
        matches = semantic_rivals(torch.randn(inputs[10].numel(), 2, 12), inputs[10])
        return inputs, matches

    def test_mass_canonical_and_hard_support_are_fixed(self):
        members = torch.arange(6).reshape(2, 3)
        canonical = torch.tensor([0, 3])
        risk = torch.zeros(1, 6, 2)
        risk[0, 2, 1] = 1
        source = torch.tensor([[[1., 1.], [.3, .2], [.9, 0.], [1., 1.], [.5, .7], [.1, .5]]], dtype=torch.float64)
        actual = redistribute(source, risk, members, canonical)
        self.assertTrue(torch.equal(actual[:, canonical], torch.ones_like(actual[:, canonical])))
        self.assertEqual(float(actual[0, 2, 1]), 0.)
        torch.testing.assert_close(actual[:, members].sum(2), (risk[:, members] == 0).sum(2).double(), atol=1e-12, rtol=0)
        self.assertGreater(float(actual[0, 2, 0]), 1.)
        self.assertLess(float(actual[0, 1, 0]), 1.)

    def test_uniform_source_and_single_survivor_recover_hard_exactly(self):
        members = torch.arange(6).reshape(2, 3)
        canonical = torch.tensor([0, 3])
        risk = torch.zeros(1, 6, 2)
        risk[0, 1:3, 1] = 1
        source = torch.full_like(risk, .317, dtype=torch.float64)
        source[risk > 0] = 0
        source[:, canonical] = 1
        self.assertTrue(torch.equal(redistribute(source, risk, members, canonical), (risk == 0).double()))

    def test_general_writer_matches_hard_and_hand_computation(self):
        members = torch.arange(6).reshape(2, 3)
        evidence = torch.tensor([[[.5, 2., 1.], [1., 0., 3.]]])
        crop = CachedCrop(evidence, None, torch.zeros((1, 1), dtype=torch.long), torch.ones(1, 1))
        valid = torch.tensor([True])
        risk = torch.zeros(1, 6, 2)
        risk[0, 2, 1] = 1
        keep = risk == 0
        expected = directed_cached([crop], members, risk, valid)
        self.assertTrue(torch.equal(allocation_directed([crop], members, keep.double(), keep, valid), expected))
        weights = keep.double()
        weights[0, 4, 0], weights[0, 5, 0] = .5, 1.5
        actual = allocation_directed([crop], members, weights, keep, valid)
        reference = torch.logsumexp(evidence[0, 1].double()+torch.tensor([1., .5, 1.5]).double().log(), 0)
        reference -= torch.logsumexp(evidence[0, 1].double(), 0)
        self.assertEqual(float(actual[0, 1, 0]), float(reference))

    def test_replays_class_control_and_primary_singleton_are_exact(self):
        for k in (20, 40):
            inputs, matches = self.fixture(k)
            values, diag = redistribution_scores(*inputs, matches=matches)
            old = retained_competition_scores(*inputs)[0]
            for m in ('NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact'):
                self.assertTrue(torch.equal(values[m], old[m]))
            self.assertTrue(torch.equal(values[MATCHED], matched_scores(*inputs, matches=matches, methods=(MATCHED,))[0][MATCHED]))
            self.assertTrue(torch.equal(values[CLASS_MEAN], old['FineRivalProjected_Exact']))
            self.assertTrue(torch.equal(values[PRIMARY], redistribution_scores(*inputs, matches=matches, methods=(PRIMARY,))[0][PRIMARY]))
            self.assertLessEqual(max(diag['all_mass_max_errors'].values()), 1e-10)
            self.assertEqual(diag['class_mean_score_max_error'], 0.)

    def test_invalid_queries_and_changed_mass(self):
        inputs, matches = self.fixture()
        inputs = list(inputs)
        inputs[9].fill_(False)
        values, _ = redistribution_scores(*inputs, matches=matches)
        self.assertTrue(all(torch.equal(values[m], values['NoAdmission_Exact']) for m in NEW_METHODS))
        members = torch.arange(4).reshape(2, 2)
        crop = CachedCrop(torch.zeros(1, 2, 2), None, torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1))
        weights = torch.ones(1, 4, 2)
        weights[0, 1, 1] = 2
        with self.assertRaisesRegex(ValueError, 'mass'):
            allocation_directed([crop], members, weights, torch.ones_like(weights, dtype=torch.bool), torch.tensor([True]))


if __name__ == '__main__':
    unittest.main()
