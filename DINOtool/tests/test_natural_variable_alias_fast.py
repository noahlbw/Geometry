import unittest
from unittest.mock import patch

import torch

from dinotool.natural_variable_alias_fast import (prepare_observations, sampled_margins_cached,
    hard_observation_cached, retained_variable_scores_fast)
from dinotool.natural_variable_alias_reader import (alias_groups, sampled_margins, hard_observation,
    retained_variable_scores)
from dinotool.stratified_soft_alias import WideCrop


class NaturalVariableAliasFastTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(104)
        torch.set_num_threads(2)

    def inputs(self, sizes=(1, 3, 5), n=19):
        parents = torch.arange(len(sizes)).repeat_interleave(torch.tensor(sizes))
        canonical = torch.tensor([sum(sizes[:c]) for c in range(len(sizes))])
        q = len(parents)
        wide = [WideCrop(torch.randn(441, q), torch.randn(q), 0, 0, 336, 336)]
        fine = [WideCrop(torch.randn(1024, q), torch.randn(q), 0, 0, 512, 512, 32, 512)]
        coords = torch.rand(n, 2) * 335
        valid = torch.ones(n, dtype=torch.bool)
        valid[-2:] = False
        return (torch.randn(n, len(sizes)), torch.eye(n)*.25, torch.randn(n, len(sizes)),
                wide, torch.ones(336, 336), fine, torch.ones(512, 512), coords, coords, valid,
                parents, canonical, (336, 336))

    def test_sampled_margins_bitwise_equal_for_nonunit_beta(self):
        args = self.inputs()
        crops, count, coordinates = args[3], args[4], args[7]
        groups = alias_groups(args[10], args[0].shape[-1])
        for beta in (.3, 1., 2.):
            prepared = prepare_observations(crops, count, coordinates, args[-1], groups, beta)
            for sl in (slice(0, 7), slice(7, 19)):
                expected = sampled_margins(crops, count, coordinates[sl], args[-1], groups, beta)
                actual = sampled_margins_cached(prepared, coordinates, groups, sl)
                self.assertTrue(torch.equal(actual, expected))

    def test_hard_observation_bitwise_equal_with_singletons(self):
        args = self.inputs()
        groups = alias_groups(args[10], args[0].shape[-1])
        risk = torch.rand(len(args[9]), len(args[10]), len(groups))
        risk[:, args[11]] = 0
        risk.scatter_(-1, args[10][None, :, None].expand(len(risk), -1, 1), 0.)
        for beta in (.3, 1., 2.):
            prepared = prepare_observations(args[3], args[4], args[7], args[-1], groups, beta)
            expected = hard_observation(args[3], args[4], args[7], args[-1], groups, risk, args[9], beta)
            actual, _ = hard_observation_cached(prepared, groups, risk, args[9], beta, slice(None))
            self.assertTrue(torch.equal(actual, expected))

    def test_full_reader_scores_and_diagnostics_bitwise_equal(self):
        for sizes in ((1, 3, 5), (4, 4, 4)):
            args = self.inputs(sizes)
            for beta in (.3, 1., 2.):
                for chunk in (7, 128):
                    expected, old_stats = retained_variable_scores(*args, beta=beta, query_chunk=chunk)
                    actual, new_stats = retained_variable_scores_fast(*args, beta=beta, query_chunk=chunk)
                    for key in expected:
                        self.assertTrue(torch.equal(actual[key], expected[key]), key)
                    self.assertEqual(new_stats, old_stats)

    def test_overlapping_multiple_crops_and_invalid_coordinates(self):
        args = list(self.inputs(n=21))
        q = len(args[10])
        args[3].append(WideCrop(torch.randn(441, q), torch.randn(q), 0, 112, 336, 336))
        args[4] = torch.ones(336, 448)
        args[4][:, 112:336] += 1
        args[-1] = (336, 448)
        args[7] = torch.rand(21, 2)*torch.tensor([335., 447.])
        args[8] = torch.rand(21, 2)*511
        expected, stats = retained_variable_scores(*args, query_chunk=8)
        actual, fast_stats = retained_variable_scores_fast(*args, query_chunk=8)
        for key in expected:
            self.assertTrue(torch.equal(actual[key], expected[key]), key)
        self.assertEqual(stats, fast_stats)

    def test_crop_profiles_are_prepared_once_not_once_per_chunk(self):
        from dinotool.natural_variable_alias_reader import profile
        args = self.inputs()
        with patch('dinotool.natural_variable_alias_fast.profile', wraps=profile) as mocked:
            retained_variable_scores_fast(*args, query_chunk=3)
        self.assertEqual(mocked.call_count, len(args[3])+len(args[5]))

    def test_rejects_invalid_configuration_and_missing_canonical(self):
        args = self.inputs()
        for options in ({'beta': 0}, {'query_chunk': 0}):
            with self.assertRaises(ValueError):
                retained_variable_scores_fast(*args, **options)
        groups = alias_groups(args[10], args[0].shape[-1])
        prepared = prepare_observations(args[3], args[4], args[7], args[-1], groups, 1.)
        with self.assertRaises(ValueError):
            hard_observation_cached(prepared, groups,
                torch.ones(len(args[9]), len(args[10]), len(groups)), args[9], 1., slice(None))


if __name__ == '__main__':
    unittest.main()
