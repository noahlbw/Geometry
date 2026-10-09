import unittest

import torch

from dinotool.fine_alias_view import (PRIMARY, SHUFFLES, OBSERVER_METHODS,
    independent_observation, physical_margins, view_sources)
from dinotool.stratified_soft_alias import WideCrop


class FineAliasViewTest(unittest.TestCase):
    def fixture(self):
        members = torch.tensor([[0, 1, 2], [3, 4, 5]])
        parents, canonical = torch.tensor([0, 0, 0, 1, 1, 1]), torch.tensor([0, 3])
        valid = torch.tensor([True, True, False])
        broad = torch.ones(3, 6, 2)
        fine = torch.ones_like(broad)
        fine[0, 1, 1], fine[0, 2, 1], fine[1, 4, 0] = -1., -3., -2.
        base = torch.zeros(3, 6, dtype=torch.float64)
        base[0] = .1
        return broad, fine, parents, canonical, valid, members, base

    def test_only_observed_opposite_view_contradiction_is_rejected(self):
        args = self.fixture()
        _, fields = view_sources(*args)
        pair = fields['pair']
        self.assertAlmostEqual(float(pair[0, 1, 1]), .5)
        self.assertAlmostEqual(float(pair[0, 2, 1]), .75)
        self.assertEqual(float(pair[0, 1, 0]), 0.)
        self.assertEqual(float(pair[:, args[3]].abs().max()), 0.)
        self.assertEqual(float(pair[~args[4]].abs().max()), 0.)
        self.assertEqual(float(pair[1, 1].abs().max()), 0.)

    def test_no_contradiction_recovers_same_class_only_model(self):
        args = list(self.fixture())
        args[1] = args[0].clone()
        sources, fields = view_sources(*args)
        torch.testing.assert_close(sources[PRIMARY], args[-1], atol=0, rtol=0)
        self.assertEqual(float(fields['view'].abs().max()), 0.)

    def test_shuffles_and_class_mean_keep_declared_budgets(self):
        args = self.fixture()
        sources, fields = view_sources(*args)
        self.assertEqual(set(fields['alias_spectrum_errors']), set(SHUFFLES))
        self.assertEqual(max(fields['alias_spectrum_errors'].values()), 0.)
        self.assertEqual(fields['spatial_view_spectrum_error'], 0.)
        self.assertLessEqual(fields['class_mean_view_budget_error'], 1e-12)
        for risk in sources.values():
            self.assertTrue(bool(((risk >= 0) & (risk <= 1)).all()))
            self.assertEqual(float(risk[~args[4]].abs().max()), 0.)

    def test_physical_margin_units_and_coverage(self):
        crops = [WideCrop(torch.tensor([[2., 1., 3., 0., 0., 0.]]).expand(1024, -1),
            torch.zeros(6), t, l, 256, 256, 32, 256) for t in (0, 256) for l in (0, 256)]
        coords = torch.tensor([[8., 8.], [264., 264.], [504., 504.]])
        members = torch.tensor([[0, 1, 2], [3, 4, 5]])
        valid = torch.tensor([True, True, False])
        margins = physical_margins(crops, torch.ones(512, 512), coords, members, valid)
        torch.testing.assert_close(margins[0, :3, 1], torch.tensor([2., 1., 3.]), atol=1e-6, rtol=0)
        torch.testing.assert_close(margins[0], margins[1], atol=0, rtol=0)
        self.assertEqual(float(margins[2].abs().max()), 0.)

    def test_pair_scope_changes_only_the_declared_rival(self):
        members = torch.arange(9).reshape(3, 3)
        parents, canonical = torch.arange(3).repeat_interleave(3), torch.tensor([0, 3, 6])
        valid = torch.tensor([True, True, False])
        broad = torch.ones(3, 9, 3)
        fine = broad.clone()
        fine[0, 1, 1] = -1
        pair = view_sources(broad, fine, parents, canonical, valid, members,
            torch.zeros(3, 9, dtype=torch.float64))[1]['pair']
        self.assertGreater(float(pair[0, 1, 1]), 0.)
        self.assertEqual(float(pair[0, 1, 2]), 0.)

    def observer_fixture(self):
        args = self.fixture()
        crop = WideCrop(torch.tensor([[2., 1., 3., 0., 1., 0.]]).expand(4, -1),
            torch.zeros(6), 0, 0, 512, 512, 2, 512)
        coords = torch.tensor([[128., 128.], [384., 384.], [128., 384.]])
        pair = view_sources(*args)[1]['pair']
        return (torch.ones(3, 2, dtype=torch.float64), [crop], torch.ones(512, 512),
            coords, (512, 512), args[5], pair, args[4], args[3], pair)

    def test_independent_observer_zero_risk_is_exact_identity(self):
        args = list(self.observer_fixture())
        args[6] = torch.zeros_like(args[6])
        args[9] = torch.zeros_like(args[9])
        values, _ = independent_observation(*args)
        self.assertEqual(set(values), set(OBSERVER_METHODS))
        for value in values.values():
            torch.testing.assert_close(value, args[0], atol=0, rtol=0)

    def test_independent_hard_shuffles_match_each_rivals_deleted_count(self):
        values, stats = independent_observation(*self.observer_fixture())
        self.assertTrue(all(bool(torch.isfinite(v).all()) for v in values.values()))
        self.assertTrue(all(s['count_max_error'] == 0 for s in stats.values()))
        self.assertTrue(all(s['canonical_risk_max'] == 0 for s in stats.values()))


if __name__ == '__main__':
    unittest.main()
