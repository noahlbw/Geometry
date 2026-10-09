import unittest
from unittest.mock import patch

import torch

from dinotool.bounded_fine_coverage import (CLASS_MEAN, SHUFFLES, FineCoverageReader,
    alias_controls, weighted_observation)
from dinotool.fine_reference_admission import fine_crop_positions
from dinotool.native_alias_noise import hard_pair_observation, signed_potential
from dinotool.stratified_soft_alias import WideCrop


class FineCoverageTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(67)
        self.members = torch.arange(12).reshape(3, 4)
        self.canonical = self.members[:, 0]
        self.valid = torch.tensor([True, True, False])
        self.coordinates = torch.tensor([[8., 8.], [24., 24.], [40., 40.]])
        self.crop = WideCrop(torch.randn(4, 12), torch.randn(12), 0, 0, 32, 32, 2, 32)
        self.count = torch.ones(32, 32)
        self.risk = torch.rand(3, 12, 3)
        self.risk[self.risk < .4] = 0
        self.risk[:, self.canonical] = 0
        self.risk[~self.valid] = 0
        self.risk.scatter_(-1, torch.arange(3).repeat_interleave(4)[None, :, None].expand(3, -1, 1), 0.)

    def test_quadrants_cover_valid_edges_with_at_most_four_crops(self):
        for h, w in ((512, 512), (384, 512), (128, 512), (1, 1)):
            canvas = torch.zeros(h, w, dtype=torch.long)
            positions = fine_crop_positions(h, w)
            self.assertLessEqual(len(positions), 4)
            for t, l, ch, cw in positions:
                canvas[t:t+ch, l:l+cw] += 1
            self.assertTrue(torch.equal(canvas, torch.ones_like(canvas)))

    def test_whole_image_encoding_cap_is_enforced_before_forward(self):
        observer = object()
        reader = FineCoverageReader(observer)
        crops = (torch.zeros(3, 512, 512),)*4
        with patch('dinotool.bounded_fine_coverage.fine_patch_features', return_value=torch.ones(1)) as mock:
            for _ in range(4):
                self.assertEqual(len(reader(observer, crops)), 4)
            self.assertEqual(mock.call_count, 16)
            with self.assertRaises(RuntimeError):
                reader(observer, crops)
            self.assertEqual(mock.call_count, 16)

    def test_invalid_reader_owner_and_shape_are_rejected(self):
        reader = FineCoverageReader(object())
        with self.assertRaises(RuntimeError):
            reader(object(), (torch.zeros(3, 512, 512),))
        with self.assertRaises(RuntimeError):
            reader(reader.observer, (torch.zeros(3, 336, 336),))
        self.assertEqual(reader.calls, 0)

    def test_controls_preserve_canonical_and_pairwise_count(self):
        controls = alias_controls(self.risk, self.members, self.canonical)
        original = (self.risk[:, self.members] == 0).double()
        shared = controls[CLASS_MEAN]
        torch.testing.assert_close(shared.sum(2), original.sum(2), atol=1e-12, rtol=0)
        self.assertTrue(torch.equal(shared[:, :, 0], torch.ones(3, 3, 3, dtype=torch.float64)))
        for name in SHUFFLES:
            changed = controls[name]
            self.assertTrue(torch.equal(changed[:, self.canonical], self.risk[:, self.canonical]))
            self.assertTrue(torch.equal(changed[:, self.members].sort(2).values, self.risk[:, self.members].sort(2).values))
            self.assertTrue(torch.equal(changed[~self.valid], torch.zeros_like(changed[~self.valid])))

    def test_controls_are_reproducible(self):
        a = alias_controls(self.risk, self.members, self.canonical)
        b = alias_controls(self.risk, self.members, self.canonical)
        for name in a:
            self.assertTrue(torch.equal(a[name], b[name]))

    def test_weighted_writer_exactly_recovers_binary_hard_writer(self):
        allocation = (self.risk[:, self.members] == 0).double()
        actual = weighted_observation([self.crop], self.count, self.coordinates, (32, 32),
                                      self.members, allocation, self.valid)
        expected = hard_pair_observation([self.crop], self.count, self.coordinates, (32, 32),
                                        self.members, self.risk, self.valid)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)

    def test_no_rejection_is_exact_identity(self):
        weights = torch.ones(3, 3, 4, 3, dtype=torch.float64)
        actual = weighted_observation([self.crop], self.count, self.coordinates, (32, 32),
                                      self.members, weights, self.valid)
        self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))

    def test_class_constant_weight_cancels_up_to_roundoff(self):
        weights = torch.rand(3, 3, 1, 3, dtype=torch.float64).expand(-1, -1, 4, -1)
        actual = weighted_observation([self.crop], self.count, self.coordinates, (32, 32),
                                      self.members, weights, self.valid)
        torch.testing.assert_close(actual, torch.zeros_like(actual), atol=1e-12, rtol=0)

    def test_invalid_donors_and_class_gauge_remain_zero(self):
        directed = hard_pair_observation([self.crop], self.count, self.coordinates, (32, 32),
                                        self.members, self.risk, self.valid)
        potential, stats = signed_potential(directed, self.valid)
        self.assertTrue(torch.equal(potential[~self.valid], torch.zeros_like(potential[~self.valid])))
        torch.testing.assert_close(potential.sum(-1), torch.zeros(3, dtype=torch.float64), atol=1e-12, rtol=0)
        self.assertLessEqual(stats['gauge_max_error'], 1e-12)

    def test_empty_survivors_are_rejected(self):
        with self.assertRaises(ValueError):
            weighted_observation([self.crop], self.count, self.coordinates, (32, 32),
                                 self.members, torch.zeros(3, 3, 4, 3), self.valid)


if __name__ == '__main__':
    unittest.main()
