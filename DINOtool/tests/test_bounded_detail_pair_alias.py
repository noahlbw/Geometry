import unittest

import torch

from dinotool.bounded_detail_pair_alias import (PRIMARY, CLASS_MEAN, SHUFFLE,
    detail_positions, detail_rgb, select_details, sparse_weights, pair_potential)


class GlobalDetailAliasTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)

    def test_candidates_partition_image_without_overlaps(self):
        for h, w in ((896, 896), (672, 896), (64, 896)):
            canvas = torch.zeros(h, w, dtype=torch.int64)
            for t, l, ch, cw in detail_positions(h, w):
                canvas[t:t+ch, l:l+cw] += 1
            self.assertTrue(torch.equal(canvas, torch.ones_like(canvas)))
            self.assertLessEqual(len(detail_positions(h, w)), 16)

    def test_detail_uses_original_rgb_and_zero_padding(self):
        image = torch.ones(3, 512, 512)
        actual = detail_rgb(image, (0, 0, 256, 256), (896, 896))
        self.assertEqual(tuple(actual.shape), (1, 3, 512, 512))
        torch.testing.assert_close(actual[:, :, 4:], torch.ones_like(actual[:, :, 4:]), atol=1e-6, rtol=0)
        edge = detail_rgb(image, (768, 768, 128, 128), (896, 896))
        self.assertTrue(torch.equal(edge[:, :, 256:], torch.zeros_like(edge[:, :, 256:])))
        self.assertTrue(torch.equal(edge[:, :, :, 256:], torch.zeros_like(edge[:, :, :, 256:])))

    def test_selection_is_image_wide_budget_two_and_skips_zero_conflict(self):
        local = torch.tensor([[5., 0.], [5., 0.], [5., 0.], [5., 0.]])
        record = dict(operator=torch.eye(4), valid=torch.ones(4, dtype=torch.bool),
                      coordinates=torch.tensor([[64., 64.], [64., 320.], [320., 64.], [320., 320.]]),
                      fields={'p': (local, local)})
        self.assertEqual(select_details([record], 896, 896), ())
        record['fields'] = {'p': (local, local.flip(-1))}
        chosen = select_details([record], 896, 896)
        self.assertEqual(chosen, ((0, 0, 256, 256), (0, 256, 256, 256)))
        self.assertEqual(len(select_details([record]*4, 896, 896)), 2)

    def test_alias_risk_depends_on_rival_and_preserves_controls(self):
        wide = torch.tensor([[[0., 3., 1.], [0., 1., 2.], [0., -3., -2.]]]*3)
        fine = wide.clone()
        fine[:, 0, 1] = -.5
        pairs = torch.tensor([[0, 1], [0, 2], [0, 1]])
        known = torch.tensor([True, True, False])
        weights = sparse_weights(wide, fine, fine, pairs, torch.zeros(3, dtype=torch.long), known)
        self.assertLess(float(weights[PRIMARY][0, 0, 1]), 1.)
        self.assertEqual(float(weights[PRIMARY][1, 0, 1]), 1.)
        self.assertTrue(torch.equal(weights[PRIMARY][2], torch.ones_like(weights[PRIMARY][2])))
        self.assertTrue(torch.equal(weights[PRIMARY][..., 0], torch.ones(3, 2)))
        torch.testing.assert_close(weights[PRIMARY].sum(-1), weights[CLASS_MEAN].sum(-1), atol=1e-6, rtol=0)
        torch.testing.assert_close(weights[PRIMARY].sort(-1).values, weights[SHUFFLE].sort(-1).values, atol=0, rtol=0)

    def test_identity_and_exact_two_class_potential(self):
        evidence = torch.randn(4, 5, 3)
        pairs = torch.tensor([[0, 1], [1, 3], [3, 2], [4, 0]])
        valid = torch.tensor([True, True, True, False])
        weights = torch.rand(4, 2, 3)*.5+.5
        zero = pair_potential(evidence, pairs, torch.ones_like(weights), valid)
        self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))
        actual = pair_potential(evidence, pairs, weights, valid)
        selected = evidence[torch.arange(4)[:, None], pairs].double()
        delta = ((selected+weights.double().log()).logsumexp(-1)-selected.logsumexp(-1)
                 -(weights.double().sum(-1)/3).log())
        contrast = .5*(delta[:, 0]-delta[:, 1])
        expected = torch.zeros(4, 5, dtype=torch.float64)
        expected.scatter_(1, pairs, torch.stack((contrast, -contrast), -1))
        expected[~valid] = 0.
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
        torch.testing.assert_close(actual.sum(-1), torch.zeros(4, dtype=torch.float64), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
