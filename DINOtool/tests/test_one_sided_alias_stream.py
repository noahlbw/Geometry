import unittest
import torch

from dinotool.one_sided_alias_stream import fixed_slot_stream
from dinotool.reciprocal_alias_admission import fixed_slot_action
from dinotool.rival_alias_fast import CachedCrop


class StreamTests(unittest.TestCase):
    def data(self, classes=5):
        torch.manual_seed(1008)
        q, k = 19, 20
        members = torch.randperm(classes*k).reshape(classes, k)
        valid = torch.arange(q) < q-2
        risk = torch.rand(q, classes*k, classes)
        risk[:, members[:, 0]] = 0
        risk[~valid] = 0
        crops = []
        for _ in range(3):
            coefficients = torch.rand(q, 4)
            coefficients /= coefficients.sum(1, keepdim=True)*3
            crops.append(CachedCrop(torch.randn(29, classes, k)*5, torch.empty(0),
                                   torch.randint(29, (q, 4)), coefficients))
        return crops, members, risk, valid

    def test_multiple_crops_stencils_alias_order_and_blocks(self):
        for classes in (2, 7, 19):
            data = self.data(classes)
            expected, _ = fixed_slot_action(*data, chunk=3)
            for q, r in ((3, 2), (1024, 16)):
                actual, _ = fixed_slot_stream(*data, query_chunk=q, rival_chunk=r)
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
                self.assertTrue(torch.equal(actual[~data[3]], torch.zeros_like(actual[~data[3]])))

    def test_zero_action_and_no_survivor(self):
        crops, members, risk, valid = self.data()
        actual, _ = fixed_slot_stream(crops, members, torch.zeros_like(risk), valid)
        self.assertEqual(float(actual.abs().max()), 0)
        with self.assertRaises(ValueError):
            fixed_slot_stream(crops, members, torch.ones_like(risk), valid)

    def test_log_domain_fallback_preserves_extreme_logits(self):
        crops, members, risk, valid = self.data()
        members = torch.arange(members.numel()).reshape_as(members)
        risk.zero_()
        risk[:, members[0, 1:], 1] = 1
        for crop in crops:
            crop.evidence[:, 0, 0] = -2000
            crop.evidence[:, 0, 1:] = 2000
        expected, old = fixed_slot_action(crops, members, risk, valid, chunk=3)
        actual, new = fixed_slot_stream(crops, members, risk, valid, query_chunk=7, rival_chunk=2)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
        self.assertEqual(old['mass_log_fallbacks'], new['mass_log_fallbacks'])
        self.assertGreater(new['mass_log_fallbacks'], 0)


if __name__ == '__main__':
    unittest.main()
