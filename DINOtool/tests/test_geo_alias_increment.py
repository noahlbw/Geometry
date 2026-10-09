import unittest

import torch

from dinotool.geo_alias_increment import (CONFIG, PRIMARY, fixed_profile, increment_predictions,
    max_excess_delta, text_discrimination, visual_discrimination, witness_fields)
from dinotool.stratified_soft_alias import WideCrop


class GeoAliasIncrementTest(unittest.TestCase):
    def test_fixed_profile_prefix(self):
        members = torch.arange(80).reshape(2, 40)
        crop = WideCrop(torch.arange(80).float()[None], torch.linspace(-1, 1, 80), 0, 0, 512, 512)
        full = fixed_profile(crop, members)
        ids = members[:, :20].flatten()
        base_crop = WideCrop(crop.alias_logits[:, ids], crop.salience[ids], 0, 0, 512, 512)
        base = fixed_profile(base_crop, torch.arange(40).reshape(2, 20))
        self.assertTrue(torch.equal(full[..., :20], base))

    def test_uniform20_identity(self):
        evidence = torch.randn(5, 3, 20)
        weights = torch.ones(5, 3, 20, 3)
        delta = max_excess_delta(evidence, weights, torch.tensor([0, 3, 7]))
        self.assertEqual(float(delta.abs().max()), 0.)

    def test_zero_addition_and_duplicate_invariance(self):
        evidence = torch.randn(4, 2, 20)
        weights = torch.rand(4, 2, 20, 2)
        positions = torch.tensor([0, 2])
        base = max_excess_delta(evidence, weights, positions)
        extra = torch.cat((evidence, torch.full((4, 2, 10), 10000.)), -1)
        off = torch.cat((weights, torch.zeros(4, 2, 10, 2)), -2)
        self.assertTrue(torch.equal(base, max_excess_delta(extra, off, positions)))
        duplicate = torch.cat((evidence, evidence[..., 4:5]), -1)
        repeated = torch.cat((weights, weights[..., 4:5, :]), -2)
        self.assertTrue(torch.equal(base, max_excess_delta(duplicate, repeated, positions)))

    def test_text_shared_or_rival_alias_has_zero_ownership(self):
        text = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [1., 0.], [0., 1.], [1., 1.]])
        members, canonical = torch.tensor([[0, 1, 2], [3, 4, 5]]), torch.tensor([0, 4])
        gate = text_discrimination(text, members, canonical)
        self.assertEqual(float(gate[0, 1, 1]), 0.)
        self.assertEqual(float(gate[0, 2, 1]), 0.)
        self.assertEqual(float(gate[0, 0, 1]), 1.)

    def test_reference_selection_does_not_use_tested_alias(self):
        raw = torch.tensor([[1., 0., .3], [1., 0., .2], [0., 1., .4], [0., 1., .5]])
        relation, canonical = torch.ones(4, 4), torch.tensor([0, 1])
        valid = torch.ones(4, dtype=torch.bool)
        before = witness_fields(relation, raw[:, canonical], raw, canonical, valid)
        changed = raw.clone()
        changed[:, 2] = 10000.
        after = witness_fields(relation, raw[:, canonical], changed, canonical, valid)
        self.assertTrue(torch.equal(before[1], after[1]))
        self.assertTrue(torch.equal(before[2], after[2]))
        self.assertTrue(torch.equal(before[0][:, :, canonical], after[0][:, :, canonical]))

    def test_shared_visual_response_is_not_discriminative(self):
        members, canonical = torch.tensor([[0, 1], [2, 3]]), torch.tensor([0, 2])
        means = torch.tensor([[[1., .8, 0., .8], [0., .8, 1., .8]]])
        gate, known = visual_discrimination(means, torch.ones(1, 2), members, canonical, torch.tensor([True]))
        self.assertTrue(bool(known[0, 0, 1]))
        self.assertEqual(float(gate[0, 0, 1, 1]), 0.)
        self.assertEqual(float(gate[0, 0, 0, 1]), 1.)

    def test_unknown_pairs_abstain_and_shuffle_is_matched(self):
        members = torch.arange(40).reshape(2, 20)
        crop = WideCrop(torch.randn(1024, 40), torch.randn(40), 0, 0, 512, 512, 32, 512)
        coords, valid = torch.tensor([[8., 8.], [504., 504.]]), torch.tensor([True, False])
        weights = torch.rand(2, 2, 20, 2)
        values, _, diag = increment_predictions(torch.ones(2, 2), torch.eye(2), [crop], torch.ones(512, 512),
            coords, (512, 512), members, members[:, 0], torch.ones(2, 20, 2), weights, torch.ones_like(weights),
            torch.zeros(2, 2, 2, dtype=torch.bool), valid)
        self.assertEqual(diag['weight_shuffle_spectrum_error'], 0.)
        self.assertTrue(torch.equal(values[PRIMARY], torch.ones(2, 2, dtype=torch.float64)))


if __name__ == '__main__':
    unittest.main()
