import itertools
import unittest

import torch
import torch.nn.functional as F

from dinotool.alias_action_capacity import (action_envelope, changed_class_prediction, evidence_caps,
    label_assisted_action, reconstruction_operator, target_margin_upper)
from dinotool.excess_alias_rejection import excess_delta


class AliasActionCapacityTest(unittest.TestCase):
    def test_caps_match_existing_writer(self):
        logits = torch.tensor([[[2., .7, -.3], [.1, .5, -.2]]])
        canonical = torch.tensor([[True, False, False], [False, True, False]])
        single, protected, all_words = evidence_caps(logits, canonical)
        for a in range(3):
            rho = torch.ones_like(logits)
            rho[..., a] = 0.
            torch.testing.assert_close(single[..., a], -excess_delta(logits, rho))
        torch.testing.assert_close(protected, -excess_delta(logits, canonical.float()[None]))
        torch.testing.assert_close(all_words, -excess_delta(logits, torch.zeros_like(logits)))

    def test_equal_evidence_has_no_action(self):
        values = evidence_caps(torch.full((2, 3, 20), 8.), torch.eye(3, 20, dtype=torch.bool))
        self.assertTrue(all(torch.equal(value, torch.zeros_like(value)) for value in values))

    def test_protected_canonical_limit(self):
        logits = torch.tensor([[[8., -8.]]])
        single, protected, all_words = evidence_caps(logits, torch.tensor([[True, False]]))
        self.assertEqual(float(protected[0, 0]), 0.)
        self.assertGreater(float(all_words[0, 0]), .6)
        self.assertEqual(float(single[0, 0, 1]), 0.)

    def test_identity_reconstruction_is_half(self):
        operator, residual = reconstruction_operator(torch.eye(3), torch.tensor([True, True, True]))
        torch.testing.assert_close(operator, torch.eye(3, dtype=torch.float64)*.5)
        self.assertLess(residual, 1e-12)

    def test_padding_is_inert(self):
        relation = torch.ones(3, 3)
        operator, _ = reconstruction_operator(relation, torch.tensor([True, True, False]))
        self.assertTrue(torch.equal(operator[2], torch.zeros(3, dtype=torch.float64)))
        self.assertTrue(torch.equal(operator[:, 2], torch.zeros(3, dtype=torch.float64)))
        small, _ = reconstruction_operator(relation[:2, :2], torch.tensor([True, True]))
        torch.testing.assert_close(operator[:2, :2], small)

    def test_box_envelope_exact_signed_corners(self):
        baseline = torch.tensor([[.1, .3], [.2, -.1]], dtype=torch.float64)
        operator = torch.tensor([[.4, -.1], [-.1, .3]], dtype=torch.float64)
        cap = torch.tensor([[.2, .5], [.6, .1]], dtype=torch.float64)
        values = torch.stack([baseline + operator @ (-cap*torch.tensor(bits).reshape(2, 2))
                             for bits in itertools.product((0, 1), repeat=4)])
        low, high = action_envelope(baseline, operator, cap)
        torch.testing.assert_close(low, values.amin(0))
        torch.testing.assert_close(high, values.amax(0))
        target = torch.tensor([0, 1])
        margin = target_margin_upper(low, high, target)
        possible = torch.stack([v.gather(1, target[:, None])[:, 0] - v.gather(1, (1-target)[:, None])[:, 0]
                                for v in values]).amax(0)
        torch.testing.assert_close(margin, possible)

    def test_zero_cap_reduces_to_original_margin(self):
        baseline = torch.tensor([[1., 2., 3.], [5., 2., 1.]])
        low, high = action_envelope(baseline, torch.eye(2, dtype=torch.float64), torch.zeros_like(baseline))
        torch.testing.assert_close(target_margin_upper(low, high, torch.tensor([2, 1])),
                                   torch.tensor([1., -3.], dtype=torch.float64))

    def test_label_policy_is_one_feasible_action_not_bound(self):
        cap = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        action = label_assisted_action(cap, torch.tensor([0, 1, -1]))
        torch.testing.assert_close(action, torch.tensor([[0., -2.], [-3., 0.], [0., 0.]]))
        self.assertTrue(bool(((action >= -cap) & (action <= 0)).all()))

    def test_changed_class_argmax_ties(self):
        baseline = torch.tensor([[1., 1., 0.], [1., 2., 2.]])
        for c in range(3):
            scores = torch.tensor([[1., 2.], [0., 3.]])
            expected = []
            for plane in scores:
                changed = baseline.clone()
                changed[:, c] = plane
                expected.append(changed.argmax(-1))
            torch.testing.assert_close(changed_class_prediction(baseline, scores, c), torch.stack(expected))

    def test_interpolation_is_linear_for_action(self):
        baseline = torch.randn(1, 3, 2, 2, dtype=torch.float64)
        action = torch.randn_like(baseline)
        resize = lambda value: F.interpolate(value, (8, 8), mode='bilinear', align_corners=False)
        torch.testing.assert_close(resize(baseline+action), resize(baseline)+resize(action))


if __name__ == '__main__':
    unittest.main()
