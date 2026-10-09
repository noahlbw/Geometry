import math
import unittest

import torch

from dinotool.matched_contribution_alias import contribution_margins, matched_reversal, stencil_margins
from dinotool.pair_context_reader import pair_observation
from dinotool.stratified_soft_alias import WideCrop


class MatchedContributionTest(unittest.TestCase):
    def source(self):
        full = torch.ones(4, 6, 2)
        kept = -torch.ones(2, 4, 6, 2)
        removed = torch.ones_like(kept)*3
        parents, canonical = torch.tensor([0, 0, 0, 1, 1, 1]), torch.tensor([0, 3])
        known = torch.tensor([True, True, True, False])
        return full, kept, removed, parents, canonical, known

    def test_alias_margin_recovers_actual_class_margin(self):
        x = torch.randn(11, 3, 5, generator=torch.Generator().manual_seed(81), dtype=torch.float64)
        for beta in (1., .5, 2.):
            m = contribution_margins(x, beta).reshape(11, 3, 5, 3)
            actual = (beta*x).logsumexp(-1)/beta
            reconstructed = (beta*m).logsumexp(2)/beta-math.log(5)/beta
            torch.testing.assert_close(reconstructed, actual[..., None]-actual[:, None], atol=1e-14, rtol=0)

    def test_common_profiled_shift_is_neutral(self):
        x = torch.randn(7, 3, 4, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
        torch.testing.assert_close(contribution_margins(x+9), contribution_margins(x), atol=1e-14, rtol=0)

    def test_reversal_strength_and_protection(self):
        args = self.source()
        risk = matched_reversal(*args)
        self.assertEqual(float(risk[0, 1, 1]), .25)
        self.assertEqual(float(risk[:, args[4]].abs().max()), 0.)
        self.assertEqual(float(risk[-1].abs().max()), 0.)
        self.assertEqual(float(risk[0, 1, 0]), 0.)

    def test_both_fills_must_accept_reversal(self):
        args = list(self.source())
        args[1][1] = 2
        self.assertEqual(float(matched_reversal(*args).abs().max()), 0.)

    def test_full_advantage_and_removed_advantage_are_required(self):
        for field in (0, 2):
            args = list(self.source())
            args[field].zero_()
            self.assertEqual(float(matched_reversal(*args).abs().max()), 0.)

    def test_class_and_alias_permutation(self):
        args = self.source()
        original = matched_reversal(*args)
        slots = torch.tensor([3, 5, 4, 0, 2, 1])
        swapped = [args[0][:, slots][:, :, [1, 0]], args[1][:, :, slots][:, :, :, [1, 0]],
                   args[2][:, :, slots][:, :, :, [1, 0]], 1-args[3][slots], torch.tensor([0, 3]), args[5]]
        torch.testing.assert_close(matched_reversal(*swapped), original[:, slots][:, :, [1, 0]], atol=0, rtol=0)

    def test_stencil_follows_aggregation_then_interpolation(self):
        x = torch.tensor([[[2., -1.], [0., 1.]], [[-2., 3.], [4., -3.]]], dtype=torch.float64)
        ids, coeff = torch.tensor([[0, 1]]), torch.tensor([[.2, .8]], dtype=torch.float64)
        result = stencil_margins(x, ids, coeff)
        expected = contribution_margins(x)[0]*.2+contribution_margins(x)[1]*.8
        torch.testing.assert_close(result[0], expected, atol=0, rtol=0)
        wrong = contribution_margins((x[0]*.2+x[1]*.8)[None])
        self.assertGreater(float((result-wrong).abs().max()), .1)

    def test_zero_risk_uses_existing_identity_writer(self):
        crops = [WideCrop(torch.randn(441, 6), torch.zeros(6), 0, 0, 336, 336)]
        coordinates = torch.tensor([[8., 8.], [24., 24.]])
        members = torch.tensor([[0, 1, 2], [3, 4, 5]])
        known = torch.ones(2, dtype=torch.bool)
        _, _, directed = pair_observation(crops, torch.ones(336, 336), coordinates,
            (336, 336), members, torch.zeros(2, 6, 2), known)
        self.assertEqual(float(directed.abs().max()), 0.)

    def test_unknown_and_nonreversal_leave_evidence_unchanged(self):
        args = list(self.source())
        args[5].zero_()
        self.assertEqual(float(matched_reversal(*args).abs().max()), 0.)
        args = list(self.source())
        args[1].fill_(1.)
        self.assertEqual(float(matched_reversal(*args).abs().max()), 0.)

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            contribution_margins(torch.ones(2, 1, 3))
        args = list(self.source())
        args[1][0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            matched_reversal(*args)
        with self.assertRaises(ValueError):
            stencil_margins(torch.ones(2, 2, 3), torch.tensor([[0, 1]]), torch.tensor([[1., -1.]]))
