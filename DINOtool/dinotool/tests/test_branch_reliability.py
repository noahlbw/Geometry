from __future__ import annotations

import unittest

import torch

from dinotool.branch_reliability import BranchReliabilitySelector, pair_margin
from dinotool.tcpr import TCPRTextBank


def bank(features=None):
    if features is None:
        features = torch.eye(4)
    return TCPRTextBank(features, torch.tensor([0, 0, 1, 1]),
                        torch.tensor([True, False, True, False]),
                        ("a", "b"), ("a", "a2", "b", "b2"))


class BranchReliabilityTests(unittest.TestCase):
    def test_normalized_reference(self):
        scores = torch.tensor([[0.2, 0.2, 0.4, 0.4]])
        own = torch.tensor([[False, False, True, True]])
        rival = ~own
        self.assertAlmostEqual(float(pair_margin(scores, own, rival, 0.07)), 0.2, places=6)

    def observations(self, height=3, width=3):
        g = torch.tensor([0.6, 0.1, 0.0, 0.0]).expand(height, width, 4).clone()
        v = torch.tensor([0.0, 0.0, 0.6, 0.4]).expand(height, width, 4).clone()
        raw = torch.ones(height, width, 4)
        gl = torch.zeros(height, width, dtype=torch.long)
        vl = torch.ones_like(gl)
        valid = torch.ones_like(gl, dtype=torch.bool)
        return g, v, raw, gl, vl, valid

    def test_supported_opposing_reference(self):
        decisions, counts = BranchReliabilitySelector(bank()).select(*self.observations())
        self.assertTrue(bool(decisions["LeaveFamilyOut"].all()))
        self.assertTrue(bool(decisions["GroundedLeaveFamilyOut"].all()))
        self.assertEqual(counts["grounded_proposed"], 9)

    def test_agreement_and_padding_do_not_trigger(self):
        g, v, raw, gl, vl, valid = self.observations()
        vl[0] = 0
        valid[1] = False
        decisions, _ = BranchReliabilitySelector(bank()).select(g, v, raw, gl, vl, valid)
        for decision in decisions.values():
            self.assertFalse(bool(decision[:2].any()))

    def test_no_neighbor_does_not_authorize_grounded_route(self):
        decisions, _ = BranchReliabilitySelector(bank()).select(*self.observations(1, 1))
        self.assertTrue(bool(decisions["LeaveFamilyOut"].any()))
        self.assertFalse(bool(decisions["GroundedLeaveFamilyOut"].any()))

    def test_empty_family_reference_is_unknown(self):
        features = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        decisions, counts = BranchReliabilitySelector(bank(features)).select(*self.observations())
        self.assertEqual(counts["identifiable_disagreements"], 0)
        self.assertFalse(any(bool(d.any()) for d in decisions.values()))


if __name__ == "__main__":
    unittest.main()
