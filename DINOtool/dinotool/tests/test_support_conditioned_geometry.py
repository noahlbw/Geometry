from __future__ import annotations

from dataclasses import replace
import math
import unittest

import torch
import torch.nn.functional as F

from dinotool.gear_ov import GearObservations, class_scores
from dinotool.support_conditioned_geometry import (
    SupportConditionedReadout, SupportConfig, SupportObservation, SupportRegion,
    choose_regions, consensus_states, crop_geometry, family_excluded_margins,
    family_exclusion_layout, partition_supports, signed_replacement, transport_crop_features,
)
from dinotool.tcpr import TCPRTextBank


class SupportConditionedTests(unittest.TestCase):
    def test_family_reference_matches_explicit_holdout(self):
        scores = torch.tensor([[2.0, 1.0, -0.5, 1.5, -1.0, 0.25],
                               [1e4, 1e4 - 1, 1e4 - 2, 1e4 + 1, 1e4, 1e4 - 3]])
        parents = torch.tensor([0, 0, 0, 1, 1, 1])
        families = [[0, 3], [1, 2], [4], [5]]
        margins, identifiable = family_excluded_margins(scores, parents, families, 2)
        self.assertTrue(bool(identifiable.all()))
        self.assertTrue(bool(torch.isfinite(margins).all()))
        for family in families:
            group = [[a for a in range(6) if a not in family and parents[a] == c]
                     for c in range(2)]
            heldout = torch.stack([torch.logsumexp(scores[:, ids], -1) - math.log(len(ids))
                                  for ids in group], -1)
            for alias in family:
                own = int(parents[alias])
                self.assertTrue(torch.allclose(margins[:, alias],
                                               heldout[:, own] - heldout[:, 1 - own]))

    def test_extinguished_class_is_unresolved_not_a_fallback_teacher(self):
        margins, available = family_excluded_margins(
            torch.tensor([[10., 0., 1.]]), torch.tensor([0, 1, 1]), [[0], [1], [2]], 2)
        self.assertFalse(bool(available[0]))
        plus, minus = consensus_states(margins[0], margins[0], available)
        self.assertEqual(float(plus[0] + minus[0]), 0)

    def test_three_states_do_not_force_acceptance_on_disagreement(self):
        plus, minus = consensus_states(torch.tensor([2., -3., 2., 0.]),
                                       torch.tensor([1., -2., -1., 4.]),
                                       torch.ones(4, dtype=torch.bool))
        self.assertAlmostEqual(float(plus[0]), math.tanh(1), places=6)
        self.assertAlmostEqual(float(minus[1]), math.tanh(2), places=6)
        self.assertEqual(float(plus[2] + minus[2] + plus[3] + minus[3]), 0)
        self.assertTrue(bool((plus * minus == 0).all()))

    def test_signed_replacement_is_convex_directional_and_support_limited(self):
        original = torch.tensor([[[1., 1., 1., 1.], [1., 1., 1., 1.]]])
        proposed = torch.tensor([[[3., -1., -2., 4.], [3., -1., -2., 4.]]])
        plus, minus = torch.tensor([.5, 0., 1., 0.]), torch.tensor([0., .5, 0., 1.])
        final = signed_replacement(original, proposed, plus, minus, torch.tensor([[True, False]]))
        self.assertTrue(torch.equal(final[0, 0], torch.tensor([2., 0., 1., 1.])))
        self.assertTrue(torch.equal(final[0, 1], original[0, 1]))
        self.assertTrue(bool(((final >= torch.minimum(original, proposed))
                              & (final <= torch.maximum(original, proposed))).all()))

    def test_negative_alias_evidence_cannot_increase_parent_logit(self):
        parents = torch.tensor([0, 0, 1, 1])
        original = torch.tensor([[[2., -2., 1., 0.]]])
        final = signed_replacement(original, original - 2, torch.zeros(4),
                                   torch.tensor([0., 1., 0., 0.]), torch.tensor([[True]]))
        before = class_scores(original, parents, 2)
        after = class_scores(final, parents, 2)
        self.assertLess(float(after[0, 0, 0]), float(before[0, 0, 0]))
        self.assertEqual(float(after[0, 0, 1]), float(before[0, 0, 1]))

    def test_crop_encloses_actual_support_not_fixed_quadrant(self):
        mask = torch.zeros(64, 64, dtype=torch.bool)
        mask[5:8, 41:43] = True
        top, left, extent = crop_geometry(mask, SupportConfig())
        self.assertLessEqual(top, 5 * 8)
        self.assertGreaterEqual(top + extent, 8 * 8)
        self.assertLessEqual(left, 41 * 8)
        self.assertGreaterEqual(left + extent, 43 * 8)
        self.assertEqual(extent % 16, 0)
        self.assertLess(extent, 256)

    def test_region_budget_class_diversity_and_padding(self):
        config = replace(SupportConfig(), maximum_regions=2, minimum_region_patches=1)
        labels = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1],
                               [2, 2, 3, 3], [-1, -1, 3, 3]])
        logits = torch.zeros(4, 4, 2)
        logits[labels == 2] = torch.tensor([0., .03])
        logits[labels == 0] = torch.tensor([.01, 0.])
        logits[labels == 1] = torch.tensor([.02, 0.])
        logits[labels == 3] = torch.tensor([.5, 0.])
        regions = choose_regions(labels, logits, config)
        self.assertEqual([r.label for r in regions], [0, 2])
        self.assertFalse(bool(torch.stack([r.mask for r in regions]).any(0)[labels < 0].any()))

    def test_raw_partition_excludes_padding(self):
        raw = torch.zeros(16, 16, 4)
        raw[:, :8, 0], raw[:, 8:, 1] = 1, 1
        valid = torch.ones(16, 16, dtype=torch.bool)
        valid[-4:] = False
        labels = partition_supports(raw, valid, replace(SupportConfig(), segments=8))
        self.assertTrue(bool((labels[~valid] == -1).all()))
        self.assertTrue(bool((labels[valid] >= 0).all()))

    def test_transport_never_uses_off_support_or_absent_donors(self):
        raw = F.normalize(torch.randn(8, 8, 4, generator=torch.Generator().manual_seed(3)), dim=-1)
        mask = torch.zeros(8, 8, dtype=torch.bool)
        mask[2:6, 2:6] = True
        region = SupportRegion(0, mask, 0, 0, 64)
        aligned = torch.zeros(8, 8, 2)
        aligned[..., 1] = 1
        aligned[mask] = torch.tensor([1., 0.])
        proposal, usable = transport_crop_features(raw, region, raw, aligned, (64, 64), SupportConfig())
        self.assertTrue(torch.equal(usable, mask))
        self.assertTrue(torch.allclose(proposal[mask], torch.tensor([1., 0.]).expand(16, -1)))
        self.assertTrue(bool((proposal[~mask] == 0).all()))
        outside = SupportRegion(0, mask, 1000, 1000, 64)
        absent, usable = transport_crop_features(raw, outside, raw, aligned, (64, 64), SupportConfig())
        self.assertFalse(bool(usable.any()))
        self.assertTrue(bool((absent == 0).all()))

    @staticmethod
    def readout_fixture():
        readout = object.__new__(SupportConditionedReadout)
        readout.config = SupportConfig()
        readout.bank = TCPRTextBank(torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]]),
                                   torch.tensor([0, 0, 1, 1]), torch.tensor([True, False, True, False]),
                                   ("a", "b"), ("a1", "a2", "b1", "b2"))
        readout.families = [[0], [1], [2], [3]]
        readout.family_layout = family_exclusion_layout(readout.bank.parent_indices,
                                                         readout.families, 2)
        readout.full_text = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.],
                                          [0., 1., 0., 0.], [0., 1., 0., 0.]])
        features = torch.tensor([1., 0.]).expand(64, 64, 2).clone()
        raw = torch.zeros(64, 64, 2)
        views = GearObservations(features[::2, ::2], features, features[::4, ::4],
                                 raw, raw[::4, ::4], 512, 512)
        baseline = torch.tensor([1., 0.]).view(1, 2, 1, 1).expand(1, 2, 512, 512).clone()
        mask = torch.zeros(64, 64, dtype=torch.bool)
        mask[8:16, 8:16] = True
        proposed = torch.zeros_like(features)
        proposed[mask] = torch.tensor([0., 1.])
        observation = SupportObservation(SupportRegion(0, mask, 48, 48, 96), proposed, mask,
                                         torch.tensor([0., 1., 0., 0.]),
                                         torch.tensor([0., 1., 0., 0.]))
        return readout, views, baseline, observation

    def test_no_observation_and_unresolved_are_exact_multiscale(self):
        readout, views, baseline, observation = self.readout_fixture()
        final, _ = readout.solve(views, baseline, [])
        self.assertIs(final, baseline)
        unknown = replace(observation, tight_global=torch.zeros(4), wide_global=torch.zeros(4))
        final, diagnostic = readout.solve(views, baseline, [unknown])
        self.assertTrue(torch.equal(final, baseline))
        self.assertEqual(diagnostic["resolved_alias_fraction"], 0)

    def test_same_verifier_changes_both_competing_sides_and_not_whole_quadrant(self):
        readout, views, baseline, observation = self.readout_fixture()
        final, diagnostic = readout.solve(views, baseline, [observation])
        self.assertLess(float(final[0, 0, 96, 96]), float(baseline[0, 0, 96, 96]))
        self.assertGreater(float(final[0, 1, 96, 96]), float(baseline[0, 1, 96, 96]))
        self.assertTrue(torch.equal(final[..., 300:, 300:], baseline[..., 300:, 300:]))
        self.assertGreater(diagnostic["positive_replacement_mean"], 0)
        self.assertGreater(diagnostic["negative_replacement_mean"], 0)


if __name__ == "__main__":
    unittest.main()
