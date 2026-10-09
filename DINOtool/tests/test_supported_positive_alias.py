import unittest

import torch

from dinotool.bounded_fine_coverage import CLASS_MEAN as OLD_CLASS_MEAN, alias_controls
from dinotool.native_alias_noise import signed_potential
from dinotool.rival_alias_fast import CachedCrop, directed_cached
from dinotool.supported_positive_alias import (CLASS_MEAN, METHODS, NO_GEOMETRY_RISK,
    OBSERVATION_MEAN, PRIMARY, SHUFFLES, SHUFFLED_WRITE, WIDE_ONLY, cached_classes,
    class_only_risk, joint_scores, permuted_operator, supported_risk)


class SupportedPositiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(206)
        self.members = torch.arange(12).reshape(3, 4)
        self.parents = torch.arange(3).repeat_interleave(4)
        self.canonical = self.members[:, 0]
        self.valid = torch.tensor([True, True, True, False])
        self.coordinates = torch.randn(4, 2)
        self.relation = torch.eye(4)
        self.operator = torch.eye(4, dtype=torch.float64)*.5
        self.operator[~self.valid] = 0
        self.operator[:, ~self.valid] = 0
        self.local, self.broad, self.field = (torch.randn(4, 3) for _ in range(3))
        self.wide_margin = torch.ones(4, 12, 3)
        self.fine_margin = -torch.ones_like(self.wide_margin)
        indices = torch.arange(4)[:, None]
        coefficients = torch.ones(4, 1)
        self.wide = [CachedCrop(torch.randn(4, 3, 4), self.wide_margin, indices, coefficients)]
        self.fine = [CachedCrop(torch.randn(4, 3, 4), self.fine_margin, indices, coefficients)]

    def scores(self, methods=METHODS[3:]):
        return joint_scores(self.local, self.operator, self.broad, self.field,
            self.wide, self.fine, self.coordinates, self.relation, self.valid,
            self.members, self.canonical, self.parents, methods=methods)

    def test_conditional_risk_protects_self_canonical_and_padding(self):
        risk = supported_risk(self.wide_margin, self.fine_margin, self.relation,
                              self.valid, self.parents, self.canonical)
        self.assertTrue(torch.equal(risk[:, self.canonical], torch.zeros(4, 3, 3)))
        self.assertTrue(torch.equal(risk[~self.valid], torch.zeros(1, 12, 3)))
        self.assertTrue(torch.equal(risk.gather(-1, self.parents[None, :, None].expand(4, -1, 1)),
                                   torch.zeros(4, 12, 1)))
        self.assertGreater(float(risk.sum()), 0)

    def test_same_alias_can_survive_one_rival_and_not_another(self):
        fine = self.fine_margin.clone()
        fine[:, 1, 2] = 1
        risk = supported_risk(self.wide_margin, fine, self.relation, self.valid,
                              self.parents, self.canonical)
        self.assertGreater(float(risk[0, 1, 1]), 0)
        self.assertEqual(float(risk[0, 1, 2]), 0)

    def test_supported_contradiction_is_required(self):
        fine = self.fine_margin.clone()
        fine[1] = 1
        relation = self.relation.clone()
        relation[0] = 0
        relation[0, 1] = 1
        risk = supported_risk(self.wide_margin, fine, relation, self.valid, self.parents, self.canonical)
        self.assertEqual(float(risk[0].sum()), 0)
        ungrounded = supported_risk(self.wide_margin, fine, relation, self.valid,
                                    self.parents, self.canonical, use_geometry=False)
        self.assertGreater(float(ungrounded[0].sum()), 0)

    def test_class_control_preserves_pairwise_survivor_mass(self):
        risk = supported_risk(self.wide_margin, self.fine_margin, self.relation,
                              self.valid, self.parents, self.canonical)
        changed = class_only_risk(risk, self.members, self.canonical)
        expected = alias_controls(risk, self.members, self.canonical)[OLD_CLASS_MEAN]
        torch.testing.assert_close(1-changed[:, self.members], expected, atol=0, rtol=0)
        torch.testing.assert_close((1-changed[:, self.members]).sum(2),
                                    (risk[:, self.members] == 0).double().sum(2), atol=1e-12, rtol=0)

    def test_class_control_supports_noncontiguous_groups(self):
        groups = self.members[:, torch.tensor([1, 3, 0, 2])].T.reshape(3, 4)
        canonical = groups[:, 0]
        risk = torch.rand(4, 12, 3)
        risk[risk < .5] = 0
        risk[:, canonical] = 0
        changed = class_only_risk(risk, groups, canonical)
        expected = alias_controls(risk, groups, canonical)[OLD_CLASS_MEAN]
        torch.testing.assert_close(1-changed[:, groups], expected, atol=1e-12, rtol=0)

    def test_positive_and_intervention_write_are_source_aligned(self):
        values, _ = self.scores()
        baseline = self.local.double()+self.operator@(self.broad.double()-self.local.double())
        innovation = .5*(self.field.double()-self.broad.double())
        innovation[~self.valid] = 0
        torch.testing.assert_close(values[OBSERVATION_MEAN], baseline+self.operator@innovation, atol=0, rtol=0)
        torch.testing.assert_close(values[PRIMARY]-values[OBSERVATION_MEAN],
                                    .5*(values[WIDE_ONLY]-baseline), atol=1e-12, rtol=0)

    def test_no_contradiction_is_exact_same_information_identity(self):
        self.fine[0].margins.copy_(self.wide_margin)
        values, stats = self.scores()
        self.assertTrue(torch.equal(values[PRIMARY], values[OBSERVATION_MEAN]))
        self.assertEqual(stats['mean_absolute_admission_potential'], 0)

    def test_singleton_scores_match_simultaneous_controls(self):
        combined, _ = self.scores()
        for method in METHODS[3:]:
            single, _ = self.scores((method,))
            self.assertTrue(torch.equal(single[method], combined[method]), method)

    def test_invalid_action_and_class_gauge_are_zero(self):
        risk = supported_risk(self.wide_margin, self.fine_margin, self.relation,
                              self.valid, self.parents, self.canonical)
        potential, stats = signed_potential(directed_cached(self.wide, self.members, risk, self.valid), self.valid)
        self.assertTrue(torch.equal(potential[~self.valid], torch.zeros(1, 3, dtype=torch.float64)))
        self.assertLessEqual(stats['gauge_max_error'], 1e-12)

    def test_write_permutation_preserves_spectrum_and_padding(self):
        first = permuted_operator(self.operator, self.valid)
        second = permuted_operator(self.operator, self.valid)
        self.assertTrue(torch.equal(first, second))
        torch.testing.assert_close(torch.linalg.eigvalsh(first), torch.linalg.eigvalsh(self.operator))
        self.assertTrue(torch.equal(first[~self.valid], torch.zeros(1, 4, dtype=torch.float64)))

    def test_cached_positive_source_is_exact_stencil_assembly(self):
        actual = cached_classes(self.fine, self.coordinates)
        expected = self.fine[0].evidence.logsumexp(-1)
        self.assertTrue(torch.equal(actual, expected))

    def test_invalid_method_is_rejected(self):
        with self.assertRaises(ValueError):
            self.scores(('UnfrozenRule',))


if __name__ == '__main__':
    unittest.main()
