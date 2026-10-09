import unittest

import torch

from dinotool.native_class_ownership import family_class_field, ownership_risk, risk_union
from dinotool.stratified_soft_alias import WideCrop


class NativeClassOwnershipTest(unittest.TestCase):
    def observation(self):
        crop = WideCrop(torch.tensor([[2., 8., 1., 1., 8., 3.]]).expand(441, -1).clone(),
                        torch.tensor([0., 2., 0., 0., 2., 0.]), 0, 0, 336, 336)
        coords = torch.tensor([[8., 8.], [24., 24.]])
        groups = torch.tensor([[0, 1, 2], [3, 4, 5]])
        exclude = torch.tensor([[False, True, False, False, True, False]])
        valid = torch.tensor([True, False])
        return [crop], torch.ones(336, 336), coords, groups, exclude, valid

    def test_excluded_logits_and_salience_cannot_confirm_reference(self):
        args = list(self.observation())
        first = family_class_field(*args)[0]
        args[0][0].alias_logits[:, [1, 4]] = 10000
        args[0][0].salience[[1, 4]] = 10000
        torch.testing.assert_close(first, family_class_field(*args)[0], atol=0, rtol=0)

    def test_reference_matches_retained_normalized_score(self):
        field, known = family_class_field(*self.observation())
        expected = torch.tensor([2., 1.], dtype=torch.float64).logsumexp(0)-torch.tensor(2., dtype=torch.float64).log()
        self.assertAlmostEqual(float(field[0, 0, 0]), float(expected), places=12)
        self.assertTrue(bool(known.all()))
        self.assertEqual(float(field[1].abs().max()), 0.)

    def test_empty_class_reference_is_unknown(self):
        args = list(self.observation())
        args[4][0, :3] = True
        field, known = family_class_field(*args)
        self.assertFalse(bool(known[0, 0]))
        risk = ownership_risk(field, known, torch.zeros(6, dtype=torch.long),
                              torch.tensor([0, 0, 0, 1, 1, 1]), torch.tensor([0, 3]),
                              torch.ones(6, 2), args[-1])
        self.assertEqual(float(risk.abs().max()), 0.)

    def test_affirmative_rival_and_text_conflict_both_required(self):
        field = torch.tensor([[[1., 3.]], [[3., 1.]]], dtype=torch.float64)
        known = torch.ones(1, 2, dtype=torch.bool)
        ids, parents, canonical = torch.zeros(4, dtype=torch.long), torch.tensor([0, 0, 1, 1]), torch.tensor([0, 2])
        conflict = torch.tensor([[0., 1.], [0., .7], [1., 0.], [.6, 0.]])
        risk = ownership_risk(field, known, ids, parents, canonical, conflict, torch.ones(2, dtype=torch.bool))
        self.assertGreater(float(risk[0, 1, 1]), 0.)
        self.assertEqual(float(risk[1, 1, 1]), 0.)
        self.assertEqual(float(risk[:, canonical].abs().max()), 0.)
        torch.testing.assert_close(risk_union(torch.zeros_like(risk), risk), risk, atol=0, rtol=0)

    def test_common_class_offset_is_irrelevant(self):
        args = (torch.tensor([[[1., 3.]]]), torch.ones(1, 2, dtype=torch.bool), torch.zeros(4, dtype=torch.long),
                torch.tensor([0, 0, 1, 1]), torch.tensor([0, 2]), torch.ones(4, 2), torch.ones(1, dtype=torch.bool))
        a = ownership_risk(*args)
        b = ownership_risk(args[0]+100, *args[1:])
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_union_preserves_unknown_and_bounds(self):
        a, b = torch.tensor([0., .2, 1.], dtype=torch.float64), torch.tensor([.5, .4, .8], dtype=torch.float64)
        torch.testing.assert_close(risk_union(a, torch.zeros_like(a)), a, atol=0, rtol=0)
        self.assertTrue(bool(((risk_union(a, b) >= a) & (risk_union(a, b) <= 1)).all()))
