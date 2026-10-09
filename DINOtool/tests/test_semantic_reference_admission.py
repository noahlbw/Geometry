from dataclasses import replace
import unittest

import torch

from dinotool.native_class_ownership import family_class_field
from dinotool.semantic_reference_admission import PRIMARY, SHUFFLES, weighted_family_field, source_controls
from dinotool.stratified_soft_alias import WideCrop


class SemanticReferenceAdmissionTest(unittest.TestCase):
    def fixture(self):
        crop = WideCrop(torch.tensor([[2., 8., 1., 1., 8., 3.]]).expand(441, -1).clone(),
            torch.tensor([0., 2., 0., 0., 2., 0.]), 0, 0, 336, 336)
        coordinates = torch.tensor([[8., 8.], [24., 24.]])
        members = torch.tensor([[0, 1, 2], [3, 4, 5]])
        excluded = torch.tensor([[False, True, False, False, True, False]])
        valid = torch.tensor([True, False])
        return [crop], torch.ones(336, 336), coordinates, members, excluded, valid

    def test_all_retained_is_exact_prior_reference(self):
        args = self.fixture()
        before = family_class_field(*args)
        after = weighted_family_field(*args, torch.ones(6))
        for a, b in zip(before, after):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_weighted_score_and_empty_unknown(self):
        args = self.fixture()
        weights = torch.tensor([1., 1., .5, 0., 0., 0.], dtype=torch.float64)
        field, known = weighted_family_field(*args, weights)
        expected = torch.tensor([2., 1.+torch.log(torch.tensor(.5, dtype=torch.float64))]).logsumexp(0)-torch.log(torch.tensor(1.5, dtype=torch.float64))
        self.assertAlmostEqual(float(field[0, 0, 0]), float(expected), places=12)
        self.assertFalse(bool(known[0, 1]))
        self.assertEqual(float(field[:, :, 1].abs().max()), 0.)
        self.assertEqual(float(field[1].abs().max()), 0.)

    def test_held_family_cannot_confirm_its_reference(self):
        args = list(self.fixture())
        weights = torch.tensor([1., .2, .5, 1., .2, .5])
        original = weighted_family_field(*args, weights)[0]
        args[0] = [replace(c, alias_logits=c.alias_logits.clone(), salience=c.salience.clone()) for c in args[0]]
        args[0][0].alias_logits[:, [1, 4]] = 10000
        args[0][0].salience[[1, 4]] = 10000
        torch.testing.assert_close(original, weighted_family_field(*args, weights)[0], atol=0, rtol=0)

    def source_fixture(self):
        args = self.fixture()
        parents, canonical = torch.tensor([0, 0, 0, 1, 1, 1]), torch.tensor([0, 3])
        semantic = torch.tensor([[0., 0.], [0., .8], [0., .3], [0., 0.], [.7, 0.], [.2, 0.]], dtype=torch.float64)
        return (*args, torch.zeros(6, dtype=torch.long), parents, canonical, semantic,
            semantic.clone(), torch.zeros(2, 6, 2))

    def test_no_semantic_conflict_is_exact_class_view_base(self):
        args = list(self.source_fixture())
        args[9] = torch.zeros_like(args[9])
        output, fields = source_controls(*args)
        torch.testing.assert_close(output[PRIMARY], fields['old_base'], atol=0, rtol=0)
        self.assertEqual(float(fields['extra'].abs().max()), 0.)

    def test_bounded_sources_keep_assignment_spectra(self):
        args = self.source_fixture()
        output, fields = source_controls(*args)
        for risk in output.values():
            self.assertTrue(bool(((risk >= 0) & (risk <= 1)).all()))
            self.assertEqual(float(risk[~args[5]].abs().max()), 0.)
        self.assertEqual(set(fields['semantic_spectrum_errors']), set(SHUFFLES))
        self.assertEqual(max(fields['semantic_spectrum_errors'].values()), 0.)
        self.assertEqual(float(fields['extra'][:, args[8]].abs().max()), 0.)

    def test_invalid_admissibility_rejected(self):
        with self.assertRaises(ValueError):
            weighted_family_field(*self.fixture(), torch.full((6,), 1.01))


if __name__ == '__main__':
    unittest.main()
