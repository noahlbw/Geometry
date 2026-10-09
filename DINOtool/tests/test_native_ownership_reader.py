import unittest

import torch

from dinotool.native_ownership_reader import (PRIMARY, ALIAS_SHUFFLES, REFERENCE_SHUFFLES,
    query_permutation, source_controls)
from dinotool.native_class_ownership import risk_union


class NativeOwnershipReaderTest(unittest.TestCase):
    def fixture(self):
        valid = torch.tensor([True, True, False])
        field = torch.tensor([[[1., 4.]], [[4., 1.]], [[0., 0.]]], dtype=torch.float64)
        parents, canonical = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 2])
        conflict = torch.tensor([[0., 0.], [0., .6], [0., 0.], [.7, 0.]])
        members = torch.tensor([[0, 1], [2, 3]])
        view = torch.zeros(3, 4, 2)
        attachment = view.clone()
        attachment[0, 1, 1], attachment[1, 3, 0] = .5, .5
        return (view, attachment, field, torch.ones(1, 2, dtype=torch.bool), torch.zeros(4, dtype=torch.long),
            parents, canonical, conflict, valid, members, field, torch.ones(1, 2, dtype=torch.bool))

    def test_primary_preserves_unknown_and_canonical(self):
        args = self.fixture()
        output = source_controls(*args)
        torch.testing.assert_close(output[PRIMARY], args[1], atol=0, rtol=0)
        for risk in output.values():
            self.assertEqual(float(risk[:, args[6]].abs().max()), 0.)
            self.assertEqual(float(risk[~args[8]].abs().max()), 0.)
            self.assertTrue(bool(((risk >= 0) & (risk <= 1)).all()))

    def test_alias_controls_match_each_class_risk_spectrum(self):
        args = self.fixture()
        output = source_controls(*args)
        expected = output[PRIMARY][:, args[9]].sort(2).values
        for name in ALIAS_SHUFFLES:
            torch.testing.assert_close(output[name][:, args[9]].sort(2).values, expected, atol=0, rtol=0)

    def test_reference_controls_do_not_modify_input(self):
        args = self.fixture()
        original = args[2].clone()
        output = source_controls(*args)
        torch.testing.assert_close(args[2], original, atol=0, rtol=0)
        self.assertTrue(all(name in output for name in REFERENCE_SHUFFLES))

    def test_union_uses_unrounded_reference_risk_before_final_cast(self):
        args = list(self.fixture())
        args[0][0, 1, 1] = .137
        args[1] = args[1].double()
        args[1][0, 1, 1] = .123456789
        output = source_controls(*args)
        torch.testing.assert_close(output[PRIMARY], risk_union(args[0], args[1]), atol=0, rtol=0)

    def test_query_permutation_keeps_invalid_indices_and_is_bijective(self):
        valid = torch.tensor([True, False, True, True, False, True])
        a, b = query_permutation(valid, 20261003), query_permutation(valid, 20261003)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(a.sort().values, torch.arange(6), atol=0, rtol=0)
        torch.testing.assert_close(a[~valid], torch.arange(6)[~valid], atol=0, rtol=0)
