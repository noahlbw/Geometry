import math
import unittest
from dataclasses import replace

import torch

from dinotool.geometry_readout_trace import alias_class_scores
from dinotool.pair_conditional_alias import PairAliasConfig, PairConditionalAliases
from dinotool.tcpr import TCPRTextBank


def example_bank():
    features = torch.tensor([[1., 0., 0.], [1., 1., .1], [1., .1, 1.], [1., 1., 1.],
                             [0., 1., 0.], [1., 1., .1], [.1, 1., 1.], [1., 1., 1.],
                             [0., 0., 1.], [1., .1, 1.], [.1, 1., 1.], [1., 1., 1.]])
    return TCPRTextBank(features, torch.arange(3).repeat_interleave(4),
                       torch.tensor([True, False, False, False] * 3), ("A", "B", "C"),
                       ("A", "AB", "AC", "ABC", "B", "BA", "BC", "BCA", "C", "CA", "CB", "CAB"))


class PairAliasTest(unittest.TestCase):
    def setUp(self):
        self.bank = example_bank()
        self.reader = PairConditionalAliases(self.bank)

    def test_alias_depends_on_rival(self):
        self.assertFalse(bool(self.reader.selected[0][1, 1]))
        self.assertTrue(bool(self.reader.selected[0][2, 1]))
        self.assertTrue(bool(self.reader.selected[0][1, 2]))
        self.assertFalse(bool(self.reader.selected[0][2, 2]))

    def test_random_control_matches_counts_and_canonical(self):
        for selected, random in zip(self.reader.selected, self.reader.random):
            self.assertTrue(torch.equal(selected.sum(-1), random.sum(-1)))
            self.assertTrue(bool(selected[:, 0].all() and random[:, 0].all()))
        other = PairConditionalAliases(self.bank)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(self.reader.random, other.random)))

    def test_no_global_deletion(self):
        self.assertTrue(all(bool(mask.diagonal()[0]) for mask in self.reader.selected))
        self.assertTrue(bool(self.reader.selected[0][:, 1].any()))
        self.assertEqual(len(self.bank.alias_names), 12)

    def test_manual_pair_lme_and_other_classes_unchanged(self):
        scores = torch.tensor([[[.3, .8, .2, .1, .45, .1, .1, .1, -2., -2., -2., -2.]]])
        original = alias_class_scores(scores, self.bank.parent_indices, 3)
        result, _ = self.reader.read(scores)
        pair = original.topk(2, -1).indices[0, 0].tolist()
        self.assertEqual(set(pair), {0, 1})
        for c, d in (pair, pair[::-1]):
            selected = scores[..., self.reader.members[c]][..., self.reader.selected[c][d]]
            expected = .07 * (torch.logsumexp(selected / .07, -1) - math.log(selected.shape[-1]))
            self.assertTrue(torch.equal(result["selected"][..., c], expected))
        self.assertTrue(torch.equal(result["selected"][..., 2], original[..., 2]))
        self.assertEqual(int(original.argmax(-1)), 0)
        self.assertEqual(int(result["selected"].argmax(-1)), 1)

    def test_count_offset_is_explicit(self):
        torch.manual_seed(1)
        scores = torch.randn(1, 20, 12)
        original = alias_class_scores(scores, self.bank.parent_indices, 3)
        result, _ = self.reader.read(scores)
        for i, pair in enumerate(original.topk(2, -1).indices[0].tolist()):
            for c, d in (pair, pair[::-1]):
                count = int(self.reader.selected[c][d].sum())
                expected = .07 * math.log(4 / count)
                self.assertAlmostEqual(float(result["selected"][0, i, c] - result["fixed_count"][0, i, c]), expected, places=6)

    def test_padding_unchanged(self):
        scores = torch.randn(1, 6, 12)
        valid = torch.tensor([[True, True, False, True, False, False]])
        original = alias_class_scores(scores, self.bank.parent_indices, 3)
        result, _ = self.reader.read(scores, valid)
        self.assertTrue(all(torch.equal(value[~valid], original[~valid]) for value in result.values()))

    def test_all_retained_exact_identity(self):
        bank = example_bank()
        bank = replace(bank, features=torch.eye(3).repeat_interleave(4, 0))
        reader = PairConditionalAliases(bank)
        scores = torch.randn(2, 4, 12)
        original = alias_class_scores(scores, bank.parent_indices, 3)
        result, _ = reader.read(scores)
        self.assertTrue(all(torch.equal(value, original) for value in result.values()))

    def test_alias_permutation(self):
        permutation = torch.tensor([3, 1, 8, 2, 10, 5, 4, 9, 0, 6, 7, 11])
        bank = TCPRTextBank(self.bank.features[permutation], self.bank.parent_indices[permutation],
                           self.bank.canonical_mask[permutation], self.bank.class_names,
                           tuple(self.bank.alias_names[i] for i in permutation))
        scores = torch.randn(1, 7, 12)
        left, _ = self.reader.read(scores)
        right, _ = PairConditionalAliases(bank).read(scores[..., permutation])
        self.assertTrue(torch.allclose(left["selected"], right["selected"], atol=1e-6))

    def test_common_score_shift(self):
        scores = torch.randn(1, 7, 12)
        left, _ = self.reader.read(scores)
        right, _ = self.reader.read(scores + 3)
        self.assertTrue(all(torch.allclose(left[key] + 3, value, atol=1e-6) for key, value in right.items()))

    def test_identical_canonicals_abstain(self):
        bank = example_bank()
        bank.features[4] = bank.features[0]
        reader = PairConditionalAliases(bank)
        self.assertTrue(bool(reader.selected[0][1].all() and reader.selected[1][0].all()))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            PairConditionalAliases(self.bank, PairAliasConfig(canonical_contrast_fraction=2))
        with self.assertRaises(ValueError):
            self.reader.read(torch.full((1, 12), float("nan")))


if __name__ == "__main__":
    unittest.main()
