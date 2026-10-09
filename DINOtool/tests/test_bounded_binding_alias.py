import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from dinotool.bounded_binding_alias import binding_bank, binding_phrase, binding_weights, supported_features
from dinotool.bounded_canonical_pair_alias import sparse_pair_potential, weight_controls
from dinotool.bounded_patch_only import readout
from dinotool.geometry_execution import prepare_image_pruned
from dinotool.stratified_soft_alias import WideCrop
from test_geometry_execution import fixture


class BindingAliasTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261006)

    def test_binding_compares_same_word_under_both_owners(self):
        self.assertEqual(binding_phrase('wall', 'roof'), 'wall, described as roof')
        calls = []

        def encode(groups, templates):
            calls.append(groups)
            n, k = len(groups), len(groups[0])
            return F.normalize(torch.randn(n*k, 5), dim=-1), torch.arange(n).repeat_interleave(k), None

        geometry = SimpleNamespace(backbone=SimpleNamespace(encode_text_aliases=encode))
        bank = SimpleNamespace(class_names=('wall', 'roof'), alias_names=('wall', 'brick wall', 'roof', 'red roof'),
                               parent_indices=torch.tensor([0, 0, 1, 1]))
        first = binding_bank(geometry, bank)
        second = binding_bank(geometry, bank)
        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(first['directions'].shape), (2, 2, 2, 5))
        self.assertEqual(calls[0][0][1], 'wall, described as brick wall')
        self.assertEqual(calls[0][1][1], 'roof, described as brick wall')
        self.assertTrue(torch.equal(first['directions'][0, 0], torch.zeros(2, 5)))

    def test_real_head_native_batch_contract(self):
        model = fixture()
        prepared = prepare_image_pruned(model, torch.rand(1, 3, 32, 32))
        self.assertEqual(prepared.native_projected.shape[0], 1)
        native = prepared.native_projected[0]
        valid = torch.ones(len(native), dtype=torch.bool)
        support, known = supported_features(native, prepared.geometry_patch_conditional[0], valid)
        directions = torch.randn(3, 3, 4, native.shape[-1])
        result = binding_weights(native, support, directions, torch.tensor([[0, 1]]*4), torch.zeros(3, dtype=torch.long), known)
        self.assertEqual(tuple(result.shape), (4, 2, 4))
        self.assertEqual(readout(model.backbone.model.visual_model.head, prepared, 2.)[0].shape[:2], (1, 4))

    def test_geometry_support_excludes_padding_and_recovers_missing_support(self):
        native = torch.randn(5, 7)
        valid = torch.tensor([True, True, True, False, False])
        relation = torch.rand(5, 5)
        relation[2] = 0
        result, known = supported_features(native, relation, valid)
        expected = relation[0, :3]@native[:3]/relation[0, :3].sum()
        torch.testing.assert_close(result[0], expected)
        self.assertEqual(known.tolist(), [True, True, False, False, False])
        changed = native.clone()
        changed[3:] += 100
        torch.testing.assert_close(result, supported_features(changed, relation, valid)[0])

    def test_weights_match_explicit_direct_and_supported_comparisons(self):
        native, supported = torch.randn(71, 5), torch.randn(71, 5)
        directions = torch.randn(3, 3, 4, 5)
        pairs = torch.tensor([[i % 3, (i+1) % 3] for i in range(71)])
        canonical = torch.tensor([0, 1, 2])
        known = torch.ones(71, dtype=torch.bool)
        actual = binding_weights(native, supported, directions, pairs, canonical, known)
        expected = torch.empty_like(actual)
        for q in range(71):
            for side in range(2):
                own, rival = pairs[q, side], pairs[q, 1-side]
                delta = directions[own, rival]
                a, b = (delta@native[q]/.07).double(), (delta@supported[q]/.07).double()
                expected[q, side] = (1-torch.sigmoid(-a)*torch.sigmoid(-b)).clamp_min(1e-6)
                expected[q, side, canonical[own]] = 1.
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_identical_alias_can_differ_across_rivals(self):
        native = torch.tensor([[1., 0.], [1., 0.]])
        directions = torch.zeros(3, 3, 3, 2)
        directions[0, 1, 1, 0], directions[0, 2, 1, 0] = -1., 1.
        weights = binding_weights(native, native, directions, torch.tensor([[0, 1], [0, 2]]),
                                  torch.zeros(3, dtype=torch.long), torch.ones(2, dtype=torch.bool))
        self.assertLess(float(weights[0, 0, 1]), 1e-4)
        self.assertGreater(float(weights[1, 0, 1]), .999)
        self.assertTrue(torch.equal(weights[..., 0], torch.ones(2, 2, dtype=torch.float64)))

    def test_missing_support_or_equal_pair_is_identity(self):
        native = torch.randn(3, 4)
        pairs = torch.tensor([[0, 1], [0, 0], [1, 0]])
        actual = binding_weights(native, native, torch.randn(2, 2, 3, 4), pairs,
            torch.zeros(2, dtype=torch.long), torch.tensor([False, True, False]))
        self.assertTrue(torch.equal(actual, torch.ones_like(actual)))

    def test_canonical_only_has_no_noncanonical_word_identity(self):
        native, supported = torch.randn(9, 5), torch.randn(9, 5)
        directions = torch.randn(3, 3, 4, 5)
        pairs = torch.tensor([[i % 3, (i+1) % 3] for i in range(9)])
        canonical = torch.tensor([0, 1, 2])
        known = torch.ones(9, dtype=torch.bool)
        actual = binding_weights(native, supported, directions, pairs, canonical, known, canonical_only=True)
        for q in range(9):
            for side in range(2):
                ids = torch.arange(4) != canonical[pairs[q, side]]
                self.assertEqual(len(actual[q, side, ids].unique()), 1)

    def test_controls_preserve_class_mass_spectrum_and_canonical(self):
        native, supported = torch.randn(9, 5), torch.randn(9, 5)
        pairs = torch.tensor([[i % 3, (i+1) % 3] for i in range(9)])
        canonical = torch.tensor([0, 1, 2])
        actual = binding_weights(native, supported, torch.randn(3, 3, 4, 5), pairs, canonical,
                                 torch.ones(9, dtype=torch.bool))
        mean, shuffled = weight_controls(actual, pairs, canonical)
        torch.testing.assert_close(mean.sum(-1), actual.sum(-1), atol=1e-12, rtol=0)
        torch.testing.assert_close(shuffled.sort(-1).values, actual.sort(-1).values, atol=0, rtol=0)

    def test_identity_weights_leave_the_wide_observation_unchanged(self):
        members = torch.arange(9).reshape(3, 3)
        crop = WideCrop(torch.randn(4, 9), torch.randn(9), 0, 0, 2, 2, 2, 2)
        result = sparse_pair_potential([crop], torch.ones(2, 2), torch.tensor([[.5, .5], [1.5, 1.5]]),
            (2, 2), members, torch.tensor([[0, 1], [1, 2]]), torch.ones(2, 2, 3), torch.ones(2, dtype=torch.bool))
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))


if __name__ == '__main__':
    unittest.main()
