from dataclasses import replace
import unittest

import torch

from dinotool.bounded_detail_projected_alias import native_detail_features, pair_field, project_pair_action
from dinotool.bounded_canonical_pair_alias import sparse_pair_potential
from dinotool.bounded_detail_pair_alias import sparse_weights, PRIMARY as WEIGHT_PRIMARY
from dinotool.geometry_execution import prepare_image_pruned
from dinotool.stratified_soft_alias import WideCrop
from test_geometry_execution import fixture


class DetailProjectedAliasTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(82)

    def test_native_only_features_are_exact_legacy_native_descriptors(self):
        model = fixture()
        state = {name: value.clone() for name, value in model.backbone.model.state_dict().items()}
        for depth in (1, 2):
            for prefix in ('preserve', 'block'):
                for index in (0, -1):
                    if depth == 2 and index == 0:
                        continue
                    model.config = replace(model.config, geometry_depth=depth, prefix_policy=prefix, head_block=index)
                    rgb = torch.rand(1, 3, 32, 32)
                    actual = native_detail_features(model, rgb)
                    expected = prepare_image_pruned(model, rgb).native_projected
                    self.assertTrue(torch.equal(actual, expected), (depth, prefix, index))
        self.assertTrue(all(torch.equal(state[name], value) for name, value in model.backbone.model.state_dict().items()))

    def test_native_only_path_encodes_once_and_refuses_training_head(self):
        model = fixture()
        original = model.backbone.model.visual_model.get_backbone_features
        calls = []
        def counted(rgb):
            calls.append(tuple(rgb.shape))
            return original(rgb)
        model.backbone.model.visual_model.get_backbone_features = counted
        actual = native_detail_features(model, torch.rand(1, 3, 32, 32))
        self.assertEqual(calls, [(1, 3, 32, 32)])
        self.assertEqual(tuple(actual.shape), (1, 4, 4))
        model.backbone.model.visual_model.head.train()
        with self.assertRaises(ValueError):
            native_detail_features(model, torch.rand(1, 3, 32, 32))

    def test_projection_rejects_reversal_and_clips_overshoot(self):
        requested = torch.tensor([8., -8., 2., -2., 0., 5., -4.], dtype=torch.float64)
        target = torch.tensor([3., -3., -4., 4., 2., 0., -6.], dtype=torch.float64)
        expected = torch.tensor([3., -3., 0., 0., 0., 0., -4.], dtype=torch.float64)
        actual = project_pair_action(requested, target)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(bool((actual.abs() <= target.abs()).all()))
        self.assertTrue(bool((actual*target >= 0).all()))
        self.assertTrue(torch.equal(project_pair_action(-requested, -target), -actual))

    def test_pair_gauge_and_class_permutation_equivariance(self):
        action = torch.randn(9, dtype=torch.float64)
        pairs = torch.tensor([[i % 4, (i+1) % 4] for i in range(9)])
        actual = pair_field(action, pairs, 4)
        torch.testing.assert_close(actual.sum(-1), torch.zeros(9, dtype=torch.float64), atol=0, rtol=0)
        torch.testing.assert_close(actual.gather(1, pairs[:, :1])[:, 0]-actual.gather(1, pairs[:, 1:])[:, 0], action, atol=0, rtol=0)
        permutation = torch.tensor([2, 0, 3, 1])
        inverse = permutation.argsort()
        permuted = pair_field(action, inverse[pairs], 4)
        torch.testing.assert_close(actual[:, permutation], permuted, atol=0, rtol=0)

    def test_sparse_h_write_matches_zero_padded_dense_write(self):
        operator = torch.randn(17, 17, dtype=torch.float64)
        ids = torch.tensor([1, 4, 5, 11, 15])
        pairs = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [1, 0]])
        values = pair_field(torch.randn(5, dtype=torch.float64), pairs, 3)
        padded = torch.zeros(17, 3, dtype=torch.float64)
        padded[ids] = values
        torch.testing.assert_close(operator[:, ids]@values, operator@padded, atol=1e-12, rtol=0)

    def test_identity_alias_weights_have_zero_projected_action(self):
        members = torch.arange(9).reshape(3, 3)
        crop = WideCrop(torch.randn(4, 9), torch.randn(9), 0, 0, 2, 2, 2, 2)
        pairs = torch.tensor([[0, 1], [1, 2]])
        valid = torch.ones(2, dtype=torch.bool)
        potential = sparse_pair_potential([crop], torch.ones(2, 2), torch.tensor([[.5, .5], [1.5, 1.5]]),
            (2, 2), members, pairs, torch.ones(2, 2, 3, dtype=torch.float64), valid)
        requested = potential.gather(1, pairs[:, :1])[:, 0]-potential.gather(1, pairs[:, 1:])[:, 0]
        action = project_pair_action(requested, torch.randn(2, dtype=torch.float64))
        self.assertTrue(torch.equal(action, torch.zeros_like(action)))

    def test_weight_rule_still_depends_on_rival_and_protects_canonical(self):
        wide = torch.tensor([[[0., 3., 1.], [0., 1., 2.], [0., -3., -2.]]]*2)
        fine = wide.clone()
        fine[:, 0, 1] = -.5
        pairs = torch.tensor([[0, 1], [0, 2]])
        weights = sparse_weights(wide, fine, fine, pairs, torch.zeros(3, dtype=torch.long), torch.ones(2, dtype=torch.bool))[WEIGHT_PRIMARY]
        self.assertLess(float(weights[0, 0, 1]), float(weights[1, 0, 1]))
        self.assertTrue(torch.equal(weights[..., 0], torch.ones(2, 2)))

    def test_h_transport_can_change_unobserved_output_rows(self):
        operator = torch.tensor([[.5, .3], [.3, .5]], dtype=torch.float64)
        action = pair_field(torch.tensor([2.], dtype=torch.float64), torch.tensor([[0, 1]]), 2)
        change = operator[:, :1]@action
        self.assertGreater(float(change[1].abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
