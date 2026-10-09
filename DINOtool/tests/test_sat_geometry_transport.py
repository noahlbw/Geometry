import unittest
from types import SimpleNamespace

import torch

from dinotool.sat_geometry_transport import correspondence_weights, native_patch_states, paired_transport


class SATTransportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261002)
        self.source = torch.randn(1, 16, 5)
        self.valid = torch.ones(1, 16, dtype=torch.bool)

    def test_same_source_recovers_native_exactly(self):
        result, _ = paired_transport(self.source, self.source, self.valid.float(), self.valid)
        torch.testing.assert_close(result, self.source, atol=0, rtol=0)

    def test_head_inputs_use_native_states_not_unit_descriptors(self):
        tokens = torch.randn(1, 19, 5)*9
        prepared = SimpleNamespace(backbone_tokens=tokens, prefix_tokens=3,
                                   raw_patch_tokens=torch.nn.functional.normalize(tokens[:, 3:], dim=-1))
        actual = native_patch_states(prepared)
        torch.testing.assert_close(actual, tokens[:, 3:], atol=0, rtol=0)
        self.assertFalse(torch.equal(actual, prepared.raw_patch_tokens))

    def test_missing_backbone_states_are_rejected(self):
        with self.assertRaises(ValueError):
            native_patch_states(SimpleNamespace(backbone_tokens=None, prefix_tokens=3))

    def test_recovers_exact_rotation_and_offset(self):
        rotation = torch.linalg.qr(torch.randn(5, 5))[0]
        target = self.source @ rotation+torch.arange(5.)
        result, stats = paired_transport(self.source, target, self.valid.float(), self.valid)
        torch.testing.assert_close(result, target, atol=2e-6, rtol=2e-6)
        self.assertLess(stats["alignment_error_after"], 1e-10)

    def test_weighted_fit_reduces_the_declared_objective(self):
        target = torch.randn_like(self.source)
        weights = torch.arange(1., 17)[None]
        _, stats = paired_transport(self.source, target, weights, self.valid)
        self.assertLessEqual(stats["alignment_error_after"], stats["alignment_error_before"]+1e-6)
        self.assertLess(stats["rotation_orthogonality_error"], 2e-6)

    def test_alignment_restores_native_tf32_setting(self):
        before = torch.backends.cuda.matmul.allow_tf32
        paired_transport(self.source, torch.randn_like(self.source), self.valid.float(), self.valid)
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, before)

    def test_invalid_patches_and_padding_remain_native(self):
        valid = self.valid.clone()
        valid[:, 2:5] = False
        target = torch.randn_like(self.source)
        result, _ = paired_transport(self.source, target, valid.float(), valid)
        torch.testing.assert_close(result[:, 2:5], target[:, 2:5], atol=0, rtol=0)

    def test_geometry_balancing_is_image_valid_and_mean_one(self):
        relation = torch.tensor([[[1., 0., 0.], [1., 0., 0.], [0., 0., 1.]]])
        valid = torch.tensor([[True, True, False]])
        weights = correspondence_weights(relation, valid)
        self.assertEqual(float(weights[0, 2]), 0.)
        self.assertAlmostEqual(float(weights[valid].mean()), 1., places=5)
        self.assertGreater(float(weights[0, 1]), float(weights[0, 0]))

    def test_uniform_geometry_returns_uniform_weights(self):
        relation = torch.ones(1, 16, 16)/16
        torch.testing.assert_close(correspondence_weights(relation, self.valid), self.valid.float())

    def test_correspondence_and_transport_are_permutation_equivariant(self):
        relation = torch.rand(1, 16, 16).softmax(-1)
        order = torch.randperm(16)
        weights = correspondence_weights(relation, self.valid)
        shuffled = correspondence_weights(relation[:, order][:, :, order], self.valid[:, order])
        torch.testing.assert_close(shuffled, weights[:, order], atol=2e-6, rtol=2e-6)
        target = torch.randn_like(self.source)
        actual, _ = paired_transport(self.source, target, weights, self.valid)
        other, _ = paired_transport(self.source[:, order], target[:, order], shuffled, self.valid[:, order])
        torch.testing.assert_close(other, actual[:, order], atol=3e-6, rtol=3e-6)

    def test_empty_image_does_not_invent_features(self):
        valid = torch.zeros_like(self.valid)
        target = torch.randn_like(self.source)
        result, _ = paired_transport(self.source, target, valid.float(), valid)
        torch.testing.assert_close(result, target)


if __name__ == "__main__":
    unittest.main()
