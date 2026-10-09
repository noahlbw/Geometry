import unittest
from types import SimpleNamespace

import torch

from dinotool.bounded_crop_head_alias import (PRIMARY, crop_head_features, quadrant_indices,
    soft_pair_delta, supported_margins, weight_controls)
from dinotool.matched_contribution_alias import contribution_margins
from dinotool.stratified_soft_alias import WideCrop
from test_frozen_semantic_path import Block


class CropHeadAliasTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)

    def test_quadrants_partition_without_new_pixels(self):
        ids = quadrant_indices(32, 32, 'cpu')
        self.assertEqual(tuple(ids.shape), (4, 256))
        torch.testing.assert_close(ids.flatten().sort().values, torch.arange(1024), atol=0, rtol=0)
        with self.assertRaises(ValueError):
            quadrant_indices(21, 21, 'cpu')

    def test_support_factorization_matches_dense_alias_rival_pooling(self):
        evidence = torch.randn(5, 3, 4)
        relation = torch.randn(5, 5).softmax(-1)
        valid = torch.tensor([True, True, True, True, False])
        native, actual, _, known = supported_margins(evidence, relation, valid)
        weights = relation.masked_fill(~valid[None], 0.)
        weights = weights / weights.sum(-1, keepdim=True)
        weights[~valid] = 0.
        expected = (weights @ contribution_margins(evidence).flatten(1)).reshape_as(actual)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(native, contribution_margins(evidence), atol=0, rtol=0)
        self.assertTrue(torch.equal(known, valid))

    def test_soft_identity_and_canonical_preserving_spectrum(self):
        risk = torch.rand(3, 6, 2)
        canonical = torch.tensor([0, 3])
        risk[:, canonical] = 0.
        weights = weight_controls(risk, canonical, 2, 3)
        self.assertTrue(torch.equal(weights[PRIMARY][:, canonical], torch.ones(3, 2, 2)))
        torch.testing.assert_close(weights[PRIMARY].reshape(3, 2, 3, 2).sort(2).values,
            weights['CropHead_AliasShuffle'].reshape(3, 2, 3, 2).sort(2).values, atol=0, rtol=0)
        torch.testing.assert_close(weights[PRIMARY].reshape(3, 2, 3, 2).sum(2),
            weights['CropHead_ClassMean'].reshape(3, 2, 3, 2).sum(2), atol=1e-6, rtol=0)

    def test_weighted_action_matches_direct_logmeanexp(self):
        crop = WideCrop(torch.randn(441, 4), torch.randn(4), 0, 0, 336, 336)
        members = torch.tensor([[0, 1], [2, 3]])
        coordinates = torch.tensor([[80., 80.], [160., 160.]])
        count = torch.ones(336, 336)
        valid = torch.ones(2, dtype=torch.bool)
        weights = torch.rand(2, 4, 2) * .5 + .5
        from dinotool.calibrated_competitive_alias import profiled_logits
        from dinotool.stratified_soft_alias import crop_stencil
        ids, coeff = crop_stencil(crop, count, coordinates, (336, 336))
        evidence = profiled_logits(crop, members)[ids].double()
        grouped = weights.reshape(2, 2, 2, 2).double()
        direct = (evidence[..., None] + grouped[:, None].log()).logsumexp(3)
        direct -= evidence.logsumexp(-1)[..., None]
        direct -= grouped.mean(2).log()[:, None]
        expected = (direct * coeff.double()[..., None, None]).sum(1)
        expected.diagonal(dim1=-2, dim2=-1).zero_()
        torch.testing.assert_close(soft_pair_delta([crop], count, coordinates, (336, 336), members, weights, valid),
            expected, atol=1e-12, rtol=1e-12)
        unchanged = soft_pair_delta([crop], count, coordinates, (336, 336), members, torch.ones_like(weights), valid)
        self.assertTrue(torch.equal(unchanged, torch.zeros_like(unchanged)))

    def test_crop_head_finite_and_masks_invalid_keys(self):
        head = SimpleNamespace(blocks=[Block(), Block()], ln_final=torch.nn.LayerNorm(4),
                               linear_projection=torch.nn.Identity())
        tokens = torch.randn(1, 1026, 4)
        valid = torch.ones(1024, dtype=torch.bool)
        valid[1::2] = False
        prepared = SimpleNamespace(backbone_tokens=tokens, prefix_tokens=2, grid_height=32, grid_width=32, block_index=1)
        actual, ids = crop_head_features(head, prepared, valid)
        changed = tokens.clone()
        changed[:, 2:][:, ~valid] += 100.
        new, _ = crop_head_features(head, SimpleNamespace(**{**vars(prepared), 'backbone_tokens': changed}), valid)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual[valid[ids]], new[valid[ids]], atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.equal(actual[~valid[ids]], torch.zeros_like(actual[~valid[ids]])))


if __name__ == '__main__':
    unittest.main()
