import unittest
from types import SimpleNamespace

import torch

from dinotool.matched_neighborhood_controls import METHODS, run_neighborhood, spatial_logits
from dinotool.matched_readout_controls import run_head
from test_frozen_semantic_path import Block


class NeighborhoodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(113)
        self.head = SimpleNamespace(blocks=[Block(), Block()], ln_final=torch.nn.LayerNorm(4),
                                    linear_projection=torch.nn.Identity())
        self.prepared = SimpleNamespace(backbone_tokens=torch.randn(1, 6, 4), prefix_tokens=2,
            geometry_patch_conditional=torch.rand(1, 4, 4).softmax(-1), block_index=1,
            grid_height=2, grid_width=2)

    def test_original_geometry_is_identical(self):
        actual = run_neighborhood(self.head, self.prepared, "Geometry")
        expected, _ = run_head(self.head, self.prepared.backbone_tokens, torch.randn(1, 4, 4),
                              self.prepared.geometry_patch_conditional, 2, "Geometry")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_spatial_only_is_feature_independent(self):
        prior = spatial_logits(2, 2, device="cpu")
        self.assertTrue(bool((prior.diag() == 0).all()))
        torch.testing.assert_close(prior, prior.T)
        self.assertTrue(bool((prior <= 0).all()))

    def test_controls_are_distinct_and_do_not_mutate_head(self):
        before = self.prepared.backbone_tokens.clone()
        results = [run_neighborhood(self.head, self.prepared, method) for method in METHODS]
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in results))
        self.assertTrue(torch.equal(before, self.prepared.backbone_tokens))
        self.assertFalse(torch.allclose(results[0], results[1]))
        self.assertFalse(torch.allclose(results[2], results[3]))


if __name__ == "__main__":
    unittest.main()
