import unittest
from types import SimpleNamespace

import torch

from dinotool.matched_semantic_innovation import counterfactual_innovation, matched_proxy_features


class MatchedInnovationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.g, self.t, self.c = [torch.randn(1, 4, 3) for _ in range(3)]
        self.a = torch.rand(1, 4, 4).softmax(-1)
        self.valid = torch.ones(1, 4, dtype=torch.bool)

    def test_equal_heads_are_exact_identity(self):
        z, _ = counterfactual_innovation(self.g, self.c, self.c, self.a, self.valid)
        self.assertTrue(torch.allclose(z, self.g, atol=1e-6))

    def test_shared_class_specific_offsets_cancel(self):
        nuisance = torch.randn(1, 4, 3)*7
        z, _ = counterfactual_innovation(self.g, self.t, self.c, self.a, self.valid)
        shifted, _ = counterfactual_innovation(self.g, self.t+nuisance, self.c+nuisance, self.a, self.valid)
        self.assertTrue(torch.allclose(z, shifted, atol=1e-5))

    def test_local_prior_is_not_replaced(self):
        prior_shift = torch.randn_like(self.g)
        z, _ = counterfactual_innovation(self.g, self.t, self.c, self.a, self.valid)
        shifted, _ = counterfactual_innovation(self.g+prior_shift, self.t, self.c, self.a, self.valid)
        self.assertTrue(torch.allclose(shifted-z, prior_shift, atol=1e-6))

    def test_identity_relation_is_half_the_contrast(self):
        z, _ = counterfactual_innovation(self.g, self.t, self.c, torch.eye(4)[None], self.valid)
        self.assertTrue(torch.allclose(z, self.g+.5*(self.t-self.c), atol=1e-6))

    def test_matched_head_preserves_parameters_tokens_and_metadata(self):
        attention = SimpleNamespace(num_heads=2, qkv=torch.nn.Linear(4, 12),
                                    proj=torch.nn.Linear(4, 4), proj_drop=torch.nn.Identity())
        block = SimpleNamespace(attn=attention, norm1=torch.nn.Identity(), norm2=torch.nn.Identity(),
                                ls1=torch.nn.Identity(), ls2=torch.nn.Identity(), mlp=torch.nn.Linear(4, 4))
        head = SimpleNamespace(blocks=[block], ln_final=torch.nn.LayerNorm(4), linear_projection=torch.nn.Identity(), patch_size=27)
        prepared = SimpleNamespace(prefix_tokens=5, grid_height=2, grid_width=2,
                                   backbone_tokens=torch.randn(1, 9, 4), raw_patch_tokens=torch.ones(1, 4, 4))
        tokens = prepared.backbone_tokens.clone()
        weights = attention.qkv.weight.clone()
        output, counts = matched_proxy_features(head, prepared)
        self.assertEqual(head.patch_size, 27)
        self.assertTrue(torch.equal(tokens, prepared.backbone_tokens))
        self.assertTrue(torch.equal(weights, attention.qkv.weight))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(counts, {"empty_rows": 4, "observed_rows": 4})

    def test_metadata_restored_after_failure(self):
        prepared = SimpleNamespace(prefix_tokens=5, grid_height=2, grid_width=2,
                                   backbone_tokens=torch.randn(1, 9, 4), raw_patch_tokens=torch.ones(1, 4, 4))
        head = SimpleNamespace(blocks=[], linear_projection=torch.nn.Identity(),
                               ln_final=lambda value: (_ for _ in ()).throw(RuntimeError("test")))
        with self.assertRaises(RuntimeError):
            matched_proxy_features(head, prepared)
        self.assertFalse(hasattr(head, "patch_size"))


if __name__ == "__main__":
    unittest.main()
