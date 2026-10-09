import unittest
from types import SimpleNamespace

import torch

from dinotool.finite_vip_observer import finite_proxy_attention


class FiniteProxyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.head = SimpleNamespace(patch_size=2)
        self.attention = SimpleNamespace(num_heads=2, qkv=torch.nn.Linear(4, 12),
                                        proj=torch.nn.Linear(4, 4), proj_drop=torch.nn.Identity())
        self.tokens = torch.randn(1, 9, 4)

    def test_identical_raw_features_read_self(self):
        output, empty, rows = finite_proxy_attention(self.head, self.attention, self.tokens, torch.ones(1, 4, 4))
        self.assertEqual((empty, rows), (4, 4))
        value = self.attention.qkv(self.tokens[:, 5:]).reshape(1, 4, 3, 2, 2)[:, :, 2].reshape(1, 4, 4)
        self.assertTrue(torch.allclose(output[:, 5:], self.attention.proj(value)))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.equal(output[:, :5], self.tokens[:, :5]))

    def test_nonempty_support_unchanged(self):
        output, empty, rows = finite_proxy_attention(self.head, self.attention, self.tokens, torch.eye(4)[None])
        self.assertEqual((empty, rows), (0, 4))
        value = self.attention.qkv(self.tokens[:, 5:]).reshape(1, 4, 3, 2, 2)[:, :, 2].reshape(1, 4, 4)
        self.assertTrue(torch.allclose(output[:, 5:], self.attention.proj(value)))


if __name__ == "__main__":
    unittest.main()
