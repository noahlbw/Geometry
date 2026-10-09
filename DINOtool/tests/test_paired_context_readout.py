import unittest
from types import SimpleNamespace

import torch

from dinotool.finite_vip_observer import finite_proxy_attention
from dinotool.geometry_semantic_innovation import anchored_innovation
from dinotool.paired_context_readout import footprint_overlap, patch_boxes, restricted_proxy_attention, transported_support


class PairedContextTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.head = SimpleNamespace(patch_size=2)
        self.attention = SimpleNamespace(qkv=torch.nn.Linear(8, 24), num_heads=2,
                                         proj=torch.nn.Linear(8, 8), proj_drop=torch.nn.Identity())
        self.tokens, self.raw = torch.randn(1, 9, 8), torch.randn(1, 4, 8)

    def test_full_support_exactly_replays_guarded_vip(self):
        old, _, _ = finite_proxy_attention(self.head, self.attention, self.tokens, self.raw)
        new, _ = restricted_proxy_attention(self.head, self.attention, self.tokens, self.raw,
                                             torch.ones(4, 4, dtype=torch.bool))
        self.assertTrue(torch.equal(old, new))

    def test_empty_support_is_finite_self_read_and_preserves_prefix(self):
        out, empty = restricted_proxy_attention(self.head, self.attention, self.tokens, self.raw,
                                                 torch.zeros(4, 4, dtype=torch.bool))
        value = self.attention.qkv(self.tokens[:, 5:]).reshape(1, 4, 3, 2, 4)[:, :, 2].reshape(1, 4, 8)
        self.assertEqual(empty, 4)
        self.assertTrue(torch.equal(out[:, :5], self.tokens[:, :5]))
        self.assertTrue(torch.allclose(out[:, 5:], self.attention.proj(value), atol=1e-6))

    def test_support_inputs_not_mutated(self):
        support = torch.eye(4, dtype=torch.bool)
        originals = [v.clone() for v in (self.tokens, self.raw, support)]
        restricted_proxy_attention(self.head, self.attention, self.tokens, self.raw, support)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(originals, (self.tokens, self.raw, support))))

    def test_fractional_patch_footprints_and_padding(self):
        broad = patch_boxes(1, 2, 0, 0, 2, 2, "cpu")
        fine = patch_boxes(1, 4, 0, 0, 2, 1, "cpu")
        overlap = footprint_overlap(broad, fine, 2, 3)
        self.assertTrue(torch.equal(overlap, torch.tensor([[2., 2., 0., 0.], [0., 0., 2., 0.]])))

    def test_identity_geometry_maps_to_identity_support(self):
        support, mapped = transported_support(torch.eye(4)[None], torch.ones(1, 4, dtype=torch.bool), torch.eye(4))
        self.assertTrue(torch.equal(support, torch.eye(4, dtype=torch.bool)))
        self.assertTrue(bool(mapped.all()))

    def test_unmapped_queries_keep_full_support_and_padding_donors_are_excluded(self):
        overlap = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        support, mapped = transported_support(torch.eye(3)[None], torch.tensor([[True, True, False]]), overlap)
        self.assertFalse(bool(mapped[-1]))
        self.assertTrue(bool(support[-1].all()))
        self.assertFalse(bool(support[0, -1]))

    def test_zero_context_exactly_recovers_geometry(self):
        g = torch.randn(1, 4, 3)
        out, _ = anchored_innovation(g, g, torch.rand(1, 4, 4), torch.ones(1, 4, dtype=torch.bool))
        self.assertTrue(torch.equal(out, g))

    def test_identity_writeback_recovers_same_information_half_update(self):
        g, delta = torch.randn(1, 4, 3), torch.randn(1, 4, 3)
        out, _ = anchored_innovation(g, g+delta, torch.eye(4)[None], torch.ones(1, 4, dtype=torch.bool))
        self.assertTrue(torch.allclose(out, g+.5*delta, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
