import unittest
from types import SimpleNamespace

import torch

from dinotool.bounded_patch_only import readout
from dinotool.matched_readout_controls import run_head
from dinotool.tcpr import _geometry_attended
from test_frozen_semantic_path import Block


class PatchOnlyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(409)
        self.tokens = torch.randn(1, 6, 4)
        self.g = torch.randn(1, 4, 4).softmax(-1)
        self.head = SimpleNamespace(blocks=[Block(), Block()], ln_final=torch.nn.LayerNorm(4),
                                    linear_projection=torch.nn.Identity())
        self.prepared = SimpleNamespace(backbone_tokens=self.tokens, prefix_tokens=2,
                                        geometry_patch_conditional=self.g, block_index=1)

    def test_unit_strength_is_exact_existing_block_prefix(self):
        expected = run_head(self.head, self.tokens, self.tokens[:, 2:], self.g, 2, 'Geometry_BlockPrefix', 1)[0]
        torch.testing.assert_close(readout(self.head, self.prepared, 1.)[0], expected, atol=0, rtol=0)

    def test_double_read_only_changes_patch_queries(self):
        native = torch.randn(1, 2, 6, 6).softmax(-1)
        values = torch.randn(1, 2, 6, 2)
        a = _geometry_attended(native, self.g, values, 2, 'block')
        b = _geometry_attended(native, self.g * 2, values, 2, 'block')
        torch.testing.assert_close(b[..., :2, :], a[..., :2, :], atol=0, rtol=0)
        torch.testing.assert_close(b[..., 2:, :], a[..., 2:, :] * 2, atol=0, rtol=0)

    def test_finite_outputs_and_original_relation_unmodified(self):
        original = self.g.clone()
        for strength in (1., 2., 'SCLIP_Two'):
            self.assertTrue(torch.isfinite(readout(self.head, self.prepared, strength)[0]).all())
        self.assertTrue(torch.equal(self.g, original))

    def test_undeclared_strength_rejected(self):
        with self.assertRaises(ValueError):
            readout(self.head, self.prepared, 1.5)


if __name__ == '__main__':
    unittest.main()
