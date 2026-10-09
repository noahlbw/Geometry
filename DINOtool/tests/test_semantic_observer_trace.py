from types import SimpleNamespace
import unittest

import torch
from torch import nn

from dinotool.semantic_observer_trace import matched_vip_features, nonself_supported


def fixture():
    block = nn.Module()
    block.norm1, block.norm2 = nn.LayerNorm(8), nn.LayerNorm(8)
    block.attn = nn.Identity()
    block.ls1, block.ls2 = nn.Identity(), nn.Identity()
    block.mlp = nn.Linear(8, 8)
    head = nn.Module()
    head.blocks = nn.ModuleList([block])
    head.ln_final, head.linear_projection = nn.LayerNorm(8), nn.Identity()
    prepared = SimpleNamespace(prefix_tokens=5, grid_height=2, grid_width=2,
                               backbone_tokens=torch.randn(1, 9, 8),
                               raw_patch_tokens=torch.randn(1, 4, 8))
    return head, prepared


class SemanticObserverTraceTest(unittest.TestCase):
    def test_matched_readout_preserves_original_runtime(self):
        head, prepared = fixture()
        head.patch_size = 21
        state = {name: tensor.clone() for name, tensor in head.state_dict().items()}
        tokens = prepared.backbone_tokens.clone()
        output = matched_vip_features(head, prepared, lambda h, a, x, r: x)
        self.assertEqual(head.patch_size, 21)
        self.assertEqual(output.shape, (1, 4, 8))
        torch.testing.assert_close(output.norm(dim=-1), torch.ones(1, 4))
        self.assertTrue(torch.equal(tokens, prepared.backbone_tokens))
        self.assertTrue(all(torch.equal(state[name], tensor) for name, tensor in head.state_dict().items()))

    def test_metadata_restored_on_proxy_error(self):
        head, prepared = fixture()
        def fail(*args):
            raise RuntimeError("fixture error")
        with self.assertRaises(RuntimeError):
            matched_vip_features(head, prepared, fail)
        self.assertFalse(hasattr(head, "patch_size"))

    def test_supported_scores_exclude_self_and_invalid_donors(self):
        scores = torch.tensor([[[1.], [2.], [5.]]])
        relation = torch.ones(1, 3, 3)
        supported = nonself_supported(scores, relation, torch.tensor([[True, False, True]]))
        torch.testing.assert_close(supported, torch.tensor([[[5.], [3.], [1.]]]))

    def test_no_donors_returns_original_score(self):
        scores = torch.randn(1, 3, 2)
        supported = nonself_supported(scores, torch.eye(3)[None], torch.ones(1, 3, dtype=torch.bool))
        self.assertTrue(torch.equal(scores, supported))


if __name__ == "__main__":
    unittest.main()
