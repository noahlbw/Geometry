import unittest
from types import SimpleNamespace

import torch
from torch import nn

from dinotool.anchored_infra import cast_anchored_visual_weights


class AnchoredInfraTests(unittest.TestCase):
    def test_bfloat16_cast_deduplicates_shared_backbone_and_keeps_norm(self):
        shared = nn.Linear(8, 8)
        local_head = nn.Sequential(nn.LayerNorm(8))
        wide_head = nn.Sequential(nn.LayerNorm(8))

        class Visual(nn.Module):
            def __init__(self, backbone, head):
                super().__init__()
                self.backbone = backbone
                self.head = head

        local_visual = Visual(shared, local_head)
        wide_visual = Visual(shared, wide_head)
        model = SimpleNamespace(
            geometry=SimpleNamespace(backbone=SimpleNamespace(
                model=SimpleNamespace(visual_model=local_visual))),
            vip=SimpleNamespace(backbone=SimpleNamespace(
                model=SimpleNamespace(visual_model=wide_visual))),
        )

        stats = cast_anchored_visual_weights(model, 'bfloat16')
        self.assertEqual(stats['weight_dtype'], 'bfloat16')
        self.assertEqual(stats['visual_modules'], 2)
        self.assertEqual(shared.weight.dtype, torch.bfloat16)
        self.assertEqual(shared.bias.dtype, torch.bfloat16)
        self.assertEqual(local_head[0].weight.dtype, torch.float32)
        self.assertEqual(wide_head[0].weight.dtype, torch.float32)
        # The shared Linear is counted once even though both visual modules
        # reference it; each independent LayerNorm has its own FP32 storage.
        self.assertEqual(stats['parameters'], 2)

    def test_none_is_a_noop(self):
        visual = nn.Linear(4, 4)
        model = SimpleNamespace(
            geometry=SimpleNamespace(backbone=SimpleNamespace(
                model=SimpleNamespace(visual_model=visual))),
            vip=SimpleNamespace(backbone=SimpleNamespace(
                model=SimpleNamespace(visual_model=visual))),
        )
        before = visual.weight.detach().clone()
        stats = cast_anchored_visual_weights(model, 'none')
        self.assertEqual(stats['parameters'], 0)
        self.assertEqual(visual.weight.dtype, torch.float32)
        torch.testing.assert_close(visual.weight, before)


if __name__ == '__main__':
    unittest.main()
