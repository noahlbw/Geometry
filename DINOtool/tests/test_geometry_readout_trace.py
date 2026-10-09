from types import SimpleNamespace
from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinotool.geometry_readout_trace import (
    alias_class_scores, class_attribution, exact_alias_attribution,
    probe_scores, trace_geometry_head,
)
from dinotool.tcpr import TCPRConfig, _intervened_head

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_geometry_readout_streams import Confusion, add_patch_attribution


class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.rand(8))

    def forward(self, x):
        return x * self.gamma


def fixture():
    torch.manual_seed(20261001)
    blocks = []
    for _ in range(2):
        block = nn.Module()
        block.norm1, block.norm2 = nn.LayerNorm(8), nn.LayerNorm(8)
        block.attn = nn.Module()
        block.attn.num_heads, block.attn.scale = 2, .5
        block.attn.qkv, block.attn.proj = nn.Linear(8, 24), nn.Linear(8, 8)
        block.attn.proj_drop = nn.Identity()
        block.ls1, block.ls2 = Scale(), Scale()
        block.mlp = nn.Sequential(nn.Linear(8, 12), nn.GELU(), nn.Linear(12, 8))
        blocks.append(block)
    head = nn.Module()
    head.blocks = nn.ModuleList(blocks)
    head.ln_final, head.linear_projection = nn.LayerNorm(8), nn.Linear(8, 6)
    head.eval().requires_grad_(False)
    prepared = SimpleNamespace(backbone_tokens=torch.randn(1, 7, 8), block_index=1,
                               prefix_tokens=1, geometry_patch_conditional=torch.randn(1, 6, 6).softmax(-1))
    bank = SimpleNamespace(features=F.normalize(torch.randn(9, 6), dim=-1),
                           parent_indices=torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2]), class_count=3)
    return head, prepared, bank


class GeometryTraceTest(unittest.TestCase):
    def test_runner_confusion_interface(self):
        import numpy as np
        matrix = Confusion(("a", "b"))
        matrix.update(np.array([[0, 1]]), np.array([[0, 1]]))
        self.assertEqual(matrix.summary()["mean_iou_percent"], 100.)

    def test_runner_patch_group_attribution(self):
        scores = torch.tensor([[[.3, .1], [.2, .4]]])
        components = {"first": scores / 2, "second": scores / 2}
        totals = {}
        add_patch_attribution(totals, scores, components, torch.tensor([[0, 0]]),
                              torch.tensor([[True, True]]))
        self.assertEqual(totals["0->1"]["patch_centers"], 1)
        self.assertAlmostEqual(totals["0->1"]["margin_sum"],
                               sum(totals["0->1"]["components"].values()))

    def test_replay_matches_existing_geometry(self):
        head, prepared, _ = fixture()
        config = TCPRConfig(geometry_depth=2)
        reference = _intervened_head(head, prepared.backbone_tokens,
                                    prepared.geometry_patch_conditional, 1, 1, 2, "preserve")
        trace = trace_geometry_head(head, prepared, config)
        self.assertTrue(torch.equal(reference, trace["projected"]))

    def test_trace_does_not_modify_parameters_or_input(self):
        head, prepared, _ = fixture()
        state = {key: value.clone() for key, value in head.state_dict().items()}
        tokens = prepared.backbone_tokens.clone()
        trace_geometry_head(head, prepared, TCPRConfig(geometry_depth=2))
        self.assertTrue(torch.equal(tokens, prepared.backbone_tokens))
        self.assertTrue(all(torch.equal(state[key], value) for key, value in head.state_dict().items()))

    def test_final_class_attribution_is_exact(self):
        head, prepared, bank = fixture()
        trace = trace_geometry_head(head, prepared, TCPRConfig(geometry_depth=2))
        scores, aliases = probe_scores(head, trace, bank, 1)
        components = exact_alias_attribution(head, trace, bank, 1)
        torch.testing.assert_close(sum(components.values()), aliases, atol=2e-6, rtol=1e-5)
        classes = class_attribution(aliases, components, bank.parent_indices, bank.class_count)
        torch.testing.assert_close(sum(classes.values()), scores["final"], atol=2e-6, rtol=1e-5)

    def test_actual_identity_projection_path(self):
        head, prepared, bank = fixture()
        head.linear_projection = nn.Identity()
        bank.features = F.normalize(torch.randn(9, 8), dim=-1)
        trace = trace_geometry_head(head, prepared, TCPRConfig(geometry_depth=2))
        scores, aliases = probe_scores(head, trace, bank, 1)
        components = exact_alias_attribution(head, trace, bank, 1)
        classes = class_attribution(aliases, components, bank.parent_indices, bank.class_count)
        torch.testing.assert_close(sum(classes.values()), scores["final"], atol=2e-6, rtol=1e-5)

    def test_no_prefix_policy_has_zero_prefix_component(self):
        head, prepared, _ = fixture()
        trace = trace_geometry_head(head, prepared, TCPRConfig(geometry_depth=2, prefix_policy="block"))
        self.assertEqual(float(trace["components"]["prefix"].abs().max()), 0.)

    def test_zero_component_cannot_change_attribution(self):
        aliases = torch.randn(1, 6, 9)
        parents = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2])
        output = class_attribution(aliases, {"all": aliases, "none": torch.zeros_like(aliases)}, parents, 3)
        self.assertEqual(float(output["none"].abs().max()), 0.)
        torch.testing.assert_close(sum(output.values()), alias_class_scores(aliases, parents, 3))


if __name__ == "__main__":
    unittest.main()
