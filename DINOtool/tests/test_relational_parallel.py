"""Numerical and CAFe interface tests for DINO relational parallel reconstruction."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.getenv("CAFE_OFFICIAL_ROOT", str(ROOT.parent / "third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c")))
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(OFFICIAL / "CAFe_DINO")]
from modeling.cafedino import CAFe_DINO
from dinotool.cafe_relational_parallel import (
    CafeRelationalParallel,
    RelationalParallelConfig,
    config_from_checkpoint,
    relational_parallel_loss,
)
from dinotool.relational_parallel import (
    AnchoredCostReconstruction,
    ClassRelationHead,
    SpatialRelationHead,
    SpatialStencil,
    relation_supervision,
)


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1024, 1)
        self.visual_model = nn.Module()
        self.visual_model.backbone = nn.Module()
        self.visual_model.backbone.blocks = nn.ModuleList(nn.Linear(1024, 1024) for _ in range(4))

    def encode_image_with_patch_tokens(self, images, normalize=False):
        tokens = self.conv(F.avg_pool2d(images, 16)).flatten(2).transpose(1, 2)
        for block in self.visual_model.backbone.blocks:
            tokens = tokens + 0.05 * torch.tanh(block(tokens))
        return tokens.mean(1), torch.tanh(tokens), tokens


class TinyUpsampler(nn.Module):
    def forward(self, image, features, q_chunk_size=None):
        return F.interpolate(features, size=image.shape[-2:], mode="bilinear", align_corners=False)


def official():
    return CAFe_DINO(TinyVision(), None, TinyUpsampler(), (7, 7), "cpu", aggregator_dim=16, aggregator_blocks=2).eval()


class DeterministicTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260924)
        torch.set_num_threads(2)
        stack = ExitStack()
        stack.enter_context(torch.backends.mkldnn.flags(enabled=False))
        stack.enter_context(torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH))
        self.addCleanup(stack.close)


class StencilTests(DeterministicTest):
    def test_adjoint_identity(self):
        value = torch.randn(2, 3, 4, 5, 7)
        edge = torch.randn(2, 4, 3, 4, 5, 7)
        valid = SpatialStencil.edge_valid(torch.ones(2, 5, 7, dtype=torch.bool))
        edge = edge * valid[:, :, None, None]
        left = (SpatialStencil.difference(value) * edge).sum()
        right = (value * SpatialStencil.adjoint(edge)).sum()
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-6)

    def test_invalid_edges_do_not_contribute(self):
        # B=2 is deliberate: it catches an otherwise silent broadcast where
        # batch was incorrectly aligned to the cost-channel axis.
        value = torch.randn(2, 3, 2, 4, 5)
        relation_head = SpatialRelationHead(3, 6, 4).eval()
        relation, weight = relation_head(value, torch.randn(2, 6, 4, 5))
        valid = SpatialStencil.edge_valid(torch.ones(2, 4, 5, dtype=torch.bool))
        relation_invalid = relation.masked_select(~valid[:, :, None, None].expand_as(relation))
        weight_invalid = weight.masked_select(~valid[:, :, None].expand_as(weight))
        self.assertTrue(torch.equal(relation_invalid, torch.zeros_like(relation_invalid)))
        self.assertTrue(torch.equal(weight_invalid, torch.zeros_like(weight_invalid)))


class ReconstructionTests(DeterministicTest):
    def test_exact_relations_leave_solution_fixed_and_residual_contracts(self):
        reconstruction = AnchoredCostReconstruction(iterations=4)
        solution = torch.randn(1, 3, 4, 5, 7)
        valid = SpatialStencil.edge_valid(torch.ones(1, 5, 7, dtype=torch.bool))
        weights = valid[:, :, None].expand(-1, -1, solution.shape[2], -1, -1).float()
        spatial = SpatialStencil.difference(solution) * valid[:, :, None, None]
        classes = reconstruction.center_queries(solution)
        rhs = solution + reconstruction.spatial_weight * SpatialStencil.adjoint(weights.unsqueeze(2) * spatial)
        rhs = rhs + reconstruction.class_weight * classes
        before = (rhs - reconstruction.operator(torch.zeros_like(solution), weights)).square().mean()
        result = reconstruction(solution, spatial, weights, classes)
        after = (rhs - reconstruction.operator(result, weights)).square().mean()
        torch.testing.assert_close(result, solution, rtol=1e-6, atol=1e-6)
        self.assertLess(float(after), float(before) * 1e-8)

    def test_class_relation_is_query_permutation_equivariant_and_singleton_zero(self):
        head = ClassRelationHead(3, 8, 4).eval()
        semantic, text = torch.randn(2, 3, 4, 5, 7), torch.randn(4, 8)
        order = torch.tensor([2, 0, 3, 1])
        reference = head(semantic, text)
        permuted = head(semantic[:, :, order], text[order])
        torch.testing.assert_close(permuted, reference[:, :, order], rtol=1e-5, atol=1e-6)
        singleton = head(semantic[:, :, :1], text[:1])
        torch.testing.assert_close(singleton, torch.zeros_like(singleton), rtol=0, atol=0)

    def test_relation_supervision_ignores_void_patches_and_is_finite(self):
        spatial = torch.randn(2, 4, 3, 4, 4, 4, requires_grad=True)
        classes = torch.randn(2, 3, 4, 4, 4, requires_grad=True)
        labels = torch.randint(0, 4, (2, 32, 32))
        labels[:, :8, :8] = 255
        readout = torch.randn(3, requires_grad=True)
        s_loss, c_loss = relation_supervision(spatial, classes, labels, readout)
        (s_loss + c_loss).backward()
        self.assertTrue(torch.isfinite(s_loss + c_loss))
        self.assertTrue(all(item.grad is not None and torch.isfinite(item.grad).all() for item in (spatial, classes, readout)))


class ModelTests(DeterministicTest):
    def test_model_gradients_and_checkpoint_roundtrip(self):
        base = official()
        restore_base = copy.deepcopy(base)
        config = RelationalParallelConfig(stages=2, relation_dim=8, solver_steps=4)
        model = CafeRelationalParallel(base, config).train()
        image, text = torch.randn(1, 3, 112, 224), torch.randn(3, 1024)
        target = torch.randint(0, 3, (1, 112, 224))
        target[:, :16, :16] = 255
        output = model(image, text, relation_target=target)
        self.assertEqual(output["logits"].shape, (1, 3, 112, 224))
        loss, _ = relational_parallel_loss(output, target, torch.randn_like(output["logits"]))
        loss.backward()
        for prefix in ("spatial.0.", "channel.0.", "fusion.0.spatial_relation.",
                       "fusion.0.class_relation.", "relation_readout", "cafe.backbone.visual_model.backbone.blocks.3."):
            grads = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
            self.assertTrue(any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads), prefix)
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        optimizer = torch.optim.AdamW(model.optimizer_groups(1e-3, 1e-3, 1e-4))
        before = model.fusion[0].spatial_relation.delta.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.fusion[0].spatial_relation.delta.weight))
        payload = dict(format=model.checkpoint_format, architecture=model.architecture(), adapted_state=model.adapted_state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "relational.pt"
            torch.save(payload, path)
            loaded = torch.load(path, weights_only=False)
        restored = CafeRelationalParallel(restore_base, config_from_checkpoint(loaded)).eval()
        restored.load_adapted_state_dict(loaded["adapted_state"])
        model.eval()
        with torch.no_grad():
            expected = model(image, text)["logits"]
            torch.testing.assert_close(restored(image, text)["logits"], expected, rtol=2e-5, atol=3e-6)
        malformed = dict(loaded["adapted_state"])
        malformed.pop(next(iter(malformed)))
        with self.assertRaises(ValueError):
            restored.load_adapted_state_dict(malformed)

    def test_cpu_bfloat16_autocast_stays_finite(self):
        model = CafeRelationalParallel(official(), RelationalParallelConfig(stages=2, relation_dim=8)).train()
        image, text = torch.randn(1, 3, 112, 112), torch.randn(3, 1024)
        labels = torch.randint(0, 3, (1, 112, 112))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(image, text, relation_target=labels)
            loss = output["logits"].float().square().mean() + output["relation_loss"].float()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
