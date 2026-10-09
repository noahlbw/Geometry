"""Partition supervision must escape equal slots and respect ignored pixels."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinotool.proposal_supervision import proposal_partition_loss


class ProposalTests(unittest.TestCase):
    def test_collapsed_slots_receive_different_spatial_gradients(self):
        logits = torch.zeros(1, 4, 3, requires_grad=True)
        target = torch.tensor([[[0, 0], [1, 1]]])
        loss = proposal_partition_loss(logits, target, 2, 2)
        loss.backward()
        self.assertFalse(torch.allclose(logits.grad[:, 0], logits.grad[:, 2]))
        updated = logits.detach() - 2 * logits.grad
        self.assertLess(proposal_partition_loss(updated, target, 2, 2).item(), loss.item())

    def test_permuting_slots_preserves_set_loss(self):
        torch.manual_seed(17)
        logits = torch.randn(1, 4, 3)
        target = torch.tensor([[[0, 0], [1, 255]]])
        a = proposal_partition_loss(logits, target, 2, 2)
        b = proposal_partition_loss(logits[:, :, [2, 0, 1]], target, 2, 2)
        torch.testing.assert_close(a, b)

    def test_ignore_pixels_do_not_receive_a_region_target(self):
        logits = torch.randn(1, 4, 3, requires_grad=True)
        loss = proposal_partition_loss(logits, torch.tensor([[[0, 1], [255, 255]]]), 2, 2)
        loss.backward()
        torch.testing.assert_close(logits.grad[:, 2:], torch.zeros_like(logits.grad[:, 2:]))
        empty = proposal_partition_loss(logits, torch.full((1, 2, 2), 255), 2, 2)
        self.assertEqual(empty.item(), 0)


if __name__ == "__main__":
    unittest.main()
