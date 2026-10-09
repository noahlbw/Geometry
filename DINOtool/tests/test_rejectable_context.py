from __future__ import annotations

import torch
import numpy as np

from scripts.eval_grounded_context import proposal_quality
from dinotool.rejectable_context import (
    accepted_logits, local_visual_graph, proposal_acceptance,
    reconstruct_acceptance, visual_witness,
)


def test_visual_witness_excludes_self_and_disconnected_tokens():
    geometry = torch.ones(1, 4, 4) / 4
    valid = torch.tensor([[True, True, True, False]])
    graph = local_visual_graph(geometry, valid, grid_height=2, grid_width=2, radius=1)
    assert torch.equal(graph, graph.transpose(-1, -2))
    assert torch.equal(graph.diagonal(dim1=1, dim2=2), torch.zeros(1, 4))
    assert graph[0, 3].sum() == 0 and graph[0, :, 3].sum() == 0
    local = torch.tensor([[[1., 0.], [0., 1.], [1., 0.], [0., 1.]]])
    witness, variance = visual_witness(local, graph, valid)
    torch.testing.assert_close(witness[0, 0], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(witness[0, 3], local[0, 3])
    assert bool((variance >= 0).all())


def test_acceptance_uses_competing_class_evidence_and_can_reject():
    local = torch.tensor([[[0.5, 0.45, 0.05], [0.5, 0.45, 0.05]]])
    bounded = torch.tensor([[[0.8, 0.2, 0.0], [0.8, 0.2, 0.0]]])
    witness = torch.tensor([[[0.75, 0.2, 0.05], [0.45, 0.5, 0.05]]])
    initial, support, revision = proposal_acceptance(
        local, bounded, witness, torch.zeros(1, 2),
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert support[0, 0] > 0 and initial[0, 0] > 0
    assert support[0, 1] < 0 and initial[0, 1] == 0
    torch.testing.assert_close(
        (accepted_logits(local, revision, torch.zeros_like(initial), 0.07) / 0.07).softmax(-1),
        local,
    )
    torch.testing.assert_close(
        (accepted_logits(local, revision, torch.ones_like(initial), 0.07) / 0.07).softmax(-1),
        bounded,
    )


def test_graph_reconstruction_keeps_rejected_node_at_identity_and_reduces_energy():
    local = torch.tensor([[[0.50, 0.50], [0.55, 0.45], [0.50, 0.50]]])
    bounded = torch.tensor([[[0.80, 0.20], [0.80, 0.20], [0.80, 0.20]]])
    witness = torch.tensor([[[0.75, 0.25], [0.72, 0.28], [0.35, 0.65]]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    graph = torch.tensor([[[0., 0.5, 0.], [0.5, 0., 0.5], [0., 0.5, 0.]]])
    variance = torch.zeros(1, 3)
    initial, support, revision = proposal_acceptance(local, bounded, witness, variance, valid)
    accepted = reconstruct_acceptance(
        local, witness, revision, variance, graph, valid, initial, support,
        strength=1.0, steps=40,
    )
    assert accepted[0, 2] == 0
    assert bool(((accepted >= 0) & (accepted <= 1)).all())

    def energy(alpha):
        changed = alpha[..., None] * revision
        fit = 0.5 * (local + changed - witness).square().sum()
        differences = changed[:, :, None, :] - changed[:, None, :, :]
        return fit + 0.25 * (graph[..., None] * differences.square()).sum()

    assert energy(accepted) <= energy(initial) + 1e-7


def test_proposal_accounting_distinguishes_retained_help_and_rejected_harm():
    base = np.array([0, 0, 1, 1, 0, 0])
    proposal = np.array([1, 1, 0, 0, 1, 1])
    target = np.array([1, 0, 1, 0, 255, 1])
    output = np.array([1, 0, 1, 0, 0, 0])
    counts = proposal_quality(base, proposal, output, target, class_count=2)
    assert counts == {
        "proposed_changed": 5,
        "proposed_beneficial": 3,
        "proposed_harmful": 2,
        "beneficial_retained": 2,
        "harmful_rejected": 2,
    }
