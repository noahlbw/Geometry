import torch

from dinotool.decide_ov import compose_decide_outputs


def test_rejected_pixels_keep_anchor_logits_exactly() -> None:
    anchor = torch.tensor([[[[2.0, 0.0]], [[0.0, 2.0]]]])
    candidate = torch.tensor([[[[0.0, 0.0]], [[2.0, 2.0]]]])
    held = torch.tensor([[[[3.0, 0.0]], [[-1.0, 2.0]]]])
    result = compose_decide_outputs(anchor, candidate, [held], [candidate], [held])
    assert torch.equal(result.logits["M1_replay_1"][..., 0], anchor[..., 0])


def test_same_action_replay_can_reject_direct_support() -> None:
    anchor = torch.tensor([[[[2.0]], [[0.0]]]])
    candidate = torch.tensor([[[[0.0]], [[2.0]]]])
    held_raw = torch.tensor([[[[0.0]], [[1.0]]]])
    held_replay = torch.tensor([[[[1.5]], [[0.5]]]])
    result = compose_decide_outputs(anchor, candidate, [held_raw], [candidate], [held_replay])
    assert result.accepted["B5_direct_1"].item()
    assert not result.accepted["M1_replay_1"].item()
    assert torch.equal(result.logits["M1_replay_1"], anchor)


def test_two_view_rule_requires_both_views() -> None:
    anchor = torch.tensor([[[[2.0]], [[0.0]]]])
    candidate = torch.tensor([[[[0.0]], [[2.0]]]])
    support = torch.tensor([[[[0.0]], [[1.0]]]])
    oppose = torch.tensor([[[[1.0]], [[0.0]]]])
    result = compose_decide_outputs(
        anchor,
        candidate,
        [support, oppose],
        [candidate, candidate],
        [support, support],
    )
    assert result.accepted["B5_direct_1"].item()
    assert not result.accepted["B5_direct_2"].item()
    assert not result.accepted["M2_replay_2"].item()

