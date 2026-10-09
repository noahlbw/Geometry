from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class DecideOutputs:
    logits: dict[str, Tensor]
    accepted: dict[str, Tensor]
    changed: Tensor


def compose_decide_outputs(
    anchor: Tensor,
    candidate: Tensor,
    held_raw: Sequence[Tensor],
    held_independent: Sequence[Tensor],
    held_replay: Sequence[Tensor],
    *,
    temperature: float = 0.07,
) -> DecideOutputs:
    """Build matched-budget DECIDE-OV baselines from one fixed TLP candidate."""
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    _validate_scores(anchor, candidate, held_raw, held_independent, held_replay)
    if len(held_raw) not in {1, 2}:
        raise ValueError("DECIDE-OV requires one or two held-out observations.")
    if len(held_independent) != len(held_raw) or len(held_replay) != len(held_raw):
        raise ValueError("Held-out raw, independent, and replay sequences must have equal length.")

    old_class = anchor.argmax(dim=1, keepdim=True)
    new_class = candidate.argmax(dim=1, keepdim=True)
    changed = old_class.ne(new_class)
    anchor_probabilities = torch.softmax(anchor.float() / temperature, dim=1)
    candidate_probabilities = torch.softmax(candidate.float() / temperature, dim=1)
    anchor_entropy = -(anchor_probabilities * anchor_probabilities.clamp_min(1e-12).log()).sum(dim=1, keepdim=True)
    candidate_entropy = -(
        candidate_probabilities * candidate_probabilities.clamp_min(1e-12).log()
    ).sum(dim=1, keepdim=True)

    direct = [_pair_margin(scores, old_class, new_class) for scores in held_raw]
    replay = [_pair_margin(scores, old_class, new_class) for scores in held_replay]
    direct_one = direct[0].gt(0)
    direct_all = torch.stack(direct, dim=0).amin(dim=0).gt(0)
    replay_one = replay[0].gt(0)
    replay_all = torch.stack(replay, dim=0).amin(dim=0).gt(0)

    acceptance = {
        "B4_entropy": changed & candidate_entropy.lt(anchor_entropy),
        "B4_confidence": changed
        & candidate_probabilities.amax(dim=1, keepdim=True).gt(
            anchor_probabilities.amax(dim=1, keepdim=True)
        ),
        "B5_direct_1": changed & direct_one,
        "M1_replay_1": changed & direct_one & replay_one,
    }
    if len(held_raw) == 2:
        acceptance["B5_direct_2"] = changed & direct_all
        acceptance["M2_replay_2"] = changed & direct_all & replay_all

    outputs = {
        "B0_raw": anchor,
        "B1_tlp": candidate,
        "B2_view_average": torch.stack((anchor, *held_raw), dim=0).mean(dim=0),
        "B3_tlp_average": torch.stack((candidate, *held_independent), dim=0).mean(dim=0),
    }
    outputs.update({name: _commit(anchor, candidate, mask) for name, mask in acceptance.items()})
    return DecideOutputs(logits=outputs, accepted=acceptance, changed=changed)


def _pair_margin(scores: Tensor, old_class: Tensor, new_class: Tensor) -> Tensor:
    old_score = scores.gather(1, old_class)
    new_score = scores.gather(1, new_class)
    return new_score - old_score


def _commit(anchor: Tensor, candidate: Tensor, accepted: Tensor) -> Tensor:
    return torch.where(accepted.expand_as(anchor), candidate, anchor)


def _validate_scores(anchor: Tensor, candidate: Tensor, *groups: Sequence[Tensor]) -> None:
    if anchor.ndim != 4:
        raise ValueError("Scores must be BCHW tensors.")
    for scores in (candidate, *(item for group in groups for item in group)):
        if scores.shape != anchor.shape:
            raise ValueError("All DECIDE-OV score tensors must have the same shape.")

