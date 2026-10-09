import torch

from dinotool.tcpr import TCPRTextBank
from dinotool.vip_alias_distillation import (
    VIPAliasAccumulator,
    VIPDistillationConfig,
    normalized_alias_activation,
    random_walk_relation,
)


def _bank():
    return TCPRTextBank(
        features=torch.nn.functional.normalize(torch.eye(4), dim=-1),
        parent_indices=torch.tensor([0, 0, 1, 1]),
        canonical_mask=torch.tensor([True, False, True, False]),
        class_names=("a", "b"),
        alias_names=("a", "a good", "b", "b good"),
    )


def test_random_walk_is_row_stochastic():
    walk = random_walk_relation(torch.tensor([[0.8, 0.2], [0.1, 0.9]]), VIPDistillationConfig())
    torch.testing.assert_close(walk.sum(-1), torch.ones(2))


def test_activation_ignores_padding():
    scores = torch.tensor([[[0.1, 0.2], [0.9, 0.8], [99.0, -99.0]]])
    result = normalized_alias_activation(scores, torch.tensor([[True, True, False]]))
    torch.testing.assert_close(result[0, :2], torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    assert result[0, 2].abs().max() == 0


def test_vip_selection_keeps_canonical_and_admits_better_alias():
    bank = _bank()
    stats = VIPAliasAccumulator(4, device=torch.device("cpu"))
    stats.vg_sum[:] = torch.tensor([0.2, 0.7, 0.4, 0.3], dtype=torch.float64)
    stats.sc_sum[:] = torch.tensor([0.6, 0.2, 0.3, 0.1], dtype=torch.float64)
    stats.observed_images[:] = 2
    assert stats.report(bank)["selected_mask"] == [True, True, True, False]
