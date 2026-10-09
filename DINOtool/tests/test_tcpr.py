from contextlib import nullcontext

import torch
from torch import nn

from dinotool.tcpr import (
    TCPRConfig,
    TCPRSegmenter,
    TCPRTextBank,
    _evidence_conditioned_attended,
    _geometry_attended,
    aggregate_alias_scores,
    candidate_edges,
    competition_evidence,
    filter_aliases,
    select_competing_pairs,
)
from dinotool.text_visual_intervention import (
    TextVisualInterventionConfig,
    text_visual_intervention_readouts,
)


class _TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = 2
        self.scale = 2 ** -0.5
        self.qkv = nn.Linear(4, 12, bias=False)
        self.proj = nn.Linear(4, 4, bias=False)
        self.proj_drop = nn.Identity()


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(4)
        self.attn = _TinyAttention()
        self.ls1 = nn.Identity()
        self.norm2 = nn.LayerNorm(4)
        self.mlp = nn.Linear(4, 4, bias=False)
        self.ls2 = nn.Identity()

    def forward(self, x):
        qkv = self.attn.qkv(self.norm1(x)).reshape(x.shape[0], x.shape[1], 3, 2, 2)
        q, k, v = (part.transpose(1, 2) for part in torch.unbind(qkv, dim=2))
        weights = torch.softmax((q @ k.transpose(-1, -2)) * self.attn.scale, dim=-1)
        attended = (weights @ v).transpose(1, 2).reshape_as(x)
        x = x + self.attn.proj(attended)
        return x + self.mlp(self.norm2(x))


class _TinyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Module()
        self.head.blocks = nn.ModuleList([_TinyBlock()])
        self.head.ln_final = nn.LayerNorm(4)
        self.head.linear_projection = nn.Linear(4, 4, bias=False)

    def get_backbone_features(self, rgb):
        batch = rgb.shape[0]
        raw = torch.tensor(
            [[[1.0, 0.0, 0.2, 0.0], [0.9, 0.1, 0.2, 0.0],
              [0.0, 1.0, 0.0, 0.2], [0.1, 0.9, 0.0, 0.2]]]
        ).expand(batch, -1, -1).clone()
        return torch.zeros((batch, 4)), raw, torch.zeros((batch, 1, 4))


class _TinyBackbone:
    patch_size = 16
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.model = nn.Module()
        self.model.visual_model = _TinyVisual()
        self._imagenet_mean = torch.zeros((1, 3, 1, 1))
        self._imagenet_std = torch.ones((1, 3, 1, 1))

    def _validate_rgb(self, rgb):
        assert rgb.shape[-2:] == (32, 32)

    @staticmethod
    def _autocast():
        return nullcontext()


def test_alias_aggregation_is_normalized_and_canonical_is_retained() -> None:
    scores = torch.tensor([[[0.3, 0.3, 0.1], [0.2, 0.2, 0.4]]])
    parents = torch.tensor([0, 0, 1])
    canonical = torch.tensor([True, False, True])
    valid = torch.ones((1, 2), dtype=torch.bool)
    config = TCPRConfig(maximum_aliases_per_class=1)
    weights = filter_aliases(scores, scores, parents, canonical, 2, valid, config)
    assert weights[0, 0] == 1.0
    assert weights[0, 1] == 0.0
    assert weights[0, 2] == 1.0

    all_weights = torch.tensor([[0.5, 0.5, 1.0]])
    classes = aggregate_alias_scores(scores, all_weights, parents, 2, temperature=0.07)
    assert torch.allclose(classes[..., 0], scores[..., 0], atol=1e-6)
    assert torch.allclose(classes[..., 1], scores[..., 2], atol=1e-6)


def test_competing_pair_keeps_geometry_first_and_uses_union_for_second() -> None:
    geometry = torch.tensor([[[0.8, 0.5, 0.1]]])
    native = torch.tensor([[[0.1, 0.2, 0.9]]])
    pair = select_competing_pairs(geometry, native)
    assert pair.tolist() == [[[0, 2]]]


def test_zero_evidence_is_exact_geometry_readout() -> None:
    generator = torch.Generator().manual_seed(17)
    batch, heads, prefix, patches, head_dim = 1, 2, 2, 5, 3
    logits = torch.randn((batch, heads, prefix + patches, prefix + patches), generator=generator)
    native = torch.softmax(logits, dim=-1)
    geometry = torch.softmax(torch.randn((batch, patches, patches), generator=generator), dim=-1)
    values = torch.randn((batch, heads, prefix + patches, head_dim), generator=generator)
    valid = torch.ones((batch, patches), dtype=torch.bool)
    indices, _ = candidate_edges(geometry, native[..., prefix:, prefix:], 3, valid)
    evidence = torch.zeros_like(indices, dtype=torch.float32)
    expected = _geometry_attended(native, geometry, values, prefix)
    actual = _evidence_conditioned_attended(
        native, geometry, values, indices, evidence, prefix, TCPRConfig()
    )
    assert torch.equal(actual, expected)


def test_competition_evidence_requires_references_for_both_classes() -> None:
    indices = torch.tensor([[[[1], [0]]]])
    affinity = torch.ones_like(indices, dtype=torch.float32)
    pairs = torch.tensor([[[0, 1], [1, 0]]])
    references = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
    reliability = torch.ones((1, 2))
    geometry = torch.tensor([[[0.8, 0.2], [0.2, 0.8]]])
    native = geometry.clone()

    blocked, blocked_mass = competition_evidence(
        indices, affinity, pairs, references, torch.tensor([[True, False]]),
        reliability, geometry, native, TCPRConfig(),
    )
    assert torch.count_nonzero(blocked) == 0
    assert torch.count_nonzero(blocked_mass) == 0

    active, active_mass = competition_evidence(
        indices, affinity, pairs, references, torch.tensor([[True, True]]),
        reliability, geometry, native, TCPRConfig(),
    )
    assert torch.isfinite(active).all()
    assert torch.isfinite(active_mass).all()
    assert torch.all(active >= 0)


def test_complete_tcpr_tensor_path_runs_once_and_returns_both_routes() -> None:
    torch.manual_seed(3)
    model = TCPRSegmenter(
        _TinyBackbone(),
        TCPRConfig(
            reference_min_quality=0.0,
            reference_diversity_radius=0,
            evidence_topk=2,
        ),
    )
    bank = TCPRTextBank(
        features=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        parent_indices=torch.tensor([0, 1]),
        canonical_mask=torch.tensor([True, True]),
        class_names=("first", "second"),
        alias_names=("first", "second"),
    )
    prepared = model.prepare_image(torch.zeros((1, 3, 32, 32)))
    result = model.read_prepared(prepared, bank)

    assert result.logits.shape == result.geometry_logits.shape == result.native_logits.shape == (1, 2, 2, 2)
    assert result.pair_ids.shape == (1, 2, 2, 2)
    assert result.evidence_mass.shape == (1, 2, 2)
    assert torch.isfinite(result.logits).all()
    assert result.reference_counts.shape == (1, 2)


def test_text_visual_intervention_matrix_returns_matched_arms() -> None:
    torch.manual_seed(5)
    model = TCPRSegmenter(_TinyBackbone(), TCPRConfig(maximum_aliases_per_class=2))
    prepared = model.prepare_image(torch.zeros((1, 3, 32, 32)))
    canonical = TCPRTextBank(
        features=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        parent_indices=torch.tensor([0, 1]),
        canonical_mask=torch.tensor([True, True]),
        class_names=("first", "second"),
        alias_names=("first", "second"),
    )
    expanded = TCPRTextBank(
        features=torch.tensor([
            [1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0],
        ]),
        parent_indices=torch.tensor([0, 0, 1, 1]),
        canonical_mask=torch.tensor([True, False, True, False]),
        class_names=("first", "second"),
        alias_names=("first", "first-like", "second", "second-like"),
    )
    result = text_visual_intervention_readouts(
        prepared, canonical, expanded, model.config, TextVisualInterventionConfig()
    )
    expected = {
        "G_canonical", "G_expanded", "TextGraph_logits", "ImageSelf_50",
        "TextSelf_25", "TextSelf_50", "Cross_25", "Cross_50",
        "Parallel_25", "Parallel_50", "TextSelf_shuffled_50", "Serial_50",
    }
    assert set(result.logits) == expected
    assert all(value.shape == (1, 2, 2, 2) for value in result.logits.values())
    assert all(torch.isfinite(value).all() for value in result.logits.values())
    assert set(result.feature_shift) == expected - {"G_canonical", "G_expanded", "TextGraph_logits"}
    assert result.relation_kl_from_geometry >= 0.0
