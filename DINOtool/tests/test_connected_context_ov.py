import torch

from dinotool.connected_context_ov import (
    ConnectedContextOVConfig,
    ConnectedContextOVSegmenter,
    supervised_route_complementarity,
)


class _FakeBackbone:
    patch_size = 16
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.context_batches = 0

    @staticmethod
    def similarity_logits(features, text):
        return torch.einsum("bdhw,cd->bchw", features, text)

    def encode_image_with_structure(self, rgb):
        batch, _, height, width = rgb.shape
        grid_h, grid_w = height // 16, width // 16
        raw = torch.zeros((batch, 2, grid_h, grid_w))
        raw[:, 0, :, : grid_w // 2] = 1.0
        raw[:, 1, :, grid_w // 2 :] = 1.0
        local = torch.zeros_like(raw)
        local[:, 0] = 0.1
        return local, raw, torch.nn.functional.normalize(local.mean((2, 3)), dim=1)

    def encode_image(self, rgb):
        self.context_batches += 1
        batch, _, height, width = rgb.shape
        grid_h, grid_w = height // 16, width // 16
        mean = rgb.mean((2, 3))[:, :2]
        feature = torch.nn.functional.normalize(mean, dim=1)[:, :, None, None]
        feature = feature.expand(batch, 2, grid_h, grid_w).contiguous()
        return feature, torch.nn.functional.normalize(mean, dim=1)


def _config(**changes):
    values = dict(
        atom_similarity_threshold=0.99,
        minimum_context_patches=2,
        maximum_context_regions=8,
        context_size=32,
        context_batch_size=8,
        context_anchor_weight=0.0,
        branch_temperature=0.5,
        region_prior_strength=0.5,
        continuity_strength=0.2,
        continuity_iterations=2,
    )
    values.update(changes)
    return ConnectedContextOVConfig(**values)


def test_model_runs_distinct_context_forward_and_exposes_both_routes() -> None:
    backbone = _FakeBackbone()
    model = ConnectedContextOVSegmenter(
        backbone=backbone,
        config=_config(region_prior_strength=2.0),
    )
    rgb = torch.zeros((1, 3, 64, 64))
    rgb[:, 0, :, :32] = 1.0
    rgb[:, 1, :, 32:] = 1.0
    result = model.segment_with_text_features(rgb, torch.eye(2))

    assert backbone.context_batches == 1
    assert result.local_logits.shape == result.context_logits.shape == result.logits.shape == (1, 2, 4, 4)
    assert result.context_valid.all()
    assert result.support_ids.unique().numel() == 2
    assert result.diagnostics[0].context_regions == 2
    assert result.diagnostics[0].branch_disagreement == 0.5
    assert torch.equal(result.logits.argmax(1)[0, :, :2], torch.zeros((4, 2), dtype=torch.long))
    assert torch.equal(result.logits.argmax(1)[0, :, 2:], torch.ones((4, 2), dtype=torch.long))
    assert torch.isfinite(result.logits).all()


def test_strong_local_exception_survives_regional_context() -> None:
    backbone = _FakeBackbone()
    model = ConnectedContextOVSegmenter(
        backbone=backbone,
        config=_config(region_prior_strength=0.15, continuity_strength=0.05, continuity_iterations=1),
    )
    rgb = torch.zeros((1, 3, 64, 64))
    local, raw, anchors = backbone.encode_image_with_structure(rgb)
    local[:, 0] = 1.0
    local[:, 1] = 0.0
    local[:, 0, 1, 1] = -1.0
    local[:, 1, 1, 1] = 3.0

    original = backbone.encode_image_with_structure
    backbone.encode_image_with_structure = lambda _: (local, raw, anchors)
    result = model.segment_with_text_features(rgb, torch.eye(2))
    backbone.encode_image_with_structure = original

    assert result.logits.argmax(1)[0, 1, 1].item() == 1
    assert (result.logits.argmax(1) == 0).sum().item() >= 15


def test_supervised_complementarity_reports_exclusive_branch_evidence() -> None:
    target = torch.tensor([[[0, 1, 0, 1]]])
    local_labels = torch.tensor([[[0, 0, 0, 0]]])
    context_labels = torch.tensor([[[0, 1, 1, 0]]])
    final_labels = torch.tensor([[[0, 1, 0, 0]]])

    def logits(labels):
        return torch.nn.functional.one_hot(labels, 2).permute(0, 3, 1, 2).float() * 4.0

    report = supervised_route_complementarity(
        logits(local_labels), logits(context_labels), logits(final_labels),
        torch.ones_like(target, dtype=torch.bool), target,
    )

    assert report.both_correct == 0.25
    assert report.local_only_correct == 0.25
    assert report.context_only_correct == 0.25
    assert report.both_wrong == 0.25
    assert report.oracle_accuracy == 0.75
    assert report.oracle_gain_over_best_branch == 0.25
