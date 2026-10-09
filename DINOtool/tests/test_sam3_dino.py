from argparse import Namespace
from types import SimpleNamespace

import pytest
from PIL import Image
import torch

from dinotool import cli
from dinotool.prompts import ClassSpec
from dinotool.sam3 import Sam3ImageComponents, Sam3PromptComponents
from dinotool.sam3_dino import (
    Sam3DinoFusionConfig,
    _build_prompt_bank,
    _class_probabilities_from_baseline,
    _pgrf_prompt_score,
    uncertainty_gated_fuse,
)
from dinotool.sam3_dino import Sam3DinoPredictor


def _prompt_components(
    semantic: float,
    *,
    instance: float | None,
    object_score: float = 1.0,
    presence: float = 1.0,
) -> Sam3PromptComponents:
    semantic_map = torch.tensor([[semantic]], dtype=torch.float32)
    if instance is None:
        instance_maps = torch.empty((0, 1, 1), dtype=torch.float32)
        instance_scores = torch.empty((0,), dtype=torch.float32)
    else:
        instance_maps = torch.tensor([[[instance]]], dtype=torch.float32)
        instance_scores = torch.tensor([object_score], dtype=torch.float32)
    return Sam3PromptComponents(
        query_index=0,
        class_index=0,
        prompt="example",
        semantic_logits=semantic_map,
        instance_mask_logits=instance_maps,
        instance_scores=instance_scores,
        instance_fused_logits=semantic_map,
        base_logits=semantic_map,
        presence_score=torch.tensor(presence, dtype=torch.float32),
        fused_logits=semantic_map * presence,
    )


def test_uncertainty_gated_fusion_keeps_confident_sam3_pixels_stable() -> None:
    reference = torch.tensor([[[[0.99, 0.50]], [[0.01, 0.50]]]])
    candidate = torch.tensor([[[[0.00, 1.00]], [[1.00, 0.00]]]])

    fused = uncertainty_gated_fuse(reference, candidate, strength=0.5, uncertainty_power=1.0)

    assert fused[0, 0, 0, 0] == pytest.approx(0.98505)
    assert fused[0, 0, 0, 1] == pytest.approx(0.625)
    assert torch.allclose(fused.sum(dim=1), torch.ones((1, 1, 2)))


def test_uncertainty_gated_fusion_accepts_zero_strength_ablation() -> None:
    reference = torch.tensor([[[[0.8]], [[0.2]]]])
    candidate = torch.tensor([[[[0.2]], [[0.8]]]])
    assert torch.equal(
        uncertainty_gated_fuse(reference, candidate, strength=0.0, uncertainty_power=1.0),
        reference,
    )


def test_fusion_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        Sam3DinoFusionConfig(dino_input_resolution=1001).validate()
    with pytest.raises(ValueError, match="semantic_strength"):
        Sam3DinoFusionConfig(semantic_strength=1.1).validate()


def test_default_fusion_preserves_the_official_sam3_path() -> None:
    config = Sam3DinoFusionConfig()

    assert config.use_constrained_prompts is False
    assert config.use_pgrf is False
    assert config.geometry.enabled is False
    assert config.use_dino_tlp is False
    assert config.use_satellite_structure is False


def test_prompt_bank_preserves_official_aliases_and_class_mapping() -> None:
    classes = (
        ClassSpec("background", ("background", "other terrain")),
        ClassSpec("building", ("building", "rooftop", "house")),
    )
    words = ("background", "building", "house")
    indices = (0, 1, 1)

    official = _build_prompt_bank(words, indices, classes, enabled=False, maximum_per_class=1)
    assert official.query_words == words
    assert official.query_class_indices == indices
    assert official.source == "official-segearth"

    constrained = _build_prompt_bank(words, indices, classes, enabled=True, maximum_per_class=4)
    constrained_official = tuple(
        (word, class_index)
        for word, class_index in zip(constrained.query_words, constrained.query_class_indices)
        if word in set(words)
    )
    assert constrained_official == tuple(zip(words, indices))
    assert "rooftop" in constrained.query_words
    assert constrained.source == "constrained-aerial"


def test_pgrf_only_raises_a_score_when_instance_evidence_is_stronger() -> None:
    semantic_dominant = _pgrf_prompt_score(
        _prompt_components(0.70, instance=0.40, object_score=0.95)
    )
    instance_supported = _pgrf_prompt_score(
        _prompt_components(0.40, instance=0.90, object_score=0.95)
    )

    assert semantic_dominant.item() == pytest.approx(0.70)
    assert 0.40 < instance_supported.item() <= 0.90


def test_baseline_component_scores_are_not_sigmoided_twice() -> None:
    class_scores = torch.tensor([[[0.10]], [[0.80]]])
    components = Sam3ImageComponents(prompt_outputs=(), class_logits=class_scores)
    assert torch.equal(_class_probabilities_from_baseline(components, 2), class_scores)


def test_region_inputs_execute_the_tlp_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDino:
        device = torch.device("cpu")
        satellite = None

        def encode_image(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.tensor([[[[1.0]], [[0.0]]]]), torch.tensor([[1.0, 0.0]])

        @staticmethod
        def similarity_logits(patch_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
            return torch.einsum("bdhw,cd->bchw", patch_features, text_features)

    called: dict[str, object] = {}

    def fake_tlp(
        logits: torch.Tensor,
        rgb: torch.Tensor,
        text_features: torch.Tensor,
        config: object,
        structure: torch.Tensor | None,
    ) -> tuple[torch.Tensor, object]:
        called["shapes"] = (tuple(logits.shape), tuple(rgb.shape), tuple(text_features.shape))
        assert structure is None
        return logits + 1.0, object()

    monkeypatch.setattr("dinotool.sam3_dino.text_aware_laplacian_propagation", fake_tlp)
    predictor = Sam3DinoPredictor.__new__(Sam3DinoPredictor)
    predictor.dino = FakeDino()
    predictor.text_features = torch.eye(2)
    predictor.fusion_config = SimpleNamespace(dino_input_resolution=16, use_dino_tlp=True)
    predictor.tlp_config = object()

    logits, patch_features, anchor = predictor._dino_region_inputs(Image.new("RGB", (16, 16)))

    assert called["shapes"] == ((1, 2, 1, 1), (1, 3, 1, 1), (2, 2))
    assert torch.equal(logits, torch.tensor([[[2.0]], [[1.0]]]))
    assert patch_features.shape == (1, 2, 1, 1)
    assert torch.equal(anchor, torch.tensor([1.0, 0.0]))


def test_loveda_commands_forward_start_index_only_to_sam3_dino(monkeypatch: pytest.MonkeyPatch) -> None:
    regular_call: dict[str, object] = {}

    class FakeRegularBenchmark:
        def __init__(self, *args: object) -> None:
            pass

        def run(self, *args: object, **kwargs: object) -> dict[str, object]:
            regular_call["kwargs"] = kwargs
            return {"ok": True}

    fusion_config_call: dict[str, object] = {}

    class FakeFusionConfig:
        def __init__(self, **kwargs: object) -> None:
            fusion_config_call.update(kwargs)

    class FakeSam3Config:
        @classmethod
        def from_root(cls, *args: object, **kwargs: object) -> object:
            return object()

    fusion_call: dict[str, object] = {}

    class FakeSam3DinoBenchmark:
        def __init__(self, *args: object) -> None:
            pass

        def run(self, *args: object, **kwargs: object) -> dict[str, object]:
            fusion_call["kwargs"] = kwargs
            return {"ok": True}

    monkeypatch.setattr(cli, "DINOTextSegmenter", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "LoveDABenchmark", FakeRegularBenchmark)
    monkeypatch.setattr(cli, "Sam3Config", FakeSam3Config)
    monkeypatch.setattr(cli, "Sam3DinoFusionConfig", FakeFusionConfig)
    monkeypatch.setattr(cli, "Sam3DinoPredictor", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "Sam3DinoLoveDABenchmark", FakeSam3DinoBenchmark)
    monkeypatch.setattr(cli, "make_checkpoints", lambda args: object())
    monkeypatch.setattr(cli, "make_inference_config", lambda args, mode: Namespace(confidence_threshold=None))
    monkeypatch.setattr(cli, "make_tlp_config", lambda args: object())
    monkeypatch.setattr(cli, "make_gsup_config", lambda args: object())

    regular_args = Namespace(
        class_config=None,
        modes=("baseline",),
        device="cpu",
        no_amp=True,
        data_root="data",
        output_dir="out",
        max_images=10,
        progress_every=1,
    )
    assert cli.run_loveda(regular_args) == {"ok": True}
    assert regular_call["kwargs"] == {"max_images": 10, "progress_every": 1}

    fusion_args = Namespace(
        sam3_root="sam3",
        sam3_checkpoint=None,
        sam3_bpe_path=None,
        sam3_class_file=None,
        device="cpu",
        sam3_resolution=1008,
        sam3_confidence_threshold=0.5,
        sam3_probability_threshold=0.5,
        no_amp=True,
        dino_input_resolution=1024,
        sam3_posterior_temperature=0.25,
        dino_posterior_temperature=0.07,
        dino_semantic_strength=0.0,
        dino_structure_strength=0.2,
        dino_uncertainty_power=0.0,
        no_dino_tlp=False,
        data_root="data",
        output_dir="out",
        max_images=10,
        start_index=1000,
        progress_every=1,
    )
    assert cli.run_loveda_sam3_dino(fusion_args) == {"ok": True}
    assert fusion_call["kwargs"] == {"max_images": 10, "start_index": 1000, "progress_every": 1}
    assert fusion_config_call["use_constrained_prompts"] is False
    assert fusion_config_call["use_pgrf"] is False
    assert fusion_config_call["use_satellite_structure"] is False
    assert fusion_config_call["legacy_pixel_fusion"] is False
    assert fusion_config_call["geometry"].enabled is False
    assert fusion_config_call["region_verifier"].enabled is True
