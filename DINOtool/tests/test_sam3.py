from dataclasses import replace

import numpy as np
import pytest
import torch

from dinotool.sam3 import (
    Sam3GeometryConfig,
    Sam3ImageContext,
    SegEarthOV3Predictor,
    _component_boxes,
    _read_query_file,
    _xyxy_to_normalized_cxcywh,
)


def test_read_query_file_preserves_synonym_class_mapping(tmp_path) -> None:
    prompts = tmp_path / "classes.txt"
    prompts.write_text("background\nbuilding, house\nroad\n", encoding="utf-8")

    words, indices = _read_query_file(prompts)

    assert words == ["background", "building", "house", "road"]
    assert indices == [0, 1, 1, 2]


def test_geometry_component_boxes_rank_by_evidence_and_normalize() -> None:
    mask = np.zeros((7, 9), dtype=bool)
    mask[1:3, 2:4] = True
    mask[4:7, 5:8] = True
    score_map = np.zeros(mask.shape, dtype=np.float32)
    score_map[1:3, 2:4] = 0.9
    score_map[4:7, 5:8] = 0.3

    boxes = _component_boxes(
        mask,
        minimum_area=3,
        maximum_components=1,
        score_map=score_map,
        expand_ratio=0.0,
    )

    assert boxes == [(2.0, 1.0, 4.0, 3.0)]
    assert _xyxy_to_normalized_cxcywh(boxes[0], width=9, height=7) == pytest.approx(
        [3.0 / 9.0, 2.0 / 7.0, 2.0 / 9.0, 2.0 / 7.0]
    )


def test_geometry_config_rejects_invalid_probability_threshold() -> None:
    with pytest.raises(ValueError, match="probability_threshold"):
        Sam3GeometryConfig(probability_threshold=1.0).validate()


def test_geometry_config_rejects_zero_prompt_budget() -> None:
    with pytest.raises(ValueError, match="maximum_prompt_calls"):
        Sam3GeometryConfig(maximum_prompt_calls=0).validate()


def test_context_grounding_supports_dynamic_class_count_and_geometry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcessor:
        def __init__(self) -> None:
            self.reset_count = 0

        def reset_all_prompts(self, state: dict[str, object]) -> None:
            self.reset_count += 1
            state.pop("prompt", None)

        def set_text_prompt(self, *, state: dict[str, object], prompt: str) -> dict[str, object]:
            value = {"road": 0.2, "water": 0.4, "building": 0.6}[prompt]
            state["prompt"] = prompt
            state["semantic_mask_logits"] = torch.full((2, 3), value)
            state["presence_score"] = torch.tensor(1.0)
            return state

    predictor = SegEarthOV3Predictor.__new__(SegEarthOV3Predictor)
    predictor.device = torch.device("cpu")
    predictor.use_amp = False
    predictor.num_classes = 7
    predictor.processor = FakeProcessor()
    calls: list[str] = []

    def fake_geometry(state, components, **kwargs):
        calls.append(components.prompt)
        return replace(components, geometry_attempted=True, geometry_accepted=True)

    monkeypatch.setattr(predictor, "_geometric_reprompt", fake_geometry)
    context = Sam3ImageContext(state={}, width=3, height=2)
    components = predictor.predict_components_from_context(
        context,
        query_words=("road", "water", "building"),
        query_class_indices=(0, 1, 2),
        class_count=3,
        geometry=Sam3GeometryConfig(enabled=True, reprompt_background=True, maximum_prompt_calls=2),
    )

    assert components.class_logits.shape == (3, 2, 3)
    assert components.class_logits[:, 0, 0].tolist() == pytest.approx([0.2, 0.4, 0.6])
    assert calls == ["road", "water"]
    assert components.geometry_attempted == 2
    assert components.geometry_accepted == 2
    assert predictor.processor.reset_count == 4
