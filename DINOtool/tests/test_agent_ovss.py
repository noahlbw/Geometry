import torch

from dinotool import cli
from dinotool import agent_tool
from dinotool.agent_ovss import (
    AgentOVSSConfig,
    AgentSam3DinoSession,
    _active_indices,
    _build_agent_prompt_bank,
    _normalize_classes,
)
from dinotool.prompts import ClassSpec


def test_agent_profiles_bound_dynamic_prompt_and_geometry_work() -> None:
    fast = AgentOVSSConfig.for_profile("fast")
    balanced = AgentOVSSConfig.for_profile("balanced")
    accurate = AgentOVSSConfig.for_profile("accurate")

    assert fast.maximum_prompts_per_class == 1
    assert fast.geometry.enabled is False
    assert fast.maximum_active_classes == 0
    assert balanced.maximum_active_classes == 0
    assert accurate.maximum_active_classes == 0
    assert balanced.geometry.maximum_prompt_calls == 2
    assert accurate.maximum_prompts_per_class == 2
    assert accurate.geometry.maximum_prompt_calls == 4
    assert accurate.use_satellite_structure is True


def test_agent_normalizes_background_and_routes_only_top_global_candidates() -> None:
    classes, background_added = _normalize_classes(
        (ClassSpec("road", ("road",)), ClassSpec("building", ("building",)))
    )

    assert background_added is True
    assert [spec.name for spec in classes] == ["background", "road", "building"]
    assert _active_indices(torch.tensor([0.1, 0.2, 0.9]), 3, 1) == (0, 2)
    assert _active_indices(torch.tensor([0.1, 0.2, 0.9]), 3, 0) == (0, 1, 2)


def test_dynamic_prompt_bank_uses_only_routed_classes_and_respects_budget() -> None:
    classes = (
        ClassSpec("background", ("background",)),
        ClassSpec("road", ("road", "street")),
        ClassSpec("building", ("building", "house")),
    )

    bank = _build_agent_prompt_bank(
        classes,
        maximum_prompts_per_class=1,
        include_aerial_variants=True,
    )

    assert bank.source == "dynamic-agent-vocabulary"
    assert bank.query_words == ("background", "road", "building")
    assert bank.query_class_indices == (0, 1, 2)


def test_agent_cli_builds_fast_profile_with_dynamic_class_limit() -> None:
    args = cli.build_parser().parse_args(
        [
            "infer-sam3-dino",
            "--image",
            "scene.png",
            "--classes",
            "road",
            "building",
            "water",
            "--output-dir",
            "output",
            "--sam3-root",
            "sam3",
            "--profile",
            "fast",
            "--max-active-classes",
            "2",
        ]
    )

    config = cli.make_agent_ovss_config(args)

    assert config.profile == "fast"
    assert config.maximum_active_classes == 2
    assert config.geometry.enabled is False
    assert config.use_dino_tlp is False


def test_persistent_agent_parser_accepts_explicit_warmup() -> None:
    args = agent_tool._service_parser().parse_args(
        ["--serve-sam3-dino", "--sam3-root", "sam3", "--warmup"]
    )

    assert args.warmup is True


def test_agent_text_cache_reuses_individual_class_prototypes() -> None:
    class FakeDino:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def encode_text(self, specs):
            self.calls.append(tuple(spec.name for spec in specs))
            return torch.tensor(
                [[float(index + 1), float(len(spec.name))] for index, spec in enumerate(specs)]
            )

    session = object.__new__(AgentSam3DinoSession)
    session.dino = FakeDino()
    session._text_cache = {}
    session._class_text_cache = {}
    initial = (ClassSpec("background", ("background",)), ClassSpec("road", ("road",)))
    expanded = (*initial, ClassSpec("building", ("building",)))

    _, full_hit, class_hits = session._text_features(initial)
    assert full_hit is False
    assert class_hits == 0
    assert session.dino.calls == [("background", "road")]

    _, full_hit, class_hits = session._text_features(initial)
    assert full_hit is True
    assert class_hits == 2
    assert session.dino.calls == [("background", "road")]

    _, full_hit, class_hits = session._text_features(expanded)
    assert full_hit is False
    assert class_hits == 2
    assert session.dino.calls == [("background", "road"), ("building",)]
