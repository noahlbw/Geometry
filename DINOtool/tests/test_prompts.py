import json

import pytest

from dinotool.prompts import ClassSpec, class_specs_from_labels, load_class_specs, prompts_for_class


def test_remote_prompts_expand_synonyms() -> None:
    spec = ClassSpec("building", ("building", "rooftop"))
    prompts = prompts_for_class(spec)
    assert len(prompts) == 12
    assert any("rooftop" in prompt for prompt in prompts)
    assert any("seen from above" in prompt for prompt in prompts)


def test_load_class_config_adds_canonical_name(tmp_path) -> None:
    path = tmp_path / "classes.json"
    path.write_text(
        json.dumps({"classes": [{"name": "bare_soil", "synonyms": ["exposed earth"]}]}),
        encoding="utf-8",
    )
    specs = load_class_specs(path)
    assert specs[0].name == "bare soil"
    assert specs[0].synonyms == ("bare soil", "exposed earth")


def test_duplicate_class_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        class_specs_from_labels(["Road", "road"])

