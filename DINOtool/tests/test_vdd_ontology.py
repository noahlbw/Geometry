from pathlib import Path

from dinotool.prompts import load_class_specs


def test_official_vdd_vocabulary_tracks_dataset_semantics():
    path = Path(__file__).resolve().parents[1] / "configs" / "grounded_vdd_official20.json"
    classes = load_class_specs(path)
    assert tuple(item.name for item in classes) == (
        "other", "wall", "road", "vegetation", "vehicle", "roof", "water"
    )
    assert all(len(item.synonyms) == 20 for item in classes)
    assert not {"roof", "rooftop"} & set(classes[1].synonyms)
