from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REMOTE_SENSING_TEMPLATES: tuple[str, ...] = (
    "a satellite image of {label}.",
    "an aerial image of {label}.",
    "a remote sensing image of {label}.",
    "an overhead view of {label}.",
    "{label} seen from above.",
    "a high resolution satellite view of {label}.",
)


@dataclass(frozen=True)
class ClassSpec:
    name: str
    synonyms: tuple[str, ...]

    @classmethod
    def from_name(cls, name: str) -> "ClassSpec":
        cleaned = clean_phrase(name)
        return cls(name=cleaned, synonyms=(cleaned,))


def clean_phrase(value: str) -> str:
    cleaned = " ".join(value.strip().replace("_", " ").split())
    if not cleaned:
        raise ValueError("Class names and synonyms must be non-empty strings.")
    return cleaned


def class_specs_from_labels(labels: Sequence[str]) -> list[ClassSpec]:
    if not labels:
        raise ValueError("At least one class is required.")
    specs = [ClassSpec.from_name(label) for label in labels]
    _validate_unique_names(specs)
    return specs


def load_class_specs(path: str | Path) -> list[ClassSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("classes") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError("Class config must contain a non-empty 'classes' list.")
    specs: list[ClassSpec] = []
    for entry in entries:
        if isinstance(entry, str):
            specs.append(ClassSpec.from_name(entry))
            continue
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError("Each class must be a string or an object with a 'name'.")
        name = clean_phrase(str(entry["name"]))
        synonyms = entry.get("synonyms", [name])
        if not isinstance(synonyms, list) or not synonyms:
            raise ValueError(f"Class '{name}' must have a non-empty synonyms list.")
        cleaned = tuple(dict.fromkeys(clean_phrase(str(item)) for item in synonyms))
        if name not in cleaned:
            cleaned = (name,) + cleaned
        specs.append(ClassSpec(name=name, synonyms=cleaned))
    _validate_unique_names(specs)
    return specs


def load_vip_official_aliases(path: str | Path) -> tuple[tuple[str, ...], ...]:
    """Match VIP's get_cls_idx: only comma followed by space splits queries."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError(f"Invalid VIP class file: {path}")
    return tuple(tuple(line.split(", ")) for line in lines)


def prompts_for_class(
    spec: ClassSpec,
    templates: Sequence[str] = REMOTE_SENSING_TEMPLATES,
) -> list[str]:
    prompts = [template.format(label=synonym) for synonym in spec.synonyms for template in templates]
    return list(dict.fromkeys(prompts))


def serialize_class_specs(specs: Iterable[ClassSpec]) -> list[dict[str, object]]:
    return [{"name": spec.name, "synonyms": list(spec.synonyms)} for spec in specs]


def _validate_unique_names(specs: Sequence[ClassSpec]) -> None:
    names = [spec.name.casefold() for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("Class names must be unique (case-insensitive).")
    if len(specs) > 254:
        raise ValueError("At most 254 classes are supported; label 255 is reserved for unknown.")

