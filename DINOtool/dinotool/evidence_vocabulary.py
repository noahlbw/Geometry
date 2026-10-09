"""Fixed, auditable evidence descriptions for training-free RER-OV."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .prompts import ClassSpec, clean_phrase


@dataclass(frozen=True)
class EvidencePhrase:
    text: str
    family: str
    anchor_class: str
    prior: float = 1.0


@dataclass(frozen=True)
class EvidenceVocabulary:
    phrases: tuple[EvidencePhrase, ...]
    families: tuple[str, ...]

    def for_classes(self, class_names: Sequence[str]) -> "EvidenceVocabulary":
        allowed = {clean_phrase(name) for name in class_names}
        phrases = tuple(item for item in self.phrases if item.anchor_class in allowed)
        if not phrases:
            raise ValueError("Evidence vocabulary has no phrases for the requested classes.")
        families = tuple(family for family in self.families if any(p.family == family for p in phrases))
        return EvidenceVocabulary(phrases=phrases, families=families)


@dataclass(frozen=True)
class EncodedEvidenceBank:
    features: Tensor
    family_indices: Tensor
    anchor_indices: Tensor
    priors: Tensor
    phrases: tuple[str, ...]
    families: tuple[str, ...]
    class_names: tuple[str, ...]

    @property
    def phrase_count(self) -> int:
        return len(self.phrases)

    @property
    def family_count(self) -> int:
        return len(self.families)

    def validate(self) -> None:
        count = self.features.shape[0]
        if self.features.ndim != 2 or count < 1:
            raise ValueError("Evidence features must have shape [R,D] with R > 0.")
        for tensor in (self.family_indices, self.anchor_indices, self.priors):
            if tensor.shape != (count,):
                raise ValueError("Evidence metadata must have one value per phrase.")
        if len(self.phrases) != count or not self.families or not self.class_names:
            raise ValueError("Evidence metadata is incomplete.")
        if int(self.family_indices.min()) < 0 or int(self.family_indices.max()) >= len(self.families):
            raise ValueError("Evidence family index is out of range.")
        if int(self.anchor_indices.min()) < 0 or int(self.anchor_indices.max()) >= len(self.class_names):
            raise ValueError("Evidence class index is out of range.")
        if not bool(torch.isfinite(self.priors).all()) or not bool((self.priors > 0).all()):
            raise ValueError("Evidence priors must be finite and positive.")


def load_evidence_vocabulary(path: str | Path) -> EvidenceVocabulary:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("phrases") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("Evidence config must contain a non-empty 'phrases' list.")
    phrases: list[EvidencePhrase] = []
    seen: set[tuple[str, str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Every evidence phrase must be an object.")
        text = clean_phrase(str(entry.get("text", "")))
        family = clean_phrase(str(entry.get("family", "")))
        anchor = clean_phrase(str(entry.get("class", "")))
        prior = float(entry.get("prior", 1.0))
        if prior <= 0:
            raise ValueError(f"Evidence phrase '{text}' has a non-positive prior.")
        key = (text.casefold(), family.casefold(), anchor.casefold())
        if key in seen:
            continue
        seen.add(key)
        phrases.append(EvidencePhrase(text=text, family=family, anchor_class=anchor, prior=prior))
    declared = payload.get("families", []) if isinstance(payload, dict) else []
    families = tuple(dict.fromkeys(clean_phrase(str(value)) for value in declared))
    discovered = tuple(dict.fromkeys(item.family for item in phrases))
    families = tuple(dict.fromkeys((*families, *discovered)))
    return EvidenceVocabulary(phrases=tuple(phrases), families=families)


def encode_evidence_vocabulary(
    backbone,
    vocabulary: EvidenceVocabulary,
    class_names: Sequence[str],
    *,
    batch_size: int = 64,
) -> EncodedEvidenceBank:
    selected = vocabulary.for_classes(class_names)
    names = tuple(clean_phrase(name) for name in class_names)
    class_lookup = {name: index for index, name in enumerate(names)}
    family_lookup = {name: index for index, name in enumerate(selected.families)}
    specs = [ClassSpec.from_name(item.text) for item in selected.phrases]
    features = backbone.encode_text(specs, batch_size=batch_size)
    device = features.device
    bank = EncodedEvidenceBank(
        features=features,
        family_indices=torch.tensor(
            [family_lookup[item.family] for item in selected.phrases], device=device, dtype=torch.long
        ),
        anchor_indices=torch.tensor(
            [class_lookup[item.anchor_class] for item in selected.phrases], device=device, dtype=torch.long
        ),
        priors=torch.tensor([item.prior for item in selected.phrases], device=device),
        phrases=tuple(item.text for item in selected.phrases),
        families=selected.families,
        class_names=names,
    )
    bank.validate()
    return bank
