"""Reciprocal Evidence Reading for frozen DINOv3/DINO.text OVSS."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .contrastive_support import (
    ContrastiveSupportConfig,
    ContrastiveSupportResult,
    build_contrastive_support,
)
from .evidence_consensus import (
    EvidenceConsensusConfig,
    EvidenceConsensusDiagnostics,
    solve_evidence_consensus,
)
from .evidence_vocabulary import EncodedEvidenceBank, EvidenceVocabulary, encode_evidence_vocabulary
from .tcpr import (
    TCPRConfig,
    TCPRPreparedImage,
    TCPRSegmenter,
    TCPRTextBank,
    _normalize_valid_mask,
    _tokens_to_map,
    aggregate_alias_scores,
    filter_aliases,
)
from .text_visual_intervention import _text_conditioned_relation


@dataclass(frozen=True)
class ReciprocalReadoutConfig:
    alias_temperature: float = 0.07
    class_temperature: float = 0.07
    semantic_relation_strength: float = 4.0
    rounds: int = 2
    feedback_temperature: float = 0.06
    feedback_damping: float = 0.50
    feedback_prior_mass: float = 0.25
    feedback_minimum: float = 0.25
    feedback_maximum: float = 4.0
    contrastive: ContrastiveSupportConfig = ContrastiveSupportConfig()
    consensus: EvidenceConsensusConfig = EvidenceConsensusConfig()

    def validate(self) -> None:
        if min(
            self.alias_temperature, self.class_temperature, self.semantic_relation_strength,
            self.rounds, self.feedback_temperature, self.feedback_minimum,
            self.feedback_maximum,
        ) <= 0:
            raise ValueError("RER temperatures, rounds, and feedback bounds must be positive.")
        if self.rounds > 3:
            raise ValueError("RER intentionally supports at most three inference rounds.")
        if not 0 <= self.feedback_damping <= 1 or not 0 <= self.feedback_prior_mass <= 1:
            raise ValueError("Feedback damping and prior mass must be in [0,1].")
        if self.feedback_minimum > self.feedback_maximum:
            raise ValueError("feedback_minimum must not exceed feedback_maximum.")
        self.contrastive.validate()
        self.consensus.validate()


@dataclass(frozen=True)
class ReciprocalReadoutDiagnostics:
    rounds: int
    changed_from_s0: float
    evidence_coverage: float
    mean_availability: float
    mean_phrase_weight: float
    consensus: tuple[EvidenceConsensusDiagnostics, ...]

    def summary(self) -> dict[str, object]:
        payload = asdict(self)
        payload["consensus"] = [asdict(value) for value in self.consensus]
        return payload


@dataclass(frozen=True)
class ReciprocalReadoutResult:
    logits: Tensor
    s0_logits: Tensor
    evidence_map: Tensor
    diagnostics: ReciprocalReadoutDiagnostics


class ReciprocalReadoutSegmenter:
    """Training-free RER-OV wrapper around the existing Geometry readout."""

    method_name = "RER-OV: reciprocal evidence reading"

    def __init__(
        self,
        backbone,
        config: ReciprocalReadoutConfig = ReciprocalReadoutConfig(),
        tcpr_config: TCPRConfig = TCPRConfig(),
    ) -> None:
        config.validate()
        self.backbone = backbone
        self.config = config
        self.base = TCPRSegmenter(backbone, tcpr_config)

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    @property
    def patch_size(self) -> int:
        return self.backbone.patch_size

    def prepare_image(self, rgb: Tensor) -> TCPRPreparedImage:
        return self.base.prepare_image(rgb)

    def encode_text(self, classes, batch_size: int = 64) -> TCPRTextBank:
        return self.base.encode_text(classes, batch_size=batch_size)

    def encode_evidence(
        self,
        vocabulary: EvidenceVocabulary,
        class_names,
        *,
        batch_size: int = 64,
    ) -> EncodedEvidenceBank:
        return encode_evidence_vocabulary(
            self.backbone, vocabulary, class_names, batch_size=batch_size
        )

    @torch.inference_mode()
    def read_prepared(
        self,
        prepared: TCPRPreparedImage,
        text_bank: TCPRTextBank,
        evidence_bank: EncodedEvidenceBank,
        *,
        valid_mask: Tensor | None = None,
    ) -> ReciprocalReadoutResult:
        text_bank.validate()
        evidence_bank.validate()
        batch, patches, _ = prepared.geometry_projected.shape
        valid = _normalize_valid_mask(
            valid_mask, batch, prepared.grid_height, prepared.grid_width,
            prepared.geometry_projected.device,
        ).reshape(batch, patches)
        if tuple(text_bank.class_names) != tuple(evidence_bank.class_names):
            raise ValueError("Recognition and evidence banks must use the same class order.")

        s0, class_text = self._initial_readout(prepared, text_bank, valid)
        phrase_weights: Tensor | None = None
        fixed_pairs: Tensor | None = None
        current = s0
        consensus_diagnostics: list[EvidenceConsensusDiagnostics] = []
        final_support: ContrastiveSupportResult | None = None
        for round_index in range(self.config.rounds):
            support = build_contrastive_support(
                prepared.raw_patch_tokens,
                prepared.geometry_projected,
                prepared.geometry_patch_conditional.float(),
                class_text,
                current,
                evidence_bank,
                valid,
                self.config.contrastive,
                phrase_weights=phrase_weights,
                pair_ids=fixed_pairs,
            )
            if fixed_pairs is None:
                fixed_pairs = support.pair_ids
            consensus = solve_evidence_consensus(
                s0, support, valid, self.config.consensus
            )
            current = consensus.scores
            consensus_diagnostics.append(consensus.diagnostics)
            final_support = support
            if round_index + 1 < self.config.rounds:
                phrase_weights = self._cross_family_feedback(
                    s0, support, evidence_bank, valid, phrase_weights
                )

        if final_support is None:
            raise RuntimeError("RER produced no inference round.")
        old_label = s0.argmax(-1)
        new_label = current.argmax(-1)
        valid_count = max(int(valid.sum()), 1)
        availability = final_support.availability
        evidence_patch = availability.sum(dim=(-1, -2)) > 0
        diagnostics = ReciprocalReadoutDiagnostics(
            rounds=self.config.rounds,
            changed_from_s0=float(((old_label != new_label) & valid).sum().item() / valid_count),
            evidence_coverage=float((evidence_patch & valid).sum().item() / valid_count),
            mean_availability=float(
                availability.sum().item() / max(int((availability > 0).sum()), 1)
            ),
            mean_phrase_weight=float(phrase_weights.mean().item()) if phrase_weights is not None else 1.0,
            consensus=tuple(consensus_diagnostics),
        )
        evidence_map = availability.sum(dim=(-1, -2)).reshape(
            batch, prepared.grid_height, prepared.grid_width
        )
        return ReciprocalReadoutResult(
            logits=_tokens_to_map(current, prepared.grid_height, prepared.grid_width),
            s0_logits=_tokens_to_map(s0, prepared.grid_height, prepared.grid_width),
            evidence_map=evidence_map,
            diagnostics=diagnostics,
        )

    def _initial_readout(
        self,
        prepared: TCPRPreparedImage,
        text_bank: TCPRTextBank,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        visual = F.normalize(prepared.geometry_projected.float(), dim=-1)
        text = F.normalize(text_bank.features.float(), dim=-1)
        geometry_alias = visual @ text.T
        native_alias = prepared.native_projected.float() @ text.T
        alias_weights = filter_aliases(
            geometry_alias,
            native_alias,
            text_bank.parent_indices,
            text_bank.canonical_mask,
            text_bank.class_count,
            valid,
            self.base.config,
        )
        expanded_scores = aggregate_alias_scores(
            geometry_alias,
            alias_weights,
            text_bank.parent_indices,
            text_bank.class_count,
            self.config.alias_temperature,
        )
        probability = torch.softmax(expanded_scores / self.config.class_temperature, dim=-1)
        geometry = prepared.geometry_patch_conditional.float()
        geometry = geometry.masked_fill(~valid[:, None, :], 0.0)
        geometry = geometry / geometry.sum(-1, keepdim=True).clamp_min(1e-8)
        relation = _text_conditioned_relation(
            geometry, probability, valid, self.config.semantic_relation_strength
        )
        s0 = relation @ expanded_scores
        canonical = []
        for class_index in range(text_bank.class_count):
            member = (text_bank.parent_indices == class_index) & text_bank.canonical_mask
            canonical.append(text[member][0])
        return s0, torch.stack(canonical)

    def _cross_family_feedback(
        self,
        s0: Tensor,
        support: ContrastiveSupportResult,
        evidence_bank: EncodedEvidenceBank,
        valid: Tensor,
        previous: Tensor | None,
    ) -> Tensor:
        batch, patches, pair_count, phrase_count = support.phrase_directions.shape
        previous = (
            torch.ones((batch, patches, pair_count, phrase_count), device=s0.device)
            if previous is None else previous
        )
        updated = previous.clone()
        support_mean = support.phrase_support.mean(-1, keepdim=True).clamp_min(1e-6)
        visual_quality = torch.sigmoid(
            2.0 * (support.phrase_support / support_mean - 1.0)
        )[:, :, None]
        for family in range(evidence_bank.family_count):
            members = evidence_bank.family_indices == family
            if not bool(members.any()):
                continue
            leave_out = solve_evidence_consensus(
                s0,
                support,
                valid,
                self.config.consensus,
                excluded_family=family,
            ).scores
            margin = _pair_margin(leave_out, support.pair_ids)
            direction = support.phrase_directions[..., members]
            consistency = torch.sigmoid(
                direction.sign() * margin[..., None] / self.config.feedback_temperature
            )
            target = consistency * (0.5 + visual_quality[..., members])
            target = target / target.mean(-1, keepdim=True).clamp_min(1e-6)
            target = (
                (1.0 - self.config.feedback_prior_mass) * target
                + self.config.feedback_prior_mass
            )
            blended = (
                (1.0 - self.config.feedback_damping) * previous[..., members]
                + self.config.feedback_damping * target
            )
            updated[..., members] = blended.clamp(
                self.config.feedback_minimum, self.config.feedback_maximum
            )
        return torch.where(valid[:, :, None, None], updated, torch.ones_like(updated))


def _pair_margin(values: Tensor, pairs: Tensor) -> Tensor:
    return values.gather(-1, pairs[..., 0]) - values.gather(-1, pairs[..., 1])
