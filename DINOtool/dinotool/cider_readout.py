"""Class-competitive influence-guided differential evidence routing."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .competitive_evidence_readout import (
    calibrate_edge_bias,
    redistribute_candidate_attention,
    _geometry_candidates,
    _normalized_geometry,
    _ordered_competitors,
    _roll_valid,
)
from .tcpr import (
    TCPRConfig,
    TCPRPreparedImage,
    TCPRSegmenter,
    TCPRTextBank,
    _finish_attention_block,
    _finish_head,
    _geometry_attended,
    _normalize_valid_mask,
    _tokens_to_map,
    aggregate_alias_scores,
    filter_aliases,
)
from .text_visual_intervention import _text_conditioned_relation


@dataclass(frozen=True)
class CIDERConfig:
    alias_temperature: float = 0.07
    class_temperature: float = 0.07
    semantic_relation_strength: float = 4.0
    pair_text_temperature: float = 0.07
    candidate_members: int = 24
    maximum_attention_kl: float = 0.03
    attention_bisection_steps: int = 10
    query_chunk: int = 32
    shuffled_roll_fraction: float = 0.5
    evidence_strength: float = 1.0
    epsilon: float = 1e-6

    def validate(self) -> None:
        positive = (
            self.alias_temperature,
            self.class_temperature,
            self.semantic_relation_strength,
            self.pair_text_temperature,
            self.candidate_members,
            self.maximum_attention_kl,
            self.attention_bisection_steps,
            self.query_chunk,
            self.epsilon,
        )
        if min(positive) <= 0:
            raise ValueError("CIDER temperatures, counts, KL, and epsilon must be positive.")
        if self.evidence_strength < 0:
            raise ValueError("evidence_strength must be non-negative.")
        if not 0.0 < self.shuffled_roll_fraction < 1.0:
            raise ValueError("shuffled_roll_fraction must be in (0,1).")


@dataclass(frozen=True)
class CIDERArmDiagnostics:
    changed_from_s0: float
    evidence_coverage: float
    mean_evidence_magnitude: float
    mean_attention_kl: float
    maximum_attention_kl: float
    influence_sign_agreement: float
    mean_predicted_margin_change: float
    mean_actual_margin_change: float

    def summary(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CIDERResult:
    s0_logits: Tensor
    image_self_logits: Tensor
    cider_logits: Tensor
    shuffled_logits: Tensor
    evidence_map: Tensor
    shuffled_evidence_map: Tensor
    diagnostics: CIDERArmDiagnostics
    shuffled_diagnostics: CIDERArmDiagnostics


@dataclass(frozen=True)
class _Initial:
    geometry: Tensor
    valid: Tensor
    visual: Tensor
    text: Tensor
    alias_weights: Tensor
    class_scores: Tensor
    s0_scores: Tensor
    pair_directions: Tensor


class CIDERSegmenter:
    """Frozen DINO.text with class-pair-specific differential visual rereading."""

    method_name = "CIDER-OV"

    def __init__(
        self,
        backbone,
        config: CIDERConfig = CIDERConfig(),
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

    def read_prepared(
        self,
        prepared: TCPRPreparedImage,
        text_bank: TCPRTextBank,
        *,
        valid_mask: Tensor | None = None,
    ) -> CIDERResult:
        initial = self._initial_readout(prepared, text_bank, valid_mask)
        pairs = _ordered_competitors(initial.s0_scores)
        directions = gather_pair_directions(initial.pair_directions, pairs)
        cider, evidence, diagnostics = self._read_arm(
            prepared, text_bank, initial, pairs, directions
        )
        shuffled_pairs = _roll_valid(pairs, initial.valid, self.config.shuffled_roll_fraction)
        shuffled_directions = _roll_valid(
            directions, initial.valid, self.config.shuffled_roll_fraction
        )
        shuffled, shuffled_evidence, shuffled_diagnostics = self._read_arm(
            prepared, text_bank, initial, shuffled_pairs, shuffled_directions
        )

        image_context = F.normalize(initial.geometry @ initial.visual, dim=-1)
        image_features = F.normalize(0.5 * initial.visual + 0.5 * image_context, dim=-1)
        image_alias = image_features @ initial.text.T
        image_scores = aggregate_alias_scores(
            image_alias,
            initial.alias_weights,
            text_bank.parent_indices,
            text_bank.class_count,
            self.config.alias_temperature,
        )
        height, width = prepared.grid_height, prepared.grid_width
        return CIDERResult(
            s0_logits=_tokens_to_map(initial.s0_scores, height, width),
            image_self_logits=_tokens_to_map(image_scores, height, width),
            cider_logits=_tokens_to_map(cider, height, width),
            shuffled_logits=_tokens_to_map(shuffled, height, width),
            evidence_map=evidence.reshape(evidence.shape[0], height, width),
            shuffled_evidence_map=shuffled_evidence.reshape(
                shuffled_evidence.shape[0], height, width
            ),
            diagnostics=diagnostics,
            shuffled_diagnostics=shuffled_diagnostics,
        )

    def _initial_readout(
        self,
        prepared: TCPRPreparedImage,
        text_bank: TCPRTextBank,
        valid_mask: Tensor | None,
    ) -> _Initial:
        text_bank.validate()
        visual = F.normalize(prepared.geometry_projected.float(), dim=-1)
        batch, patches, _ = visual.shape
        valid = _normalize_valid_mask(
            valid_mask,
            batch,
            prepared.grid_height,
            prepared.grid_width,
            visual.device,
        ).reshape(batch, patches)
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
        class_scores = aggregate_alias_scores(
            geometry_alias,
            alias_weights,
            text_bank.parent_indices,
            text_bank.class_count,
            self.config.alias_temperature,
        )
        geometry = _normalized_geometry(prepared.geometry_patch_conditional.float(), valid)
        probability = torch.softmax(class_scores / self.config.class_temperature, dim=-1)
        relation = _text_conditioned_relation(
            geometry, probability, valid, self.config.semantic_relation_strength
        )
        s0 = relation @ class_scores
        directions = pairwise_text_directions(
            text,
            alias_weights,
            text_bank.parent_indices,
            text_bank.canonical_mask,
            text_bank.class_count,
            self.config.pair_text_temperature,
            self.config.epsilon,
        )
        return _Initial(
            geometry=geometry,
            valid=valid,
            visual=visual,
            text=text,
            alias_weights=alias_weights,
            class_scores=class_scores,
            s0_scores=s0,
            pair_directions=directions,
        )

    def _read_arm(
        self,
        prepared: TCPRPreparedImage,
        text_bank: TCPRTextBank,
        initial: _Initial,
        pairs: Tensor,
        directions: Tensor,
    ) -> tuple[Tensor, Tensor, CIDERArmDiagnostics]:
        indices, _ = _geometry_candidates(
            initial.geometry,
            initial.valid,
            self.config.candidate_members,
            self.config.epsilon,
        )
        influence, base_class_margin = self._decision_influence(
            prepared, text_bank, initial, pairs, indices
        )
        candidate_visual = gather_candidates(initial.visual, indices)
        query_semantic = (initial.visual * directions).sum(-1)
        semantic = (
            torch.einsum("bqkd,bqd->bqk", candidate_visual, directions)
            - query_semantic[..., None]
        )
        class_probability = torch.softmax(
            initial.class_scores / self.config.class_temperature, dim=-1
        )
        first_probability = class_probability.gather(-1, pairs[..., 0, None]).squeeze(-1)
        second_probability = class_probability.gather(-1, pairs[..., 1, None]).squeeze(-1)
        pair_mass = first_probability + second_probability
        ambiguity = 4.0 * first_probability * second_probability / pair_mass.square().clamp_min(
            self.config.epsilon
        )
        evidence_demand = pair_mass * ambiguity.clamp(0.0, 1.0).sqrt()
        edge, magnitude = decision_consistent_edges(
            semantic, influence, evidence_demand, initial.valid, self.config.epsilon
        )
        calibrated = calibrate_edge_bias(edge, magnitude, self.config.epsilon)
        modified, row_kl = redistribute_candidate_attention(
            initial.geometry,
            indices,
            calibrated * self.config.evidence_strength,
            initial.valid,
            maximum_kl=self.config.maximum_attention_kl,
            bisection_steps=self.config.attention_bisection_steps,
            epsilon=self.config.epsilon,
        )
        final_scores, _, final_class_scores = self._frozen_reread(
            prepared, text_bank, initial, modified
        )
        final_class_margin = (
            final_class_scores.gather(-1, pairs[..., 0, None]).squeeze(-1)
            - final_class_scores.gather(-1, pairs[..., 1, None]).squeeze(-1)
        )
        candidate_delta = modified.gather(-1, indices) - initial.geometry.gather(-1, indices)
        predicted_delta = (candidate_delta * influence).sum(-1)
        actual_delta = final_class_margin - base_class_margin
        active_prediction = initial.valid & (predicted_delta.abs() > self.config.epsilon)
        agreement = (
            (predicted_delta.sign() == actual_delta.sign()) & active_prediction
        ).sum().item() / max(int(active_prediction.sum()), 1)

        valid_count = max(int(initial.valid.sum()), 1)
        changed = (
            (final_scores.argmax(-1) != initial.s0_scores.argmax(-1)) & initial.valid
        ).sum().item() / valid_count
        active = initial.valid & (magnitude > self.config.epsilon)
        diagnostics = CIDERArmDiagnostics(
            changed_from_s0=float(changed),
            evidence_coverage=float(active.sum().item() / valid_count),
            mean_evidence_magnitude=float(
                magnitude[active].mean().item() if bool(active.any()) else 0.0
            ),
            mean_attention_kl=float(row_kl[initial.valid].mean().item()),
            maximum_attention_kl=float(row_kl.max().item()),
            influence_sign_agreement=float(agreement),
            mean_predicted_margin_change=float(
                predicted_delta[initial.valid].abs().mean().item()
            ),
            mean_actual_margin_change=float(
                actual_delta[initial.valid].abs().mean().item()
            ),
        )
        return final_scores, magnitude, diagnostics

    def _decision_influence(
        self,
        prepared: TCPRPreparedImage,
        text_bank: TCPRTextBank,
        initial: _Initial,
        pairs: Tensor,
        indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        prefix = prepared.prefix_tokens
        with torch.inference_mode(False), torch.enable_grad():
            native_attention = prepared.native_attention.detach().clone()
            geometry = initial.geometry.detach().clone()
            values = prepared.value_tokens.detach().clone()
            shared = prepared.shared_tokens.detach().clone()
            normal_valid = initial.valid.detach().clone()
            normal_text = initial.text.detach().clone()
            normal_weights = initial.alias_weights.detach().clone()
            normal_parents = text_bank.parent_indices.detach().clone()
            normal_pairs = pairs.detach().clone()
            attended = _geometry_attended(native_attention, geometry, values, prefix)
            attended = attended.detach().clone().requires_grad_(True)
            head = self.backbone.model.visual_model.head
            with self.backbone._autocast():
                block = _finish_attention_block(
                    head.blocks[prepared.block_index], shared, attended
                )
                projected = _finish_head(head, block, prepared.block_index)
            visual = F.normalize(projected[:, prefix:].float(), dim=-1)
            alias_scores = visual @ normal_text.T
            class_scores = aggregate_alias_scores(
                alias_scores,
                normal_weights,
                normal_parents,
                text_bank.class_count,
                self.config.alias_temperature,
            )
            margin = (
                class_scores.gather(-1, normal_pairs[..., 0, None]).squeeze(-1)
                - class_scores.gather(-1, normal_pairs[..., 1, None]).squeeze(-1)
            )
            objective = (margin * normal_valid).sum()
            gradient = torch.autograd.grad(objective, attended, retain_graph=False)[0]

        gradient = gradient[:, :, prefix:].detach().float()
        value_patch = prepared.value_tokens[..., prefix:, :].detach().float()
        native_patch_mass = prepared.native_attention[..., prefix:, prefix:].sum(
            -1, keepdim=True
        ).detach().float()
        base_context = torch.einsum(
            "bqk,bhkd->bhqd", initial.geometry.float(), value_patch
        )
        batch, patches, count = indices.shape
        influence = torch.zeros((batch, patches, count), device=indices.device)
        b = torch.arange(batch, device=indices.device)[:, None, None, None]
        h = torch.arange(value_patch.shape[1], device=indices.device)[None, :, None, None]
        for start in range(0, patches, self.config.query_chunk):
            stop = min(start + self.config.query_chunk, patches)
            selected = value_patch[b, h, indices[:, None, start:stop], :]
            delta = selected - base_context[:, :, start:stop, None, :]
            delta = delta * native_patch_mass[:, :, start:stop, None, :]
            local_gradient = gradient[:, :, start:stop, None, :]
            influence[:, start:stop] = (local_gradient * delta).sum(dim=(1, 4))
        return influence, margin.detach()

    def _frozen_reread(
        self,
        prepared: TCPRPreparedImage,
        text_bank: TCPRTextBank,
        initial: _Initial,
        geometry: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        head = self.backbone.model.visual_model.head
        with torch.inference_mode(), self.backbone._autocast():
            attended = _geometry_attended(
                prepared.native_attention,
                geometry,
                prepared.value_tokens,
                prepared.prefix_tokens,
            )
            block = _finish_attention_block(
                head.blocks[prepared.block_index], prepared.shared_tokens, attended
            )
            projected = _finish_head(head, block, prepared.block_index)
        visual = F.normalize(projected[:, prepared.prefix_tokens:].float(), dim=-1)
        alias_scores = visual @ initial.text.T
        class_scores = aggregate_alias_scores(
            alias_scores,
            initial.alias_weights,
            text_bank.parent_indices,
            text_bank.class_count,
            self.config.alias_temperature,
        )
        probability = torch.softmax(class_scores / self.config.class_temperature, dim=-1)
        relation = _text_conditioned_relation(
            geometry, probability, initial.valid, self.config.semantic_relation_strength
        )
        output = relation @ class_scores
        output = torch.where(initial.valid[..., None], output, initial.s0_scores)
        return output, visual, class_scores


def pairwise_text_directions(
    text: Tensor,
    alias_weights: Tensor,
    parents: Tensor,
    canonical_mask: Tensor,
    class_count: int,
    temperature: float,
    epsilon: float,
) -> Tensor:
    """Build image-adaptive A-vs-B text bases from all retained aliases."""
    canonical = torch.stack(
        [text[(parents == index) & canonical_mask][0] for index in range(class_count)]
    )
    batch, _ = alias_weights.shape
    output = torch.zeros(
        (batch, class_count, class_count, text.shape[-1]),
        device=text.device,
        dtype=torch.float32,
    )
    for first in range(class_count):
        first_members = parents == first
        first_text = text[first_members]
        for second in range(class_count):
            if first == second:
                continue
            second_members = parents == second
            second_text = text[second_members]
            text_difference = canonical[first] - canonical[second]
            first_logits = (
                first_text @ text_difference / temperature
                + alias_weights[:, first_members].clamp_min(epsilon).log()
            )
            second_logits = (
                second_text @ (-text_difference) / temperature
                + alias_weights[:, second_members].clamp_min(epsilon).log()
            )
            first_prototype = torch.einsum(
                "bm,md->bd", torch.softmax(first_logits, dim=-1), first_text
            )
            second_prototype = torch.einsum(
                "bm,md->bd", torch.softmax(second_logits, dim=-1), second_text
            )
            output[:, first, second] = F.normalize(
                first_prototype - second_prototype, dim=-1
            )
    return output


def gather_pair_directions(directions: Tensor, pairs: Tensor) -> Tensor:
    batch = torch.arange(directions.shape[0], device=directions.device)[:, None]
    return directions[batch, pairs[..., 0], pairs[..., 1]]


def gather_candidates(values: Tensor, indices: Tensor) -> Tensor:
    batch = torch.arange(values.shape[0], device=values.device)[:, None, None]
    return values[batch, indices]


def decision_consistent_edges(
    semantic: Tensor,
    influence: Tensor,
    demand: Tensor,
    valid: Tensor,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    """Keep edges whose text sign agrees with their frozen-decoder influence."""
    if semantic.shape != influence.shape:
        raise ValueError("semantic and influence edges must have identical shape.")
    semantic_scale = semantic.square().mean(-1, keepdim=True).sqrt().clamp_min(epsilon)
    influence_scale = influence.square().mean(-1, keepdim=True).sqrt().clamp_min(epsilon)
    semantic_unit = torch.tanh(semantic / semantic_scale)
    influence_unit = torch.tanh(influence / influence_scale)
    product = F.relu(semantic_unit * influence_unit)
    product = product.masked_fill(~valid[..., None], 0.0)
    magnitude = product.mean(-1) * demand.clamp(0.0, 1.0)
    magnitude = magnitude * valid
    return product, magnitude
