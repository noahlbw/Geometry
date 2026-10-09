"""Training-free text-conditioned parallel reading for frozen DINO.text."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .parallel_readout import structural_logits
from .prompts import ClassSpec, REMOTE_SENSING_TEMPLATES, clean_phrase


@dataclass(frozen=True)
class TCPRConfig:
    geometry_temperature: float = 0.10
    spatial_sigma: float = 0.25
    head_block: int = -1
    geometry_depth: int = 1
    prefix_policy: str = "preserve"
    relation_policy: str = "dense"
    alias_temperature: float = 0.07
    alias_filter_temperature: float = 0.20
    maximum_aliases_per_class: int = 8
    canonical_alias_prior: float = 0.20
    reference_count: int = 4
    reference_margin: float = 0.015
    reference_temperature: float = 0.03
    reference_min_quality: float = 0.20
    reference_diversity_radius: int = 2
    evidence_topk: int = 32
    evidence_strength: float = 1.0
    visual_evidence_temperature: float = 0.08
    semantic_evidence_temperature: float = 0.05
    evidence_epsilon: float = 1e-6
    sparse_query_chunk: int = 256

    def validate(self) -> None:
        temperatures = (
            self.geometry_temperature, self.spatial_sigma, self.alias_temperature,
            self.alias_filter_temperature, self.reference_temperature,
            self.visual_evidence_temperature, self.semantic_evidence_temperature,
            self.evidence_epsilon,
        )
        if any(value <= 0 for value in temperatures):
            raise ValueError("TCPR temperatures and epsilon must be positive.")
        counts = (self.maximum_aliases_per_class, self.reference_count,
                  self.evidence_topk, self.sparse_query_chunk)
        if min(counts) < 1:
            raise ValueError("TCPR alias, reference, evidence, and chunk counts must be positive.")
        if self.reference_diversity_radius < 0:
            raise ValueError("reference_diversity_radius must be non-negative.")
        if not 0.0 <= self.reference_min_quality <= 1.0:
            raise ValueError("reference_min_quality must be in [0, 1].")
        if self.canonical_alias_prior < 0 or self.evidence_strength < 0:
            raise ValueError("Alias prior and evidence strength must be non-negative.")
        if self.geometry_depth not in (1, 2) or self.prefix_policy not in ("preserve", "block"):
            raise ValueError("Geometry depth and patch-to-prefix policy are invalid.")
        if self.relation_policy not in ("dense", "sparse"):
            raise ValueError("Geometry relation policy must be dense or sparse.")


@dataclass(frozen=True)
class TCPRTextBank:
    features: Tensor
    parent_indices: Tensor
    canonical_mask: Tensor
    class_names: tuple[str, ...]
    alias_names: tuple[str, ...]

    @property
    def class_count(self) -> int:
        return len(self.class_names)

    def validate(self) -> None:
        aliases = self.features.shape[0]
        if self.features.ndim != 2 or aliases < 1:
            raise ValueError("Alias features must have shape [M,D] with M > 0.")
        if self.parent_indices.shape != (aliases,) or self.canonical_mask.shape != (aliases,):
            raise ValueError("Alias parent and canonical tensors must have shape [M].")
        if len(self.alias_names) != aliases or not self.class_names:
            raise ValueError("Alias metadata does not match encoded features.")
        if self.parent_indices.dtype != torch.long or self.canonical_mask.dtype != torch.bool:
            raise ValueError("Alias parents must be long and canonical_mask must be bool.")
        if int(self.parent_indices.min()) < 0 or int(self.parent_indices.max()) >= self.class_count:
            raise ValueError("Alias parent index is outside the semantic class range.")
        for class_index in range(self.class_count):
            members = self.parent_indices == class_index
            if not bool(members.any()) or int((members & self.canonical_mask).sum()) != 1:
                raise ValueError("Every class must contain exactly one canonical alias.")


@dataclass(frozen=True)
class TCPRPreparedImage:
    shared_tokens: Tensor
    raw_patch_tokens: Tensor
    value_tokens: Tensor
    native_attention: Tensor
    native_patch_conditional: Tensor
    geometry_patch_conditional: Tensor
    native_projected: Tensor
    geometry_projected: Tensor
    prefix_tokens: int
    grid_height: int
    grid_width: int
    block_index: int
    native_global: Tensor | None = None
    backbone_tokens: Tensor | None = None


@dataclass(frozen=True)
class TCPRDiagnostics:
    alias_kept_per_class: tuple[tuple[int, ...], ...]
    reference_counts: tuple[tuple[int, ...], ...]
    reference_class_coverage: float
    evidence_patch_coverage: float
    mean_evidence_mass: float
    changed_from_geometry: float
    mean_geometry_native_disagreement: float

    def summary(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TCPRResult:
    logits: Tensor
    geometry_logits: Tensor
    native_logits: Tensor
    pair_ids: Tensor
    evidence_mass: Tensor
    reference_counts: Tensor
    alias_weights: Tensor
    diagnostics: TCPRDiagnostics


class TCPRSegmenter:
    """Frozen DINO.text with geometry and competition-conditioned readouts."""

    method_name = "TCPR: text-conditioned parallel reading"

    def __init__(self, backbone, config: TCPRConfig = TCPRConfig()) -> None:
        config.validate()
        self.backbone = backbone
        self.config = config

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    @property
    def patch_size(self) -> int:
        return self.backbone.patch_size

    def encode_text(self, classes: Sequence[ClassSpec], batch_size: int = 64) -> TCPRTextBank:
        aliases_by_class = [
            tuple(dict.fromkeys(clean_phrase(item) for item in (spec.name, *spec.synonyms)))
            for spec in classes
        ]
        features, parents, canonical = self.backbone.encode_text_aliases(
            aliases_by_class, templates=REMOTE_SENSING_TEMPLATES, batch_size=batch_size
        )
        bank = TCPRTextBank(
            features=features,
            parent_indices=parents,
            canonical_mask=canonical,
            class_names=tuple(spec.name for spec in classes),
            alias_names=tuple(alias for aliases in aliases_by_class for alias in aliases),
        )
        bank.validate()
        return bank

    @torch.inference_mode()
    def prepare_image(self, rgb: Tensor) -> TCPRPreparedImage:
        self.backbone._validate_rgb(rgb)
        normalized = (
            rgb.to(self.device, non_blocking=True) - self.backbone._imagenet_mean
        ) / self.backbone._imagenet_std
        with self.backbone._autocast():
            visual = self.backbone.model.visual_model
            cls, raw, registers = visual.get_backbone_features(normalized)
            tokens = torch.cat((cls.unsqueeze(1), registers, raw), dim=1)
            head = visual.head
            block_index = _resolve_block_index(head, self.config.head_block)
            prefix = tokens.shape[1] - raw.shape[1]
            shared = tokens
            for earlier in head.blocks[:block_index]:
                shared = earlier(shared)

            block = head.blocks[block_index]
            attention = block.attn
            qkv = attention.qkv(block.norm1(shared))
            batch, token_count, _ = qkv.shape
            channels = attention.qkv.in_features
            qkv = qkv.reshape(
                batch, token_count, 3, attention.num_heads, channels // attention.num_heads
            )
            query, key, value = torch.unbind(qkv, dim=2)
            query, key, value = (part.transpose(1, 2) for part in (query, key, value))
            native_logits = (query.float() @ key.float().transpose(-1, -2)) * attention.scale
            native_attention = torch.softmax(native_logits, dim=-1).to(value.dtype)
            native_patch = torch.softmax(native_logits[..., prefix:, prefix:], dim=-1).to(value.dtype)
            geometry_logits = structural_logits(
                raw,
                rgb.shape[-2] // self.patch_size,
                rgb.shape[-1] // self.patch_size,
                temperature=self.config.geometry_temperature,
                spatial_sigma=self.config.spatial_sigma,
            )
            geometry_patch = geometry_relation(
                geometry_logits, raw, self.config.relation_policy
            ).to(value.dtype)
            native_block = _finish_attention_block(block, shared, native_attention @ value)
            geometry_block = _finish_attention_block(
                block, shared, _geometry_attended(native_attention, geometry_patch, value, prefix)
            )
            native_projected = _finish_head(head, native_block, block_index)
            geometry_projected = _finish_head(head, geometry_block, block_index)
            if self.config.geometry_depth != 1 or self.config.prefix_policy != "preserve":
                geometry_projected = _intervened_head(
                    head, tokens, geometry_patch, prefix, block_index,
                    self.config.geometry_depth, self.config.prefix_policy,
                )
        return TCPRPreparedImage(
            shared_tokens=shared,
            raw_patch_tokens=F.normalize(raw.float(), dim=-1),
            value_tokens=value,
            native_attention=native_attention,
            native_patch_conditional=native_patch,
            geometry_patch_conditional=geometry_patch,
            native_projected=F.normalize(native_projected[:, prefix:].float(), dim=-1),
            geometry_projected=F.normalize(geometry_projected[:, prefix:].float(), dim=-1),
            prefix_tokens=prefix,
            grid_height=rgb.shape[-2] // self.patch_size,
            grid_width=rgb.shape[-1] // self.patch_size,
            block_index=block_index,
            native_global=F.normalize(torch.cat((
                native_projected[:, 0].float(),
                native_projected[:, prefix:].float().mean(dim=1),
            ), dim=-1), dim=-1),
            backbone_tokens=tokens,
        )

    @torch.inference_mode()
    def read_prepared(
        self,
        prepared: TCPRPreparedImage,
        text_bank: TCPRTextBank,
        *,
        valid_mask: Tensor | None = None,
    ) -> TCPRResult:
        text_bank.validate()
        if text_bank.features.shape[1] != prepared.geometry_projected.shape[-1]:
            raise ValueError("Text aliases and image patches must share feature dimension.")
        batch, patches, _ = prepared.geometry_projected.shape
        valid = _normalize_valid_mask(
            valid_mask, batch, prepared.grid_height, prepared.grid_width,
            prepared.geometry_projected.device,
        ).reshape(batch, patches)
        text = F.normalize(text_bank.features.float(), dim=-1)
        geometry_alias = prepared.geometry_projected @ text.T
        native_alias = prepared.native_projected @ text.T
        alias_weights = filter_aliases(
            geometry_alias, native_alias, text_bank.parent_indices,
            text_bank.canonical_mask, text_bank.class_count, valid, self.config,
        )
        geometry_classes = aggregate_alias_scores(
            geometry_alias, alias_weights, text_bank.parent_indices,
            text_bank.class_count, self.config.alias_temperature,
        )
        native_classes = aggregate_alias_scores(
            native_alias, alias_weights, text_bank.parent_indices,
            text_bank.class_count, self.config.alias_temperature,
        )
        pairs = select_competing_pairs(geometry_classes, native_classes)
        reference_scores, reference_counts, reference_reliability = build_reference_scores(
            prepared.raw_patch_tokens, geometry_classes, native_classes,
            geometry_alias, native_alias, alias_weights, text_bank.parent_indices,
            valid, prepared.grid_height, prepared.grid_width, self.config,
        )
        indices, affinity = candidate_edges(
            prepared.geometry_patch_conditional, prepared.native_patch_conditional,
            self.config.evidence_topk, valid,
        )
        evidence, evidence_mass = competition_evidence(
            indices, affinity, pairs, reference_scores, reference_counts > 0,
            reference_reliability, geometry_classes, native_classes, self.config,
        )
        head = self.backbone.model.visual_model.head
        with self.backbone._autocast():
            attended = _evidence_conditioned_attended(
                prepared.native_attention, prepared.geometry_patch_conditional,
                prepared.value_tokens, indices, evidence, prepared.prefix_tokens, self.config,
            )
            final_block = _finish_attention_block(
                head.blocks[prepared.block_index], prepared.shared_tokens, attended
            )
            projected = _finish_head(head, final_block, prepared.block_index)
        final_patches = F.normalize(projected[:, prepared.prefix_tokens:].float(), dim=-1)
        final_alias = final_patches @ text.T
        final_classes = aggregate_alias_scores(
            final_alias, alias_weights, text_bank.parent_indices,
            text_bank.class_count, self.config.alias_temperature,
        )
        final_classes = torch.where(valid[..., None], final_classes, geometry_classes)
        evidence_mass = evidence_mass.masked_fill(~valid[:, None], 0.0)

        geometry_labels = geometry_classes.argmax(-1)
        native_labels = native_classes.argmax(-1)
        final_labels = final_classes.argmax(-1)
        valid_count = max(int(valid.sum()), 1)
        evidence_patch = evidence_mass.mean(1) > self.config.evidence_epsilon
        kept = tuple(
            tuple(int(((text_bank.parent_indices == cls) & (alias_weights[item] > 0)).sum())
                  for cls in range(text_bank.class_count))
            for item in range(batch)
        )
        diagnostics = TCPRDiagnostics(
            alias_kept_per_class=kept,
            reference_counts=tuple(tuple(int(value) for value in row.tolist())
                                   for row in reference_counts.cpu()),
            reference_class_coverage=float((reference_counts > 0).float().mean()),
            evidence_patch_coverage=float(((evidence_patch & valid).sum() / valid_count).item()),
            mean_evidence_mass=float(
                evidence_mass.sum().item() / max(valid_count * evidence_mass.shape[1], 1)
            ),
            changed_from_geometry=float(((geometry_labels != final_labels) & valid).sum() / valid_count),
            mean_geometry_native_disagreement=float(
                ((geometry_labels != native_labels) & valid).sum() / valid_count
            ),
        )
        return TCPRResult(
            logits=_tokens_to_map(final_classes, prepared.grid_height, prepared.grid_width),
            geometry_logits=_tokens_to_map(geometry_classes, prepared.grid_height, prepared.grid_width),
            native_logits=_tokens_to_map(native_classes, prepared.grid_height, prepared.grid_width),
            pair_ids=pairs.transpose(1, 2).reshape(batch, 2, prepared.grid_height, prepared.grid_width),
            evidence_mass=evidence_mass.mean(1).reshape(batch, prepared.grid_height, prepared.grid_width),
            reference_counts=reference_counts,
            alias_weights=alias_weights,
            diagnostics=diagnostics,
        )

    def segment(
        self,
        rgb: Tensor,
        classes: Sequence[ClassSpec],
        *,
        valid_mask: Tensor | None = None,
        text_batch_size: int = 64,
    ) -> TCPRResult:
        bank = self.encode_text(classes, batch_size=text_batch_size)
        return self.read_prepared(self.prepare_image(rgb), bank, valid_mask=valid_mask)


def aggregate_alias_scores(
    alias_scores: Tensor,
    alias_weights: Tensor,
    parent_indices: Tensor,
    class_count: int,
    temperature: float,
) -> Tensor:
    """Weighted log-mean-exp over aliases without alias-count bias."""
    if alias_scores.ndim != 3:
        raise ValueError("alias_scores must have shape [B,N,M].")
    if alias_weights.shape != (alias_scores.shape[0], alias_scores.shape[2]):
        raise ValueError("alias_weights must have shape [B,M].")
    output = []
    for class_index in range(class_count):
        members = parent_indices == class_index
        weights = alias_weights[:, members]
        scores = alias_scores[:, :, members]
        log_weights = weights.clamp_min(torch.finfo(scores.dtype).tiny).log()[:, None]
        output.append(temperature * torch.logsumexp(scores / temperature + log_weights, dim=-1))
    return torch.stack(output, dim=-1)


def filter_aliases(
    geometry_alias: Tensor,
    native_alias: Tensor,
    parent_indices: Tensor,
    canonical_mask: Tensor,
    class_count: int,
    valid: Tensor,
    config: TCPRConfig,
) -> Tensor:
    """Filter aliases by pilot agreement and spatial concentration."""
    batch, patches, aliases = geometry_alias.shape
    if native_alias.shape != geometry_alias.shape or valid.shape != (batch, patches):
        raise ValueError("Pilot alias scores or valid mask have incompatible shape.")
    weights = torch.zeros((batch, aliases), device=geometry_alias.device, dtype=torch.float32)
    for image_index in range(batch):
        mask = valid[image_index]
        for class_index in range(class_count):
            members = (parent_indices == class_index).nonzero(as_tuple=False).flatten()
            g = geometry_alias[image_index, mask][:, members].float()
            n = native_alias[image_index, mask][:, members].float()
            if g.shape[0] > 1:
                gc = g - g.mean(0, keepdim=True)
                nc = n - n.mean(0, keepdim=True)
                consistency = F.cosine_similarity(gc.T, nc.T, dim=-1).add(1.0).mul(0.5)
                probability = torch.softmax((g + n) / (2.0 * config.alias_temperature), dim=0)
                entropy = -(probability * probability.clamp_min(1e-8).log()).sum(0)
                saliency = 1.0 - entropy / math.log(g.shape[0])
            else:
                consistency = saliency = torch.ones(len(members), device=g.device)
            reliability = 0.65 * consistency.clamp(0, 1) + 0.35 * saliency.clamp(0, 1)
            local_canonical = canonical_mask[members]
            reliability = reliability + local_canonical.float() * config.canonical_alias_prior
            keep = min(config.maximum_aliases_per_class, len(members))
            chosen = torch.topk(reliability, keep).indices
            canonical = local_canonical.nonzero(as_tuple=False).flatten()[0]
            if not bool((chosen == canonical).any()):
                chosen[-1] = canonical
            selected = torch.softmax(reliability[chosen] / config.alias_filter_temperature, dim=0)
            weights[image_index, members[chosen]] = selected
    return weights


def select_competing_pairs(geometry_scores: Tensor, native_scores: Tensor) -> Tensor:
    if geometry_scores.shape != native_scores.shape or geometry_scores.ndim != 3:
        raise ValueError("Class pilot scores must share shape [B,N,C].")
    first = geometry_scores.argmax(-1)
    if geometry_scores.shape[-1] == 1:
        return torch.stack((first, first), dim=-1)
    candidates = torch.maximum(geometry_scores, native_scores).clone()
    candidates.scatter_(-1, first[..., None], -torch.inf)
    return torch.stack((first, candidates.argmax(-1)), dim=-1)


def build_reference_scores(
    raw: Tensor,
    geometry_classes: Tensor,
    native_classes: Tensor,
    geometry_alias: Tensor,
    native_alias: Tensor,
    alias_weights: Tensor,
    parent_indices: Tensor,
    valid: Tensor,
    height: int,
    width: int,
    config: TCPRConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build a small, class-balanced image reference bank without labels."""
    batch, patches, _ = raw.shape
    classes = geometry_classes.shape[-1]
    g_top = torch.topk(geometry_classes, min(2, classes), dim=-1).values
    n_top = torch.topk(native_classes, min(2, classes), dim=-1).values
    g_margin = g_top[..., 0] - (g_top[..., 1] if classes > 1 else 0.0)
    n_margin = n_top[..., 0] - (n_top[..., 1] if classes > 1 else 0.0)
    g_label, n_label = geometry_classes.argmax(-1), native_classes.argmax(-1)
    consensus = (g_label == n_label) & valid
    local_support = _local_label_support(g_label, valid, height, width)

    alias_agreement = torch.ones((batch, patches, classes), device=raw.device)
    for class_index in range(classes):
        members = parent_indices == class_index
        if int(members.sum()) <= 1:
            continue
        values = 0.5 * (geometry_alias[..., members] + native_alias[..., members])
        weights = alias_weights[:, members]
        mean = (values * weights[:, None]).sum(-1)
        variance = ((values - mean[..., None]).square() * weights[:, None]).sum(-1)
        alias_agreement[..., class_index] = torch.exp(
            -variance / config.semantic_evidence_temperature**2
        )
    margin_quality = torch.sigmoid(
        (torch.minimum(g_margin, n_margin) - config.reference_margin)
        / config.reference_temperature
    )
    predicted_alias = alias_agreement.gather(-1, g_label[..., None]).squeeze(-1)
    quality = margin_quality * local_support * (0.5 + 0.5 * predicted_alias)
    quality = quality * consensus.float()

    raw = F.normalize(raw.float(), dim=-1)
    scores = torch.zeros((batch, classes, patches), device=raw.device)
    counts = torch.zeros((batch, classes), device=raw.device, dtype=torch.long)
    reliability = torch.zeros((batch, classes), device=raw.device)
    for image_index in range(batch):
        for class_index in range(classes):
            candidates = (
                consensus[image_index]
                & (g_label[image_index] == class_index)
                & (quality[image_index] >= config.reference_min_quality)
            ).nonzero(as_tuple=False).flatten()
            if not candidates.numel():
                continue
            order = candidates[torch.argsort(quality[image_index, candidates], descending=True)]
            selected = _spatially_diverse_indices(
                order, width, config.reference_count, config.reference_diversity_radius
            )
            selected = torch.tensor(selected, device=raw.device, dtype=torch.long)
            selected_quality = quality[image_index, selected]
            weights = selected_quality / selected_quality.sum().clamp_min(config.evidence_epsilon)
            similarity = raw[image_index] @ raw[image_index, selected].T
            scores[image_index, class_index] = config.visual_evidence_temperature * torch.logsumexp(
                similarity / config.visual_evidence_temperature
                + weights.clamp_min(1e-8).log()[None], dim=-1,
            )
            counts[image_index, class_index] = selected.numel()
            reliability[image_index, class_index] = selected_quality.mean()
    return scores, counts, reliability


def candidate_edges(
    geometry: Tensor, native: Tensor, topk: int, valid: Tensor
) -> tuple[Tensor, Tensor]:
    """Take the sparse union of Geometry and Native relation candidates."""
    if native.ndim != 4 or geometry.ndim != 3:
        raise ValueError("Expected native [B,H,N,N] and geometry [B,N,N].")
    base = torch.maximum(native.float(), geometry[:, None].float())
    patches = base.shape[-1]
    base = base.masked_fill(~valid[:, None, None, :], -torch.inf)
    if patches > 1:
        diagonal = torch.eye(patches, device=base.device, dtype=torch.bool)[None, None]
        base = base.masked_fill(diagonal, -torch.inf)
    count = min(topk, patches - 1 if patches > 1 else 1)
    affinity, indices = torch.topk(base, count, dim=-1)
    affinity = affinity.masked_fill(~torch.isfinite(affinity), 0.0)
    affinity = affinity * valid[:, None, :, None]
    return indices, affinity


def competition_evidence(
    indices: Tensor,
    affinity: Tensor,
    pairs: Tensor,
    reference_scores: Tensor,
    reference_available: Tensor,
    reference_reliability: Tensor,
    geometry_classes: Tensor,
    native_classes: Tensor,
    config: TCPRConfig,
) -> tuple[Tensor, Tensor]:
    """Construct symmetric A-not-B and B-not-A evidence on sparse edges."""
    batch, heads, patches, count = indices.shape
    first, second = pairs[..., 0], pairs[..., 1]
    g_first = geometry_classes.gather(-1, first[..., None]).squeeze(-1)
    g_second = geometry_classes.gather(-1, second[..., None]).squeeze(-1)
    n_first = native_classes.gather(-1, first[..., None]).squeeze(-1)
    n_second = native_classes.gather(-1, second[..., None]).squeeze(-1)
    refs_by_patch = reference_scores.transpose(1, 2)
    ref_first_query = refs_by_patch.gather(-1, first[..., None]).squeeze(-1)
    ref_second_query = refs_by_patch.gather(-1, second[..., None]).squeeze(-1)
    available = (
        reference_available.gather(1, first)
        & reference_available.gather(1, second)
        & (first != second)
    )
    local_delta = 0.5 * ((g_first - g_second) + (n_first - n_second))
    local_delta = local_delta + 0.5 * (ref_first_query - ref_second_query)
    local_first = torch.sigmoid(local_delta / config.semantic_evidence_temperature)
    local_second = 1.0 - local_first

    b = torch.arange(batch, device=indices.device)[:, None, None, None]
    first_edge = first[:, None, :, None].expand(-1, heads, -1, count)
    second_edge = second[:, None, :, None].expand(-1, heads, -1, count)
    ref_first = reference_scores[b, first_edge, indices]
    ref_second = reference_scores[b, second_edge, indices]
    class_scores = torch.maximum(geometry_classes, native_classes)
    score_first = class_scores[b, indices, first_edge]
    score_second = class_scores[b, indices, second_edge]
    mean_score = class_scores.mean(-1)[b, indices]
    visual_delta = ref_first - ref_second
    semantic_delta = score_first - score_second
    reliability_first = reference_reliability.gather(1, first)[:, None, :, None]
    reliability_second = reference_reliability.gather(1, second)[:, None, :, None]
    quality_first = reliability_first * torch.sigmoid(
        (score_first - mean_score) / config.semantic_evidence_temperature
    )
    quality_second = reliability_second * torch.sigmoid(
        (score_second - mean_score) / config.semantic_evidence_temperature
    )

    def positive(value: Tensor, temperature: float) -> Tensor:
        return torch.tanh(F.relu(value) / temperature)

    first_evidence = (
        quality_first
        * positive(visual_delta, config.visual_evidence_temperature)
        * positive(semantic_delta, config.semantic_evidence_temperature)
    )
    second_evidence = (
        quality_second
        * positive(-visual_delta, config.visual_evidence_temperature)
        * positive(-semantic_delta, config.semantic_evidence_temperature)
    )
    combined = 0.5 * (
        local_first[:, None, :, None] * first_evidence
        + local_second[:, None, :, None] * second_evidence
    )
    evidence = affinity * combined * available[:, None, :, None]
    return evidence, evidence.sum(-1)


def _evidence_conditioned_attended(
    native_attention: Tensor,
    geometry: Tensor,
    values: Tensor,
    indices: Tensor,
    evidence: Tensor,
    prefix: int,
    config: TCPRConfig,
) -> Tensor:
    """Read values with sparse evidence; zero evidence exactly returns Geometry."""
    value_patch = values[..., prefix:, :]
    patch_mass = native_attention[..., prefix:, prefix:].sum(-1, keepdim=True)
    geometry_value = geometry[:, None].to(values.dtype) @ value_patch
    extra_value = _sparse_weighted_values(
        value_patch, indices, evidence.to(values.dtype), config.sparse_query_chunk
    )
    evidence_mass = evidence.sum(-1, keepdim=True).to(values.dtype)
    conditioned = (
        geometry_value + config.evidence_strength * extra_value
    ) / (1.0 + config.evidence_strength * evidence_mass)
    special = native_attention[..., prefix:, :prefix] @ values[..., :prefix, :]
    changed = special + patch_mass * conditioned
    geometry_attended = special + patch_mass * geometry_value
    changed = torch.where(evidence_mass > config.evidence_epsilon, changed, geometry_attended)
    prefix_attended = native_attention[..., :prefix, :] @ values
    return torch.cat((prefix_attended, changed), dim=2)


def _sparse_weighted_values(values: Tensor, indices: Tensor, weights: Tensor, chunk: int) -> Tensor:
    batch, heads, patches, head_dim = values.shape
    output = torch.zeros((batch, heads, patches, head_dim), device=values.device, dtype=values.dtype)
    b = torch.arange(batch, device=values.device)[:, None, None, None]
    h = torch.arange(heads, device=values.device)[None, :, None, None]
    for start in range(0, patches, chunk):
        stop = min(start + chunk, patches)
        selected = values[b, h, indices[:, :, start:stop], :]
        output[:, :, start:stop] = (
            selected * weights[:, :, start:stop, :, None]
        ).sum(-2)
    return output


def _geometry_attended(native_attention: Tensor, geometry: Tensor, values: Tensor,
                       prefix: int, prefix_policy: str = "preserve") -> Tensor:
    prefix_attended = native_attention[..., :prefix, :] @ values
    patch = geometry[:, None].to(values.dtype) @ values[..., prefix:, :]
    if prefix_policy == "block":
        return torch.cat((prefix_attended, patch), dim=2)
    if prefix_policy != "preserve":
        raise ValueError(f"Unknown patch-to-prefix policy: {prefix_policy}")
    special = native_attention[..., prefix:, :prefix] @ values[..., :prefix, :]
    patch_mass = native_attention[..., prefix:, prefix:].sum(-1, keepdim=True)
    return torch.cat((prefix_attended, special + patch_mass * patch), dim=2)


def geometry_relation(logits: Tensor, raw: Tensor, policy: str) -> Tensor:
    """Keep the Geometry logits fixed while changing patch connectivity only."""
    if policy == "dense":
        return torch.softmax(logits, dim=-1)
    if policy != "sparse" or logits.shape != (raw.shape[0], raw.shape[1], raw.shape[1]):
        raise ValueError("Invalid sparse Geometry relation inputs.")
    normalized = F.normalize(raw.float(), dim=-1)
    similarity = normalized @ normalized.transpose(-1, -2)
    active = similarity > 1.5 * similarity.mean(dim=-1, keepdim=True)
    diagonal = torch.eye(raw.shape[1], device=raw.device, dtype=torch.bool)[None]
    active = active | diagonal
    return torch.softmax(logits.masked_fill(~active, float("-inf")), dim=-1)


def _intervened_head(head, tokens: Tensor, geometry: Tensor, prefix: int,
                     target_block: int, depth: int, prefix_policy: str) -> Tensor:
    first = target_block - depth + 1
    if first < 0:
        raise ValueError("Requested geometry depth exceeds preceding vision-head blocks.")
    current = tokens
    for index, block in enumerate(head.blocks[:target_block + 1]):
        if index < first:
            current = block(current)
            continue
        attention = block.attn
        qkv = attention.qkv(block.norm1(current))
        batch, token_count, channels3 = qkv.shape
        channels = channels3 // 3
        qkv = qkv.reshape(batch, token_count, 3, attention.num_heads,
                          channels // attention.num_heads)
        query, key, value = torch.unbind(qkv, dim=2)
        query, key, value = (part.transpose(1, 2) for part in (query, key, value))
        logits = (query.float() @ key.float().transpose(-1, -2)) * attention.scale
        native_attention = torch.softmax(logits, dim=-1).to(value.dtype)
        attended = _geometry_attended(native_attention, geometry, value, prefix, prefix_policy)
        current = _finish_attention_block(block, current, attended)
    return _finish_head(head, current, target_block)


def _finish_attention_block(block, x: Tensor, attended: Tensor) -> Tensor:
    batch, heads, tokens, head_dim = attended.shape
    merged = attended.transpose(1, 2).reshape(batch, tokens, heads * head_dim)
    residual = block.attn.proj_drop(block.attn.proj(merged))
    x_attn = x + block.ls1(residual)
    return x_attn + block.ls2(block.mlp(block.norm2(x_attn)))


def _finish_head(head, current: Tensor, block_index: int) -> Tensor:
    for block in head.blocks[block_index + 1:]:
        current = block(current)
    return head.linear_projection(head.ln_final(current))


def _resolve_block_index(head, requested: int) -> int:
    index = requested if requested >= 0 else len(head.blocks) + requested
    if index < 0 or index >= len(head.blocks):
        raise ValueError(f"head_block {requested} is outside the vision head.")
    return index


def _normalize_valid_mask(
    valid_mask: Tensor | None,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
) -> Tensor:
    if valid_mask is None:
        return torch.ones((batch, height, width), device=device, dtype=torch.bool)
    if valid_mask.ndim == 2 and batch == 1:
        valid_mask = valid_mask[None]
    if valid_mask.shape != (batch, height, width):
        raise ValueError("valid_mask must have shape [B,H,W] on the patch grid.")
    return valid_mask.to(device=device, dtype=torch.bool)


def _local_label_support(labels: Tensor, valid: Tensor, height: int, width: int) -> Tensor:
    batch = labels.shape[0]
    labels = labels.reshape(batch, height, width)
    valid = valid.reshape(batch, height, width)
    support = torch.ones_like(labels, dtype=torch.float32)
    count = torch.ones_like(labels, dtype=torch.float32)
    if width > 1:
        agreement = (labels[:, :, :-1] == labels[:, :, 1:]).float()
        pair = valid[:, :, :-1] & valid[:, :, 1:]
        support[:, :, :-1] += agreement * pair
        support[:, :, 1:] += agreement * pair
        count[:, :, :-1] += pair
        count[:, :, 1:] += pair
    if height > 1:
        agreement = (labels[:, :-1] == labels[:, 1:]).float()
        pair = valid[:, :-1] & valid[:, 1:]
        support[:, :-1] += agreement * pair
        support[:, 1:] += agreement * pair
        count[:, :-1] += pair
        count[:, 1:] += pair
    return (support / count.clamp_min(1.0)).reshape(batch, -1)


def _spatially_diverse_indices(ordered: Tensor, width: int, maximum: int, radius: int) -> list[int]:
    selected: list[int] = []
    for value in ordered.tolist():
        row, column = divmod(int(value), width)
        if radius and any(
            max(abs(row - divmod(other, width)[0]), abs(column - divmod(other, width)[1])) <= radius
            for other in selected
        ):
            continue
        selected.append(int(value))
        if len(selected) == maximum:
            break
    if not selected and ordered.numel():
        selected.append(int(ordered[0]))
    return selected


def _tokens_to_map(tokens: Tensor, height: int, width: int) -> Tensor:
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], height, width)
