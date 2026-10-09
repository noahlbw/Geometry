"""Suppress contradicted contextual alias excess without redistributing weights."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
import torch.nn.functional as F

from .calibrated_competitive_alias import profiled_logits
from .stratified_soft_alias import crop_stencil


IMPLEMENTATION = "geometry-canonical-excess-alias-rejection-v1-20261003"
PRIMARY = "ExcessReject_Coupled"
SHUFFLED = tuple("ShuffledReject" + str(i) + "_Coupled" for i in range(3))
METHODS = ("Geometry", "BroadVIP", "Anchored_VIP", PRIMARY,
           "TextOnlyReject_Coupled", "HardDelete_Coupled", *SHUFFLED)
REPLAY = METHODS[:3]


@dataclass(frozen=True)
class ExcessRejectConfig:
    local_temperature: float = .07
    cell_pixels: int = 64
    minimum_effective_support: float = 4.
    query_chunk: int = 128
    random_seed: int = 20261003
    epsilon: float = 1e-6


def excess_delta(logits, retention, beta=1.):
    if logits.shape != retention.shape or logits.shape[-1] < 1 or beta <= 0:
        raise ValueError("Matching alias evidence and retention required.")
    if not bool(torch.isfinite(logits).all() and torch.isfinite(retention).all()):
        raise ValueError("Finite evidence and retention required.")
    if not bool(((retention >= 0) & (retention <= 1)).all()):
        raise ValueError("Retention must lie in [0,1].")
    k = logits.shape[-1]
    responsibility = (beta * logits).softmax(-1)
    rejected = ((1 - retention) * (responsibility - 1 / k).clamp_min(0)).sum(-1)
    return torch.log1p(-rejected.clamp_max(1 - 1 / k)) / beta


def hard_delete_delta(logits, retention, beta=1.):
    keep = retention == 1
    count = keep.sum(-1)
    if bool((count < 1).any()):
        raise ValueError("At least one alias must remain.")
    original = (beta * logits).logsumexp(-1)
    changed = (beta * logits).masked_fill(~keep, -torch.inf).logsumexp(-1)
    delta = (changed - original + (logits.shape[-1] / count).log()) / beta
    return torch.where(keep.all(-1), 0., delta)


def semantic_contradictions(features, parents, canonical, epsilon=1e-6):
    text = F.normalize(features.float(), dim=-1)
    similarity = text @ text[canonical].T
    own = similarity.gather(1, parents[:, None])
    span = similarity.amax(-1, keepdim=True) - similarity.amin(-1, keepdim=True)
    conflict = (similarity - own).clamp_min(0) / span.clamp_min(epsilon)
    conflict = torch.where(span > epsilon, conflict, 0.)
    conflict[canonical] = 0.
    return conflict.clamp(0, 1)


def canonical_support(local_logits, relation, coordinates, valid, config=ExcessRejectConfig()):
    n = len(local_logits)
    if relation.shape != (n, n) or coordinates.shape != (n, 2) or valid.shape != (n,):
        raise ValueError("Matching Geometry support tensors required.")
    cells = (coordinates / config.cell_pixels).floor().long()
    same_cell = (cells[:, None] == cells[None]).all(-1)
    weights = relation.float().masked_fill(same_cell | ~valid[None] | ~valid[:, None], 0.)
    mass = weights.sum(-1, keepdim=True)
    weights = weights / mass.clamp_min(config.epsilon)
    effective = weights.square().sum(-1).clamp_min(config.epsilon ** 2).reciprocal()
    supported = valid & (mass[:, 0] > config.epsilon) & (effective >= config.minimum_effective_support)
    support = weights @ local_logits.float().softmax(-1)
    return support, supported, effective


def contextual_retention(local_aliases, wide_aliases, support, supported, conflict, parents,
                         config=ExcessRejectConfig()):
    if local_aliases.shape != wide_aliases.shape or conflict.shape != (len(parents), support.shape[-1]):
        raise ValueError("Matching local/wide aliases and canonical conflicts required.")
    own = support[:, parents, None]
    rivals = support[:, None]
    leakage = (rivals - own).clamp_min(0) / (rivals + own).clamp_min(config.epsilon)
    contradiction = (leakage * conflict[None]).amax(-1)
    gain = ((wide_aliases - local_aliases) / config.local_temperature).clamp(0, 1)
    risk = contradiction * gain * supported[:, None]
    return (1 - risk).clamp(0, 1)


def vocabulary_variants(specs):
    """Matched-count, class-name-only context stress; not fresh LLM generation."""
    from .prompts import ClassSpec
    names = tuple(spec.name for spec in specs)
    if len(names) < 2 or any(len(spec.synonyms) != 20 for spec in specs):
        raise ValueError("Multi-class, exactly20-alias vocabulary required.")
    variants = {"clean": list(specs), "wrong_parent": [], "paraphrase": []}
    manifest = {"clean": [], "wrong_parent": [], "paraphrase": []}
    for c, spec in enumerate(specs):
        aliases = list(spec.synonyms)
        slots = [i for i, phrase in enumerate(aliases) if phrase != spec.name][-4:]
        if len(slots) != 4:
            raise ValueError("Four noncanonical replacement slots required.")
        rival = names[(c + 1) % len(names)]
        candidates = (rival, "a view of " + rival, rival + " seen from above", "an area of " + rival,
                      "an example of " + rival, "the visible " + rival, "a region of " + rival)
        retained = {phrase for i, phrase in enumerate(aliases) if i not in slots}
        replacements = [phrase for phrase in candidates if phrase not in retained][:4]
        if len(replacements) != 4:
            raise ValueError("Insufficient distinct wrong-parent replacements.")
        for arm in ("wrong_parent", "paraphrase"):
            changed = aliases.copy()
            for j, slot in enumerate(slots):
                changed[slot] = replacements[j] if arm == "wrong_parent" else "an overhead view of " + aliases[slot]
            if len(set(changed)) != 20:
                raise ValueError("Replacement creates duplicate aliases: " + spec.name)
            variants[arm].append(ClassSpec(spec.name, tuple(changed)))
            manifest[arm].append({"class": spec.name, "slots": slots,
                "before": [aliases[i] for i in slots], "after": [changed[i] for i in slots],
                "declared_rival": rival if arm == "wrong_parent" else None})
    return variants, manifest


class ExcessAliasReader:
    def __init__(self, bank, config=ExcessRejectConfig()):
        bank.validate()
        self.bank, self.config = bank, config
        self.members = torch.stack([(bank.parent_indices == c).nonzero().flatten() for c in range(bank.class_count)])
        self.count = self.members.shape[-1]
        self.canonical = torch.stack([(bank.canonical_mask & (bank.parent_indices == c)).nonzero().flatten()[0]
                                     for c in range(bank.class_count)])
        self.conflict = semantic_contradictions(bank.features, bank.parent_indices, self.canonical, config.epsilon)
        self.permutations = []
        for seed in range(config.random_seed, config.random_seed + 3):
            generator = torch.Generator().manual_seed(seed)
            self.permutations.append(torch.stack([torch.randperm(self.count, generator=generator)
                for _ in range(bank.class_count)]).to(bank.features.device))

    def report(self):
        return {"implementation": IMPLEMENTATION, "config": asdict(self.config),
                "aliases": self.bank.alias_names, "canonical_indices": self.canonical.tolist(),
                "semantic_conflicts": self.conflict.tolist(),
                "permutations": [value.tolist() for value in self.permutations],
                "target_labels_used": False, "local_anchor": "unchanged original20 vocabulary",
                "source": "canonical class competition on other Geometry cells, matched-template contextual gain, text contradiction"}

    @torch.inference_mode()
    def read(self, local_aliases, relation, coordinates, valid, broad, crops, auxiliary, crop_count, image_size, beta=1.):
        cfg = self.config
        support, supported, effective = canonical_support(local_aliases[:, self.canonical] / cfg.local_temperature,
                                                         relation, coordinates, valid, cfg)
        output = {method: [] for method in METHODS[3:]}
        total_risk, active, gains, correction, shuffle_error = 0., 0, 0, 0., 0.
        class_ids = torch.arange(self.bank.class_count, device=broad.device)
        for start in range(0, len(local_aliases), cfg.query_chunk):
            sl = slice(start, start + cfg.query_chunk)
            coords = coordinates[sl]
            fields = []
            wide = torch.zeros_like(local_aliases[sl])
            for crop, cosines in zip(crops, auxiliary):
                indices, coefficients = crop_stencil(crop, crop_count, coords, image_size)
                wide += (cosines[indices] * coefficients[..., None]).sum(1)
                evidence = profiled_logits(crop, self.members)[indices]
                fields.append((evidence, coefficients))
            retention = contextual_retention(local_aliases[sl], wide, support[sl], supported[sl],
                                             self.conflict, self.bank.parent_indices, cfg)
            risk = 1 - retention
            grouped = retention[:, self.members]
            variants = {PRIMARY: grouped,
                "TextOnlyReject_Coupled": (1 - self.conflict.amax(-1))[self.members][None].expand_as(grouped),
                "HardDelete_Coupled": grouped}
            for method, permutation in zip(SHUFFLED, self.permutations):
                variants[method] = grouped.gather(-1, permutation[class_ids][None].expand_as(grouped))
                shuffle_error = max(shuffle_error, float((variants[method].sort(-1).values - grouped.sort(-1).values).abs().max()))
            for method, values in variants.items():
                delta = torch.zeros_like(broad[sl])
                for evidence, coefficients in fields:
                    writer = hard_delete_delta if method == "HardDelete_Coupled" else excess_delta
                    delta += (writer(evidence, values[:, None].expand_as(evidence), beta) * coefficients[..., None]).sum(1)
                delta = delta.masked_fill(~valid[sl, None], 0.)
                if method == PRIMARY:
                    correction += float(delta[valid[sl]].abs().sum())
                output[method].append(broad[sl] + delta)
            total_risk += float(risk[valid[sl]].sum())
            active += int((risk[valid[sl]] > 1e-6).sum())
            gains += int((wide[valid[sl]] > local_aliases[sl][valid[sl]]).sum())
        n = max(int(valid.sum()), 1)
        return {method: torch.cat(values) for method, values in output.items()}, {
            "supported_patch_fraction": float(supported[valid].float().mean()) if bool(valid.any()) else 0.,
            "effective_support": float(effective[supported].mean()) if bool(supported.any()) else 0.,
            "mean_rejection": total_risk / (n * len(self.bank.alias_names)),
            "rejected_alias_fraction": active / (n * len(self.bank.alias_names)),
            "positive_gain_fraction": gains / (n * len(self.bank.alias_names)),
            "mean_score_suppression": correction / (n * self.bank.class_count),
            "shuffle_spectrum_error": shuffle_error,
        }
