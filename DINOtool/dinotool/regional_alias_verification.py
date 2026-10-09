"""Region-verified alias evidence for the frozen Geometry readout."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from .competitive_ownership import alias_families, positive_excess_readout
from .gear_ov import GearObservations, GearText, class_scores
from .tcpr import TCPRTextBank


IMPLEMENTATION = "geometry-regional-alias-verification-v1-20260930"
METHODS = ("Geometry", "Multiscale", "ScreenOnly", "ContextOnly", "VerifiedGeometry")
DIAGNOSTICS = (
    "mean_alias_rejection", "screen_mean_abs_correction",
    "context_mean_abs_correction", "final_mean_abs_correction",
    "active_region_fraction", "changed_patch_fraction",
)


@dataclass(frozen=True)
class RegionalVerificationConfig:
    alias_temperature: float = 0.07
    family_cosine: float = 0.97
    structural_temperature: float = 0.10
    prototype_fraction: float = 0.25
    correction_strength: float = 2.0

    def validate(self) -> None:
        if (self.alias_temperature <= 0 or self.structural_temperature <= 0
                or self.correction_strength < 0 or not 0 < self.prototype_fraction <= 1
                or not 0 <= self.family_cosine <= 1):
            raise ValueError("Invalid regional verification configuration.")


def _resize_grid(values: Tensor, height: int, width: int) -> Tensor:
    return F.interpolate(values.permute(2, 0, 1)[None], size=(height, width),
                         mode="bilinear", align_corners=False)[0].permute(1, 2, 0)


def fixed_denominator_scores(scores: Tensor, reference: Tensor, retention: Tensor,
                             parents: Tensor, classes: int) -> Tensor:
    """Suppress positive alias excess without inflating a class by deleting weak slots."""
    return positive_excess_readout(scores, reference, retention, parents, classes)


class RegionalAliasVerificationReadout:
    def __init__(self, bank: TCPRTextBank, backbone, template_source: str | Path,
                 config: RegionalVerificationConfig = RegionalVerificationConfig()):
        config.validate()
        bank.validate()
        if bank.class_count < 2:
            raise ValueError("Regional verification requires competing classes.")
        self.config = config
        self.bank = bank
        self.parents = bank.parent_indices
        self.families = alias_families(bank, config.family_cosine)
        self.classes = bank.class_count
        self.aliases = len(bank.alias_names)
        device = bank.features.device
        family_index = torch.empty(self.aliases, dtype=torch.long, device=device)
        for index, members in enumerate(self.families):
            family_index[members] = index
        self.family_index = family_index
        self.class_count = torch.bincount(self.parents, minlength=self.classes)
        same_class = self.parents[:, None] == self.parents[None, :]
        same_family = family_index[:, None] == family_index[None, :]
        self.excluded_alias_matrix = (same_class & same_family).float()
        self.excluded_alias_count = self.excluded_alias_matrix.sum(0)
        self.parent_matrix = F.one_hot(self.parents, self.classes).float()
        flat_group_class = family_index * self.classes + self.parents
        self.group_class_matrix = F.one_hot(
            flat_group_class, len(self.families) * self.classes).float()
        self.group_class_count = self.group_class_matrix.sum(0).reshape(
            len(self.families), self.classes)
        source = Path(template_source)
        self.template_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        self.full_text = self._encode_full_text(backbone, source)

    def config_dict(self) -> dict[str, object]:
        return {**asdict(self.config), "family_count": len(self.families),
                "verifier_templates": "pinned OpenAI ImageNet templates",
                "template_sha256": self.template_sha256}

    def _encode_full_text(self, backbone, source: Path) -> Tensor:
        spec = importlib.util.spec_from_file_location("regional_verifier_templates", source)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load fixed verifier templates: {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        templates = module.get_text_template("openai_imagenet_template")
        prompts = [template(alias) for alias in self.bank.alias_names for template in templates]
        encoded = []
        for start in range(0, len(prompts), 64):
            tokens = backbone.tokenize(prompts[start:start + 64]).to(backbone.device)
            with torch.inference_mode(), backbone._autocast():
                features = backbone.model.encode_text(tokens, normalize=False)
            encoded.append(F.normalize(features.float(), dim=-1))
        vectors = torch.cat(encoded).reshape(self.aliases, len(templates), -1).mean(1)
        return F.normalize(vectors, dim=-1)

    def _leave_family_out(self, scores: Tensor, *, need_teacher: bool = True
                          ) -> tuple[Tensor | None, Tensor]:
        """Class references exclude the tested phrase family in every class."""
        shape = scores.shape[:-1]
        flat = scores.reshape(-1, self.aliases)
        shift = flat.amax(dim=-1, keepdim=True)
        mass = (flat - shift).exp()
        class_mass = mass @ self.parent_matrix
        excluded_mass = mass @ self.excluded_alias_matrix
        count = self.class_count[self.parents] - self.excluded_alias_count
        reference = ((class_mass[:, self.parents] - excluded_mass).clamp_min(1e-30).log()
                     + shift - count.clamp_min(1).log())
        if bool((count == 0).any()):
            original = class_scores(flat, self.parents, self.classes)
            reference[:, count == 0] = original[:, self.parents[count == 0]]
        if not need_teacher:
            return None, reference.reshape(*shape, self.aliases)
        removed = (mass @ self.group_class_matrix).reshape(
            flat.shape[0], len(self.families), self.classes)
        group_count = self.class_count[None] - self.group_class_count
        teacher = ((class_mass[:, None] - removed).clamp_min(1e-30).log()
                   + shift[:, None] - group_count.clamp_min(1).log())
        if bool((group_count == 0).any()):
            original = class_scores(flat, self.parents, self.classes)
            teacher = torch.where(group_count[None] > 0, teacher, original[:, None])
        return (teacher.reshape(*shape, len(self.families), self.classes),
                reference.reshape(*shape, self.aliases))

    def _region_retention(self, local: Tensor, detail: Tensor) -> tuple[Tensor, Tensor]:
        scores = torch.cat((local[None], detail), dim=0) @ self.full_text.T
        scores = scores / self.config.alias_temperature
        teacher, _ = self._leave_family_out(scores)
        close = teacher[1:, self.family_index]
        broad = teacher[0, self.family_index]
        own_close = close.gather(2, self.parents[None, :, None].expand(4, -1, 1))
        own_broad = broad.gather(1, self.parents[:, None])
        margin = torch.minimum(close - own_close, broad[None] - own_broad[None])
        margin.scatter_(2, self.parents[None, :, None].expand(4, -1, 1), -torch.inf)
        retention = 1 - torch.tanh(margin.amax(-1).clamp_min(0))
        # The common class reference is used only for the signed correction.
        common = teacher.mean(1)
        return retention, common

    @staticmethod
    def _retention_map(retention: Tensor, height: int, width: int) -> Tensor:
        y = torch.arange(height, device=retention.device)[:, None] >= height // 2
        x = torch.arange(width, device=retention.device)[None, :] >= width // 2
        return retention[(2 * y.long() + x.long()).expand(height, width)]

    def _screen_view(self, features: Tensor, retention: Tensor) -> Tensor:
        scores = (features.float() @ self.bank.features.float().T)
        scores = scores / self.config.alias_temperature
        _, reference = self._leave_family_out(scores, need_teacher=False)
        corrected = fixed_denominator_scores(
            scores, reference, self._retention_map(retention, *scores.shape[:2]),
            self.parents, self.classes,
        )
        original = class_scores(scores, self.parents, self.classes)
        return self.config.alias_temperature * (corrected - original)

    def _screen(self, views: GearObservations, retention: Tensor,
                multiscale: Tensor) -> Tensor:
        corrections = []
        for features in (views.local_aligned, views.detail_aligned,
                         views.context_aligned):
            grid = self._screen_view(features, retention)
            corrections.append(_resize_grid(grid, 512, 512).permute(2, 0, 1)[None])
        return multiscale + sum(corrections) / len(corrections)

    def _correct(self, views: GearObservations, baseline: Tensor,
                 common: Tensor) -> tuple[Tensor, float]:
        fine = F.interpolate(baseline.float(), size=(64, 64),
                             mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        change = torch.zeros_like(fine)
        active = 0
        for region in range(4):
            y0, x0 = (region // 2) * 32, (region % 2) * 32
            region_scores = fine[y0:y0 + 32, x0:x0 + 32]
            raw = F.normalize(views.detail_raw[y0:y0 + 32, x0:x0 + 32].float(), dim=-1)
            region_valid_h = max(0, min(32, (views.actual_height + 7) // 8 - y0))
            region_valid_w = max(0, min(32, (views.actual_width + 7) // 8 - x0))
            if not region_valid_h or not region_valid_w:
                continue
            # The tight and broad native observations must agree on the rival.
            candidate = int(torch.minimum(common[0], common[region + 1]).argmax())
            margins = torch.minimum(common[0, candidate] - common[0],
                                    common[region + 1, candidate] - common[region + 1])
            trust = torch.tanh(margins.clamp_min(0))
            if not bool((trust > 0).any()):
                continue
            valid_scores = region_scores[:region_valid_h, :region_valid_w].reshape(-1, self.classes)
            valid_raw = raw[:region_valid_h, :region_valid_w].reshape(-1, raw.shape[-1])
            k = max(1, int(valid_scores.shape[0] * self.config.prototype_fraction))
            prototypes = []
            for class_index in range(self.classes):
                seeds = valid_scores[:, class_index].topk(k).indices
                prototypes.append(F.normalize(valid_raw[seeds].mean(0), dim=0))
            similarity = valid_raw @ torch.stack(prototypes).T
            rival = valid_scores.argmax(-1)
            row = torch.arange(rival.numel(), device=rival.device)
            support = torch.sigmoid((similarity[:, candidate] - similarity[row, rival])
                                    / self.config.structural_temperature)
            gap = (valid_scores[row, rival] - valid_scores[:, candidate]).clamp_min(0)
            magnitude = self.config.correction_strength * trust[rival] * support * gap
            magnitude = torch.where(rival == candidate, 0, magnitude)
            if bool((magnitude > 0).any()):
                active += 1
            update = torch.zeros_like(valid_scores)
            update[:, candidate] += magnitude / 2
            update.scatter_add_(1, rival[:, None], (-magnitude / 2)[:, None])
            change[y0:y0 + region_valid_h, x0:x0 + region_valid_w] = \
                update.reshape(region_valid_h, region_valid_w, self.classes)
        dense_change = _resize_grid(change, 512, 512).permute(2, 0, 1)[None]
        return baseline + dense_change, active / 4

    @torch.inference_mode()
    def solve_all(self, views: GearObservations, text: GearText,
                  multiscale: Tensor) -> tuple[dict[str, Tensor], dict[str, float]]:
        if any(item is None for item in (views.local_global, views.detail_global,
                                        views.context_global)):
            raise ValueError("Native global image observations are required.")
        if not torch.allclose(text.features, self.bank.features, atol=1e-5):
            raise ValueError("Full verifier and Geometry must share the same alias bank.")
        if views.local_global.shape[-1] != self.full_text.shape[-1]:
            raise ValueError("Native image and text dimensions do not match.")
        retention, common = self._region_retention(views.local_global,
                                                    views.detail_global)
        screened = self._screen(views, retention, multiscale)
        context_only, active = self._correct(views, multiscale, common)
        final, _ = self._correct(views, screened, common)
        coarse_base = F.interpolate(multiscale, size=(64, 64), mode="bilinear",
                                    align_corners=False).argmax(1)
        coarse_final = F.interpolate(final, size=(64, 64), mode="bilinear",
                                     align_corners=False).argmax(1)
        valid_y = torch.arange(64, device=multiscale.device) * 8 < views.actual_height
        valid_x = torch.arange(64, device=multiscale.device) * 8 < views.actual_width
        valid = valid_y[:, None] & valid_x[None, :]
        diagnostics = {
            "mean_alias_rejection": float((1 - retention).mean()),
            "screen_mean_abs_correction": float((screened - multiscale).abs().mean()),
            "context_mean_abs_correction": float((context_only - multiscale).abs().mean()),
            "final_mean_abs_correction": float((final - multiscale).abs().mean()),
            "active_region_fraction": active,
            "changed_patch_fraction": float(((coarse_final != coarse_base) & valid[None]).sum()
                                            / valid.sum().clamp_min(1)),
        }
        return {"ScreenOnly": screened, "ContextOnly": context_only,
                "VerifiedGeometry": final}, diagnostics
