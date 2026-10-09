"""Class-relative Geometry contradiction without an absolute context-gain gate."""
from __future__ import annotations

import torch

from .calibrated_competitive_alias import profiled_logits
from .excess_alias_rejection import (ExcessAliasReader, ExcessRejectConfig,
    canonical_support, contextual_retention, excess_delta, hard_delete_delta)
from .stratified_soft_alias import crop_stencil


IMPLEMENTATION = "geometry-class-relative-excess-rejection-v1-20261003"
PRIMARY = "ClassRelativeReject_Coupled"
SHUFFLED = tuple("ShuffledRelativeReject" + str(i) + "_Coupled" for i in range(3))
METHODS = ("Geometry", "BroadVIP", "Anchored_VIP", "ExcessReject_Coupled", PRIMARY,
           "TextOnlyReject_Coupled", "HardDelete_Coupled", *SHUFFLED)
REPLAY = METHODS[:4]


def relative_retention(support, supported, conflict, parents, epsilon=1e-6):
    if support.ndim != 2 or conflict.shape != (len(parents), support.shape[-1]) or supported.shape != support.shape[:1]:
        raise ValueError("Matching canonical class support and alias contradiction required.")
    own, rivals = support[:, parents, None], support[:, None]
    leakage = (rivals-own).clamp_min(0) / (rivals+own).clamp_min(epsilon)
    risk = (leakage * conflict[None]).amax(-1) * supported[:, None]
    return (1-risk).clamp(0, 1)


class ClassRelativeAliasReader(ExcessAliasReader):
    def report(self):
        result = super().report()
        result.update(implementation=IMPLEMENTATION,
            source="canonical text contradiction times Geometry-supported rival leakage; no absolute response gain",
            historical_control="exact original gain-conditioned ExcessReject_Coupled")
        return result

    @torch.inference_mode()
    def read(self, local_aliases, relation, coordinates, valid, broad, crops, auxiliary, crop_count, image_size, beta=1.):
        cfg = self.config
        support, supported, effective = canonical_support(local_aliases[:, self.canonical] / cfg.local_temperature,
                                                         relation, coordinates, valid, cfg)
        output = {method: [] for method in METHODS[3:]}
        total_risk, active, gains, correction, shuffle_error, old_risk = 0., 0, 0, 0., 0., 0.
        for start in range(0, len(local_aliases), cfg.query_chunk):
            sl = slice(start, start+cfg.query_chunk)
            coords = coordinates[sl]
            wide = torch.zeros_like(local_aliases[sl])
            fields = []
            for crop, cosines in zip(crops, auxiliary):
                indices, coefficients = crop_stencil(crop, crop_count, coords, image_size)
                wide += (cosines[indices] * coefficients[..., None]).sum(1)
                fields.append((profiled_logits(crop, self.members)[indices], coefficients))
            retention = relative_retention(support[sl], supported[sl], self.conflict, self.bank.parent_indices, cfg.epsilon)
            previous = contextual_retention(local_aliases[sl], wide, support[sl], supported[sl],
                                            self.conflict, self.bank.parent_indices, cfg)
            grouped = retention[:, self.members]
            variants = {PRIMARY: grouped, "ExcessReject_Coupled": previous[:, self.members],
                "TextOnlyReject_Coupled": (1-self.conflict.amax(-1))[self.members][None].expand_as(grouped),
                "HardDelete_Coupled": grouped}
            for method, permutation in zip(SHUFFLED, self.permutations):
                variants[method] = grouped.gather(-1, permutation[None].expand_as(grouped))
                shuffle_error = max(shuffle_error, float((variants[method].sort(-1).values-grouped.sort(-1).values).abs().max()))
            for method, values in variants.items():
                delta = torch.zeros_like(broad[sl])
                for evidence, coefficients in fields:
                    writer = hard_delete_delta if method == "HardDelete_Coupled" else excess_delta
                    delta += (writer(evidence, values[:, None].expand_as(evidence), beta)*coefficients[..., None]).sum(1)
                delta = delta.masked_fill(~valid[sl, None], 0.)
                if method == PRIMARY:
                    correction += float(delta[valid[sl]].abs().sum())
                output[method].append(broad[sl]+delta)
            risk = 1-retention
            total_risk += float(risk[valid[sl]].sum())
            old_risk += float((1-previous)[valid[sl]].sum())
            active += int((risk[valid[sl]] > 1e-6).sum())
            gains += int((wide[valid[sl]] > local_aliases[sl][valid[sl]]).sum())
        n = max(int(valid.sum()), 1)
        denominator = n*len(self.bank.alias_names)
        return {method: torch.cat(values) for method, values in output.items()}, {
            "supported_patch_fraction": float(supported[valid].float().mean()) if bool(valid.any()) else 0.,
            "effective_support": float(effective[supported].mean()) if bool(supported.any()) else 0.,
            "mean_rejection": total_risk/denominator, "previous_mean_rejection": old_risk/denominator,
            "rejected_alias_fraction": active/denominator, "positive_gain_fraction": gains/denominator,
            "mean_score_suppression": correction/(n*self.bank.class_count), "shuffle_spectrum_error": shuffle_error,
        }
