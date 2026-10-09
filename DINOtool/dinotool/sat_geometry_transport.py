"""Geometry-balanced paired-coordinate transport from a frozen aerial backbone."""
from dataclasses import asdict, dataclass
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .matched_readout_controls import run_head
from .parallel_readout import structural_logits


IMPLEMENTATION = "geometry-balanced-sat-procrustes-v1-20261002"
PRIMARY = "Geometry_SATTransport"
METHODS = ("Geometry", "SCLIP_Two", "VIPProxy_Two", "SAT_Relation",
           "SAT_Unaligned", "SAT_UniformTransport", "MeanLogit_SATTransport", PRIMARY)


def native_patch_states(prepared):
    if prepared.backbone_tokens is None:
        raise ValueError("Native backbone states are required, not normalized relation descriptors.")
    return prepared.backbone_tokens[:, prepared.prefix_tokens:]


@contextmanager
def full_precision_alignment():
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@dataclass(frozen=True)
class SATTransportConfig:
    epsilon: float = 1e-6

    def signature(self):
        return {**asdict(self), "implementation": IMPLEMENTATION,
                "source": "frozen SAT-493M, satellite mean/std, same native512 coordinates",
                "target": "frozen LVD native backbone patches; same-position image-only pairs",
                "state": "window-local orthogonal map and two weighted means; no network updates",
                "pair_weight": "inverse valid-donor incoming mass in original Geometry G; mean one",
                "objective": "min_R sum_i w_i ||(SAT_i-muSAT)R-(LVD_i-muLVD)||^2, R^T R=I",
                "alignment_precision": "FP32 SVD (CUDA gesvd) and matmul; TF32 disabled only inside fit",
                "head": "original two-block Geometry head and original G; native LVD prefixes retained",
                "padding": "invalid patch states revert to original LVD",
                "text": "unchanged20 aliases/six RS templates/normalized LME .07",
                "view": "native512/128 overlap/Hann probability blending"}


def correspondence_weights(relation, valid, epsilon=1e-6):
    if relation.ndim != 3 or relation.shape[1] != relation.shape[2] or valid.shape != relation.shape[:2]:
        raise ValueError("Require original [B,N,N] relation and image-only [B,N] validity.")
    if not bool(torch.isfinite(relation).all()) or bool((relation < 0).any()):
        raise ValueError("Geometry must be finite and nonnegative.")
    geometry = relation.float().masked_fill(~valid[:, None], 0.)
    geometry = geometry/geometry.sum(-1, keepdim=True).clamp_min(epsilon)
    geometry = geometry.masked_fill(~valid[..., None], 0.)
    incoming = geometry.sum(-2)
    weights = torch.where(valid, incoming.clamp_min(epsilon).reciprocal(), torch.zeros_like(incoming))
    return weights*valid.sum(-1, keepdim=True)/weights.sum(-1, keepdim=True).clamp_min(epsilon)


def paired_transport(source, target, weights, valid):
    """Exact weighted orthogonal Procrustes fit of inference states, not labels."""
    if source.shape != target.shape or source.ndim != 3 or weights.shape != source.shape[:2] or valid.shape != weights.shape:
        raise ValueError("Paired frozen feature arrays and correspondence weights must match.")
    if (not all(bool(torch.isfinite(value).all()) for value in (source, target, weights))
            or bool((weights < 0).any()) or bool((weights[~valid] != 0).any())):
        raise ValueError("Require finite features and nonnegative image-valid weights.")
    if torch.equal(source, target):
        return target.clone(), {"alignment_error_before": 0., "alignment_error_after": 0.,
                                "rotation_orthogonality_error": 0., "mean_feature_cosine": 1.}
    outputs, records = [], []
    with torch.autocast(source.device.type, enabled=False), full_precision_alignment():
        for s, t, w, ok in zip(source.float(), target.float(), weights.float(), valid):
            total = w.sum()
            if int(ok.sum()) < 2 or float(total) <= 0:
                outputs.append(t)
                records.append((0., 0., 0., 1.))
                continue
            mu_s, mu_t = (w[:, None]*s).sum(0)/total, (w[:, None]*t).sum(0)/total
            xs, xt = s-mu_s, t-mu_t
            covariance = (xs*w[:, None]).T @ xt
            left, _, right = torch.linalg.svd(covariance, full_matrices=False,
                                             driver="gesvd" if covariance.is_cuda else None)
            rotation = left @ right
            aligned = xs @ rotation+mu_t
            aligned = torch.where(ok[:, None], aligned, t)
            before = (w[:, None]*(xs-xt).square()).sum()/total
            after = (w[:, None]*(aligned-t).square()).sum()/total
            if float(after) > float(before)+1e-4*max(float(before), 1.):
                raise ValueError("The exact correspondence fit increased its objective.")
            identity = torch.eye(rotation.shape[-1], device=rotation.device)
            orthogonal = float((rotation.T @ rotation-identity).abs().max())
            cosine = float(F.cosine_similarity(aligned[ok], t[ok], dim=-1).mean())
            outputs.append(aligned)
            records.append((float(before), float(after), orthogonal, cosine))
    diagnostics = dict(zip(("alignment_error_before", "alignment_error_after",
                            "rotation_orthogonality_error", "mean_feature_cosine"),
                           (sum(row[index] for row in records)/len(records) for index in range(4))))
    result = torch.stack(outputs).to(target.dtype)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Nonfinite transported features.")
    return result, diagnostics


class SATGeometryTransport:
    def __init__(self, backbone, config=SATTransportConfig()):
        self.backbone, self.config = backbone, config
        if not backbone.checkpoints.sat_weights.exists():
            raise FileNotFoundError(backbone.checkpoints.sat_weights)
        from dinov3.hub.backbones import Weights, dinov3_vitl16
        # SAT requires its untied global/local norms; load the explicit local state strictly.
        self.satellite = dinov3_vitl16(pretrained=False, weights=Weights.SAT493M)
        state = torch.load(backbone.checkpoints.sat_weights, map_location="cpu", weights_only=True)
        self.satellite.load_state_dict(state, strict=True)
        self.satellite.to(backbone.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def read(self, prepared, rgb, valid, *, controls=True):
        with self.backbone._autocast():
            normalized = (rgb.to(self.backbone.device)-self.backbone._satellite_mean)/self.backbone._satellite_std
            satellite = self.satellite.forward_features(normalized)["x_norm_patchtokens"]
        return self.read_from_satellite(prepared, satellite, valid, controls=controls)

    @torch.inference_mode()
    def read_from_satellite(self, prepared, satellite, valid, *, controls=True):
        # L2-normalized relation descriptors have the wrong magnitude for the pretrained head.
        raw = native_patch_states(prepared)
        if satellite.shape != raw.shape or valid.shape != raw.shape[:2]:
            raise ValueError("The aerial and native patch grids must have identical actual coordinates.")
        weights = correspondence_weights(prepared.geometry_patch_conditional, valid, self.config.epsilon)
        aligned, diagnostics = paired_transport(satellite, raw, weights, valid)
        diagnostics.update(mean_correspondence_weight=float(weights[valid].mean()) if valid.any() else 0.,
                           maximum_correspondence_weight=float(weights.max()),
                           native_prefix_input_change=0.)
        prefix, head = prepared.prefix_tokens, self.backbone.model.visual_model.head
        def semantic(patches, relation):
            tokens = torch.cat((prepared.backbone_tokens[:, :prefix], patches), 1)
            return run_head(head, tokens, patches, relation, prefix, "Geometry", prepared.block_index)[0]
        with self.backbone._autocast():
            features = {PRIMARY: semantic(aligned, prepared.geometry_patch_conditional)}
            if controls:
                uniform, uniform_stats = paired_transport(satellite, raw, valid.float(), valid)
                for name, value in uniform_stats.items():
                    diagnostics["uniform_"+name] = value
                features["Geometry"] = prepared.geometry_projected
                features["SAT_UniformTransport"] = semantic(uniform, prepared.geometry_patch_conditional)
                unaligned = torch.where(valid[..., None], satellite.to(raw.dtype), raw)
                features["SAT_Unaligned"] = semantic(unaligned, prepared.geometry_patch_conditional)
                relation = structural_logits(satellite, prepared.grid_height, prepared.grid_width,
                                             temperature=.10, spatial_sigma=.25).softmax(-1).to(raw.dtype)
                features["SAT_Relation"] = semantic(raw, relation)
                for name in ("SCLIP_Two", "VIPProxy_Two"):
                    features[name] = run_head(head, prepared.backbone_tokens, raw,
                        prepared.geometry_patch_conditional, prefix, name, prepared.block_index)[0]
        return features, diagnostics
