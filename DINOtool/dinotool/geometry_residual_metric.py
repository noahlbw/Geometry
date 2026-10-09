"""Geometry-conditioned bounded semantic metric, without spatial label fusion."""
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .geometry_readout_trace import alias_class_scores
from .matched_readout_controls import run_head


IMPLEMENTATION = "geometry-residual-semantic-metric-v1-20261002"
PRIMARY = "Geometry_ResidualMetric"
METHODS = ("Geometry", "SCLIP_Two", "VIPProxy_Two", "UniformMetric",
           "SpatialMetric", "ShuffledMetric", "MeanLogit_UniformMetric", PRIMARY)


@dataclass(frozen=True)
class ResidualMetricConfig:
    basis_relative_tolerance: float = 1e-6
    spatial_sigma: float = .25

    def signature(self):
        return {"implementation": IMPLEMENTATION, "metric": asdict(self),
            "source": "original fine512 Geometry descriptors and relation; no extra encoder/view",
            "basis": "uncentered span of the union of all original encoded aliases; numerical SVD rank only",
            "measurement": "R=(I-G_valid) Y B; C=R^T R/n_valid; no subtraction from deployed descriptors",
            "metric_rule": "M=I+C/trace(C); zero C gives exact original alias scores",
            "numerical_neutrality": "trace below64*fp32_eps^2*mean projected norm^2 is roundoff-neutral",
            "readout": "joint visual/text Mahalanobis cosine in text span, unchanged orthogonal visual component",
            "bound": "inverse metric eigenvalues in [1/2,1]; no signed donor coefficients or class thresholds",
            "output": "one metric-based class field; all20 aliases and original normalized LME .07",
            "adaptation": "image-window inference-state covariance; network weights and encoded text remain frozen",
            "limits": "Geometry residuals can contain true discriminative variation; covariance is not certified nuisance",
            "novelty": "whitening, Mahalanobis metrics and graph residuals have precedents; no priority claim"}


def text_span(texts, config=ResidualMetricConfig()):
    if not texts or config.basis_relative_tolerance <= 0:
        raise ValueError("Expected nonempty encoded text banks and positive numerical tolerance.")
    values = torch.cat(list(texts.values()))
    if values.ndim != 2 or not bool(torch.isfinite(values).all()):
        raise ValueError("Expected finite [aliases,channels] texts.")
    with torch.autocast(device_type=values.device.type, enabled=False):
        _, singular, vectors = torch.linalg.svd(values.double(), full_matrices=False,
            **({"driver": "gesvd"} if values.is_cuda else {}))
        count = int((singular > singular.max()*config.basis_relative_tolerance).sum())
        if not count:
            raise ValueError("Encoded aliases have zero span.")
        basis = vectors[:count].T.float()
        error = float((values.float()-values.float() @ basis @ basis.T).abs().max())
        if error > 2e-5:
            raise RuntimeError("Numerical text span loses encoded alias content.")
    return basis, error


def valid_relation(relation, valid):
    if (relation.ndim != 3 or relation.shape[1] != relation.shape[2]
            or valid.shape != relation.shape[:2] or valid.dtype != torch.bool
            or not bool(torch.isfinite(relation).all()) or bool((relation < 0).any())):
        raise ValueError("Expected finite nonnegative [B,N,N] relation and boolean [B,N] validity.")
    weights = relation.float().masked_fill(~valid[:, None], 0.)
    mass = weights.sum(-1, keepdim=True)
    normalized = weights/mass.clamp_min(torch.finfo(torch.float32).tiny)
    identity = torch.eye(relation.shape[1], device=relation.device)[None]
    return torch.where(valid[..., None] & (mass > 0), normalized, identity)


def spatial_relation(count, device, sigma=.25):
    extent = int(count**.5)
    if extent*extent != count or sigma <= 0:
        raise ValueError("Expected a square patch lattice and positive spatial sigma.")
    yy, xx = torch.meshgrid(torch.arange(extent, device=device), torch.arange(extent, device=device), indexing="ij")
    xy = torch.stack((yy.flatten(), xx.flatten()), -1).float()/extent
    distance = (xy[:, None]-xy[None]).square().sum(-1)
    return (-distance/(2*sigma**2)).softmax(-1)[None]


def residual_covariance(projected, relation, valid):
    if (projected.ndim != 3 or projected.shape[:2] != valid.shape
            or not bool(torch.isfinite(projected).all())):
        raise ValueError("Expected finite [B,N,rank] projected descriptors.")
    weights = valid_relation(relation, valid)
    residual = (projected.float()-weights @ projected.float()).masked_fill(~valid[..., None], 0.)
    covariance = residual.transpose(-1, -2) @ residual
    covariance /= valid.sum(-1)[:, None, None].clamp_min(1)
    return .5*(covariance+covariance.transpose(-1, -2))


def metric_alias_scores(features, text, basis, covariance, valid):
    if (features.ndim != 3 or text.ndim != 2 or basis.ndim != 2
            or features.shape[-1] != text.shape[-1] or basis.shape[0] != text.shape[-1]
            or covariance.shape != (features.shape[0], basis.shape[1], basis.shape[1])
            or valid.shape != features.shape[:2] or valid.dtype != torch.bool
            or not all(bool(torch.isfinite(x).all()) for x in (features, text, basis, covariance))):
        raise ValueError("Expected finite compatible descriptors, text span, covariance and validity.")
    with torch.autocast(device_type=features.device.type, enabled=False):
        original = features.float() @ text.float().T
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
        if bool((trace < 0).any()):
            raise ValueError("Covariance trace must be nonnegative.")
        visual = features.float() @ basis
        roundoff = 64*torch.finfo(torch.float32).eps**2*(visual.square().sum(-1).masked_fill(~valid, 0.)
            .sum(-1)/valid.sum(-1).clamp_min(1))
        observed = trace > roundoff
        if not bool(observed.any()):
            return original, {"covariance_trace": 0., "metric_condition_upper_bound": 1.,
                              "mean_absolute_alias_change": 0.}
        identity = torch.eye(basis.shape[1], device=basis.device)[None]
        normalized = covariance/trace[:, None, None].clamp_min(torch.finfo(torch.float32).tiny)
        metric = identity+normalized.masked_fill(~observed[:, None, None], 0.)
        factor = torch.linalg.cholesky(metric)
        semantic = text.float() @ basis
        visual_new = torch.linalg.solve_triangular(factor, visual.transpose(-1, -2), upper=False).transpose(-1, -2)
        semantic_new = torch.linalg.solve_triangular(factor,
            semantic.T[None].expand(features.shape[0], -1, -1), upper=False).transpose(-1, -2)
        # Preserve the component perpendicular to the numerical text span.
        visual_extra = (features.float().square().sum(-1)-visual.square().sum(-1)).clamp_min(0.)
        semantic_extra = (text.float().square().sum(-1)-semantic.square().sum(-1)).clamp_min(0.)
        numerator = original-visual @ semantic.T+visual_new @ semantic_new.transpose(-1, -2)
        denominator = ((visual_new.square().sum(-1)+visual_extra).sqrt()[..., None]
            * (semantic_new.square().sum(-1)+semantic_extra[None]).sqrt()[:, None])
        scores = numerator/denominator.clamp_min(torch.finfo(torch.float32).tiny)
        scores = torch.where(valid[..., None] & observed[:, None, None], scores, original)
        change = (scores-original).abs().masked_fill(~valid[..., None], 0.)
    return scores, {"covariance_trace": float(trace.mean()), "metric_condition_upper_bound": 2.,
        "mean_absolute_alias_change": float(change.sum()/max(int(valid.sum())*text.shape[0], 1))}


class ResidualMetricReader:
    def __init__(self, geometry, banks):
        self.geometry = geometry
        self.texts = {key: F.normalize(bank.features.float(), dim=-1) for key, bank in banks.items()}
        self.basis, self.span_error = text_span(self.texts)
        self.text_fingerprints = {key: value.detach().clone() for key, value in self.texts.items()}

    @torch.inference_mode()
    def __call__(self, geometry, prepared, banks, texts, valid):
        feature = prepared.geometry_projected.float()
        with torch.autocast(device_type=feature.device.type, enabled=False):
            projected = feature @ self.basis
            relation = prepared.geometry_patch_conditional.float()
            spatial = spatial_relation(feature.shape[1], feature.device).expand_as(relation)
            active = valid[0].nonzero().flatten()
            permutation = torch.arange(feature.shape[1], device=feature.device)
            permutation[active] = active.roll(len(active)//2)
            relations = {PRIMARY: relation, "UniformMetric": torch.ones_like(relation),
                "SpatialMetric": spatial, "ShuffledMetric": relation[:, permutation][:, :, permutation]}
            covariances = {method: residual_covariance(projected, weights, valid)
                          for method, weights in relations.items()}
        head = geometry.backbone.model.visual_model.head
        with geometry.backbone._autocast():
            raw = prepared.backbone_tokens[:, prepared.prefix_tokens:]
            controls = {method: run_head(head, prepared.backbone_tokens, raw, relation,
                prepared.prefix_tokens, method, prepared.block_index)[0]
                for method in ("SCLIP_Two", "VIPProxy_Two")}
        scores, diagnostics = {}, {"text_span_rank": self.basis.shape[1], "text_span_max_error": self.span_error}
        for key, bank in banks.items():
            if not torch.equal(texts[key], self.text_fingerprints[key]):
                raise RuntimeError("Original encoded text changed.")
            aliases = feature @ texts[key].T
            group = {"Geometry": alias_class_scores(aliases, bank.parent_indices, bank.class_count)}
            for method, values in controls.items():
                group[method] = alias_class_scores(values.float() @ texts[key].T, bank.parent_indices, bank.class_count)
            for method, covariance in covariances.items():
                response, current = metric_alias_scores(feature, texts[key], self.basis, covariance, valid)
                group[method] = alias_class_scores(response, bank.parent_indices, bank.class_count)
                if method == PRIMARY and key == next(iter(banks)):
                    diagnostics.update(current)
            group["MeanLogit_UniformMetric"] = .5*(group["Geometry"]+group["UniformMetric"])
            if any(not bool(torch.isfinite(value).all()) for value in group.values()):
                raise RuntimeError("Nonfinite metric class field.")
            scores[key] = group
        return scores, diagnostics
