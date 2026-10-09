"""Joint nonnegative text decoding under a Geometry residual metric.

Geometry measures unexplained feature residuals, not agreement of class labels.
The convex decoder and projected-gradient solver are established machinery;
whether this coupling adds useful segmentation evidence is an experimental claim.
"""
from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F

from .geometry_readout_trace import alias_class_scores
from .matched_readout_controls import run_head


IMPLEMENTATION = "geometry-joint-semantic-residual-demixing-v1-20261002"
PRIMARY = "Geometry_SemanticDemix"
BASELINES = ("Geometry", "SCLIP_Two", "VIPProxy_Two")
METHODS = (*BASELINES, "TextOnlyDemix", "MeanLogit_Demix", "ShuffledGeometryDemix", PRIMARY)


@dataclass(frozen=True)
class DemixConfig:
    ridge: float = .01
    maximum_iterations: int = 2048
    check_every: int = 8
    kkt_tolerance: float = 5e-4

    def validate(self):
        if (not math.isfinite(self.ridge) or self.ridge <= 0 or self.maximum_iterations < 1
                or self.check_every < 1 or not math.isfinite(self.kkt_tolerance)
                or self.kkt_tolerance <= 0):
            raise ValueError("Invalid fixed demixing solver settings.")

    def signature(self):
        return {"demixing": asdict(self), "implementation": IMPLEMENTATION,
            "objective": "min_X>=0 .25(||Y-XT||_F^2+||G(Y-XT)||_F^2)+.5*ridge*||X||_F^2",
            "Y": "unchanged final normalized Geometry descriptors, no additional encoder or crop",
            "T": "all20 normalized alias descriptors, jointly decoded across every class",
            "G": "original Geometry restricted to valid patches and row-normalized",
            "scores": "per-patch norm of each class's explained descriptor X_c T_c",
            "neutral": "zero explained descriptor or invalid patch retains exact original Geometry scores",
            "control_identity": "G=I exactly recovers TextOnlyDemix, not original Geometry",
            "solver": "FP32 monotone restarted FISTA with quadratic backtracking and projected KKT check",
            "padding": "invalid patches excluded from both residual observations and coefficients",
            "text": "unchanged20 aliases/six RS templates; no hard alias deletion or per-class quota",
            "view": "native512/128/Hann probability blending at .07",
            "limits": "deterministic reorganization of existing information, not an independent semantic teacher; "
                      "text directions and group norm are not calibrated posterior probabilities"}


def residual_metric(value, relation):
    if relation is None:
        return value
    return .5 * (value + relation.T @ (relation @ value))


def reconstruction_objective(features, coefficients, text, relation, ridge):
    residual = coefficients @ text - features
    return .5 * (residual * residual_metric(residual, relation)).sum() + .5 * ridge * coefficients.square().sum()


class JointSemanticDecoder:
    def __init__(self, text, parents, classes, config=DemixConfig()):
        config.validate()
        if (text.ndim != 2 or parents.shape != text.shape[:1] or parents.dtype != torch.long
                or classes < 1 or not bool(torch.isfinite(text).all())
                or bool((text.norm(dim=-1) <= 0).any())
                or set(parents.tolist()) != set(range(classes))):
            raise ValueError("Finite nonzero text and a complete class assignment are required.")
        self.config, self.parents, self.classes = config, parents.clone(), classes
        self.text = F.normalize(text.detach().float(), dim=-1).clone()
        self.gram = self.text @ self.text.T
        self.text_lipschitz = float(torch.linalg.eigvalsh(self.gram).amax())

    @torch.inference_mode()
    def solve(self, features, relation=None):
        if (features.ndim != 2 or features.shape[-1] != self.text.shape[-1]
                or not bool(torch.isfinite(features).all())):
            raise ValueError("Finite feature matrix must match text dimension.")
        n = len(features)
        if relation is not None:
            if (relation.shape != (n, n) or not bool(torch.isfinite(relation).all())
                    or bool((relation < 0).any())
                    or not bool(torch.isclose(relation.sum(-1), torch.ones(n, device=relation.device),
                                              atol=1e-5, rtol=1e-5).all())):
                raise ValueError("Residual observation requires a nonnegative stochastic Geometry.")
            # Exact control recovery avoids roundoff differences in the solver path.
            if torch.equal(relation, torch.eye(n, device=relation.device, dtype=relation.dtype)):
                relation = None
        with torch.autocast(features.device.type, enabled=False):
            y = features.float()
            g = None if relation is None else relation.float()
            correlations = y @ self.text.T
            metric = None if g is None else .5*(torch.eye(n, device=y.device) + g.T @ g)
            rhs = correlations if metric is None else metric @ correlations
            ridge = self.config.ridge
            scale = max(float(rhs.abs().max()), 1e-8)
            lipschitz = self.text_lipschitz + ridge
            current = correlations.clamp_min(0) / lipschitz
            extrapolated, momentum = current.clone(), 1.

            def hessian(x):
                semantic = x @ self.gram
                return (semantic if metric is None else metric @ semantic) + ridge*x

            def gradient(x):
                return hessian(x) - rhs

            def inner(first, second):
                return (first.double()*second.double()).sum()

            def kkt_error(x):
                grad = gradient(x)
                violation = torch.where(x > 0, grad.abs(), (-grad).clamp_min(0))
                return float(violation.amax())/scale

            initial_objective = float(reconstruction_objective(y, current, self.text, g, ridge))
            backtracks, restarts, kkt = 0, 0, float("inf")
            for iteration in range(self.config.maximum_iterations):
                grad = gradient(extrapolated)
                for _ in range(32):
                    proposed = (extrapolated-grad/lipschitz).clamp_min(0)
                    displacement = proposed-extrapolated
                    curvature = inner(displacement, hessian(displacement))
                    bound = lipschitz*inner(displacement, displacement)
                    # For a quadratic, this is the exact backtracking condition.
                    # Subtracting two near-equal total objectives is unstable.
                    if float(curvature) <= float(bound)*(1+2e-6)+1e-24:
                        break
                    lipschitz *= 2
                    backtracks += 1
                else:
                    raise RuntimeError("Semantic decoder quadratic backtracking failed.")
                step = proposed-current
                objective_delta = inner(gradient(current), step)+.5*inner(step, hessian(step))
                if float(objective_delta) > 1e-12:
                    extrapolated, momentum = current.clone(), 1.
                    restarts += 1
                    kkt = kkt_error(current)
                    if kkt <= self.config.kkt_tolerance:
                        break
                    continue
                previous, current = current, proposed
                next_momentum = .5*(1+math.sqrt(1+4*momentum**2))
                extrapolated = current + (momentum-1)/next_momentum*(current-previous)
                momentum = next_momentum
                if (iteration+1) % self.config.check_every == 0 or iteration+1 == self.config.maximum_iterations:
                    kkt = kkt_error(current)
                    if kkt <= self.config.kkt_tolerance:
                        break
            final_objective = float(reconstruction_objective(y, current, self.text, g, ridge))
            if (not bool(torch.isfinite(current).all()) or kkt > self.config.kkt_tolerance
                    or final_objective > initial_objective+1e-5*max(initial_objective, 1.)):
                raise RuntimeError(f"Semantic decoder invariant/convergence failed: KKT={kkt}, "
                                   f"objective={initial_objective}->{final_objective}, "
                                   f"iterations={iteration+1}, restarts={restarts}, backtracks={backtracks}")
            return current, {"solver_iterations": float(iteration+1), "solver_kkt_relative": kkt,
                "solver_backtracks": float(backtracks), "solver_restarts": float(restarts),
                "initial_objective": initial_objective/n, "final_objective": final_objective/n,
                "active_coefficient_fraction": float((current > 0).float().mean())}

    @torch.inference_mode()
    def decode(self, features, relation, valid):
        if valid.shape != features.shape[:1] or valid.dtype != torch.bool:
            raise ValueError("Image-only validity must match the feature grid.")
        base = alias_class_scores(features.float() @ self.text.T, self.parents, self.classes)
        positions = valid.nonzero(as_tuple=False).flatten()
        if not len(positions):
            return base, {"solver_iterations": 0., "solver_kkt_relative": 0.}
        g = None
        if relation is not None:
            if relation.shape != (len(features), len(features)):
                raise ValueError("Geometry and feature grid differ.")
            g = relation[positions][:, positions].float()
            if bool((g.sum(-1) <= 0).any()):
                raise ValueError("Valid query has no valid Geometry support.")
            g = g/g.sum(-1, keepdim=True)
        coefficients, diagnostics = self.solve(features[positions], g)
        score = torch.stack([(coefficients[:, self.parents == c] @ self.text[self.parents == c]).norm(dim=-1)
                             for c in range(self.classes)], -1)
        active = coefficients.sum(-1) > 0
        output = base.clone()
        output[positions] = torch.where(active[:, None], score, base[positions])
        diagnostics["no_explained_evidence_fraction"] = float((~active).float().mean())
        diagnostics["mean_absolute_score_displacement"] = float((output-base).abs().mean())
        return output, diagnostics


@torch.inference_mode()
def read_semantic_demixing(geometry, prepared, banks, texts, valid, decoders, *, controls=True):
    if prepared.geometry_projected.shape[0] != 1 or valid.shape != prepared.geometry_projected.shape[:2]:
        raise ValueError("Demixing expects a single image window.")
    head = geometry.backbone.model.visual_model.head
    features = {"Geometry": prepared.geometry_projected}
    if controls:
        with geometry.backbone._autocast():
            for method in BASELINES[1:]:
                features[method] = run_head(head, prepared.backbone_tokens,
                    prepared.backbone_tokens[:, prepared.prefix_tokens:], prepared.geometry_patch_conditional,
                    prepared.prefix_tokens, method, prepared.block_index)[0]
    scores, diagnostics = {}, {}
    relation = prepared.geometry_patch_conditional[0]
    for key, bank in banks.items():
        group = {method: alias_class_scores(feature.float() @ texts[key].T, bank.parent_indices, bank.class_count)
                 for method, feature in features.items()}
        decoded, current = decoders[key].decode(prepared.geometry_projected[0], relation, valid[0])
        group[PRIMARY] = decoded[None]
        for field, value in current.items():
            diagnostics[field] = max(diagnostics.get(field, 0.), value) if "kkt" in field else diagnostics.get(field, 0.)+value/len(banks)
        if controls:
            local, _ = decoders[key].decode(prepared.geometry_projected[0], None, valid[0])
            group["TextOnlyDemix"] = local[None]
            group["MeanLogit_Demix"] = .5*(group["Geometry"]+group["TextOnlyDemix"])
            positions = valid[0].nonzero(as_tuple=False).flatten()
            permutation = torch.arange(len(relation), device=relation.device)
            permutation[positions] = positions.roll(max(len(positions)//2, 1))
            shuffled = relation[permutation][:, permutation]
            group["ShuffledGeometryDemix"] = decoders[key].decode(prepared.geometry_projected[0], shuffled, valid[0])[0][None]
        if any(not bool(torch.isfinite(value).all()) for value in group.values()):
            raise RuntimeError("Nonfinite semantic demixing scores.")
        scores[key] = group
    return scores, diagnostics
