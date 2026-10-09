"""Use one Geometry support operator for semantic interventions and reconstruction."""
import torch
import torch.nn.functional as F

from .geometry_readout_trace import alias_class_scores
from .prompts import REMOTE_SENSING_TEMPLATES
from .support_conditioned_geometry import SupportConfig, choose_regions, partition_supports


IMPLEMENTATION = "geometry-support-erasure-semantic-readout-v3-score-influence-20261001"
METHODS = ("Geometry", "SupportGlobal", "SupportErasedGlobal", "SupportErasure",
           "BalancedGlobal", "BalancedErasedGlobal", "BalancedErasure", "ScoreGain", "ScoreInfluence")


def reconstruct_support_scores(local, operator, observation, *, normalize_rows=False):
    """Solve .5||z-g||^2+.5||H z-b||^2 via the small support-space system."""
    if (local.ndim != 2 or operator.ndim != 2 or observation.ndim != 2
            or operator.shape[1] != local.shape[0]
            or observation.shape != (operator.shape[0], local.shape[1])):
        raise ValueError("Expected [N,C], [R,N] and [R,C] tensors.")
    if not all(bool(torch.isfinite(value).all()) for value in (local, operator, observation)):
        raise ValueError("Scores and supports must be finite.")
    if bool((operator < 0).any()):
        raise ValueError("Support weights must be nonnegative.")
    if operator.shape[0] == 0:
        return local.clone()
    with torch.autocast(device_type=local.device.type, enabled=False):
        g, h, b = local.float(), operator.float(), observation.float()
        residual = b-h @ g
        if normalize_rows:
            scale = h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            h, residual = h/scale, residual/scale
        small = torch.eye(len(h), device=h.device)+h @ h.T
        return g+h.T @ torch.linalg.solve(small, residual)


def balanced_support_scores(local, operator, observation):
    """L2-normalize measurement rows and targets, not their semantic confidence.

    For a uniform N-token support, unscaled reconstruction writes (b-g)/(N+1)
    per token. This fixed normalization gives (b-g)/2 independent of N.
    """
    return reconstruct_support_scores(local, operator, observation, normalize_rows=True)


def semantic_score_influence(local, operator, score_gain, valid, *, normalize_area=True):
    """Write measured native class-score changes, not an absolute class prototype.

    Divide by actual soft-erasure area for the fixed per-area variant. This is
    a finite-difference hypothesis, not exact patch attribution through the
    nonlinear image encoder. Small-support replacement artifacts can amplify.
    """
    if valid.shape != local.shape[:1] or valid.dtype != torch.bool:
        raise ValueError("Requires image-valid token centers, without label validity.")
    if not bool(valid.any()) or len(operator) == 0:
        return local.clone()
    gain = score_gain.float()
    if normalize_area:
        h = operator.float().masked_fill(~valid[None], 0.)
        area = h.sum(-1, keepdim=True)/h.amax(-1, keepdim=True).clamp_min(1e-12)/valid.sum()
        gain = gain/area.clamp_min(1e-12)
    delta = reconstruct_support_scores(torch.zeros_like(local), operator, gain, normalize_rows=True)
    return torch.where(valid[:, None], local+delta, local)


def erase_supported_rgb(rgb, patch_mask, valid):
    """Keep coordinates/FOV fixed; replace support by the valid-window RGB mean."""
    if (rgb.ndim != 4 or rgb.shape[:2] != (1, 3) or patch_mask.ndim != 2
            or valid.shape != patch_mask.shape or not bool(valid.any())):
        raise ValueError("Requires one RGB window and a nonempty valid patch grid.")
    if (not bool(torch.isfinite(patch_mask).all()) or bool((patch_mask < 0).any())
            or bool((patch_mask > 1).any())):
        raise ValueError("Erasure strength must be finite and in [0,1].")
    pixel_valid = F.interpolate(valid[None, None].float(), rgb.shape[-2:], mode="nearest")
    mask = F.interpolate((patch_mask*valid)[None, None].float(), rgb.shape[-2:], mode="nearest")
    fill = (rgb.float()*pixel_valid).sum((-2, -1), keepdim=True)/pixel_valid.sum().clamp_min(1.)
    return rgb.float()*(1-mask)+fill*mask


def support_operator(raw, relation, local, valid, config):
    """Supports use only original raw features, scores and image bounds."""
    if raw.ndim != 2 or relation.shape != (len(raw), len(raw)):
        raise ValueError("Expected raw tokens and matching relation.")
    height, width = valid.shape
    if height*width != len(raw):
        raise ValueError("Image-valid patch grid does not match tokens.")
    labels = partition_supports(raw.reshape(height, width, -1), valid, config)
    regions = choose_regions(labels, local.reshape(height, width, -1), config)
    rows = []
    for region in regions:
        selected = region.mask.reshape(-1)
        row = relation[selected].float().mean(0).masked_fill(~valid.reshape(-1), 0.)
        if float(row.sum()) > 0:
            rows.append(row/row.sum())
    return torch.stack(rows) if rows else local.new_zeros((0, len(raw)))


@torch.inference_mode()
def full_text_queries(backbone, bank):
    prompts = [template.format(label=alias) for alias in bank.alias_names for template in REMOTE_SENSING_TEMPLATES]
    values = []
    for start in range(0, len(prompts), 64):
        tokens = backbone.tokenize(prompts[start:start+64]).to(backbone.device)
        with backbone._autocast():
            encoded = backbone.model.encode_text(tokens, normalize=False).float()
        values.append(F.normalize(encoded, dim=-1))
    return F.normalize(torch.cat(values).reshape(len(bank.alias_names), len(REMOTE_SENSING_TEMPLATES), -1).mean(1), dim=-1)


@torch.inference_mode()
def read_support_erasure(geometry, prepared, bank, full_text, rgb, valid,
                         config=SupportConfig()):
    """Observe actual native image-embedding changes, not a virtual region CLS.

    Normalized embedding differences are an unverified semantic observation.
    A support intervention can affect context and create replacement artifacts.
    No correctness probability or guaranteed causal semantic identity is assumed.
    """
    config.validate()
    if prepared.native_global is None or valid.shape != (1, prepared.grid_height*prepared.grid_width):
        raise ValueError("Requires native full-image descriptors and image-valid centers.")
    text = F.normalize(bank.features.float(), dim=-1)
    local = alias_class_scores(prepared.geometry_projected.float() @ text.T,
                               bank.parent_indices, bank.class_count)[0]
    grid_valid = valid[0].reshape(prepared.grid_height, prepared.grid_width)
    if not bool(grid_valid.any()):
        return {method: local[None].clone() for method in METHODS}, {"regions": 0, "resolved": 0}
    operator = support_operator(prepared.raw_patch_tokens[0], prepared.geometry_patch_conditional[0],
                                local, grid_valid, config)
    reference = operator @ local
    observations = {method: [] for method in ("SupportGlobal", "SupportErasedGlobal", "SupportErasure")}
    original = prepared.native_global.float()
    resolved = 0
    norms = []
    for index, row in enumerate(operator):
        strength = (row/row.max()).reshape(grid_valid.shape)
        erased_rgb = erase_supported_rgb(rgb, strength, grid_valid)
        erased = geometry.prepare_image(erased_rgb).native_global.float()
        difference = original-erased
        norm = float(difference.norm())
        norms.append(norm)
        usable = norm > 1e-12
        resolved += int(usable)
        descriptors = {"SupportGlobal": original, "SupportErasedGlobal": erased,
                       "SupportErasure": F.normalize(difference, dim=-1)}
        for method, descriptor in descriptors.items():
            observed = alias_class_scores(descriptor @ full_text.T, bank.parent_indices, bank.class_count)[0]
            observations[method].append(observed if usable else reference[index])
    output = {"Geometry": local[None]}
    balanced_names = {"SupportGlobal": "BalancedGlobal", "SupportErasedGlobal": "BalancedErasedGlobal",
                      "SupportErasure": "BalancedErasure"}
    targets = {}
    for method, rows in observations.items():
        observation = torch.stack(rows) if rows else reference
        targets[method] = observation
        scores = reconstruct_support_scores(local, operator, observation)
        scores = torch.where(valid[0, :, None], scores, local)
        output[method] = scores[None]
        scores = balanced_support_scores(local, operator, observation)
        output[balanced_names[method]] = torch.where(valid[0, :, None], scores, local)[None]
    gain = targets["SupportGlobal"]-targets["SupportErasedGlobal"]
    output["ScoreGain"] = semantic_score_influence(local, operator, gain, valid[0], normalize_area=False)[None]
    output["ScoreInfluence"] = semantic_score_influence(local, operator, gain, valid[0])[None]
    return output, {"regions": len(operator), "resolved": resolved,
                    "mean_erasure_embedding_norm": sum(norms)/max(len(norms), 1),
                    "mean_abs_class_delta": float((output["SupportErasure"]-output["Geometry"]).abs().mean()),
                    "balanced_mean_abs_class_delta": float((output["BalancedErasure"]-output["Geometry"]).abs().mean()),
                    "influence_mean_abs_class_delta": float((output["ScoreInfluence"]-output["Geometry"]).abs().mean()),
                    "support_effective_tokens": operator.square().sum(-1).clamp_min(1e-12).reciprocal().cpu().tolist(),
                    "support_reference": reference.cpu().tolist(),
                    "semantic_observations": {method: observation.cpu().tolist() for method, observation in targets.items()}}
