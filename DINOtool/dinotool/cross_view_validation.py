"""Frozen native wide observations with cross-view validation of Geometry updates."""
import torch
import torch.nn.functional as F

from .geometry_readout_trace import alias_class_scores
from .geometry_semantic_innovation import anchored_innovation


IMPLEMENTATION = "geometry-native-cross-view-validation-v1-20261004"
PRIMARY = "Validated_Native"
METHODS = ("Geometry", "NativeWide", "MeanLogit_Native", "Anchored_Native", PRIMARY)
CONFIG = {
    "wide_long_edge": 448,
    "wide_crop": 336,
    "wide_shift": 56,
    "wide_stride": 112,
    "wide_encoder": "unchanged frozen DINOv3/DINO.text native head",
    "wide_text": "same local all20 bank; six RS templates; normalized LME .07",
    "outside_image": "zero RGB padding; only image-valid pixels enter assembled maps",
    "centering": "subtract the class mean independently at each patch",
    "validation": "one bounded analytic coefficient per local tile and view; opposite view target",
    "writeback": "mean of two validated unchanged anchored Geometry innovations",
    "local_assembly": "unchanged512/overlap128/Hann; interpolation before temperature .07",
}


def shifted_starts(length, crop=336, shift=56):
    starts = [-shift]
    while starts[-1] + crop < length:
        starts.append(starts[-1] + CONFIG["wide_stride"])
    return starts


def padded_crop(image, top, left, crop=336):
    h, w = image.shape[-2:]
    y0, x0 = max(top, 0), max(left, 0)
    y1, x1 = min(top + crop, h), min(left + crop, w)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("Wide crop does not intersect the image.")
    rgb = F.pad(image[:, y0:y1, x0:x1],
                (x0-left, left+crop-x1, y0-top, top+crop-y1))
    return rgb, (y0, y1, x0, x1, y0-top, x0-left)


@torch.inference_mode()
def native_patch_features(backbone, rgb):
    backbone._validate_rgb(rgb)
    normalized = (rgb.to(backbone.device) - backbone._imagenet_mean) / backbone._imagenet_std
    with backbone._autocast():
        visual = backbone.model.visual_model
        cls, raw, registers = visual.get_backbone_features(normalized)
        tokens = torch.cat((cls[:, None], registers, raw), 1)
        projected = visual.head(tokens)
    return F.normalize(projected[:, tokens.shape[1]-raw.shape[1]:].float(), dim=-1)


@torch.inference_mode()
def assemble_native_view(image, geometry, banks, texts, y_starts, x_starts):
    h, w = image.shape[-2:]
    maps = {key: torch.zeros(bank.class_count, h, w, device=geometry.device)
            for key, bank in banks.items()}
    count = torch.zeros(h, w, device=geometry.device)
    for top in y_starts:
        for left in x_starts:
            rgb, (y0, y1, x0, x1, cy, cx) = padded_crop(image, top, left)
            features = native_patch_features(geometry.backbone, rgb[None])
            for key, bank in banks.items():
                scores = alias_class_scores(features @ texts[key].T, bank.parent_indices, bank.class_count)
                dense = F.interpolate(scores.transpose(1, 2).reshape(1, bank.class_count, 21, 21),
                                      size=(336, 336), mode="bilinear", align_corners=False)[0]
                maps[key][:, y0:y1, x0:x1] += dense[:, cy:cy+y1-y0, cx:cx+x1-x0]
            count[y0:y1, x0:x1] += 1
    if not bool((count > 0).all()):
        raise ValueError("Incomplete native wide-view coverage.")
    return {key: value/count[None] for key, value in maps.items()}, len(y_starts)*len(x_starts)


def valid_relation(relation, valid):
    weights = relation.float().masked_fill(~valid[:, None], 0.)
    weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-12)
    return weights.masked_fill(~valid[..., None], 0.)


def validation_coefficient(direction, target, weights):
    """Exact bounded minimizer of ||R(a*direction-target)|| squared."""
    projected, residual_target = weights @ direction, weights @ target
    denominator = projected.double().square().sum((1, 2), keepdim=True)
    numerator = (projected.double()*residual_target.double()).sum((1, 2), keepdim=True)
    coefficient = torch.where(denominator > 0,
                              numerator/denominator.clamp_min(torch.finfo(torch.float64).tiny), 0.)
    coefficient = coefficient.clamp(0., 1.).to(direction.dtype)
    before = residual_target.double().square().sum((1, 2))
    after = (coefficient*projected-residual_target).double().square().sum((1, 2))
    return coefficient, before, after


@torch.inference_mode()
def validated_scores(local, first, second, relation, valid):
    if first.shape != local.shape or second.shape != local.shape:
        raise ValueError("Both observations must match local class scores.")
    if not bool(torch.isfinite(first).all() and torch.isfinite(second).all()):
        raise ValueError("Wide scores must be finite.")
    center = lambda value: value.float()-value.float().mean(-1, keepdim=True)
    g, w1, w2 = map(center, (local, first, second))
    corrected1, diagnostic1 = anchored_innovation(g, w1, relation, valid)
    corrected2, diagnostic2 = anchored_innovation(g, w2, relation, valid)
    d1, d2 = corrected1-g, corrected2-g
    weights = valid_relation(relation, valid)
    a1, before1, after1 = validation_coefficient(d1, w2-g, weights)
    a2, before2, after2 = validation_coefficient(d2, w1-g, weights)
    outputs = {
        "Geometry": local,
        "NativeWide": .5*(first+second),
        "MeanLogit_Native": .5*(local+.5*(first+second)),
        "Anchored_Native": local+.5*(d1+d2),
        PRIMARY: local+.5*(a1*d1+a2*d2),
    }
    diagnostics = {
        "validation_coefficient_view1": float(a1.mean()),
        "validation_coefficient_view2": float(a2.mean()),
        "coefficient_zero_fraction": float(torch.cat((a1.flatten(), a2.flatten())).eq(0).float().mean()),
        "coefficient_one_fraction": float(torch.cat((a1.flatten(), a2.flatten())).eq(1).float().mean()),
        "validation_error_before": float((before1+before2).mean()),
        "validation_error_after": float((after1+after2).mean()),
        "mean_absolute_ungated_innovation": float((.5*(d1+d2))[valid].abs().mean()),
        "mean_absolute_validated_innovation": float((outputs[PRIMARY]-local)[valid].abs().mean()),
        "solver_relative_residual": max(diagnostic1["solver_relative_residual"], diagnostic2["solver_relative_residual"]),
    }
    if not all(bool(torch.isfinite(value).all()) for value in outputs.values()):
        raise RuntimeError("Nonfinite cross-view output.")
    return outputs, diagnostics
