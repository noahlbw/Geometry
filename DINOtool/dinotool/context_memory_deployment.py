"""Opt-in whole-image memory branch with the existing local Geometry reader."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .compact_deployment import local_observations
from .context_memory_observer import context_assignments, read_response_before_head


@torch.inference_mode()
def memory_rgb(image, *, wide_policy='long448'):
    if wide_policy == 'natural_short336_cap672':
        from .natural_wide_resolution import target_size
        h, w = target_size(*image.shape[-2:])
        array = (image.permute(1, 2, 0).cpu().numpy()*255).round().clip(0, 255).astype(np.uint8)
        rgb = torch.from_numpy(np.ascontiguousarray(cv2.resize(array, (w, h), interpolation=cv2.INTER_LINEAR)))
        rgb = rgb.permute(2, 0, 1).float()/255
    elif wide_policy == 'long448' or (wide_policy.startswith('long') and wide_policy[4:].isdigit()):
        from eval_matched_head_fov import resize_rgb
        edge = 448 if wide_policy == 'long448' else int(wide_policy[4:])
        if edge < 16:
            raise ValueError('Wide long edge must be at least one patch.')
        rgb = resize_rgb(image, edge)
        h, w = rgb.shape[-2:]
    else:
        raise ValueError('Unknown frozen wide policy.')
    ph, pw = (h+15)//16*16, (w+15)//16*16
    return F.pad(rgb, (0, pw-w, 0, ph-h)), (h,w)


@torch.inference_mode()
def memory_wide_observations(image, observer, *, wide_policy='long448'):
    rgb, (h,w) = memory_rgb(image,wide_policy=wide_policy)
    ph,pw = rgb.shape[-2:]
    features = observer.crop_patch_features(rgb)
    return dict(wide_size=(h, w), wide=[dict(top=0, left=0, ah=h, aw=w,
        grid_hw=(ph//16, pw//16), output_hw=(ph, pw), features=features)])


@torch.inference_mode()
def response_before_wide_observations(image, observer, *, wide_policy='long448'):
    rgb, (h, w) = memory_rgb(image, wide_policy=wide_policy)
    ph, pw = rgb.shape[-2:]
    tokens, raw = observer.raw_context_tokens(rgb)
    assignment = context_assignments(raw, observer.config)
    with observer.backbone._autocast():
        features = read_response_before_head(observer.backbone.model.visual_model.head, tokens, assignment)
    features = features[:, -raw.shape[1]:]
    return dict(wide_size=(h, w), wide=[dict(top=0, left=0, ah=h, aw=w,
        grid_hw=(ph//16, pw//16), output_hw=(ph, pw), features=features)])


@torch.inference_mode()
def context_memory_observations(image, geometry, observer, strengths, *, wide_policy='long448', reconstruction=None):
    source = local_observations(image, geometry, strengths, reconstruction=reconstruction)
    return dict(source, **memory_wide_observations(image, observer, wide_policy=wide_policy))


@torch.inference_mode()
def response_before_observations(image, geometry, observer, strengths, *, wide_policy='long448', reconstruction=None):
    source = local_observations(image, geometry, strengths, reconstruction=reconstruction)
    return dict(source, **response_before_wide_observations(image, observer, wide_policy=wide_policy))
