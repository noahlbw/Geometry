"""Geometry coupling on VIP's bounded resized-image/crop protocol."""
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .alias_action_capacity import reconstruction_operator
from .geometry_readout_trace import alias_class_scores
from .inference import hann_blend_window
from .rival_alias_count import canonical_indices
from .shared_rival_soft_alias import METHODS, PRIMARY, alias_permutation, cache_wide, shared_scores
from .sparse_alias_reuse import alias_layout, profile_aliases
from .stratified_soft_alias import WideCrop
from .vip_official_adapter import VIPSettings


IMPLEMENTATION = 'geometry-shared-rival-soft-vip448-v1-20261005'
OBSERVATION = VIPSettings(tau=1., tem=1., background=False, prob_thd=0.)
VIEW_PROTOCOL = dict(resize_long_edge=448, local_crop=336, local_stride=112,
                     token_grid=[21, 21], maximum_geometry_encodings=4,
                     maximum_wide_encodings=4, fine_forwards=0,
                     resize='VIP uint8/OpenCV INTER_LINEAR/half-up dimensions',
                     stitching='retained Hann-weighted class probabilities',
                     restore='bilinear probabilities before argmax',
                     coordinates='resized image only; no native-resolution windows')


@dataclass(frozen=True)
class CropWindow:
    top: int
    left: int
    height: int
    width: int


def resize_for_vip(image):
    if image.ndim != 3 or image.shape[0] != 3 or min(image.shape[-2:]) < 1:
        raise ValueError('Nonempty CHW RGB image required.')
    height, width = image.shape[-2:]
    ratio = OBSERVATION.resize_long_edge / max(height, width)
    resized_size = (int(height * ratio + .5), int(width * ratio + .5))
    if min(resized_size) < 1:
        raise ValueError('VIP resizing would produce an empty short edge.')
    array = (image.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
    resized = cv2.resize(array, resized_size[::-1], interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).float().div_(255)


def crop_windows(height, width):
    crop, stride = OBSERVATION.slide_crop, OBSERVATION.slide_stride
    if min(height, width) < 1 or max(height, width) != OBSERVATION.resize_long_edge:
        raise ValueError('Crop layout requires an already resized VIP image.')
    rows = max(height - crop + stride - 1, 0) // stride + 1
    columns = max(width - crop + stride - 1, 0) // stride + 1
    result = []
    for row in range(rows):
        for column in range(columns):
            bottom = min(row * stride + crop, height)
            right = min(column * stride + crop, width)
            top, left = max(bottom - crop, 0), max(right - crop, 0)
            result.append(CropWindow(top, left, bottom - top, right - left))
    if not 1 <= len(result) <= VIEW_PROTOCOL['maximum_geometry_encodings']:
        raise RuntimeError('VIP-resolution visual budget exceeded.')
    return tuple(result)


def padded_crop(image, window):
    side = OBSERVATION.slide_crop
    rgb = image[:, window.top:window.top + window.height, window.left:window.left + window.width]
    return F.pad(rgb, (0, side - window.width, 0, side - window.height))


def patch_coordinates(window, device):
    axis = (torch.arange(21, device=device).float() + .5) * 16
    yy, xx = torch.meshgrid(axis + window.top, axis + window.left, indexing='ij')
    coordinates = torch.stack((yy, xx), -1).reshape(-1, 2)
    valid = ((coordinates[:, 0] < window.top + window.height)
             & (coordinates[:, 1] < window.left + window.width))
    return coordinates, valid


def alias_crop(features, query, window):
    with torch.autocast(device_type=features.device.type, dtype=torch.bfloat16,
                        enabled=features.device.type == 'cuda'):
        similarity = torch.einsum('bnd,mtd->bnmt', features, query.features.float()).mean(-1)[0]
        patch_mean = F.normalize(features.mean(1), dim=-1)
        text_mean = F.normalize(query.features.float().mean(1), dim=-1)
        salience = ((patch_mean @ text_mean.T)[0] / OBSERVATION.tem).float()
        logits = (similarity * OBSERVATION.logit_scale).float()
    return WideCrop(logits, salience, window.top, window.left, window.height, window.width)


def class_map(crop, layout):
    evidence = profile_aliases(crop.alias_logits, crop.salience, layout)
    scores = (OBSERVATION.tau * evidence).logsumexp(-1) / OBSERVATION.tau
    return F.interpolate(scores.T.reshape(1, len(layout.members), 21, 21), (336, 336),
                         mode='bilinear', align_corners=False)[0].float()


def sample_map(values, coordinates):
    height, width = values.shape[-2:]
    scale = coordinates.new_tensor((width, height))
    grid = (coordinates.flip(-1) / scale * 2 - 1).reshape(1, 21, 21, 2)
    return F.grid_sample(values[None], grid, mode='bilinear', padding_mode='border',
                         align_corners=False)[0].permute(1, 2, 0).reshape(-1, len(values))


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work=None, *, methods=METHODS,
                  cache=None):
    if not methods or not set(methods).issubset(METHODS) or set(banks) != set(queries):
        raise ValueError('Declared methods and matching frozen text banks required.')
    original_size = tuple(image.shape[-2:])
    resized = resize_for_vip(image)
    height, width = resized.shape[-2:]
    windows = crop_windows(height, width)
    device = geometry.device
    layouts = {p: alias_layout(q.parents, canonical_indices(q.class_names, q.aliases, q.parents),
                               len(q.class_names)) for p, q in queries.items()}
    for p, bank in banks.items():
        if (bank.alias_names != queries[p].aliases or not torch.equal(bank.parent_indices, queries[p].parents)
                or not bool((layouts[p].counts == 20).all())):
            raise ValueError('Unchanged matched20 alias groups required.')
    texts = {p: F.normalize(bank.features.float(), dim=-1) for p, bank in banks.items()}
    active = tuple(m for m in methods if m in METHODS[2:])
    crops = {p: [] for p in banks}
    broad = {p: torch.zeros(bank.class_count, height, width, device=device) for p, bank in banks.items()}
    count = torch.zeros(height, width, device=device)
    for window in windows:
        features = vip.crop_patch_features(padded_crop(resized, window))
        if features.shape[1] != 441:
            raise RuntimeError('Wide observer changed the21x21 token grid.')
        for p, query in queries.items():
            crop = alias_crop(features, query, window)
            crops[p].append(crop)
            broad[p][:, window.top:window.top + window.height, window.left:window.left + window.width] += (
                class_map(crop, layouts[p])[:, :window.height, :window.width])
        count[window.top:window.top + window.height, window.left:window.left + window.width] += 1
    if not bool((count > 0).all()):
        raise RuntimeError('Uncovered resized wide pixels.')
    broad = {p: values / count[None] for p, values in broad.items()}
    observations = {p: cache_wide(crops[p], layouts[p]) for p in banks} if active else {}
    permutations = {p: alias_permutation(layouts[p]) for p in banks} if active else {}
    blend = torch.from_numpy(hann_blend_window(336)).to(device)
    normalizer = torch.zeros(height, width, device=device)
    totals = {(p, m): torch.zeros(bank.class_count, height, width, device=device)
              for p, bank in banks.items() for m in methods}
    risks = {p: {} for p in banks}
    for window in windows:
        prepared = geometry.prepare_image(padded_crop(resized, window)[None].to(device))
        if (prepared.grid_height, prepared.grid_width) != (21, 21):
            raise RuntimeError('Geometry must run directly on336 pixels, without crop enlargement.')
        coordinates, valid = patch_coordinates(window, device)
        operator, residual = reconstruction_operator(prepared.geometry_patch_conditional[0], valid)
        for p, bank in banks.items():
            raw = alias_class_scores((prepared.geometry_projected.float() @ texts[p].T)[0],
                                     bank.parent_indices, bank.class_count)
            local = raw / .07
            b = sample_map(broad[p], coordinates)
            scores = dict(Geometry=raw, NoAdmission_Exact=local.double() + operator @ (b.double() - local.double()))
            if active:
                native = alias_crop(prepared.native_projected, queries[p], window)
                grounded = alias_crop(prepared.geometry_projected, queries[p], window)
                result, stats = shared_scores(local, operator, b, observations[p], count, coordinates,
                    (height, width), profile_aliases(native.alias_logits, native.salience, layouts[p]),
                    profile_aliases(grounded.alias_logits, grounded.salience, layouts[p]), valid, layouts[p],
                    methods=active, permutation=permutations[p])
                scores.update(result)
                for name, value in stats.items():
                    risks[p][name] = risks[p].get(name, 0.) + value / len(windows)
            weights = blend[:window.height, :window.width]
            for method in methods:
                dense = F.interpolate(scores[method].T.reshape(1, bank.class_count, 21, 21),
                                      (336, 336), mode='bilinear', align_corners=False)[0]
                if method == 'Geometry':
                    dense = dense / .07
                totals[p, method][:, window.top:window.top + window.height,
                                  window.left:window.left + window.width] += (
                    dense[:, :window.height, :window.width].softmax(0).float() * weights[None])
        normalizer[window.top:window.top + window.height, window.left:window.left + window.width] += weights
    if not bool((normalizer > 0).all()):
        raise RuntimeError('Uncovered resized Geometry pixels.')
    predictions = {p: {} for p in banks}
    for (p, method), accumulated in totals.items():
        probabilities = accumulated / normalizer[None]
        restored = F.interpolate(probabilities[None], original_size, mode='bilinear', align_corners=False)[0]
        predictions[p][method] = restored.argmax(0).to(torch.uint8).cpu().numpy()
    diagnostics = {p: dict(risks[p], original_size=list(original_size), resized_size=[height, width],
        tiles=len(windows), geometry_encodings=len(windows), wide_encodings=len(windows),
        fine_forwards=0, token_grid=[21, 21], crop_windows=[asdict(w) for w in windows],
        operator_residual=residual, native_resolution_encodings=0) for p in banks}
    return predictions, diagnostics
