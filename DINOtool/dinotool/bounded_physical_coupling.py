"""Bounded physical-scale adapter for the frozen Geometry/wide coupling."""
import cv2
import numpy as np
import torch

from .inference import tile_starts
from .shared_rival_soft_alias import METHODS, PRIMARY


IMPLEMENTATION = 'geometry-shared-rival-soft-bounded896-v1-20261005'
VIEW_PROTOCOL = dict(geometry_long_edge=896, local_crop=512, local_tiles=[512, 128],
    local_stride=384, wide_long_edge=448, wide_crop=336, wide_stride=112,
    geometry_token_grid=[32, 32], wide_token_grid=[21, 21],
    maximum_geometry_encodings=4, maximum_wide_encodings=4, fine_forwards=0,
    resize='uint8/OpenCV INTER_LINEAR/half-up dimensions',
    wide_source='original RGB resized independently to448; no double resampling',
    stitching='retained Hann-weighted class probabilities',
    restore='bilinear probabilities before argmax',
    coordinates='896 resized-image local coordinates; no native-resolution windows')


def resize_geometry(image):
    if image.ndim != 3 or image.shape[0] != 3 or min(image.shape[-2:]) < 1:
        raise ValueError('Nonempty CHW RGB image required.')
    height, width = image.shape[-2:]
    ratio = VIEW_PROTOCOL['geometry_long_edge'] / max(height, width)
    size = int(width * ratio + .5), int(height * ratio + .5)
    if min(size) < 1:
        raise ValueError('Resizing would produce an empty short edge.')
    array = (image.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
    resized = cv2.resize(array, size, interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).float().div_(255)


def geometry_windows(height, width):
    if min(height, width) < 1 or max(height, width) != VIEW_PROTOCOL['geometry_long_edge']:
        raise ValueError('Geometry windows require the bounded896 input.')
    windows = tuple((top, left) for top in tile_starts(height, 512, 128)
                    for left in tile_starts(width, 512, 128))
    if not 1 <= len(windows) <= VIEW_PROTOCOL['maximum_geometry_encodings']:
        raise RuntimeError('Geometry encoding budget exceeded.')
    return windows


class _BoundedBranch:
    def __init__(self, branch, side, limit, method):
        self.branch, self.side, self.limit, self.method = branch, side, limit, method
        self.device = branch.device
        self.calls = 0

    def _call(self, image):
        expected = (1, 3, self.side, self.side) if self.method == 'prepare_image' else (3, self.side, self.side)
        self.calls += 1
        if tuple(image.shape) != expected or self.calls > self.limit:
            raise RuntimeError('Actual visual input/call budget exceeded: ' + self.method)
        return getattr(self.branch, self.method)(image)

    def prepare_image(self, image):
        return self._call(image)

    def crop_patch_features(self, image):
        return self._call(image)


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, work=None, *, local_predictor,
                  methods=METHODS, cache=None):
    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared frozen methods required.')
    resized = resize_geometry(image)
    windows = geometry_windows(*resized.shape[-2:])
    local = _BoundedBranch(geometry, 512, 4, 'prepare_image')
    wide = _BoundedBranch(vip, 336, 4, 'crop_patch_features')
    predictions, diagnostics = local_predictor(resized, local, banks, wide, queries, work,
        methods=methods, cache=cache, output_size=tuple(image.shape[-2:]), wide_image=image)
    if local.calls != len(windows) or not 1 <= wide.calls <= 4:
        raise RuntimeError('Actual branch calls differ from the bounded layout.')
    for protocol, values in diagnostics.items():
        if values['tiles'] != local.calls or values.get('fine_forwards', 0):
            raise RuntimeError('Geometry/fine visual budget differs.')
        if any(v.shape != tuple(image.shape[-2:]) for v in predictions[protocol].values()):
            raise RuntimeError('Original prediction dimensions not restored.')
        values.update(geometry_encodings=local.calls, wide_encodings=wide.calls,
            native_resolution_encodings=0, fine_forwards=0,
            original_size=list(image.shape[-2:]), resized_size=list(resized.shape[-2:]))
    return predictions, diagnostics
