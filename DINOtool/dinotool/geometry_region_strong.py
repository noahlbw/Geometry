"""Geometry-supported frozen semantic observation with matched patch footprints."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .gear_ov import _crop_at
from .region_semantic_readout import FrozenRegionObserver, RegionReadoutConfig


IMPLEMENTATION = "geometry-region-so400m-footprint-joint-v1-20261002"
PRIMARY = "Geometry_RegionStrong"
BASELINES = ("Geometry", "SCLIP_Two", "VIPProxy_Two")
METHODS = (*BASELINES, "CropGlobalFusion", "RegionSemantic", "RegionFusion", PRIMARY)
REPOSITORY = "google/siglip2-so400m-patch14-384"
REVISION = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
SOURCE = Path("/data/test/code/ovss_cafe_ped_v2_20260919/ckpt/SigLIP2_so400m384_region_20261002")


@dataclass(frozen=True)
class StrongRegionConfig(RegionReadoutConfig):
    encoder_size: int = 384
    encoder_patch: int = 14

    def signature(self):
        return {"region": asdict(self), "implementation": IMPLEMENTATION,
                "observer_repository": REPOSITORY, "observer_revision": REVISION,
                "primary": PRIMARY, "geometry_readout": "unchanged original two-block readout",
                "supports": "all connected fine64/coarse16 native DINO supports; Geometry-restricted mass",
                "footprints": "pixel-center projection then actual stride14 patch-area integration; 27x27 tokens",
                "observation": "real context and detail RGB; original trained probe/K/V/MLP with support log-prior",
                "temperature": "each encoder's own fixed scale; no target-label calibration",
                "energy": "KL(p||g)+sum_r n_r KL(q_r||z_r)+sum_ri W_ri KL(p_i||z_r)",
                "text": "unchanged20 aliases/six RS templates/normalized LME",
                "view": "native512/128 overlap/Hann probability blending"}


def project_patch_support(support, *, image_shape, top=0, left=0, extent=None,
                          encoder_size=384, patch_size=14):
    """Integrate support over actual non-overlapping convolution footprints.

    The last six pixels of a384 input do not enter a stride14 convolution. Do
    not stretch the27-token grid to represent those unseen border pixels.
    """
    if support.ndim != 3 or len(image_shape) != 2:
        raise ValueError("Expected [R,H,W] supports and image height/width.")
    if encoder_size < patch_size or patch_size <= 0:
        raise ValueError("Invalid trained encoder geometry.")
    height, width = image_shape
    if min(height, width) <= 0 or (extent is not None and extent <= 0):
        raise ValueError("Invalid image/crop extent.")
    y_extent, x_extent = (height, width) if extent is None else (extent, extent)
    yy = top+(torch.arange(encoder_size, device=support.device)+.5)*y_extent/encoder_size
    xx = left+(torch.arange(encoder_size, device=support.device)+.5)*x_extent/encoder_size
    y, x = torch.meshgrid(yy, xx, indexing="ij")
    coordinates = torch.stack((2*x/width-1, 2*y/height-1), -1)
    projected = F.grid_sample(support[:, None].float(),
        coordinates[None].expand(len(support), -1, -1, -1),
        mode="bilinear", padding_mode="border", align_corners=False)
    inside = ((y >= 0) & (y < height) & (x >= 0) & (x < width)).float()
    projected = projected*inside[None, None]
    return F.avg_pool2d(projected, patch_size, stride=patch_size).flatten(1)


def footprint_region_crops(rgb, operator, masks, config):
    if rgb.ndim != 4 or len(rgb) != 1 or operator.shape != (len(masks), masks[0].numel()):
        raise ValueError("Single RGB window and matched support operator required.")
    support = operator.reshape(-1, *masks.shape[-2:])
    strides = torch.tensor([rgb.shape[-2]/masks.shape[-2], rgb.shape[-1]/masks.shape[-1]],
                           device=rgb.device)
    crops, projected = [], []
    for index, mask in enumerate(masks):
        coordinates = mask.nonzero()
        if not len(coordinates):
            raise ValueError("Cannot crop an empty Geometry support.")
        lower, upper = coordinates.amin(0), coordinates.amax(0)+1
        extent = max(config.crop_minimum,
                     math.ceil(float(((upper-lower)*strides).max())+2*config.crop_margin))
        extent = math.ceil(extent/16)*16
        center = (lower+upper).float()*strides/2
        top, left = math.floor(float(center[0])-extent/2), math.floor(float(center[1])-extent/2)
        crop = _crop_at(rgb[0], top, left, extent)
        crops.append(F.interpolate(crop, (config.encoder_size,)*2, mode="bilinear",
                                   align_corners=False, antialias=True)[0])
        projected.append(project_patch_support(support[index:index+1],
            image_shape=rgb.shape[-2:], top=top, left=left, extent=extent,
            encoder_size=config.encoder_size, patch_size=config.encoder_patch)[0])
    return torch.stack(crops), torch.stack(projected)


class FrozenStrongRegionObserver(FrozenRegionObserver):
    encoder_size = 384

    def __init__(self, source, banks, device, config=StrongRegionConfig()):
        super().__init__(source, banks, device, config)
        vision = self.model.config.vision_config
        if (self.manifest["repository"] != REPOSITORY or self.manifest["revision"] != REVISION
                or vision.image_size != config.encoder_size or vision.patch_size != config.encoder_patch):
            raise ValueError("Observer must use the pinned trained SO400M384/patch14 checkpoint.")
        if not math.isfinite(self.semantic_temperature) or self.semantic_temperature <= 0:
            raise ValueError("Frozen pretrained source temperature must be finite and positive.")
        self.encoder_size = config.encoder_size

    def context_supports(self, operator, grid_valid):
        return project_patch_support(operator.reshape(-1, *grid_valid.shape), image_shape=(512, 512),
            encoder_size=self.config.encoder_size, patch_size=self.config.encoder_patch)

    def detail_crops(self, rgb, operator, masks):
        return footprint_region_crops(rgb, operator, masks, self.config)

    @torch.inference_mode()
    def visual(self, images):
        hidden, pooled = super().visual(images)
        expected = (self.config.encoder_size//self.config.encoder_patch)**2
        if hidden.shape[1] != expected or not bool(torch.isfinite(hidden).all()):
            raise ValueError("Actual observer tokens differ from the trained patch footprints.")
        return hidden, pooled

    @torch.inference_mode()
    def __call__(self, *args, **kwargs):
        scores, diagnostics = super().__call__(*args, **kwargs)
        for group in scores.values():
            group[PRIMARY] = group.pop("RegionJoint")
        return scores, diagnostics
