"""Geometry patch-only read strength controls with retained broad calibration."""
import torch

from .bounded_local_contrast import PROTOCOL as BASE_PROTOCOL, predict_image as predict_readout
from .matched_readout_controls import run_head


IMPLEMENTATION = 'geometry-bounded896-patch-only-strength2-coupled-v1-20261005'
PRIMARY = 'Geometry_PatchOnly2Coupled'
HEADS = {'Geometry_PatchOnlyCoupled': 1., PRIMARY: 2., 'SCLIP_Coupled': 'SCLIP_Two'}
METHODS = ('Geometry', 'NoAdmission_Exact', *HEADS)
PROTOCOL = {k: v for k, v in BASE_PROTOCOL.items() if k not in ('local_readout', 'controls')}
PROTOCOL.update(local_readout='Patch queries read fixed original G @ V_patch at total strength1 or2; prefix queries stay native',
    controls='original Geometry, patch-only strength1, patch-only strength2, published SCLIP_Two',
    novelty='read strength and prefix route are configuration controls, not a new innovation claim')


def readout(head, prepared, strength):
    if strength not in (1., 2., 'SCLIP_Two'):
        raise ValueError('Only the two declared strengths or matched SCLIP control are allowed.')
    relation = prepared.geometry_patch_conditional
    method = 'SCLIP_Two' if strength == 'SCLIP_Two' else 'Geometry_BlockPrefix'
    if method == 'Geometry_BlockPrefix':
        relation = relation if strength == 1. else relation * 2
    return run_head(head, prepared.backbone_tokens, prepared.backbone_tokens[:, prepared.prefix_tokens:],
                    relation, prepared.prefix_tokens, method, prepared.block_index)


@torch.inference_mode()
def predict_image(*args, methods=METHODS, **kwargs):
    return predict_readout(*args, methods=methods, head_names=HEADS, readout=readout, primary=PRIMARY, **kwargs)
