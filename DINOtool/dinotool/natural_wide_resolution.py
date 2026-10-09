"""Task-level natural-view scale prior with the retained four-crop budget."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from .inference import tile_starts

IMPLEMENTATION='geometry-natural-wide-short336-cap672-v1-20261007'


def target_size(height,width):
    if min(height,width)<=0:
        raise ValueError('Positive RGB dimensions required.')
    ratio=min(336/min(height,width),672/max(height,width))
    return max(1,int(height*ratio+.5)),max(1,int(width*ratio+.5))


@torch.inference_mode()
def replace_wide(source,image,vip):
    """Shares local Geometry observations; encodes only the alternative wide RGB."""
    nh,nw=target_size(*image.shape[-2:])
    array=(image.permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)
    rgb=torch.from_numpy(np.ascontiguousarray(cv2.resize(array,(nw,nh),interpolation=cv2.INTER_LINEAR))).permute(2,0,1).float()/255
    wide=[]
    for top in tile_starts(nh,336,224):
        for left in tile_starts(nw,336,224):
            ah,aw=min(336,nh-top),min(336,nw-left)
            crop=F.pad(rgb[:,top:top+ah,left:left+aw],(0,336-aw,0,336-ah))
            wide.append(dict(top=top,left=left,ah=ah,aw=aw,features=vip.crop_patch_features(crop)))
    if not 1<=len(wide)<=4:
        raise RuntimeError('Natural wide budget exceeded.')
    return dict(source,wide=wide,wide_size=(nh,nw),wide_policy='natural_short336_cap672')
