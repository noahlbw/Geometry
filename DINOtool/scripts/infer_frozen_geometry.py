"""Run a frozen YAML model on an RGB image without historical experiment files."""
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from dinotool.frozen_deployment import FrozenGeometry


def main(args):
    model=FrozenGeometry(args.config,checkpoint_dir=args.checkpoint_dir,dinov3_repo=args.dinov3_repo,
        upstream_root=args.upstream_root,text_cache=args.text_cache,compact=not args.reference,device=args.device)
    array=np.asarray(Image.open(args.input).convert('RGB')).copy()
    image=torch.from_numpy(array).permute(2,0,1).float()/255
    prediction=model.predict(image)
    output=Path(args.output)
    if output.exists():raise RuntimeError('Preserve existing prediction.')
    output.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(prediction).save(output)
    print(json.dumps(dict(output=str(output),shape=list(prediction.shape),cache_identity=model.cache_identity,
        compact=not args.reference,target_masks_loaded=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','checkpoint-dir','dinov3-repo','upstream-root','text-cache','input','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--reference',action='store_true')
    main(parser.parse_args())
