"""Unlabelled, fixed-eight-image audit of native vs geometric value-read energy."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import json
import gc
import time
import numpy as np
import torch
import eval_development_readout as base
from dinotool.bounded_physical_coupling import resize_geometry,geometry_windows
from dinotool.gear_ov import _crop_at
from eval_rival_fine_full import save,check_frozen,frozen_state


@torch.inference_mode()
def main(args):
    output=Path(args.output)
    if output.exists():
        raise RuntimeError('Existing diagnostic; preserve it.')
    protocol=json.loads((Path(args.source_root)/'protocol.json').read_text())
    rows={}
    started=time.time()
    for d in ('vdd','potsdam','voc21','context60','ade150'):
        entry=protocol['datasets'][d]
        options=SimpleNamespace(dataset=d,data_root=entry['data_root'],suite_root=args.source_root,
            original_cache=str(Path(args.original_root)/'text_cache'/(d+'.pt')),
            upstream_root=args.upstream_root,checkpoint_dir=args.checkpoint_dir,dinov3_repo=args.dinov3_repo,
            mode='smoke',device='cuda',sample_seed=20260923,vdd_ontology='official',shard_index=0,num_shards=1)
        validation,_,load_image,_=base.load_samples(options,entry)
        geometry,vip,banks,queries,_=base.load_models(options,entry,('original',))
        frozen=frozen_state(geometry,vip)
        values={k:[] for k in ('native_full_over_geo','native_conditional_over_geo','raw_value_over_geo',
                              'native_patch_mass','geometry_row_entropy','native_geo_cosine')}
        per_image=[]
        for sample in validation[:8]:
            rgb=resize_geometry(load_image(sample))
            h,w=rgb.shape[-2:]
            windows=geometry_windows(h,w)
            for top,left in windows:
                prepared=geometry.prepare_image(_crop_at(rgb,top,left,512).to(geometry.device))
                prefix=prepared.prefix_tokens
                v=prepared.value_tokens.float()
                geo=prepared.geometry_patch_conditional.float()[:,None]@v[...,prefix:,:]
                native=prepared.native_attention.float()[...,prefix:,:]@v
                conditional=prepared.native_patch_conditional.float()@v[...,prefix:,:]
                gn=geo.square().mean(-1).sqrt().clamp_min(1e-8)
                ratios=dict(native_full_over_geo=native.square().mean(-1).sqrt()/gn,
                            native_conditional_over_geo=conditional.square().mean(-1).sqrt()/gn,
                            raw_value_over_geo=v[...,prefix:,:].square().mean(-1).sqrt()/gn,
                            native_patch_mass=prepared.native_attention.float()[...,prefix:,prefix:].sum(-1),
                            geometry_row_entropy=-(prepared.geometry_patch_conditional.float().clamp_min(1e-12).log()
                                *prepared.geometry_patch_conditional.float()).sum(-1),
                            native_geo_cosine=torch.nn.functional.cosine_similarity(native,geo,dim=-1))
                # Exclude reflected/padded patch centres from summary statistics.
                y,x=torch.meshgrid(torch.arange(32,device=geometry.device)*16+8+top,
                                   torch.arange(32,device=geometry.device)*16+8+left,indexing='ij')
                valid=((y<h)&(x<w)).flatten()
                for k,value in ratios.items():
                    selected=value[...,valid].flatten()
                    if not bool(torch.isfinite(selected).all()):
                        raise RuntimeError('Nonfinite diagnostic.')
                    values[k].extend(selected.cpu().tolist())
                del prepared,geo,native,conditional,ratios
            per_image.append(dict(key=sample.key,local_encodings=len(windows),mask_loaded=False))
        rows[d]=dict(samples=per_image,statistics={k:dict(mean=float(np.mean(v)),
            q10=float(np.quantile(v,.1)),median=float(np.median(v)),q90=float(np.quantile(v,.9))) for k,v in values.items()},
            **check_frozen(frozen,geometry,vip))
        save(output,dict(status='running',datasets=rows,target_masks_loaded=False,
                         caveat='Final native adapter block values are a diagnostic reference, not an exact per-block adaptive Geometry execution.'))
        print(json.dumps(dict(dataset=d,statistics=rows[d]['statistics'])),flush=True)
        del geometry,vip,banks,queries,frozen,values
        gc.collect()
        torch.cuda.empty_cache()
    save(output,dict(status='complete',datasets=rows,target_masks_loaded=False,wall_seconds=time.time()-started,
                     caveat='Final native adapter block values are a diagnostic reference, not an exact per-block adaptive Geometry execution.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','original-root','output','upstream-root','checkpoint-dir','dinov3-repo'):
        parser.add_argument('--'+name,required=True)
    main(parser.parse_args())
