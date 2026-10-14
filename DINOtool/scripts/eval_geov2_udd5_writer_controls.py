"""Same-observation controls separating visual structure from attenuation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_geov2_udd5_fixed05 import Experiment,metric,write
from eval_alias_finalization import load_samples
from eval_development_readout import confusion
from eval_rival_fine_full import tile_coordinates
from dinotool.efficient_deployment import reconstruction


METHODS=('Full','EqualLogitFusion','IsotropicHalfH','DiagonalH','SpatialH')


@torch.inference_mode()
def main(args):
    root=Path(args.experiment)
    protocol=json.loads((root/'protocol.json').read_text())
    frozen=json.loads((root/'writer_controls_protocol.json').read_text())
    if frozen['methods']!=list(METHODS) or frozen['g0']!=.5:
        raise ValueError('Writer control protocol changed.')
    out=root/'writer_controls'
    if out.exists():
        raise RuntimeError('Preserve existing writer controls.')
    out.mkdir()
    exp=Experiment(protocol,['Full'])
    m=exp.model
    args.dataset,args.mode,args.data_root='udd5','full',protocol['data_root']
    args.sample_seed,args.vdd_ontology=20260923,'official'
    args.shard_index,args.num_shards=0,1
    entry=dict(m.config,banks={'original':next(iter(m.vocabulary['banks'].values()))})
    validation,samples,images,masks,names=load_samples(args,entry)
    if len(samples)!=40:
        raise ValueError('Require the full 40-image split.')
    matrices=np.zeros((5,5,5),np.int64)
    keys,statistics=[],[]
    diagnostics=dict(patches=0,valid_h_trace=0.,difference_energy=0.,correction_energy=0.)
    for number,sample in enumerate(samples,1):
        image=images(sample)
        source=exp.source(image,'full')
        cache={}
        predictions=[exp.read(source,'Full',cache),exp.read(source,'EqualLogitFusion',cache)]
        alternatives={name:[] for name in METHODS[2:]}
        height,width=source['size']
        spatial=exp.model.compact_geometry.spatial[(32,32)]
        spatial_relation=(-spatial).softmax(-1).to(torch.bfloat16)
        from eval_geometry_vip_reliability import sample_broad
        for i,tile in enumerate(source['local']):
            coordinates=tile_coordinates(tile['top'],tile['left'],m.geometry.device)
            valid=(coordinates[:,0]<height)&(coordinates[:,1]<width)
            h=tile['operator']
            half_i=torch.diag(valid.to(h.dtype)*.5)
            diagonal=torch.diag(h.diagonal())
            spatial_h,_=reconstruction(spatial_relation,valid)
            for name,operator in zip(METHODS[2:],(half_i,diagonal,spatial_h)):
                alternatives[name].append(dict(tile,operator=operator))
            local=cache[('local',m.profile.bank,'original',m.profile.background_rule,i)]/m.profile.temperature
            broad=cache[('wide',m.profile.bank,m.profile.tau,m.profile.tem)]
            wide=sample_broad(broad,tile['top'],tile['left'],height,width).reshape_as(local)
            delta=wide.double()-local.double()
            diagnostics['patches']+=int(valid.sum())
            diagnostics['valid_h_trace']+=float(h.trace())
            diagnostics['difference_energy']+=float(delta[valid].square().sum())
            diagnostics['correction_energy']+=float((h@delta).square().sum())
        for name in METHODS[2:]:
            predictions.append(exp.read(dict(source,local=alternatives[name]),'Full',cache))
        target=masks(sample,'udd5',tuple(image.shape[-2:]))
        cms=np.stack([confusion(p,target,5) for p in predictions])
        matrices+=cms
        statistics.append(cms)
        keys.append(sample.key)
        if number==1 or number%5==0:
            progress=dict(processed=number,total=40)
            print(json.dumps(progress),flush=True)
            write(out/'progress.json',progress)
    np.savez_compressed(out/'confusions.npz',sample_keys=np.asarray(keys),confusions=np.stack(statistics))
    write(out/'results.json',dict(status='complete',g0=.5,methods=METHODS,sample_keys=keys,
        class_names=names['udd5'],processed_images=40,profile=m.profile.record(),diagnostics=diagnostics,
        metrics={name:dict(mean_iou_percent=metric(cm),confusion_matrix=cm.tolist()) for name,cm in zip(METHODS,matrices)},
        masks_used_only_for_scoring=True,observations_shared=True,protocol=frozen))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',required=True)
    main(p.parse_args())
