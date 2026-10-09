"""Fixed eight-image mechanism audit; no online labels or gate fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from unittest.mock import patch
import eval_development_readout as base
from dinotool.taxonomy_inference import TaxonomyInference
from dinotool.correction_reliability_audit import observations,summarize,SIGNALS
from eval_rival_fine_full import save,frozen_state,check_frozen
from dinotool.device_probability_accumulator import DeviceProbabilityAccumulator
from dinotool.geometry_support_audit import relation_support
import dinotool.taxonomy_inference as inference


def geometry_predict(model,source,yy,xx):
    """Observe the final coupled call of each tile; preserve original arithmetic."""
    original=inference.coupled;calls=[]
    def observe(local,wide,operator,gain):
        result=original(local,wide,operator,gain)
        # Protected residual route calls coupling twice per tile; final call is used.
        calls.append(relation_support(local,wide,operator))
        return result
    with patch.object(inference,'coupled',observe):
        result=model.predict_observations(source,return_probability=True)
    stride=2 if model.residual_reader is not None else 1
    if len(calls)!=stride*len(source['local']):raise RuntimeError('Unexpected coupled call count.')
    support=calls[stride-1::stride]
    h,w=source['size'];classes=support[0].shape[-1]
    with DeviceProbabilityAccumulator(classes,h,w,model.geometry.device) as accumulator:
        for tile,evidence in zip(source['local'],support):
            dense=F.interpolate(evidence.T.reshape(1,classes,32,32),(512,512),mode='bilinear',align_corners=False)[0]
            top,left=tile['top'],tile['left'];ah,aw=min(512,h-top),min(512,w-left)
            accumulator.add(dense[:,:ah,:aw].float(),model.blend[:ah,:aw],left,top)
        dense=F.interpolate((accumulator.probabilities/accumulator.normalizer[None])[None],source['output_size'],mode='bilinear',align_corners=False)[0]
        grid=dense[:,yy[:,None],xx[None,:]].cpu().numpy()
    return result,grid


@torch.inference_mode()
def main(args):
    root,output=Path(args.suite_root),Path(args.output_dir)
    if output.exists():raise RuntimeError('Existing diagnostic output.')
    protocol=json.loads((root/'protocol.json').read_text());entry=protocol['datasets'][args.dataset]
    args.mode='full'
    validation,all_samples,load_image,load_mask=base.load_samples(args,entry)
    lookup={s.key:s for s in all_samples};samples=[lookup[k] for k in entry['audit_keys']]
    needed=('original_imagenet','focused20') if entry['family']=='remote_sensing' else ('semantic_segmentation',)
    geometry,vip,banks,queries,checkpoints=base.load_models(args,entry,needed)
    frozen=frozen_state(geometry,vip);residual=None
    if entry.get('residual_identity') is not None:
        cache=torch.load(root/'residual_cache.pt',map_location=args.device,weights_only=True)
        if cache['identity']!=entry['residual_identity']:raise RuntimeError('Residual identity changed.')
        residual=cache['wide']
    kwargs=dict(family=entry['family'],background=entry['background_index'],residual_features=residual,local_background='union_max')
    model=TaxonomyInference.for_task(geometry,vip,banks,queries,**kwargs)
    names=next(iter(banks.values())).class_names
    records={k:[] for k in ('LocalEndpoint','WideEndpoint')};output.mkdir(parents=True)
    witnessed=False;audited_keys=[]
    for number,sample in enumerate(samples,1):
        image=load_image(sample)
        source=base.observations(image,geometry,vip,model.strengths,wide_policy=model.wide_policy)
        h,w=source['output_size'];ys=((np.arange(min(128,h))+.5)*h/min(128,h)).astype(int)
        xs=((np.arange(min(128,w))+.5)*w/min(128,w)).astype(int)
        yy=torch.tensor(ys,device=geometry.device);xx=torch.tensor(xs,device=geometry.device)
        grids={};geometry_grid=None
        for name,correction in (('Frozen','frozen'),('LocalEndpoint','local_endpoint'),('WideEndpoint','wide_endpoint')):
            model.correction=correction
            if name=='Frozen' and getattr(args,'geometry_support',False):
                (prediction,diagnostic,p),geometry_grid=geometry_predict(model,source,yy,xx)
            else:prediction,diagnostic,p=model.predict_observations(source,return_probability=True)
            if diagnostic['geometry_encodings']>4 or diagnostic['wide_encodings']>4 or diagnostic['fine_forwards']:
                raise RuntimeError('Visual budget exceeded.')
            if name=='Frozen' and number==1:
                direct=TaxonomyInference(geometry,vip,banks,queries,**kwargs,soft=entry['family']=='remote_sensing',
                    residual_mode='protected' if residual is not None else 'max',wide_policy=model.wide_policy)
                replay=direct.predict_observations(source,return_probability=True)[2]
                if not torch.equal(p,replay):raise RuntimeError('Factory/direct probability witness differs.')
                witnessed=True;del replay,direct
            grids[name]=p[:,yy[:,None],xx[None,:]].cpu().numpy()
            del prediction,p
        # Predictions are fixed before masks; labels only identify fixed/broken outcomes.
        target=load_mask(sample,source['output_size'])[ys[:,None],xs[None,:]]
        for name in records:records[name].append(observations(grids['LocalEndpoint'],grids['WideEndpoint'],grids['Frozen'],target,name,geometry_grid))
        audited_keys.append(sample.key)
        save(output/'results.json',dict(status='running',processed_images=number,total_images=len(samples),sample_keys=audited_keys,
            labels_used_for_audit_only=True,factory_direct_probability_witness=witnessed))
        print(json.dumps(dict(dataset=args.dataset,processed=number,total=len(samples))),flush=True)
        del source,grids,target,geometry_grid
    signals=SIGNALS+('geometry_corroboration',) if getattr(args,'geometry_support',False) else SIGNALS
    arrays={base_name+'__'+key:np.concatenate([r[key] for r in rows]) for base_name,rows in records.items()
            for key in ('true_class','beneficial',*signals)}
    np.savez_compressed(output/'audit_scores.npz',**arrays)
    save(output/'results.json',dict(status='complete',processed_images=len(samples),total_images=len(samples),sample_keys=audited_keys,
        labels_used_for_audit_only=True,labels_not_used_to_select_samples=True,factory_direct_probability_witness=witnessed,
        scores=summarize(records,names,entry['background_index'],signals),classes=names,
        geometry_support_rule='row-normalized squared off-diagonal H influence applied to mean endpoint patch softmax; zero isolated support; same interpolation/blending as predictions' if getattr(args,'geometry_support',False) else None,
        checkpoints=base.checkpoint_manifest(checkpoints),vocabulary=entry['banks'],residual_identity=entry.get('residual_identity'),
        grid='centres of uniform max128x128 grid in restored output; spatial score uses3x3 grid neighbours, not Geometry H',
        **check_frozen(frozen,geometry,vip)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','original-cache','dinov3-repo','checkpoint-dir','upstream-root'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--mode',default='audit');p.add_argument('--device',default='cuda')
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20260923);p.add_argument('--vdd-ontology',default='official')
    p.add_argument('--geometry-support',action='store_true')
    main(p.parse_args())
