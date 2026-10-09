"""Audit frozen residual winners at patch centres; not causal alias ablation."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import eval_development_readout as base
from eval_residual_ontology import residual_wide,residual_probability
from dinotool.taxonomy_readout import natural_profile,canonical_features
from dinotool.vip_official_adapter import VIPSettings
from eval_rival_fine_full import frozen_state,check_frozen,save
from run_evidence_adaptive_readout import TOOL,BASE,OLD,idle


def event(gt,old,new):
    if gt==0:
        return 'background_rescued' if old!=0 and new==0 else 'background_still_missed' if new!=0 else 'background_retained'
    return 'correct_foreground_destroyed' if old==gt and new==0 else 'other_foreground_to_background' if new==0 else 'foreground_retained'


@torch.inference_mode()
def main(args):
    if not idle(args.physical_gpu):
        raise RuntimeError('Diagnostic GPU occupied.')
    root,output=Path(args.suite_root),Path(args.output)
    if output.exists():
        raise RuntimeError('Preserve existing diagnosis.')
    manifest=json.loads((root/'protocol.json').read_text())
    entry=manifest['datasets']['context60']
    args.dataset='context60'
    args.data_root=entry['data_root']
    _,all_samples,load_image,load_mask=base.load_samples(args,entry)
    lookup={s.key:s for s in all_samples}
    selected=[lookup[k] for k in entry['development_keys']]
    if len(selected)!=64:
        raise RuntimeError('Use exactly the prior frozen64 development images.')
    geometry,vip,banks,queries,_=base.load_models(args,entry,('semantic_segmentation',))
    frozen=frozen_state(geometry,vip)
    bank=banks['semantic_segmentation']
    p,_=natural_profile(bank)
    residual=torch.load(root/'residual_cache.pt',map_location=args.device,weights_only=True)
    if residual['identity']!=manifest['residual_identity']:
        raise RuntimeError('Residual queries changed.')
    null_text=F.normalize(residual['wide'].float().mean(1),dim=-1)
    names=residual['identity']['names']
    anchors=canonical_features(bank)
    semantic=(null_text@anchors.T).cpu().numpy()
    bank_text=F.normalize(bank.features.float(),dim=-1)
    counters=defaultdict(Counter)
    moments=defaultdict(lambda:np.zeros(5,dtype=np.float64))
    matrices={m:np.zeros((60,60),dtype=np.int64) for m in ('automatic','residual_max')}
    keys=[]
    for sample in selected:
        image=load_image(sample)
        source=base.observations(image,geometry,vip,(p.strength,))
        broad=base.wide_scores(source,queries[p.bank],p.tau,p.tem)
        old=base.probabilities(source,p,bank,broad,{})
        new_broad=broad.clone()
        new_broad[0]=residual_wide(source,residual['wide'])['max'][0]
        new=residual_probability(source,p,bank,new_broad,0,null_text,'max')
        # Predictions are frozen before target access; this is labelled audit only.
        old_prediction=old.argmax(0).cpu().numpy()
        new_prediction=new.argmax(0).cpu().numpy()
        target=load_mask(sample,source['output_size'])
        matrices['automatic']+=base.confusion(old_prediction,target,60)
        matrices['residual_max']+=base.confusion(new_prediction,target,60)
        oh,ow=target.shape
        h,w=source['size']
        seen=set()
        for tile in source['local']:
            features=tile['features'][p.strength].float()
            scores=features@null_text.T
            maximum,winner=scores.max(-1)
            foreground=base.alias_class_scores(features@bank_text.T,bank.parent_indices,60)
            values=torch.stack((maximum,foreground[:,1:].amax(-1)),dim=-1).cpu().numpy()
            winners=winner.cpu().tolist()
            for i,alias in enumerate(winners):
                y=tile['top']+(i//32)*16+8
                x=tile['left']+(i%32)*16+8
                if y>=h or x>=w or (y,x) in seen:
                    continue
                seen.add((y,x))
                yy=min(oh-1,int(y*oh/h));xx=min(ow-1,int(x*ow/w))
                gt,previous,current=(int(a[yy,xx]) for a in (target,old_prediction,new_prediction))
                group=event(gt,previous,current)
                c=gt if gt else previous
                label=bank.class_names[c]
                key=('local',group,label)
                counters[key][names[alias]]+=1
                moments[key]+=np.asarray([1,values[i,0],values[i,1],float(semantic[alias,c]),values[i,0]-values[i,1]])
        wh,ww=source['wide_size']
        for crop in source['wide']:
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                raw=torch.einsum('bnd,mtd->bnmt',crop['features'],residual['wide'].float()).mean(-1)[0]*VIPSettings().logit_scale
            winners=raw.argmax(-1).cpu().tolist()
            for i,alias in enumerate(winners):
                y=crop['top']+(i//21)*16+8
                x=crop['left']+(i%21)*16+8
                if y>=wh or x>=ww:
                    continue
                yy=min(oh-1,int(y*oh/wh));xx=min(ow-1,int(x*ow/ww))
                gt,previous,current=(int(a[yy,xx]) for a in (target,old_prediction,new_prediction))
                group=event(gt,previous,current)
                c=gt if gt else previous
                counters[('wide',group,bank.class_names[c])][names[alias]]+=1
        keys.append(sample.key)
        print(json.dumps(dict(processed=len(keys),total=len(selected))),flush=True)
        del source,old,new,broad,new_broad
    rows=[]
    for key,count in counters.items():
        stat=moments.get(key)
        rows.append(dict(branch=key[0],event=key[1],foreground_class=key[2],patch_centres=sum(count.values()),
            aliases=count.most_common(20),local_means=dict(null_cosine=float(stat[1]/stat[0]),
                foreground_cosine=float(stat[2]/stat[0]),null_foreground_text_cosine=float(stat[3]/stat[0]),
                null_foreground_margin=float(stat[4]/stat[0])) if stat is not None and stat[0] else None))
    np.savez_compressed(output.with_suffix('.npz'),sample_keys=np.asarray(keys),**matrices)
    save(output,dict(status='complete',sample_keys=keys,samples=len(keys),diagnostics=rows,
        role='Labelled audit after frozen predictions; no selector fitting or threshold selection.',
        winner_scope='Local32/wide21 patch-centre provenance, not causal final-output alias attribution. Local overlap centres deduplicated; wide overlaps retained.',
        residual_identity=residual['identity'],**check_frozen(frozen,geometry,vip)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--physical-gpu',type=int,required=True)
    p.add_argument('--mode',default='full')
    p.add_argument('--device',default='cuda')
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20260923)
    p.add_argument('--vdd-ontology',default='official')
    p.add_argument('--original-cache',default=str(OLD/'text_cache/context60.pt'))
    p.add_argument('--dinov3-repo',default=str(TOOL/'dinov3_hub'))
    p.add_argument('--checkpoint-dir',default=str(BASE/'ckpt/DINO'))
    p.add_argument('--upstream-root',default=str(BASE/'third_party/VIP_official_5bd25ee'))
    main(p.parse_args())
