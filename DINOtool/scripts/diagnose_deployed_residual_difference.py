"""Mask-free diagnosis of the first failed packed residual comparison."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import eval_development_readout as base
from benchmark_residual_inference import protected_probability
from dinotool.taxonomy_inference import TaxonomyInference
from dinotool.development_readout import coupled
from eval_residual_ontology import residual_wide
from eval_geometry_vip_reliability import sample_broad
from eval_rival_fine_full import save
from run_evidence_adaptive_readout import BASE,TOOL,OLD,idle


@torch.inference_mode()
def main(args):
    if not idle(args.physical_gpu):
        raise RuntimeError('Diagnostic GPU occupied.')
    root,output=Path(args.suite_root),Path(args.output)
    if output.exists():
        raise RuntimeError('Existing diagnostic.')
    entry=json.loads((root/'protocol.json').read_text())['datasets']['context60']
    args.dataset='context60'
    args.data_root=entry['data_root']
    args.original_cache=str(OLD/'text_cache/context60.pt')
    _,samples,load_image,_=base.load_samples(args,entry)
    geometry,vip,banks,queries,_=base.load_models(args,entry,('semantic_segmentation',))
    residual=torch.load(root/'residual_cache.pt',map_location=args.device,weights_only=True)['wide']
    model=TaxonomyInference(geometry,vip,banks,queries,family='natural',background=0,
        soft=False,residual_features=residual,residual_mode='protected')
    p=model.profile
    bank=banks[p.bank]
    text=F.normalize(bank.features.float(),dim=-1)
    rows=[]
    for index in range(1890,min(1900,len(samples))):
        sample=samples[index]
        image=load_image(sample)
        source=base.observations(image,geometry,vip,model.strengths)
        broad=base.wide_scores(source,queries[p.bank],p.tau,p.tem)
        changed_broad=broad.clone()
        changed_broad[0]=residual_wide(source,residual)['max'][0]
        reference=protected_probability(source,p,bank,broad,changed_broad,model.residual_text,model.residual_reader)
        _,_,deployed=model.predict_observations(source,return_probability=True)
        good=torch.isclose(reference,deployed,rtol=1e-5,atol=1e-6)
        tiles=[]
        h,w=source['size']
        for tile in source['local']:
            features=tile['features'][p.strength].float()
            alias=features@text.T
            scalar=base.alias_class_scores(alias,bank.parent_indices,bank.class_count)/p.temperature
            packed=model.scorers[p.bank].uniform(alias)/p.temperature
            wide=sample_broad(broad,tile['top'],tile['left'],h,w).reshape_as(scalar)
            a=coupled(scalar,wide,tile['operator'],p.coupling)[:,1:]
            b=coupled(packed,wide,tile['operator'],p.coupling)[:,1:]
            rival_a=a.argmax(-1)+1
            rival_b=b.argmax(-1)+1
            unequal=rival_a!=rival_b
            scores=features@model.residual_text.T
            gate_a=model.residual_reader.score(scores,rival_a)
            gate_b=model.residual_reader.score(scores,rival_b)
            top=a.topk(2,dim=-1).values
            tiles.append(dict(top=tile['top'],left=tile['left'],
                maximum_local_difference=float((scalar-packed).abs().max()),
                rival_changed_patches=int(unequal.sum()),
                maximum_scalar_rival_gap=float((top[:,0]-top[:,1])[unequal].max()) if bool(unequal.any()) else 0.,
                maximum_residual_raw_difference=float((gate_a-gate_b).abs().max())))
        prediction_difference=reference.argmax(0)!=deployed.argmax(0)
        top=reference.topk(2,dim=0).values
        row=dict(key=sample.key,shard_sample_index=index,within_original_tolerance=bool(good.all()),
            maximum_probability_error=float((reference-deployed).abs().max()),
            probability_elements_outside_tolerance=int((~good).sum()),
            changed_prediction_pixels=int(prediction_difference.sum()),
            maximum_changed_prediction_gap=float((top[0]-top[1])[prediction_difference].max()) if bool(prediction_difference.any()) else 0.,tiles=tiles)
        rows.append(row)
        print(json.dumps(row),flush=True)
        if not row['within_original_tolerance']:
            break
    save(output,dict(status='complete',target_masks_loaded=False,rows=rows,
        source='Failed full PC60 shard1, first ten images after logged1890; models/words/rules unchanged.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--physical-gpu',type=int,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--mode',default='full')
    p.add_argument('--num-shards',type=int,default=2)
    p.add_argument('--shard-index',type=int,default=1)
    p.add_argument('--sample-seed',type=int,default=20260923)
    p.add_argument('--vdd-ontology',default='official')
    p.add_argument('--dinov3-repo',default=str(TOOL/'dinov3_hub'))
    p.add_argument('--checkpoint-dir',default=str(BASE/'ckpt/DINO'))
    p.add_argument('--upstream-root',default=str(BASE/'third_party/VIP_official_5bd25ee'))
    main(p.parse_args())
