"""Check deployed residual maximum against the full evaluator on real images."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import eval_development_readout as base
from eval_residual_ontology import residual_wide,residual_probability
from eval_residual_rival_protection import protected_probability
from benchmark_taxonomy_inference import timed
from dinotool.taxonomy_inference import TaxonomyInference
from eval_rival_fine_full import frozen_state,check_frozen,save
from run_evidence_adaptive_readout import BASE,TOOL,OLD,idle


@torch.inference_mode()
def main(args):
    if not idle(args.physical_gpu):
        raise RuntimeError('Benchmark GPU occupied.')
    root,output=Path(args.suite_root),Path(args.output)
    if output.exists():
        raise RuntimeError('Existing output.')
    entry=json.loads((root/'protocol.json').read_text())['datasets']['context60']
    args.dataset='context60'
    args.data_root=entry['data_root']
    args.original_cache=str(OLD/'text_cache/context60.pt')
    _,samples,load_image,_=base.load_samples(args,entry)
    geometry,vip,banks,queries,_=base.load_models(args,entry,('semantic_segmentation',))
    frozen=frozen_state(geometry,vip)
    residual=torch.load(root/'residual_cache.pt',map_location=args.device,weights_only=True)['wide']
    model=TaxonomyInference(geometry,vip,banks,queries,family='natural',
        background=entry['background_index'],soft=False,residual_features=residual,residual_mode=args.residual_mode)
    original=TaxonomyInference(geometry,vip,banks,queries,family='natural',
        background=entry['background_index'],soft=False)
    text=F.normalize(residual.float().mean(1),dim=-1)
    rows=[]
    for index in np.linspace(0,len(samples)-1,3,dtype=int):
        sample=samples[index]
        image=load_image(sample)
        source=base.observations(image,geometry,vip,model.strengths)
        p=model.profile
        broad=base.wide_scores(source,queries[p.bank],p.tau,p.tem)
        old_broad=broad.clone()
        broad[model.background]=residual_wide(source,residual)['max'][0]
        reference=(residual_probability(source,p,banks[p.bank],broad,model.background,text,'max')
            if args.residual_mode=='max' else protected_probability(source,p,banks[p.bank],old_broad,broad,text,model.residual_reader))
        prediction,diagnostic,probability=model.predict(image,return_probability=True)
        torch.testing.assert_close(reference,probability,rtol=1e-5,atol=1e-6)
        changed=reference.argmax(0)!=probability.argmax(0)
        gap=reference.topk(2,dim=0).values
        max_gap=float((gap[0]-gap[1])[changed].max()) if bool(changed.any()) else 0.
        if max_gap>2e-6:
            raise RuntimeError('Non-tie label mismatch.')
        row=dict(key=sample.key,changed_pixels=int(changed.sum()),
            maximum_changed_gap=max_gap,maximum_probability_error=float((reference-probability).abs().max()),
            diagnostic=diagnostic)
        del source,broad,reference,probability,gap,changed
        original.predict(image)
        row['uniform']=timed(lambda:original.predict(image),3)
        row[args.residual_mode]=timed(lambda:model.predict(image),3)
        rows.append(row)
        print(json.dumps(row),flush=True)
    save(output,dict(status='complete',samples=rows,target_masks_loaded=False,
        timing='Warmed synchronized full inference; excludes loading/image I/O/text encoding. Three spread PC60 images, not full-dataset latency or mIoU equality.',
        physical_gpu=args.physical_gpu,**check_frozen(frozen,geometry,vip)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--physical-gpu',type=int,required=True)
    p.add_argument('--residual-mode',choices=('max','protected'),default='max')
    p.add_argument('--device',default='cuda')
    p.add_argument('--mode',default='full')
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20260923)
    p.add_argument('--vdd-ontology',default='official')
    p.add_argument('--dinov3-repo',default=str(TOOL/'dinov3_hub'))
    p.add_argument('--checkpoint-dir',default=str(BASE/'ckpt/DINO'))
    p.add_argument('--upstream-root',default=str(BASE/'third_party/VIP_official_5bd25ee'))
    main(p.parse_args())
