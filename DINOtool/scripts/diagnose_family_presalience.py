"""Six frozen whole images, no mask reads, before deciding a full trial."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import eval_family_lme_alias as inherited
from benchmark_natural_sense_fast import require_available_gpu
from dinotool import family_mass_alias as old
from dinotool import family_presalience as trial
from dinotool.fixed20_task_readout import POLICIES
from dinotool.region_residual_alias import raw_wide
from eval_development_readout import observations,probabilities
from eval_geometry_vip_reliability import sample_broad
from eval_rival_fine_full import frozen_state,check_frozen,save


def difference(a,b):
    d=(a-b).double()
    return dict(max_abs=float(d.abs().max()),rms=float(d.square().mean().sqrt()),mean_abs=float(d.abs().mean()))


@torch.inference_mode()
def main(args):
    root=Path(args.suite_root);output=Path(args.output_dir)
    if output.exists():raise RuntimeError('Existing diagnostic output; preserve it.')
    device=torch.device(args.device);require_available_gpu(device);lease=torch.empty(1,device=device)
    output.mkdir(parents=True);save(output/'worker_status.json',dict(status='running',phase='loading'))
    setup=inherited.curated_setup if args.dataset=='ade150' else inherited.original_setup
    protocol,entry,samples,loader,masks,g,old_b,v,old_q,words,checkpoints,reference,_=setup(args)
    if reference is not None:raise RuntimeError('Unexpected extra model.')
    for name,expected in protocol['method_sources'].items():
        if hashlib.sha256((Path(__file__).parents[1]/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Frozen source differs: '+name)
    inherited.prepare_plan=old.prepare_plan
    banks,queries,plans,identity,text_cost=inherited.input_banks(root,args.dataset,g,v,old_b,old_q,checkpoints,protocol)
    meta=json.loads((root/'vocabularies'/(args.dataset+'.json')).read_text())
    families=meta['Complete20']['family_ids'];shuffled=trial.shuffled_families(families)
    shuffled_plan=old.prepare_plan(banks['Complete20'],shuffled)
    policy=POLICIES[args.dataset];frozen=frozen_state(g,v);rows=[]
    for index in sorted({0,len(samples)//2,len(samples)-1}):
        sample=samples[index];image=loader(sample);require_available_gpu(device)
        source=observations(image,g,v,(policy.strength,),wide_policy=policy.wide_policy)
        crops=raw_wide(source,queries['Complete20'])
        fields,diag=trial.fields(source,crops,plans['Complete20'],shuffled_plan,families,shuffled,policy)
        expected=old.wide_fields(source,queries['Complete20'],plans['Complete20'],('uniform','sum'),policy.tau,policy.tem)
        if not torch.equal(fields['Slot_Uniform'],expected['uniform']) or not torch.equal(fields['Slot_Family'],expected['sum']):
            raise RuntimeError('Exact00/01 predecessor wide fields differ.')
        cache={};predictions={}
        for name in trial.METHODS:
            p=probabilities(source,policy.profile(),banks['Complete20'],fields[name],cache)
            predictions[name]=p.argmax(0).cpu().numpy()
        comparisons={}
        for a,b in (('Family_Uniform','Slot_Uniform'),('Family_Family','Slot_Family'),('Family_Family','Slot_Uniform'),('Family_Family','Family_Shuffled')):
            writes=[]
            for tile in source['local']:
                h,w=source['size'];delta=sample_broad(fields[a]-fields[b],tile['top'],tile['left'],h,w).reshape(1024,-1)
                writes.append((policy.coupling*tile['operator'].double()@delta.double()).flatten())
            correction=torch.cat(writes)
            comparisons[a+'-'+b]=dict(wide=difference(fields[a],fields[b]),
                writeback=difference(correction,torch.zeros_like(correction)),
                changed_original_pixels=int(np.count_nonzero(predictions[a]!=predictions[b])))
        row=dict(sample_key=sample.key,output_shape=list(image.shape[-2:]),comparisons=comparisons,
            amplification_max_abs=max(x['amplification_max_abs'] for x in diag),
            amplification_mean_rms=float(np.mean([x['amplification_rms'] for x in diag])),
            mean_normalized_slot_entropy=float(np.mean([x['normalized_slot_entropy'] for x in diag])),
            exact00_01_predecessor_fields_replayed=True,geometry_encodings=len(source['local']),
            wide_encodings=len(source['wide']),fine_forwards=0,additional_visual_forwards=0,
            per_crop_class_amplification=diag)
        rows.append(row);save(output/'diagnostic.json',dict(status='running',dataset=args.dataset,
            images=rows,target_masks_loaded=False,text_identity=identity,checkpoints=checkpoints))
    # This is a mask-free signal/cost decision, never a claim of improved mIoU.
    nonrounding=any(row['amplification_max_abs']>1e-6 and
        row['comparisons']['Family_Family-Slot_Family']['changed_original_pixels']>0 for row in rows)
    save(output/'diagnostic.json',dict(status='complete',implementation=trial.IMPLEMENTATION,dataset=args.dataset,
        images=rows,nonrounding_candidate_action_observed=nonrounding,target_masks_loaded=False,
        frozen_family_membership=families,shuffled_membership=shuffled,text_identity=identity,
        checkpoints=checkpoints,text_cost=text_cost,**check_frozen(frozen,g,v)))
    save(output/'worker_status.json',dict(status='complete',phase='mask_free_diagnostic',processed=3,total=3))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','output-dir','dinov3-repo','checkpoint-dir','upstream-root','vocabulary-config'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--mode',default='benchmark')
    p.add_argument('--sample-seed',type=int,default=20260923);p.add_argument('--vdd-ontology',default='official')
    main(p.parse_args())
