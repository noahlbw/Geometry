"""Factor the existing expression-family action before/after alias amplification.

This is a narrow mechanism diagnostic, not another semantic reliability gate.
Class mass G, Geometry, local evidence and correction path stay fixed.
"""
import math
import torch
import torch.nn.functional as F


IMPLEMENTATION='geometry-fixed20-family-presalience-factor-diagnostic-v1-20261009'
SEED=20261011
METHODS=('Slot_Uniform','Slot_Family','Family_Uniform','Family_Family','Family_Shuffled')


def shuffled_families(families):
    rng=torch.Generator().manual_seed(SEED); output=[]
    for labels in families:
        p=torch.randperm(20,generator=rng).tolist()
        output.append([labels[i] for i in p])
    return output


def amplification(salience,labels,tem):
    # Match original mean-one floating arithmetic, not merely its real formula.
    slot=(salience/tem).softmax(0); slot=slot/slot.mean()
    groups=sorted(set(labels));means=torch.stack([salience[[i for i,g in enumerate(labels) if g==j]].mean() for j in groups])
    probability=(means/tem).softmax(0); family=probability/probability.mean()
    mapping=torch.tensor(labels,device=salience.device)
    broadcast=family[mapping]
    # The inherited float mean can round even uniform softmax slightly below1.
    # Retain its exact neutral response rather than introducing that difference.
    broadcast=torch.where((salience==salience[0]).all(),slot,broadcast)
    return slot,broadcast


@torch.inference_mode()
def fields(source,crops,plan,shuffled_plan,families,shuffled,policy):
    h,w=source['wide_size'];classes=len(plan.offsets);device=crops[0]['raw'].device
    totals={name:torch.zeros(classes,h,w,device=device) for name in METHODS};count=torch.zeros(h,w,device=device)
    diagnostic=[]
    for crop in crops:
        rows={name:[] for name in METHODS}
        for c,ids in enumerate(plan.mean.members):
            raw=crop['scaled_raw'][ids];s=crop['salience'][ids]
            slot,family=amplification(s,families[c],policy.tem)
            _,null=amplification(s,shuffled[c],policy.tem)
            original=raw*slot[:,None,None];changed=raw*family[:,None,None]
            e0=(policy.tau*original).logsumexp(0)/policy.tau
            e1=(policy.tau*changed).logsumexp(0)/policy.tau
            from .family_mass_alias import reduce_routes
            rows['Slot_Uniform'].append(e0+plan.offsets[c]/policy.tau)
            rows['Slot_Family'].append(reduce_routes(original,plan,c,('sum',),policy.tau)['sum'])
            rows['Family_Uniform'].append(e1+plan.offsets[c]/policy.tau)
            rows['Family_Family'].append(reduce_routes(changed,plan,c,('sum',),policy.tau)['sum'])
            null_evidence=raw*null[:,None,None]
            rows['Family_Shuffled'].append(reduce_routes(null_evidence,shuffled_plan,c,('sum',),policy.tau)['sum'])
            probability=(s/policy.tem).softmax(0)
            diagnostic.append(dict(class_index=c,amplification_max_abs=float((family-slot).abs().max()),
                amplification_rms=float((family-slot).square().mean().sqrt()),
                normalized_slot_entropy=float(-(probability*probability.log()).sum()/math.log(20)),
                scaled_raw_dtype=str(raw.dtype)))
        for name in METHODS:
            dense=F.interpolate(torch.stack(rows[name])[None],(336,336),mode='bilinear',align_corners=False)[0]
            top,left,ah,aw=(crop[k] for k in ('top','left','ah','aw'))
            totals[name][:,top:top+ah,left:left+aw]+=dense[:,:ah,:aw]
        count[crop['top']:crop['top']+crop['ah'],crop['left']:crop['left']+crop['aw']]+=1
    if not bool((count>0).all()):raise RuntimeError('Wide coverage gap.')
    return {name:value/count[None] for name,value in totals.items()},diagnostic
