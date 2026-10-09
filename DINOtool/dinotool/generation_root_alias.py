"""Read existing bare-root representatives of documented expression families.

Static language provenance, not a claim of image-conditioned correctness.
All twenty slots retain their original salience calculation. No provenance
means exact inherited-score fallback, never guessed linguistic grouping.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .fixed20_task_readout import POLICIES
from .generation_family_alias import quotient


IMPLEMENTATION = 'geometry-fixed20-existing-root-representative-v1-20261008'
BASE, QUOTIENT, PRIMARY, SHUFFLE, CANONICAL = ('Fixed20_TaskCoupled',
    'GeneratedFamily_Quotient','GeneratedRoot','GeneratedRoot_RepresentativeShuffle','GeneratedRoot_CanonicalOnly')
METHODS = (BASE,QUOTIENT,PRIMARY,SHUFFLE,CANONICAL)
SEED = 20261008


@dataclass(frozen=True)
class RootPlan:
    members: tuple
    canonical: tuple
    labels: object
    roots: object
    shuffled: object
    counts: tuple


def prepare_plan(bank,provenance=None):
    members=tuple((bank.parent_indices==c).nonzero().flatten() for c in range(bank.class_count))
    if any(len(ids)!=20 for ids in members):raise ValueError('Twenty existing slots per class required.')
    canonical=[]
    for ids in members:
        slot=bank.canonical_mask[ids].nonzero().flatten().cpu().tolist()
        if len(slot)!=1:raise ValueError('One existing canonical slot required.')
        canonical.append(slot[0])
    if provenance is None:
        return RootPlan(members,tuple(canonical),None,None,None,tuple([20]*bank.class_count))
    if (list(bank.class_names)!=provenance['class_names'] or list(bank.alias_names)!=provenance['aliases']):
        raise ValueError('Provenance strings/order do not match the supplied bank.')
    rng=torch.Generator().manual_seed(SEED)
    labels=[];roots=[];shuffled=[];counts=[]
    for c,ids in enumerate(members):
        group=provenance['family_ids'][c];words=[bank.alias_names[i] for i in ids.cpu().tolist()]
        bare=provenance['roots'][c]
        if len(group)!=20 or sorted(set(group))!=list(range(len(bare))):
            raise ValueError('Complete nonempty family provenance required.')
        root=[];control=[]
        for g,word in enumerate(bare):
            matches=[i for i,a in enumerate(words) if a==word and group[i]==g]
            if len(matches)!=1:raise ValueError('Each family requires one exact existing bare root.')
            root.append(matches[0]);slots=[i for i,v in enumerate(group) if v==g]
            # Canonical-family representative may change in this control.
            control.append(slots[int(torch.randint(len(slots),(1,),generator=rng))])
        if canonical[c] not in root:raise ValueError('Bare-root method must retain the existing canonical.')
        labels.append(torch.tensor(group,device=ids.device));counts.append(len(bare))
        roots.append(torch.tensor(root,device=ids.device));shuffled.append(torch.tensor(control,device=ids.device))
    return RootPlan(members,tuple(canonical),tuple(labels),tuple(roots),tuple(shuffled),tuple(counts))


def class_fields(e,plan,c,methods,tau):
    """Original profiled [20,patch...] evidence, no surviving-mass correction."""
    original=e.mul(tau).logsumexp(0)/tau
    fields={}
    for name in methods:
        if name==BASE or plan.labels is None and name!=CANONICAL:
            fields[name]=original
        elif name==QUOTIENT:
            fields[name]=quotient(e,plan.labels[c],tau,groups=plan.counts[c])
        elif name in (PRIMARY,SHUFFLE):
            ids=plan.roots[c] if name==PRIMARY else plan.shuffled[c]
            fields[name]=(torch.logsumexp(tau*e[ids],0)+math.log(20)-math.log(plan.counts[c]))/tau
        elif name==CANONICAL:
            fields[name]=e[plan.canonical[c]]+math.log(20)/tau
        else:raise ValueError('Unknown representative arm.')
    return fields


def crop_fields(features,query,plan,methods,tau,tem):
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        raw=torch.einsum('bnd,mtd->bnmt',features,query.features.float()).mean(-1)[0]
        patch_mean=F.normalize(features.mean(1),dim=-1)
        text_mean=F.normalize(query.features.float().mean(1),dim=-1)
        salience=((patch_mean@text_mean.T)[0]/tem).float()
        alias=(raw*40.).T.reshape(-1,21,21)
        values={name:[] for name in methods}
        for c,ids in enumerate(plan.members):
            weights=salience[ids].softmax(0)
            e=alias[ids]*(weights/weights.mean())[:,None,None]
            for name,field in class_fields(e,plan,c,methods,tau).items():values[name].append(field)
    return {name:torch.stack(fields).float() for name,fields in values.items()}


def wide_fields(source,query,plan,methods,tau,tem):
    from eval_development_readout import wide_scores

    if methods==(BASE,) or plan.labels is None and CANONICAL not in methods:
        original=wide_scores(source,query,tau,tem)
        return {name:original for name in methods}
    h,w=source['wide_size']
    totals={name:torch.zeros(len(query.class_names),h,w,device=query.features.device) for name in methods}
    counts=torch.zeros(h,w,device=query.features.device)
    for crop in source['wide']:
        fields=crop_fields(crop['features'],query,plan,methods,tau,tem)
        top,left,ah,aw=(crop[k] for k in ('top','left','ah','aw'))
        for name,field in fields.items():
            dense=F.interpolate(field[None],(336,336),mode='bilinear',align_corners=False)[0]
            totals[name][:,top:top+ah,left:left+aw]+=dense[:,:ah,:aw]
        counts[top:top+ah,left:left+aw]+=1
    if not bool((counts>0).all()):raise RuntimeError('Wide coverage gap.')
    return {name:values/counts[None] for name,values in totals.items()}


@torch.inference_mode()
def predict_image(image,geometry,banks,vip,queries,plans,dataset,*,methods=METHODS):
    from eval_development_readout import observations
    from eval_geometry_vip_reliability import sample_broad
    from .device_probability_accumulator import DeviceProbabilityAccumulator
    from .geometry_readout_trace import alias_class_scores
    from .inference import hann_blend_window
    from .development_readout import coupled

    if not methods or not set(methods).issubset(METHODS):raise ValueError('Declared representative arms required.')
    policy=POLICIES[dataset]
    source=observations(image,geometry,vip,(policy.strength,),wide_policy=policy.wide_policy)
    h,w=source['size'];blend=torch.from_numpy(hann_blend_window(512)).to(geometry.device)
    predictions={};diagnostics={}
    for partition,bank in banks.items():
        plan=plans[partition]
        broad=wide_fields(source,queries[partition],plan,methods,policy.tau,policy.tem)
        text=F.normalize(bank.features.float(),dim=-1);fields=[]
        for tile in source['local']:
            alias=tile['features'][policy.strength].float()@text.T
            local=alias_class_scores(alias,bank.parent_indices,bank.class_count)/policy.temperature
            values={}
            for name in methods:
                wide=sample_broad(broad[name],tile['top'],tile['left'],h,w).reshape_as(local)
                values[name]=coupled(local,wide,tile['operator'],policy.coupling)
            if not all(bool(torch.isfinite(x).all()) for x in values.values()):raise RuntimeError('Nonfinite root logits.')
            fields.append((tile,values))
        predictions[partition]={}
        for name in methods:
            with DeviceProbabilityAccumulator(bank.class_count,h,w,geometry.device) as accumulator:
                for tile,values in fields:
                    dense=F.interpolate(values[name].T.reshape(1,bank.class_count,32,32),(512,512),
                        mode='bilinear',align_corners=False)[0]
                    top,left=tile['top'],tile['left'];ah,aw=min(512,h-top),min(512,w-left)
                    accumulator.add(dense[:,:ah,:aw].softmax(0).float(),blend[:ah,:aw],left,top)
                predictions[partition][name]=accumulator.finalize_resized(source['output_size'])
        diag=dict(geometry_encodings=len(source['local']),wide_encodings=len(source['wide']),
            fine_forwards=0,additional_visual_forwards=0,additional_semantic_heads=0,input_alias_slots=20,
            provenance_available=int(plan.labels is not None),mean_root_representatives=sum(plan.counts)/bank.class_count)
        if BASE in broad and PRIMARY in broad:
            diag['root_minus_base_abs']=float((broad[PRIMARY]-broad[BASE]).abs().mean())
        diagnostics[partition]=diag
    return predictions,diagnostics
