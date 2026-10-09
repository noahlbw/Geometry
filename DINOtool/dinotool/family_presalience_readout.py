"""Predesignated full11 trial after the frozen mask-free action diagnostic."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import family_mass_alias as old
from .family_presalience import amplification,shuffled_families
from .region_residual_alias import raw_wide


IMPLEMENTATION='geometry-fixed20-family-presalience-full-v1-20261009'
CURRENT_BASE='Current20_Base';COMPLETE_BASE='Slot_Uniform'
LEGACY='Slot_Family';PRE_ONLY='Family_Uniform';PRIMARY='Family_Family'
SHUFFLE='Family_Shuffled';VIP='VIP_Complete20'
METHODS=(CURRENT_BASE,COMPLETE_BASE,LEGACY,PRE_ONLY,PRIMARY,SHUFFLE)


@dataclass(frozen=True)
class Plan:
    mass:object
    families:object
    shuffled:object
    shuffled_mass:object
    @property
    def changed_classes(self):return self.mass.changed_classes


def prepare_plan(bank,families):
    shuffled=shuffled_families(families)
    return Plan(old.prepare_plan(bank,families),families,shuffled,old.prepare_plan(bank,shuffled))


@torch.inference_mode()
def wide_fields(source,crops,plan,wanted,policy):
    h,w=source['wide_size'];classes=len(plan.mass.offsets);device=crops[0]['raw'].device
    totals={name:torch.zeros(classes,h,w,device=device) for name in wanted};count=torch.zeros(h,w,device=device)
    for crop in crops:
        rows={name:[] for name in wanted}
        for c,ids in enumerate(plan.mass.mean.members):
            raw=crop['scaled_raw'][ids];s=crop['salience'][ids]
            slot,family=amplification(s,plan.families[c],policy.tem)
            for name in wanted:
                if name==SHUFFLE:
                    _,amp=amplification(s,plan.shuffled[c],policy.tem);mass=plan.shuffled_mass
                else:amp=slot if name in (COMPLETE_BASE,LEGACY) else family;mass=plan.mass
                evidence=raw*amp[:,None,None]
                route='uniform' if name in (COMPLETE_BASE,PRE_ONLY) else 'sum'
                rows[name].append(old.reduce_routes(evidence,mass,c,(route,),policy.tau)[route])
        for name in wanted:
            dense=F.interpolate(torch.stack(rows[name])[None],(336,336),mode='bilinear',align_corners=False)[0]
            top,left,ah,aw=(crop[k] for k in ('top','left','ah','aw'))
            totals[name][:,top:top+ah,left:left+aw]+=dense[:,:ah,:aw]
        count[crop['top']:crop['top']+crop['ah'],crop['left']:crop['left']+crop['aw']]+=1
    if not bool((count>0).all()):raise RuntimeError('Wide coverage gap.')
    return {name:value/count[None] for name,value in totals.items()}


@torch.inference_mode()
def predict(image,geometry,vip,banks,queries,plans,dataset,methods=METHODS,*,verify_panel=False):
    from eval_development_readout import observations,probabilities
    from .fixed20_task_readout import POLICIES
    if not methods or not set(methods).issubset(METHODS):raise ValueError('Frozen methods required.')
    policy=POLICIES[dataset];source=observations(image,geometry,vip,(policy.strength,),wide_policy=policy.wide_policy)
    outputs={};wanted=tuple(name for name in methods if name!=CURRENT_BASE)
    if wanted:
        crops=raw_wide(source,queries['Complete20']);plan=plans['Complete20']
        fields=wide_fields(source,crops,plan,wanted,policy);cache={}
        # Paired/full inference must exactly reproduce the already frozen panel.
        if verify_panel and len(wanted)==5:
            from .family_presalience import fields as diagnostic
            reference,_=diagnostic(source,crops,plan.mass,plan.shuffled_mass,plan.families,plan.shuffled,policy)
            if any(not torch.equal(fields[name],reference[name]) for name in wanted):
                raise RuntimeError('Full readout differs from frozen diagnostic.')
        for name in wanted:
            probability=probabilities(source,policy.profile(),banks['Complete20'],fields[name],cache)
            outputs[name]=probability.argmax(0).cpu().numpy()
    if CURRENT_BASE in methods:
        field=old.wide_fields(source,queries['Current20'],plans['Current20'].mass,('base',),policy.tau,policy.tem)['base']
        probability=probabilities(source,policy.profile(),banks['Current20'],field,{})
        outputs[CURRENT_BASE]=probability.argmax(0).cpu().numpy()
    return outputs,dict(geometry_encodings=len(source['local']),wide_encodings=len(source['wide']),
        fine_forwards=0,additional_visual_forwards=0,additional_semantic_heads=0,alias_slots_per_class=20,
        changed_family_classes=plans['Complete20'].changed_classes)
