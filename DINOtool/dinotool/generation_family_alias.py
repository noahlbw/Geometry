"""Diagnostic of generated expression families, with no new visual evidence.

Family identities come from the frozen word generator, not target responses.
This is an input/aggregation diagnostic, not yet a conditional alias module.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .fixed20_task_readout import POLICIES


IMPLEMENTATION = 'geometry-generated-family-input-diagnostic-v1-20261008'
CC, HH, HC, CH = 'CC', 'HH', 'HC', 'CH'
FLAT, PRIMARY, SHUFFLE, MEAN = 'CH-flat', 'CQ', 'CQ-shuffle', 'C-mean'
METHODS = (CC, HH, HC, CH, FLAT, PRIMARY, SHUFFLE, MEAN)
ROUTES = {CC:('C','C'), HH:('H','H'), HC:('H','C'), CH:('C','H'),
          FLAT:('C','flat'), PRIMARY:('C','family'),
          SHUFFLE:('C','shuffle'), MEAN:('C','mean')}
SEED = 20261008


@dataclass(frozen=True)
class FamilyPlan:
    members: tuple
    family: tuple
    shuffled: tuple
    counts: tuple


def prepare_plan(bank, family_ids):
    members = tuple((bank.parent_indices == c).nonzero().flatten()
                    for c in range(bank.class_count))
    if len(family_ids) != bank.class_count or any(len(ids) != 20 for ids in members):
        raise ValueError('Frozen twenty-slot generation provenance required.')
    family, shuffled = [], []
    rng = torch.Generator().manual_seed(SEED)
    for c, (ids, groups) in enumerate(zip(members, family_ids)):
        if len(groups) != 20 or sorted(set(groups)) != list(range(max(groups)+1)):
            raise ValueError('Contiguous nonempty family IDs required.')
        protected = bank.canonical_mask[ids].detach().cpu()
        if protected.sum().item() != 1:
            raise ValueError('One existing canonical slot required.')
        slots = (~protected).nonzero().flatten()
        perm = torch.arange(20)
        perm[slots] = slots[torch.randperm(19, generator=rng)]
        g = torch.tensor(groups, dtype=torch.long, device=ids.device)
        family.append(g)
        shuffled.append(g[perm.to(ids.device)])
        if not torch.equal(torch.bincount(g), torch.bincount(shuffled[-1])):
            raise RuntimeError('Shuffle changed family sizes.')
    return FamilyPlan(members, tuple(family), tuple(shuffled),tuple(max(g)+1 for g in family_ids))


def quotient(values, labels, tau=1., *, groups=None):
    """[alias,patch...] profiled logits; expression means, then family LME.

    A common twenty-slot offset makes equal responses independent of family
    count. This removes within-family response inflation, not just slot counts.
    """
    groups = int(labels.max())+1 if groups is None else groups
    means = torch.stack([values[labels == g].mean(0) for g in range(groups)])
    return (math.log(20)+torch.logsumexp(tau*means, 0)-math.log(len(means)))/tau


def crop_fields(features, query, plan, wanted, tau=1., tem=1.):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        raw = torch.einsum('bnd,mtd->bnmt', features, query.features.float()).mean(-1)[0]
        patch_mean = F.normalize(features.mean(1), dim=-1)
        text_mean = F.normalize(query.features.float().mean(1), dim=-1)
        salience = ((patch_mean @ text_mean.T)[0]/tem).float()
        alias = (raw*40.).T.reshape(-1,21,21)
        out = {name:[] for name in wanted}
        for c in range(len(query.class_names)):
            ids = (query.parents == c).nonzero().flatten() if plan is None else plan.members[c]
            weight = salience[ids].softmax(0)
            e = alias[ids]*(weight/weight.mean())[:,None,None]
            original = torch.logsumexp(tau*e, 0)/tau
            for name in wanted:
                if name in ('C','H'):
                    score = original
                elif name == 'mean':
                    score = e.mean(0)+math.log(20)/tau
                elif name in ('family','shuffle') and plan is not None:
                    labels = plan.family[c] if name == 'family' else plan.shuffled[c]
                    score = quotient(e, labels, tau,groups=plan.counts[c])
                else:
                    raise ValueError('Unknown source/aggregation: '+name)
                out[name].append(score)
    return {name:torch.stack(fields).float() for name,fields in out.items()}


def wide_fields(source, query, plan, wanted, tau, tem):
    from eval_development_readout import wide_scores

    if plan is None:
        # The count-only control acts on the SAME completed historical W.
        # Adding its constant before interpolation/overlap changes rounding.
        original = wide_scores(source,query,tau,tem)
        fields = {'H':original} if 'H' in wanted else {}
        if 'flat' in wanted:
            counts = torch.bincount(query.parents,minlength=len(query.class_names)).float()
            offset = (math.log(20)-counts.log())/tau
            fields['flat'] = original+offset[:,None,None]
        if not set(wanted).issubset(('H','flat')):
            raise ValueError('Historical source only supports inherited/count-only controls.')
        return fields
    if wanted == ('C',) or wanted == ('H',):
        return {wanted[0]:wide_scores(source,query,tau,tem)}
    h,w = source['wide_size']
    out = {name:torch.zeros(len(query.class_names),h,w,device=query.features.device)
           for name in wanted}
    counts = torch.zeros(h,w,device=query.features.device)
    for crop in source['wide']:
        fields = crop_fields(crop['features'],query,plan,wanted,tau,tem)
        top,left,ah,aw = (crop[k] for k in ('top','left','ah','aw'))
        for name,patches in fields.items():
            dense = F.interpolate(patches[None],(336,336),mode='bilinear',align_corners=False)[0]
            out[name][:,top:top+ah,left:left+aw] += dense[:,:ah,:aw]
        counts[top:top+ah,left:left+aw] += 1
    if not bool((counts>0).all()):
        raise RuntimeError('Wide coverage gap.')
    return {name:values/counts[None] for name,values in out.items()}


@torch.inference_mode()
def predict_image(image, geometry, banks, vip, queries, plan, historical_reader, *, methods=METHODS):
    from eval_development_readout import observations
    from eval_geometry_vip_reliability import sample_broad
    from .device_probability_accumulator import DeviceProbabilityAccumulator
    from .geometry_readout_trace import alias_class_scores
    from .inference import hann_blend_window
    from .development_readout import coupled

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared diagnostic arms required.')
    policy = POLICIES['ade150']
    source = observations(image,geometry,vip,(policy.strength,),wide_policy=policy.wide_policy)
    h,w = source['size']
    required = tuple(dict.fromkeys(ROUTES[name][1] for name in methods))
    broad = {}
    for bank_name, keys in (('C',('C','family','shuffle','mean')),('H',('H','flat'))):
        wanted = tuple(k for k in keys if k in required)
        if wanted:
            broad.update(wide_fields(source,queries[bank_name],plan if bank_name=='C' else None,
                                     wanted,policy.tau,policy.tem))
    texts = {k:F.normalize(bank.features.float(),dim=-1) for k,bank in banks.items()
             if k in {ROUTES[name][0] for name in methods}}
    blend = torch.from_numpy(hann_blend_window(512)).to(geometry.device)
    fields = []
    diagnostics = dict(geometry_encodings=len(source['local']),wide_encodings=len(source['wide']),
        fine_forwards=0,additional_visual_forwards=0,additional_semantic_heads=0,
        fixed20_candidate_slots=20,historical_diagnostic_queries=len(queries['H'].aliases))
    if 'family' in broad:
        # Diagnostics are image means, separate from singleton deployment cost.
        for ref in ('C','shuffle','mean'):
            if ref in broad:
                diagnostics['family_minus_'+ref+'_mean'] = float((broad['family']-broad[ref]).mean())
                diagnostics['family_minus_'+ref+'_abs'] = float((broad['family']-broad[ref]).abs().mean())
    for tile in source['local']:
        local = {}
        for bank_name,text in texts.items():
            alias = tile['features'][policy.strength].float()@text.T
            bank = banks[bank_name]
            # Preserve the exact historical packed reduction for H.
            local[bank_name] = (historical_reader.uniform(alias) if bank_name=='H' else
                alias_class_scores(alias,bank.parent_indices,bank.class_count))/policy.temperature
        values = {}
        for name in methods:
            lk,wk = ROUTES[name]
            wide = sample_broad(broad[wk],tile['top'],tile['left'],h,w).reshape_as(local[lk])
            values[name] = coupled(local[lk],wide,tile['operator'],policy.coupling)
        if not all(bool(torch.isfinite(x).all()) for x in values.values()):
            raise RuntimeError('Nonfinite family diagnostic logits.')
        fields.append((tile,values))
    predictions = {}
    for name in methods:
        classes = banks[ROUTES[name][0]].class_count
        with DeviceProbabilityAccumulator(classes,h,w,geometry.device) as accumulator:
            for tile,values in fields:
                dense = F.interpolate(values[name].T.reshape(1,classes,32,32),(512,512),
                                      mode='bilinear',align_corners=False)[0]
                top,left = tile['top'],tile['left'];ah,aw = min(512,h-top),min(512,w-left)
                accumulator.add(dense[:,:ah,:aw].softmax(0).float(),blend[:ah,:aw],left,top)
            predictions[name] = accumulator.finalize_resized(source['output_size'])
    return predictions,diagnostics
