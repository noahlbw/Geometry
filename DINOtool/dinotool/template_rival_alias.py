"""Canonical-relative robustness of alias margins under the current rival.

Reuse the existing template responses before their mean is taken. Template
stability is an observable, not a guarantee of semantic correctness.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .fixed20_task_readout import POLICIES


IMPLEMENTATION = 'geometry-fixed20-template-rival-margin-v1-20261008'
BASE = 'Fixed20_TaskCoupled'
PRIMARY = 'TemplateRival_Robust'
POOLED = 'TemplateRival_Pooled'
SHUFFLED = 'TemplateRival_AliasShuffle'
UNPAIRED = 'TemplateRival_TemplateUnpair'
METHODS = (BASE, PRIMARY, POOLED, SHUFFLED, UNPAIRED)
SEED = 20261008
CLASS_CHUNK = 8


@dataclass(frozen=True)
class TemplatePlan:
    members: torch.Tensor
    canonical: torch.Tensor
    protected: torch.Tensor
    permutation: torch.Tensor


@torch.inference_mode()
def prepare_plan(bank):
    members = torch.stack([(bank.parent_indices == c).nonzero().flatten()
                          for c in range(bank.class_count)])
    if bank.class_count < 2 or members.shape[1] != 20:
        raise ValueError('At least two classes with twenty aliases each required.')
    protected = bank.canonical_mask[members]
    if not bool((protected.sum(-1) == 1).all()):
        raise ValueError('One existing canonical slot per class required.')
    canonical = members[protected].reshape(-1)
    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.arange(20, device=members.device).expand_as(members).clone()
    for c in range(bank.class_count):
        ids = (~protected[c]).nonzero().flatten()
        permutation[c, ids] = ids[torch.randperm(19, generator=generator).to(ids.device)]
    return TemplatePlan(members, canonical, protected, permutation)


def rival_indices(base_logits):
    """A single frozen current rival per patch/class, from unmodified Base."""
    order = torch.argsort(base_logits, dim=-1, descending=True, stable=True)[:, :2]
    own = torch.arange(base_logits.shape[-1], device=base_logits.device)[None]
    return torch.where(order[:, :1] != own, order[:, :1], order[:, 1:2])


def margin_correction(responses, own, rival, protected):
    """[patch,chunk,20,template], [patch,chunk,template] canonical fields.

    Use population SD in logit units, with a fixed coefficient of one. The
    correction is bounded by SD(alias-own), by the reverse triangle inequality.
    """
    anchor_sd = (own-rival).std(-1, correction=0)
    alias_sd = (responses-rival[:, :, None]).std(-1, correction=0)
    correction = anchor_sd[:, :, None]-alias_sd
    return correction.masked_fill(protected[None], 0.)


def reduce_alias(raw, salience, plan, tau, tem, correction=None):
    """Preserve Base's original class reduction and salience multiplication."""
    result = []
    for c, ids in enumerate(plan.members):
        weights = (salience[ids]/tem).softmax(0)
        values = raw[:, ids] if correction is None else raw[:, ids]+correction[:, c]
        # Preserve the old [alias,21,21] CUDA reduction layout as well as its
        # equation. Synthetic CPU inputs can have a different patch count.
        values = values.T.contiguous()
        if raw.shape[0] == 441:
            values = values.reshape(20,21,21)
            scaled = values*(weights/weights.mean())[:,None,None]
        else:
            scaled = values*(weights/weights.mean())[:,None]
        result.append(((tau*scaled).logsumexp(0)/tau).reshape(-1))
    return torch.stack(result, -1)


def patch_fields(template_similarities, salience, plan, methods, tau=1., tem=1.):
    # Match the historical BF16 mean-then-scale order for Base raw logits.
    raw = (template_similarities.mean(-1)*40.).float()
    base = reduce_alias(raw, salience, plan, tau, tem)
    rivals = rival_indices(base)
    anchors = template_similarities[:, plan.canonical].float()*40.
    corrections = {name:torch.zeros(raw.shape[0], len(plan.members), 20,
                                    device=raw.device) for name in methods if name != BASE}
    templates = template_similarities.shape[-1]
    generator = torch.Generator().manual_seed(SEED)
    template_order = torch.randperm(templates, generator=generator).to(raw.device)
    if templates > 1 and bool(torch.equal(template_order, torch.arange(templates, device=raw.device))):
        template_order = template_order.roll(1)
    stats = dict(mean_abs_alias_correction=0., mean_abs_unpair_difference=0.,
                 positive_alias_fraction=0., canonical_correction_max=0.,
                 bound_violation_max=0., maximum_margin_elements=0.,
                 maximum_existing_template_elements=template_similarities.numel(),
                 template_count=templates)
    total_slots = raw.shape[0]*len(plan.members)*19
    for start in range(0, len(plan.members), CLASS_CHUNK):
        end = min(start+CLASS_CHUNK, len(plan.members))
        response = template_similarities[:, plan.members[start:end]].float()*40.
        own = anchors[:, start:end]
        rival = anchors.gather(1, rivals[:, start:end, None].expand(-1,-1,templates))
        protected = plan.protected[start:end]
        correction = margin_correction(response, own, rival, protected)
        stats['maximum_margin_elements'] = max(stats['maximum_margin_elements'], response.numel())
        stats['mean_abs_alias_correction'] += float(correction.abs().sum())/total_slots
        stats['positive_alias_fraction'] += float((correction>0).sum())/total_slots
        stats['canonical_correction_max'] = max(stats['canonical_correction_max'],
                                               float(correction[:, protected].abs().max()))
        bound = (response-own[:, :, None]).std(-1, correction=0)
        stats['bound_violation_max'] = max(stats['bound_violation_max'],
                                          float((correction.abs()-bound).clamp_min(0).max()))
        for name in corrections:
            use = correction
            if name == POOLED:
                use = correction.sum(-1, keepdim=True).expand_as(correction)/19
                use = use.masked_fill(protected[None], 0.)
            elif name == SHUFFLED:
                use = correction.gather(-1, plan.permutation[None,start:end].expand_as(correction))
            elif name == UNPAIRED:
                use = margin_correction(response, own, rival[:, :, template_order], protected)
                stats['mean_abs_unpair_difference'] += float((use-correction).abs().sum())/total_slots
            corrections[name][:,start:end] = use
    fields = {BASE:base} if BASE in methods else {}
    fields.update({name:reduce_alias(raw,salience,plan,tau,tem,value)
                   for name,value in corrections.items()})
    return fields,stats


@torch.inference_mode()
def wide_fields(source, query, plan, methods, tau, tem):
    from eval_development_readout import wide_scores

    if methods == (BASE,):
        return {BASE:wide_scores(source,query,tau,tem)}, {}
    height,width = source['wide_size']
    accumulators = {name:torch.zeros(len(query.class_names),height,width,
                                    device=query.features.device) for name in methods}
    counts = torch.zeros(height,width,device=query.features.device)
    totals = {}
    for crop in source['wide']:
        features = crop['features']
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            similarities = torch.einsum('bnd,mtd->bnmt',features,query.features.float())[0]
            patch_mean = F.normalize(features.mean(1),dim=-1)
            text_mean = F.normalize(query.features.float().mean(1),dim=-1)
            salience = (patch_mean@text_mean.T)[0].float()
        values,stats = patch_fields(similarities,salience,plan,methods,tau,tem)
        top,left,ah,aw = (crop[k] for k in ('top','left','ah','aw'))
        for name,value in values.items():
            dense = F.interpolate(value.T.reshape(1,len(query.class_names),21,21),
                                  (336,336),mode='bilinear',align_corners=False)[0]
            accumulators[name][:,top:top+ah,left:left+aw] += dense[:,:ah,:aw]
        counts[top:top+ah,left:left+aw] += 1
        for key,value in stats.items():
            totals[key] = max(totals.get(key,0.),value) if key.startswith('maximum_') or key.endswith('_max') else totals.get(key,0.)+value
    if not bool((counts>0).all()):
        raise RuntimeError('Wide coverage gap.')
    for key in totals:
        if not key.startswith('maximum_') and not key.endswith('_max'):
            totals[key] /= len(source['wide'])
    return {name:value/counts[None] for name,value in accumulators.items()},totals


@torch.inference_mode()
def predict_image(image,geometry,banks,vip,queries,plans,dataset,*,methods=METHODS):
    from eval_development_readout import observations
    from eval_geometry_vip_reliability import sample_broad
    from .device_probability_accumulator import DeviceProbabilityAccumulator
    from .geometry_readout_trace import alias_class_scores
    from .inference import hann_blend_window

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared template-margin arms required.')
    policy = POLICIES[dataset]
    source = observations(image,geometry,vip,(policy.strength,),wide_policy=policy.wide_policy)
    height,width = source['size']
    blend = torch.from_numpy(hann_blend_window(512)).to(geometry.device)
    predictions,diagnostics = {},{}
    for partition,bank in banks.items():
        broad,stats = wide_fields(source,queries[partition],plans[partition],methods,policy.tau,policy.tem)
        text = F.normalize(bank.features.float(),dim=-1)
        fields = []
        for tile in source['local']:
            cosine = tile['features'][policy.strength].float()@text.T
            local = alias_class_scores(cosine,bank.parent_indices,bank.class_count,policy.temperature)/policy.temperature
            values = {}
            for name in methods:
                wide = sample_broad(broad[name],tile['top'],tile['left'],height,width).reshape_as(local)
                values[name] = local.double()+policy.coupling*(tile['operator'].double()@(wide.double()-local.double()))
            if not all(bool(torch.isfinite(value).all()) for value in values.values()):
                raise RuntimeError('Nonfinite template-margin prediction.')
            fields.append((tile,values))
        predictions[partition] = {}
        for name in methods:
            with DeviceProbabilityAccumulator(bank.class_count,height,width,geometry.device) as accumulator:
                for tile,values in fields:
                    dense = F.interpolate(values[name].T.reshape(1,bank.class_count,32,32),
                                          (512,512),mode='bilinear',align_corners=False)[0]
                    top,left = tile['top'],tile['left']
                    ah,aw = min(512,height-top),min(512,width-left)
                    accumulator.add(dense[:,:ah,:aw].softmax(0).float(),blend[:ah,:aw],left,top)
                predictions[partition][name] = accumulator.finalize_resized(source['output_size'])
        diagnostics[partition] = dict(stats,geometry_encodings=len(source['local']),
            wide_encodings=len(source['wide']),fine_forwards=0,additional_visual_forwards=0,
            additional_semantic_heads=0,fixed_alias_slots=20)
    return predictions,diagnostics
