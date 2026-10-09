"""One Base-selected rival, bounded fine confirmation, fixed-slot attenuation.

All observation caches are [token,class,20] or [token,class]. Interpolate
alias and class fields first, then gather one rival. No all-rival tensor is
created, even transiently. Fine RGB comes from resized896, not native detail.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .fixed20_task_readout import POLICIES
from .stratified_soft_alias import WideCrop, crop_stencil
from .template_rival_alias import prepare_plan, rival_indices


IMPLEMENTATION = 'geometry-fixed20-single-rival-fine-slot-v1-20261008'
BASE = 'Fixed20_TaskCoupled'
NOALIAS = 'SingleRivalFine_NoAlias'
PRIMARY = 'SingleRivalFine_Soft'
POOLED = 'SingleRivalFine_Pooled'
SHUFFLED = 'SingleRivalFine_Shuffle'
METHODS = (BASE, NOALIAS, PRIMARY, POOLED, SHUFFLED)
QUERY_CHUNK = 32


@dataclass(frozen=True)
class SparseCrop:
    evidence: torch.Tensor
    classes: torch.Tensor
    indices: torch.Tensor
    coefficients: torch.Tensor


def cache_crop(crop, count, coordinates, image_size, plan):
    # Same salience multiplication as the original wide class reduction.
    weights = crop.salience[plan.members].softmax(-1)
    evidence = crop.alias_logits[:, plan.members]*(weights/weights.mean(-1, keepdim=True))
    classes = evidence.logsumexp(-1)
    indices, coefficients = crop_stencil(crop, count, coordinates, image_size)
    return SparseCrop(evidence, classes, indices, coefficients)


def sample_cache(crops, *, aliases=True):
    if not crops:
        raise ValueError('Nonempty sparse observation cache required.')
    first = crops[0]
    classes = first.classes.new_zeros(len(first.indices), first.classes.shape[-1])
    evidence = first.evidence.new_zeros(len(first.indices), *first.evidence.shape[1:]) if aliases else None
    maximum = 0
    for crop in crops:
        for start in range(0, len(classes), QUERY_CHUNK):
            sl = slice(start, start+QUERY_CHUNK)
            ids, coefficients = crop.indices[sl], crop.coefficients[sl]
            classes[sl] += (crop.classes[ids]*coefficients[..., None]).sum(1)
            if aliases:
                sampled = crop.evidence[ids]
                maximum = max(maximum, sampled.numel())
                evidence[sl] += (sampled*coefficients[..., None, None]).sum(1)
    return evidence, classes, maximum


def reversal_risk(wide_alias, wide_class, fine_alias, fine_class, rivals, protected, valid):
    # The rival class is log-MEAN-exp; the positive class observation is LSE.
    b = wide_alias-(wide_class.gather(1, rivals)-math.log(20))[..., None]
    f = fine_alias-(fine_class.gather(1, rivals)-math.log(20))[..., None]
    active = (b > 0)&(f < 0)&valid[:, None, None]&~protected[None]
    # Where inactive, no division by a zero or negative interval is needed.
    denominator = torch.where(active, b-f, torch.ones_like(b))
    return torch.where(active, -f/denominator, torch.zeros_like(b)).clamp_(0, 1)


def risk_control(risk, plan, name):
    if name == PRIMARY:
        return risk
    if name == POOLED:
        return (risk.sum(-1, keepdim=True)/19).expand_as(risk).masked_fill(plan.protected[None], 0.)
    if name == SHUFFLED:
        return risk.gather(-1, plan.permutation[None].expand_as(risk))
    raise ValueError('Declared attenuation arm required.')


def attenuated_delta(crops, risk, valid):
    """LSE at the crop tokens before the unchanged interpolation stencil.

    No remaining-mass compensation. A protected canonical slot always
    survives. Untouched classes are explicitly zero, including roundoff.
    """
    result = risk.new_zeros(risk.shape[:2], dtype=torch.float64)
    maximum = 0
    for start in range(0, len(risk), QUERY_CHUNK):
        sl = slice(start, start+QUERY_CHUNK)
        weights = (1-risk[sl].double()).log()[:, None]
        untouched = (risk[sl] == 0).all(-1)
        for crop in crops:
            evidence = crop.evidence[crop.indices[sl]].double()
            maximum = max(maximum, evidence.numel())
            delta = (evidence+weights).logsumexp(-1)-evidence.logsumexp(-1)
            delta.masked_fill_(untouched[:, None], 0.)
            result[sl] += (delta*crop.coefficients[sl].double()[..., None]).sum(1)
    result.masked_fill_(~valid[:, None], 0.)
    return result, maximum


def tile_fields(local, operator, broad, wide_cache, fine_cache, valid, plan, methods):
    base = local.double()+.5*(operator.double()@(broad.double()-local.double()))
    values = {BASE:base} if BASE in methods else {}
    if methods == (BASE,):
        return values, {}
    needs_risk = bool(set(methods)&{PRIMARY,POOLED,SHUFFLED})
    fine_alias, fine_class, maximum = sample_cache(fine_cache, aliases=needs_risk)
    innovation = (fine_class.double()-broad.double()).masked_fill(~valid[:, None], 0.)
    noalias = base+.25*(operator.double()@innovation)
    if NOALIAS in methods:
        values[NOALIAS] = noalias
    stats = dict(maximum_stencil_elements=maximum, maximum_risk_elements=0,
                 canonical_risk_max=0., active_alias_fraction=0., risk_mean=0.,
                 pooled_mass_error=0., shuffled_mass_error=0., delta_positive_max=0.,
                 mean_abs_wide_attenuation=0., mean_abs_fine_innovation=float(innovation.abs().mean()))
    if needs_risk:
        wide_alias, wide_class, workspace = sample_cache(wide_cache)
        rivals = rival_indices(base)
        risk = reversal_risk(wide_alias,wide_class,fine_alias,fine_class,rivals,plan.protected,valid)
        stats.update(maximum_stencil_elements=max(maximum,workspace),maximum_risk_elements=risk.numel(),
            canonical_risk_max=float(risk[:,plan.protected].abs().max()),
            active_alias_fraction=float((risk[valid]>0).float().mean()),risk_mean=float(risk[valid].mean()))
        for name in methods:
            if name not in (PRIMARY,POOLED,SHUFFLED):
                continue
            use = risk_control(risk,plan,name)
            delta,workspace = attenuated_delta(wide_cache,use,valid)
            values[name] = noalias+.25*(operator.double()@delta)
            stats['maximum_stencil_elements'] = max(stats['maximum_stencil_elements'],workspace)
            stats['delta_positive_max'] = max(stats['delta_positive_max'],float(delta.clamp_min(0).max()))
            if name == PRIMARY:
                stats['mean_abs_wide_attenuation'] = float(delta.abs().mean())
            if name in (POOLED,SHUFFLED):
                key = 'pooled_mass_error' if name == POOLED else 'shuffled_mass_error'
                stats[key] = float((use.sum(-1)-risk.sum(-1)).abs().max())
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError('Nonfinite single-rival scores.')
    return values,stats


@torch.inference_mode()
def predict_image(image,geometry,banks,vip,queries,plans,dataset,*,methods=METHODS,execution=None):
    from eval_development_readout import observations, wide_scores
    from eval_geometry_vip_reliability import sample_broad
    from eval_matched_contribution_alias import crop_from_features
    from eval_rival_fine_full import observe_fine, tile_coordinates
    from .bounded_fine_coverage import FineCoverageReader
    from .bounded_physical_coupling import resize_geometry
    from .device_probability_accumulator import DeviceProbabilityAccumulator
    from .geometry_readout_trace import alias_class_scores
    from .inference import hann_blend_window
    from .rival_fine_full import tile_reference_coordinates

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared frozen single-rival arms required.')
    policy = POLICIES[dataset]
    source = observations(image,geometry,vip,(policy.strength,),wide_policy=policy.wide_policy)
    height,width = source['size']
    extra = methods != (BASE,)
    needs_risk = bool(set(methods)&{PRIMARY,POOLED,SHUFFLED})
    resized = resize_geometry(image) if extra else None
    reader = (FineCoverageReader(vip) if execution is None else execution.reader(vip)) if extra else None
    blend = torch.from_numpy(hann_blend_window(512)).to(geometry.device)
    predictions,diagnostics = {},{}
    # VDD/ADE each has one partition; avoid multiplying fine encodings by partitions.
    if len(banks)!=1:
        raise ValueError('This frozen two-domain trial requires one class partition.')
    for partition,bank in banks.items():
        query,plan = queries[partition],plans[partition]
        broad = wide_scores(source,query,policy.tau,policy.tem)
        text = F.normalize(bank.features.float(),dim=-1)
        count = torch.zeros(source['wide_size'],device=geometry.device)
        for crop in source['wide']:
            top,left,ah,aw = (crop[k] for k in ('top','left','ah','aw'))
            count[top:top+ah,left:left+aw] += 1
        wide_crops = []
        if needs_risk:
            for crop in source['wide']:
                layout = WideCrop(None,None,crop['top'],crop['left'],crop['ah'],crop['aw'])
                wide_crops.append(crop_from_features(crop['features'],query,layout))
        fields,totals = [],{}
        for tile in source['local']:
            cosine = tile['features'][policy.strength].float()@text.T
            local = alias_class_scores(cosine,bank.parent_indices,bank.class_count,policy.temperature)/policy.temperature
            coordinates = tile_coordinates(tile['top'],tile['left'],geometry.device)
            fine_coordinates = tile_reference_coordinates(coordinates,tile['top'],tile['left'])
            valid = (coordinates[:,0]<height)&(coordinates[:,1]<width)
            wide = sample_broad(broad,tile['top'],tile['left'],height,width).reshape_as(local)
            wc = [cache_crop(crop,count,coordinates,(height,width),plan) for crop in wide_crops]
            fc,costs = [],{}
            if extra:
                fine,fine_count,costs = observe_fine(resized[:,tile['top']:tile['top']+512,tile['left']:tile['left']+512],
                    vip,queries,fine_coordinates,valid,feature_batch=reader)
                fc = [cache_crop(crop,fine_count,fine_coordinates,(512,512),plan) for crop in fine[partition]]
            values,stats = tile_fields(local,tile['operator'],wide,wc,fc,valid,plan,methods)
            fields.append((tile,values))
            stats.update(costs)
            for key,value in stats.items():
                totals[key] = max(totals.get(key,0.),value) if key.startswith('maximum_') or key.endswith('_max') or key.endswith('_error') else totals.get(key,0.)+value
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
        for key in totals:
            if not key.startswith('maximum_') and not key.endswith('_max') and not key.endswith('_error'):
                totals[key] /= len(source['local'])
        calls = 0 if reader is None else reader.calls
        if extra and not 1<=calls<=4*len(source['local'])<=16:
            raise RuntimeError('Fine observation budget violated.')
        diagnostics[partition] = dict(totals,geometry_encodings=len(source['local']),wide_encodings=len(source['wide']),
            fine_forwards=calls,additional_visual_forwards=calls,additional_semantic_heads=0,
            fixed_alias_slots=20,online_rivals_per_class=1)
    return predictions,diagnostics
