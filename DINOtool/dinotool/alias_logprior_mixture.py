"""Use wide salience as a mixture prior, rather than a logit multiplier.

This reader does not certify semantic correctness. Its uniform and shuffled
controls distinguish alias identity from a change in score calibration.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .fixed20_task_readout import POLICIES


IMPLEMENTATION = 'geometry-original20-wide-logprior-mixture-v1-20261008'
BASE = 'Fixed20_TaskCoupled'
PRIMARY = 'AliasLogPrior_Mix'
UNIFORM = 'AliasLogPrior_Uniform'
SHUFFLED = 'AliasLogPrior_Shuffle'
METHODS = (BASE, PRIMARY, UNIFORM, SHUFFLED)
SEED = 20261008


@dataclass(frozen=True)
class MixturePlan:
    members: torch.Tensor
    permutation: torch.Tensor


@torch.inference_mode()
def prepare_plan(bank):
    members = torch.stack([(bank.parent_indices == c).nonzero().flatten()
                          for c in range(bank.class_count)])
    if members.shape[1] != 20:
        raise ValueError('Exactly twenty original aliases per class required.')
    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.stack([torch.randperm(20, generator=generator)
                               for _ in range(bank.class_count)]).to(members.device)
    return MixturePlan(members, permutation)


def mixture_logits(raw_logits, salience, plan, method, tau=1., tem=1.):
    """[patch,alias] raw logits, [alias] salience; return [patch,class]."""
    if method not in (PRIMARY, UNIFORM, SHUFFLED) or tau <= 0 or tem <= 0:
        raise ValueError('Declared mixture arm and positive temperatures required.')
    values = raw_logits[:, plan.members].float()
    prior = (salience[plan.members].float()/tem).log_softmax(-1)+math.log(20)
    if method == UNIFORM:
        prior = torch.zeros_like(prior)
    elif method == SHUFFLED:
        prior = prior.gather(-1, plan.permutation)
    return (tau*values+prior[None]).logsumexp(-1)/tau


@torch.inference_mode()
def wide_fields(source, query, plan, methods, tau, tem):
    from eval_development_readout import wide_scores

    height, width = source['wide_size']
    output = {}
    if BASE in methods:
        # Exact old scorer/reduction order remains the independent control.
        output[BASE] = wide_scores(source, query, tau, tem)
    requested = tuple(name for name in methods if name != BASE)
    stats = dict(max_logprior_abs=0., mean_salience_entropy_fraction=0.,
                 mean_abs_mix_uniform_wide=0., maximum_alias_elements=0.)
    if not requested:
        return output, stats
    accumulators = {name:torch.zeros(len(query.class_names),height,width,
                                    device=query.features.device) for name in requested}
    counts = torch.zeros(height,width,device=query.features.device)
    for crop in source['wide']:
        features = crop['features']
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            similarities = torch.einsum('bnd,mtd->bnmt',features,query.features.float()).mean(-1)[0]
            patch_mean = F.normalize(features.mean(1),dim=-1)
            text_mean = F.normalize(query.features.float().mean(1),dim=-1)
            salience = (patch_mean@text_mean.T)[0].float()
            raw = similarities*40.
        prior = (salience[plan.members]/tem).log_softmax(-1)
        stats['max_logprior_abs'] = max(stats['max_logprior_abs'],float((prior+math.log(20)).abs().max()))
        stats['mean_salience_entropy_fraction'] += float(-(prior.exp()*prior).sum(-1).mean()/math.log(20))
        values = {name:mixture_logits(raw,salience,plan,name,tau,tem) for name in requested}
        if PRIMARY in values and UNIFORM in values:
            stats['mean_abs_mix_uniform_wide'] += float((values[PRIMARY]-values[UNIFORM]).abs().mean())
        stats['maximum_alias_elements'] = max(stats['maximum_alias_elements'],raw.shape[0]*plan.members.numel())
        top,left,ah,aw = (crop[k] for k in ('top','left','ah','aw'))
        for name,value in values.items():
            dense = F.interpolate(value.T.reshape(1,len(query.class_names),21,21),
                                  (336,336),mode='bilinear',align_corners=False)[0]
            accumulators[name][:,top:top+ah,left:left+aw] += dense[:,:ah,:aw]
        counts[top:top+ah,left:left+aw] += 1
    if not bool((counts>0).all()):
        raise RuntimeError('Wide mixture coverage gap.')
    output.update({name:value/counts[None] for name,value in accumulators.items()})
    for key in ('mean_salience_entropy_fraction','mean_abs_mix_uniform_wide'):
        stats[key] /= len(source['wide'])
    return output,stats


@torch.inference_mode()
def predict_image(image,geometry,banks,vip,queries,plans,dataset,*,methods=METHODS):
    from eval_development_readout import observations
    from eval_geometry_vip_reliability import sample_broad
    from .device_probability_accumulator import DeviceProbabilityAccumulator
    from .geometry_readout_trace import alias_class_scores
    from .inference import hann_blend_window

    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared mixture arms required.')
    policy = POLICIES[dataset]
    source = observations(image,geometry,vip,(policy.strength,),wide_policy=policy.wide_policy)
    height,width = source['size']
    blend = torch.from_numpy(hann_blend_window(512)).to(geometry.device)
    predictions,diagnostics = {},{}
    for partition,bank in banks.items():
        broad,stats = wide_fields(source,queries[partition],plans[partition],methods,policy.tau,policy.tem)
        text = F.normalize(bank.features.float(),dim=-1)
        fields = []
        write_delta = 0.
        for tile in source['local']:
            cosine = tile['features'][policy.strength].float()@text.T
            local = alias_class_scores(cosine,bank.parent_indices,bank.class_count,policy.temperature)/policy.temperature
            values = {}
            for name in methods:
                wide = sample_broad(broad[name],tile['top'],tile['left'],height,width).reshape(len(local),bank.class_count)
                values[name] = local.double()+policy.coupling*(tile['operator'].double()@(wide.double()-local.double()))
            if not all(bool(torch.isfinite(value).all()) for value in values.values()):
                raise RuntimeError('Nonfinite log-prior mixture prediction.')
            if PRIMARY in values and UNIFORM in values:
                write_delta += float((values[PRIMARY]-values[UNIFORM]).abs().mean())
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
        diagnostics[partition] = dict(stats,mean_abs_mix_uniform_writeback=write_delta/len(fields),
            geometry_encodings=len(source['local']),wide_encodings=len(source['wide']),fine_forwards=0,
            additional_visual_forwards=0,additional_semantic_heads=0,fixed_alias_slots=20)
    return predictions,diagnostics
