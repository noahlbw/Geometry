"""Equivalent execution of the frozen single-rival fine20 reader.

Keep batch-one encodings, BF16 projection, chunk32, FP64 crop LSE differences
and crop accumulation order. Only cache independent fields, reuse the fine
coverage stencil, vectorize its fixed16 neighbors, and graph the same reader.
"""
from dataclasses import replace
import time

import torch
import torch.nn.functional as F

from . import single_rival_fine_alias as rule
from .bounded_fine_execution import FineCoverageExecution
from .stratified_soft_alias import WideCrop,crop_stencil


IMPLEMENTATION = 'single-rival-fine20-equivalent-execution-v1-20261008'


def vector_stencil(crop,count,coordinates,image_size):
    rh,rw = count.shape
    position = coordinates*coordinates.new_tensor((rh/image_size[0],rw/image_size[1]))-.5
    position[:,0].clamp_(0,rh-1);position[:,1].clamp_(0,rw-1)
    lower = position.floor().long();fraction = position-lower
    offsets = torch.tensor([[0,0],[0,1],[1,0],[1,1]],device=coordinates.device)
    dy,dx = offsets[:,0],offsets[:,1]
    y = (lower[:,0,None]+dy).clamp_max(rh-1)
    x = (lower[:,1,None]+dx).clamp_max(rw-1)
    weight = torch.where(dy.bool()[None],fraction[:,0,None],1-fraction[:,0,None])
    weight = weight*torch.where(dx.bool()[None],fraction[:,1,None],1-fraction[:,1,None])
    inside = (y>=crop.top)&(y<crop.top+crop.actual_height)&(x>=crop.left)&(x<crop.left+crop.actual_width)
    weight = weight*inside/count[y,x]
    local = torch.stack((y-crop.top,x-crop.left),-1).float()
    local = ((local+.5)*crop.grid_side/crop.crop_side-.5).clamp(0,crop.grid_side-1)
    base = local.floor().long();offset = local-base
    gy = (base[:,:,0,None]+dy).clamp_max(crop.grid_side-1)
    gx = (base[:,:,1,None]+dx).clamp_max(crop.grid_side-1)
    coefficient = weight[:,:,None]*torch.where(dy.bool()[None,None],offset[:,:,0,None],1-offset[:,:,0,None])
    coefficient = coefficient*torch.where(dx.bool()[None,None],offset[:,:,1,None],1-offset[:,:,1,None])
    return (gy*crop.grid_side+gx).reshape(len(coordinates),16),coefficient.reshape(len(coordinates),16)


def profile_crop(crop,plan):
    weights = crop.salience[plan.members].softmax(-1)
    evidence = crop.alias_logits[:,plan.members]*(weights/weights.mean(-1,keepdim=True))
    return evidence,evidence.logsumexp(-1)


def sparse_crop(crop,evidence,classes,count,coordinates,image_size,verify=False):
    ids,coefficients = vector_stencil(crop,count,coordinates,image_size)
    if verify:
        old_ids,old_coefficients = crop_stencil(crop,count,coordinates,image_size)
        if not torch.equal(ids,old_ids) or not torch.equal(coefficients,old_coefficients):
            raise RuntimeError('Vector stencil is not bitwise equal to original16-neighbor stencil.')
    return rule.SparseCrop(evidence,classes,ids,coefficients)


@torch.inference_mode()
def observe_fine_cached(image,vip,query,coordinates,valid,plan,reader,verify=False):
    from eval_matched_contribution_alias import crop_from_features
    from .fine_reference_admission import fine_crop_positions

    positions = fine_crop_positions(*image.shape[-2:])
    count = torch.zeros(512,512,device=vip.device)
    rgbs = []
    for top,left,ch,cw in positions:
        count[top:top+ch,left:left+cw] += 1
        rgb = F.pad(image[:,top:top+ch,left:left+cw].to(vip.device),(0,256-cw,0,256-ch))
        rgbs.append(F.interpolate(rgb[None],(512,512),mode='bilinear',align_corners=False)[0])
    count.clamp_min_(1)
    features = reader(vip,tuple(rgbs))
    coverage = torch.zeros(len(valid),device=vip.device)
    caches = []
    for value,(top,left,ch,cw) in zip(features,positions):
        layout = WideCrop(None,None,top,left,ch,cw,32,256)
        crop = replace(crop_from_features(value,query,layout),grid_side=32,crop_side=256)
        evidence,classes = profile_crop(crop,plan)
        cached = sparse_crop(crop,evidence,classes,count,coordinates,(512,512),verify)
        coverage += cached.coefficients.sum(-1)
        caches.append(cached)
    error = (coverage-1).abs().masked_fill(~valid,0.).max()
    return caches,error


def tensor_fields(local,operator,broad,wide_cache,fine_cache,valid,plan,methods):
    """Same arithmetic and order as rule.tile_fields; checks remain on device."""
    base = local.double()+.5*(operator.double()@(broad.double()-local.double()))
    fine_alias,fine_class,maximum = rule.sample_cache(fine_cache,aliases=bool(set(methods)&{rule.PRIMARY,rule.POOLED,rule.SHUFFLED}))
    innovation = (fine_class.double()-broad.double()).masked_fill(~valid[:,None],0.)
    noalias = base+.25*(operator.double()@innovation)
    values = {rule.BASE:base,rule.NOALIAS:noalias}
    zero = local.new_zeros(())
    checks = dict(canonical_risk_max=zero,active_alias_fraction=zero,risk_mean=zero,
        pooled_mass_error=zero,shuffled_mass_error=zero,delta_positive_max=zero,
        mean_abs_wide_attenuation=zero,mean_abs_fine_innovation=innovation.abs().mean())
    traces = []
    if set(methods)&{rule.PRIMARY,rule.POOLED,rule.SHUFFLED}:
        wide_alias,wide_class,workspace = rule.sample_cache(wide_cache)
        rivals = rule.rival_indices(base)
        risk = rule.reversal_risk(wide_alias,wide_class,fine_alias,fine_class,rivals,plan.protected,valid)
        traces = [fine_alias,fine_class,wide_alias,wide_class,risk]
        denominator = valid.sum().clamp_min(1)*risk.shape[1]*risk.shape[2]
        checks.update(canonical_risk_max=risk.abs().masked_fill(~plan.protected[None],0.).max(),
            active_alias_fraction=((risk>0)&valid[:,None,None]).float().sum()/denominator,
            risk_mean=(risk*valid[:,None,None]).sum()/denominator)
        maximum = max(maximum,workspace)
        for name in methods:
            if name not in (rule.PRIMARY,rule.POOLED,rule.SHUFFLED):continue
            use = rule.risk_control(risk,plan,name)
            delta,workspace = rule.attenuated_delta(wide_cache,use,valid)
            values[name] = noalias+.25*(operator.double()@delta)
            traces.append(delta)
            maximum = max(maximum,workspace)
            checks['delta_positive_max'] = torch.maximum(checks['delta_positive_max'],delta.clamp_min(0).max())
            if name==rule.PRIMARY:checks['mean_abs_wide_attenuation'] = delta.abs().mean()
            if name in (rule.POOLED,rule.SHUFFLED):
                key = 'pooled_mass_error' if name==rule.POOLED else 'shuffled_mass_error'
                checks[key] = (use.sum(-1)-risk.sum(-1)).abs().max()
    outputs = [values[name] for name in methods]
    checks['nonfinite_scores'] = (~torch.stack([torch.isfinite(value).all() for value in outputs])).float().max()
    return (*outputs,torch.stack([value.double() for value in checks.values()]),*traces)


STAT_NAMES = ('canonical_risk_max','active_alias_fraction','risk_mean','pooled_mass_error',
    'shuffled_mass_error','delta_positive_max','mean_abs_wide_attenuation','mean_abs_fine_innovation','nonfinite_scores')


class ReaderGraph:
    def __init__(self):
        self.graphs = {};self.setup_seconds = {};self.setup_calls = {}

    @torch.inference_mode()
    def __call__(self,local,operator,broad,wc,fc,valid,plan,methods,verify=False):
        key = (methods,tuple(local.shape),tuple(c.evidence.shape for c in wc),tuple(c.evidence.shape for c in fc))
        dynamic = (local,operator,broad,valid,*[value for crop in (*wc,*fc)
            for value in (crop.evidence,crop.classes,crop.indices,crop.coefficients)])
        if key not in self.graphs:
            started = time.perf_counter()
            buffers = tuple(value.clone() for value in dynamic)
            def run():
                groups = []
                for n in range(len(wc)+len(fc)):
                    groups.append(rule.SparseCrop(*buffers[4+4*n:8+4*n]))
                return tensor_fields(*buffers[:3],groups[:len(wc)],groups[len(wc):],buffers[3],plan,methods)
            stream = torch.cuda.Stream(device=local.device)
            stream.wait_stream(torch.cuda.current_stream(local.device))
            with torch.cuda.stream(stream):
                for _ in range(3):run()
            torch.cuda.current_stream(local.device).wait_stream(stream)
            torch.cuda.synchronize(local.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):outputs = run()
            torch.cuda.synchronize(local.device)
            self.graphs[key] = (graph,buffers,outputs)
            label = str(key)
            self.setup_seconds[label] = time.perf_counter()-started
            self.setup_calls[label] = 4
        graph,buffers,outputs = self.graphs[key]
        for buffer,value in zip(buffers,dynamic):buffer.copy_(value)
        graph.replay()
        if verify:
            eager = tensor_fields(local,operator,broad,wc,fc,valid,plan,methods)
            if any(not torch.equal(a,b) for a,b in zip(outputs,eager)):
                raise RuntimeError('Reader graph changed sample fields/risk/DeltaW/logits.')
            legacy,_ = rule.tile_fields(local,operator,broad,wc,fc,valid,plan,methods)
            if any(not torch.equal(outputs[n],legacy[name]) for n,name in enumerate(methods)):
                raise RuntimeError('Pure tensor writer changed frozen patch logits.')
        # Later replays must not mutate a previously returned tile.
        return {name:outputs[n].clone() for n,name in enumerate(methods)},outputs[len(methods)].clone()


class Execution:
    def __init__(self,*,cached=True,burst=True):
        if not cached or not burst:
            raise ValueError('Only the frozen cached/burst execution is declared.')
        self.fine = FineCoverageExecution(cached=True,burst=True)
        self.reader = ReaderGraph()

    def report(self):
        return dict(implementation=IMPLEMENTATION,fine_encoder=self.fine.report(),
            reader_graph_setup_seconds=self.reader.setup_seconds,reader_graph_setup_calls=self.reader.setup_calls,
            visual_budget_unchanged=16,semantic_rule=rule.IMPLEMENTATION)


@torch.inference_mode()
def predict_image(image,geometry,banks,vip,queries,plans,dataset,*,methods=rule.METHODS,execution=None,verify=False):
    from eval_development_readout import observations,wide_scores
    from eval_geometry_vip_reliability import sample_broad
    from eval_matched_contribution_alias import crop_from_features
    from eval_rival_fine_full import tile_coordinates
    from .bounded_physical_coupling import resize_geometry
    from .device_probability_accumulator import DeviceProbabilityAccumulator
    from .geometry_readout_trace import alias_class_scores
    from .inference import hann_blend_window
    from .rival_fine_full import tile_reference_coordinates

    if methods==(rule.BASE,):
        return rule.predict_image(image,geometry,banks,vip,queries,plans,dataset,methods=methods)
    if not methods or not set(methods).issubset(rule.METHODS) or len(banks)!=1:
        raise ValueError('Frozen five-arm/two-domain execution required.')
    execution = Execution() if execution is None else execution
    policy = rule.POLICIES[dataset]
    source = observations(image,geometry,vip,(policy.strength,),wide_policy=policy.wide_policy)
    height,width = source['size'];resized = resize_geometry(image)
    reader = execution.fine.reader(vip)
    needs_risk = bool(set(methods)&{rule.PRIMARY,rule.POOLED,rule.SHUFFLED})
    blend = torch.from_numpy(hann_blend_window(512)).to(geometry.device)
    predictions,diagnostics = {},{}
    for p,bank in banks.items():
        query,plan = queries[p],plans[p]
        broad = wide_scores(source,query,policy.tau,policy.tem)
        text = F.normalize(bank.features.float(),dim=-1)
        count = torch.zeros(source['wide_size'],device=geometry.device)
        for crop in source['wide']:
            top,left,ah,aw = (crop[k] for k in ('top','left','ah','aw'))
            count[top:top+ah,left:left+aw] += 1
        crops = []
        if needs_risk:
            for crop in source['wide']:
                layout = WideCrop(None,None,crop['top'],crop['left'],crop['ah'],crop['aw'])
                value = crop_from_features(crop['features'],query,layout)
                evidence,classes = profile_crop(value,plan)
                crops.append((value,evidence,classes))
        fields,metrics,coverage = [],[],[]
        for tile in source['local']:
            cosine = tile['features'][policy.strength].float()@text.T
            local = alias_class_scores(cosine,bank.parent_indices,bank.class_count,policy.temperature)/policy.temperature
            coordinates = tile_coordinates(tile['top'],tile['left'],geometry.device)
            fine_coordinates = tile_reference_coordinates(coordinates,tile['top'],tile['left'])
            valid = (coordinates[:,0]<height)&(coordinates[:,1]<width)
            wide = sample_broad(broad,tile['top'],tile['left'],height,width).reshape_as(local)
            wc = [sparse_crop(crop,evidence,classes,count,coordinates,(height,width),verify) for crop,evidence,classes in crops]
            fc,error = observe_fine_cached(resized[:,tile['top']:tile['top']+512,tile['left']:tile['left']+512],
                vip,query,fine_coordinates,valid,plan,reader,verify)
            values,checks = execution.reader(local,tile['operator'],wide,wc,fc,valid,plan,methods,verify)
            fields.append((tile,values));metrics.append(checks);coverage.append(error)
        checks = torch.stack(metrics).cpu().numpy()
        cov = float(torch.stack(coverage).max())
        if checks[:,-1].max()!=0 or cov>1e-6:
            raise RuntimeError('Nonfinite scores or changed fine coverage.')
        diagnostics[p] = {name:float(checks[:,n].max() if name.endswith('_max') or name.endswith('_error')
            else checks[:,n].mean()) for n,name in enumerate(STAT_NAMES[:-1])}
        diagnostics[p].update(geometry_encodings=len(source['local']),wide_encodings=len(source['wide']),
            fine_forwards=reader.calls,additional_visual_forwards=reader.calls,additional_semantic_heads=0,
            fixed_alias_slots=20,online_rivals_per_class=1,fine_coverage_max_error=cov,
            maximum_stencil_elements=32*16*bank.class_count*20,
            maximum_risk_elements=1024*bank.class_count*20 if needs_risk else 0)
        if not 1<=reader.calls<=4*len(source['local'])<=16:
            raise RuntimeError('Equivalent execution changed fine observation budget.')
        predictions[p] = {}
        for name in methods:
            with DeviceProbabilityAccumulator(bank.class_count,height,width,geometry.device) as accumulator:
                for tile,values in fields:
                    dense = F.interpolate(values[name].T.reshape(1,bank.class_count,32,32),(512,512),
                                          mode='bilinear',align_corners=False)[0]
                    top,left = tile['top'],tile['left'];ah,aw = min(512,height-top),min(512,width-left)
                    accumulator.add(dense[:,:ah,:aw].softmax(0).float(),blend[:ah,:aw],left,top)
                predictions[p][name] = accumulator.finalize_resized(source['output_size'])
    return predictions,diagnostics
