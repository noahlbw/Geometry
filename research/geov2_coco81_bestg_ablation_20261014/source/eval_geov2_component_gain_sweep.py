"""Matched component ablations with the same declared g0 search for every arm.

Reuses the audited frozen readout, recomputes visual interventions, and stores
image TP/predicted/target counts instead of a prohibitively large CM tensor.
"""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from eval_alias_finalization import load_samples
from eval_development_readout import confusion
from eval_geov2_fixed05_components import ARMS, Experiment
from eval_geov2_g0_sweep import metric,write

WRITER_CONTROLS=('UniformBlend','DiagonalH')


class GainExperiment(Experiment):
    def __init__(self,protocol,methods):
        # The base harness freezes .5; this subclass explicitly overrides
        # only coupling for each declared readout, preserving historical code.
        super().__init__(dict(protocol,g0=.5),[m for m in methods if m in ARMS])
        self.model.profile=replace(self.model.profile,coupling=protocol['reference_g0'])

    def read_gain(self,source,method,gain,cache):
        previous=self.model.profile
        self.model.profile=replace(previous,coupling=previous.coupling if gain is None else gain)
        try:
            return self.read(source,'Full' if method in WRITER_CONTROLS else method,cache)
        finally:
            self.model.profile=previous

    def predict_gain(self,image,method,gain):
        kind,_,fusion=ARMS[method]
        source=self.source(image,kind,solve=fusion=='coupled')
        return self.read_gain(source,method,gain,{})


def cells(protocol):
    result=[]
    for method in protocol['component_arms']:
        for gain in protocol['gains']:
            result.append(dict(key=method+'__g'+format(gain,'g'),method=method,g0=gain))
    result.extend(dict(key=method,method=method,g0=None) for method in protocol['endpoint_arms'])
    return result


@torch.inference_mode()
def main(args):
    root=Path(args.experiment)
    protocol=json.loads((root/'protocol.json').read_text())
    if not 0<=args.shard_index<args.num_shards:
        raise ValueError('Invalid shard.')
    if args.mode=='cost' and (args.arm not in protocol['cost_arms'] or args.gain not in protocol['gains']):
        raise ValueError('Cost is outside declared protocol.')
    methods=protocol['component_arms']+protocol['endpoint_arms'] if args.mode!='cost' else [args.arm]
    out=root/args.mode/(f's{args.shard_index}' if args.mode=='full' else args.arm if args.mode=='cost' else 'check')
    if out.exists():
        raise RuntimeError('Preserve prior output: '+str(out))
    out.mkdir(parents=True)
    exp=GainExperiment(protocol,methods)
    m=exp.model
    args.dataset,args.data_root=protocol['dataset'],protocol['data_root']
    args.sample_seed,args.vdd_ontology=20260923,'official'
    entry=dict(m.config,banks={'original':next(iter(m.vocabulary['banks'].values()))})
    validation,samples,images,masks,names_map=load_samples(args,entry)
    if len(validation)!=protocol['total_images']:
        raise ValueError('Validation size differs.')
    common=dict(dataset=args.dataset,reference_g0=protocol['reference_g0'],profile=m.profile.record(),
        config=m.config,alias_counts=exp.alias_counts,cache_identity=m.cache_identity,
        adaptive_gain_enabled=False)
    if args.mode=='cost':
        lookup={s.key:s for s in validation}
        rows=[]
        for key in protocol['cost_keys']:
            image=images(lookup[key])
            for _ in range(2):
                exp.predict_gain(image,args.arm,args.gain)
            times,allocated,reserved=[],[],[]
            for _ in range(7):
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start=time.perf_counter()
                result=exp.predict_gain(image,args.arm,args.gain)
                torch.cuda.synchronize()
                times.append((time.perf_counter()-start)*1000)
                allocated.append(torch.cuda.max_memory_allocated()/1048576)
                reserved.append(torch.cuda.max_memory_reserved()/1048576)
            if result.dtype!=np.uint8 or result.shape!=tuple(image.shape[-2:]):
                raise ValueError('Invalid cost output.')
            rows.append(dict(key=key,shape=list(image.shape[-2:]),milliseconds=times,
                peak_allocated_mib=max(allocated),peak_reserved_mib=max(reserved)))
            print(json.dumps(dict(arm=args.arm,processed=len(rows))),flush=True)
        write(out/'results.json',dict(status='complete',**common,arm=args.arm,g0=args.gain,images=rows,
            median_ms=float(np.median([t for row in rows for t in row['milliseconds']])),
            peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows),
            peak_reserved_mib=max(r['peak_reserved_mib'] for r in rows),
            gpu=torch.cuda.get_device_name(),target_masks_loaded=False,
            boundary='CPU float RGB to original-size CPU uint8 IDs; exclude decode/init/offline text; fresh process per arm, 7 fixed images x 2 warmups x 7 synchronized repeats. All cost arms use Full-selected g0. Beta-zero retains unused routing computations.'))
        return
    if args.mode=='smoke':
        samples=samples[:1]
    grid=cells(protocol)
    names=names_map[args.dataset]
    matrices=np.zeros((len(grid),len(names),len(names)),np.int64)
    tps,predicted,targets,keys=[],[],[],[]
    start=time.perf_counter()
    for number,sample in enumerate(samples,1):
        image=images(sample)
        predictions={}
        for kind in dict.fromkeys(ARMS[m][0] for m in methods if m in ARMS):
            source=exp.source(image,kind)
            caches={}
            for cell in grid:
                method=cell['method']
                if method in WRITER_CONTROLS or ARMS[method][0]!=kind:
                    continue
                language=ARMS[method][1]
                predictions[cell['key']]=exp.read_gain(source,method,cell['g0'],caches.setdefault(language,{}))
            if kind=='full':
                for control in WRITER_CONTROLS:
                    changed=[]
                    for tile in source['local']:
                        h=tile['operator']
                        operator=torch.eye(len(h),device=h.device,dtype=h.dtype)*.5 if control=='UniformBlend' else torch.diag(h.diagonal())
                        changed.append(dict(tile,operator=operator))
                    altered=dict(source,local=changed)
                    for cell in grid:
                        if cell['method']==control:
                            predictions[cell['key']]=exp.read_gain(altered,control,cell['g0'],caches['retained'])
                    del altered,changed
                if args.mode=='smoke':
                    import dinotool.query_slot_memory as qs
                    original=qs.query_slot_wide_observations
                    qs.query_slot_wide_observations=lambda *_a,**_k:source
                    try:
                        direct=m.predict(image)
                    finally:
                        qs.query_slot_wide_observations=original
                    fullkey='Full__g'+format(protocol['reference_g0'],'g')
                    check=dict(same_observation_frozen_vs_full_pixels=int((direct!=predictions[fullkey]).sum()),
                        uniform_g1_vs_equal_pixels=int((predictions['UniformBlend__g1']!=predictions['EqualLogitFusion']).sum()),
                        shape=list(direct.shape))
                    if check['same_observation_frozen_vs_full_pixels'] or check['uniform_g1_vs_equal_pixels']:
                        raise RuntimeError('Prediction parity failed: '+str(check))
                    write(out/'prediction_check.json',check)
                    print(json.dumps(check),flush=True)
            del source,caches
        if set(predictions)!={cell['key'] for cell in grid}:
            raise RuntimeError('A grid cell was not computed.')
        target=masks(sample,args.dataset,tuple(image.shape[-2:]))
        cms=np.stack([confusion(predictions[cell['key']],target,len(names)) for cell in grid])
        if not np.all(cms.sum(-1)==cms[:1].sum(-1)):
            raise ValueError('Target support differs across cells.')
        matrices+=cms
        tps.append(np.diagonal(cms,axis1=-2,axis2=-1).copy())
        predicted.append(cms.sum(-2))
        targets.append(cms[0].sum(-1))
        keys.append(sample.key)
        if number==1 or number%25==0 or number==len(samples):
            progress=dict(shard=args.shard_index,processed=number,total=len(samples),wall_seconds=time.perf_counter()-start)
            write(out/'progress.json',progress)
            print(json.dumps(progress),flush=True)
    np.savez_compressed(out/'statistics.npz',sample_keys=np.asarray(keys),tp=np.stack(tps),
        predicted=np.stack(predicted),target=np.stack(targets))
    write(out/'results.json',dict(status='complete',**common,cells=grid,methods=[c['key'] for c in grid],
        processed_images=len(keys),global_sample_count=len(validation),sample_keys=keys,
        shard_index=args.shard_index,num_shards=args.num_shards,class_names=names,gains=protocol['gains'],
        metrics={cell['key']:dict(mean_iou_percent=metric(cm),confusion_matrix=cm.tolist()) for cell,cm in zip(grid,matrices)},
        masks_used_only_for_scoring=True,wall_seconds=time.perf_counter()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',required=True)
    p.add_argument('--mode',choices=('smoke','full','cost'),required=True)
    p.add_argument('--arm',default='Full')
    p.add_argument('--gain',type=float,default=.5)
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    main(p.parse_args())
