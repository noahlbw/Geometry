"""One isolated model per process, fixed full-image synchronized timing."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
import eval_development_readout as base
from eval_alias_finalization import load_samples
from benchmark_natural_sense_fast import require_available_gpu
from dinotool.finite_vip_observer import FiniteVIPObserver
from dinotool.model import DINOTextSegmenter,checkpoint_manifest
from dinotool.natural_text_adaptation import CLASS_FILES
from dinotool.taxonomy_inference import TaxonomyInference
from dinotool.vip_official_adapter import VIPQueries,VIPSettings,upstream_aliases,upstream_settings,_load_upstream
from eval_natural_text_adaptation import official_options,official_prediction
from eval_rival_fine_full import save
from eval_stride_ov_loveda_e1 import make_checkpoints
from eval_vip_official_eight import PINNED_COMMIT,OFFICIAL_CLASSES


def query_from_row(row,classes,device):
    names=tuple(c['name'] for c in classes);aliases=tuple(a for c in classes for a in c['synonyms'])
    return VIPQueries(row['wide'].to(device),row['parents'].to(device),names,aliases)


@torch.inference_mode()
def vip_model(args,entry):
    checkpoints=make_checkpoints(args);upstream=Path(args.upstream_root)
    commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    if commit!=PINNED_COMMIT:raise RuntimeError('Pinned VIP source differs.')
    vip=FiniteVIPObserver(DINOTextSegmenter(checkpoints,device=args.device),upstream)
    names=tuple(c['name'] for c in entry['banks']['original']['classes'])
    if args.cost_mode=='VIPMatched':
        k='semantic_segmentation' if entry['family']=='natural' else 'original_imagenet'
        cache=torch.load(Path(args.suite_root)/'text_cache'/(args.dataset+'.pt'),map_location=args.device,weights_only=True)
        if cache['identity']!=dict(banks=entry['banks'],checkpoints=checkpoint_manifest(checkpoints),upstream_commit=commit):
            raise RuntimeError('Matched frozen cache differs.')
        query=query_from_row(cache['encoded'][k],entry['banks'][k]['classes'],vip.device)
        return vip,query,VIPSettings(tau=1.,tem=1.),dict(words='Frozen final bank '+k,
            scope='Query-pool matched; standard VIP long448 views and no Geometry-based bank choice. Not a same-observation readout control.')
    if entry['family']=='natural' or args.dataset in OFFICIAL_CLASSES:
        if entry['family']=='natural':
            groups=upstream_aliases(upstream/'configs'/('cls_'+CLASS_FILES[args.dataset]+'.txt'))
            settings,template=official_options(upstream,args.dataset)
        else:
            groups=upstream_aliases(upstream/'configs'/OFFICIAL_CLASSES[args.dataset])
            settings=upstream_settings(args.dataset);template='openai_imagenet_template'
            if args.dataset=='vaihingen':names=(*names,'clutter')
        prompt=_load_upstream(upstream/'prompts/imagenet_template.py','finalization_cost_prompts')
        vip.templates=prompt.get_text_template(template)
        if len(names)!=len(groups):raise RuntimeError('Official queries differ from taxonomy.')
        identity=dict(checkpoints=checkpoint_manifest(checkpoints),upstream_commit=commit,groups=groups,
                      names=names,template=template)
        path=Path(args.suite_root)/'cost_text'/(args.dataset+'.pt')
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            encoded=torch.load(path,map_location=args.device,weights_only=True)
            # JSON-equivalent sequence comparison tolerates serialized tuples.
            if json.dumps(encoded['identity'],sort_keys=True)!=json.dumps(identity,sort_keys=True):
                raise RuntimeError('Frozen official text identity differs.')
            query=VIPQueries(encoded['features'],encoded['parents'],names,tuple(a for g in groups for a in g))
        else:
            query=vip.encode_queries(names,groups)
            temporary=path.with_suffix('.tmp')
            torch.save(dict(identity=identity,features=query.features.cpu(),parents=query.parents.cpu()),temporary)
            temporary.rename(path)
        return vip,query,settings,dict(words='Pinned upstream official queries',template=template,
            scope='Paper configuration reimplementation with explicit all-masked-row self-Value repair; not published benchmark timing.')
    cache=torch.load(args.original_cache,map_location=args.device,weights_only=True)
    key='D__original20' if args.dataset=='loveda' else args.dataset+'__original20'
    groups=[c['synonyms'] for c in entry['banks']['original']['classes']]
    if cache['identity']['aliases'][key]!=groups:raise RuntimeError('External20 words differ.')
    query=query_from_row(cache['encoded'][key],entry['banks']['original']['classes'],vip.device)
    return vip,query,upstream_settings(args.dataset),dict(words='External original20; no official VIP vocabulary exists',
        scope='Cross-domain VIP adaptation, not published/official-domain benchmark or distilled-query timing.')


@torch.inference_mode()
def main(args):
    require_available_gpu(torch.device(args.device))
    output=Path(args.output)
    if output.exists():raise RuntimeError('Existing cost output; preserve it.')
    entry=json.loads((Path(args.suite_root)/'protocol.json').read_text())['datasets'][args.dataset]
    validation,_,loader,_,_=load_samples(args,entry)
    init_start=time.perf_counter()
    if args.cost_mode in ('Retained','Uniform'):
        if args.dataset=='loveda':args.original_cache_key='D__original20'
        needed=('original_imagenet','focused20') if entry['family']=='remote_sensing' else ('semantic_segmentation',)
        geometry,vip,banks,queries,checkpoints=base.load_models(args,entry,needed)
        residual=None
        if entry.get('residual_identity') is not None:
            cache=torch.load(Path(args.suite_root)/'residual_cache.pt',map_location=args.device,weights_only=True)
            if cache['identity']!=entry['residual_identity']:raise RuntimeError('Residual identity differs.')
            residual=cache['wide']
        model=TaxonomyInference.for_finalization(geometry,vip,banks,queries,
            ordinary_alias_policy=args.cost_mode.lower(),family=entry['family'],background=entry['background_index'],
            residual_features=residual)
        predict=model.predict;metadata=dict(wide_policy=model.wide_policy,extra_soft_weights=model.soft,
            residual_queries=0 if residual is None else len(residual),backbones_resident=2)
        calls={'geometry':0,'wide':0}
        original_geometry=geometry.prepare_image
        def counted_geometry(*a,**k):
            calls['geometry']+=1;return original_geometry(*a,**k)
        geometry.prepare_image=counted_geometry
    else:
        vip,query,settings,metadata=vip_model(args,entry)
        if args.cost_mode=='VIPOfficialProtocol' and entry['family']=='natural':
            predict=lambda image:official_prediction(image,vip,query,settings)
            metadata['view_policy']='Official natural short-edge336/maxlong2048/crop336/overlap224/actual stride112'
        else:
            predict=lambda image:vip.predict(image,query,settings)
            metadata['view_policy']='Adapter long-edge448/crop336/stride112'
            if args.cost_mode=='VIPOfficial' and entry['family']=='natural':
                metadata['scope']='Legacy official-query long448 timing; not the official natural preprocessing. Preserved as a scale-specific timing control.'
        metadata.update(settings=asdict(settings),backbones_resident=1,aliases=len(query.aliases))
        calls={'geometry':0,'wide':0}
    original_wide=vip.crop_patch_features
    def counted_wide(*a,**k):
        calls['wide']+=1;return original_wide(*a,**k)
    vip.crop_patch_features=counted_wide
    init_seconds=time.perf_counter()-init_start
    rows=[]
    indexes=np.unique(np.linspace(0,len(validation)-1,args.samples,dtype=int))
    for index in indexes:
        sample=validation[int(index)];decode_start=time.perf_counter();image=loader(sample)
        decode_seconds=time.perf_counter()-decode_start
        torch.cuda.empty_cache()
        predict(image);predict(image);torch.cuda.synchronize()
        repeats=[];peaks=[];reserved=[];observed=[]
        for _ in range(args.repeats):
            require_available_gpu(torch.device(args.device))
            calls.update(geometry=0,wide=0)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter();result=predict(image);torch.cuda.synchronize()
            repeats.append(1000*(time.perf_counter()-started))
            peaks.append(torch.cuda.max_memory_allocated()/1048576)
            reserved.append(torch.cuda.max_memory_reserved()/1048576)
            observed.append(calls.copy())
            if tuple(result[0].shape)!=tuple(image.shape[-2:]):raise RuntimeError('Original size not restored.')
            if args.cost_mode in ('Retained','Uniform') and (calls['geometry']>4 or calls['wide']>4):
                raise RuntimeError('Whole-image observation cap exceeded.')
            del result
        rows.append(dict(key=sample.key,shape=list(image.shape[-2:]),milliseconds=repeats,
            median_ms=float(np.median(repeats)),p95_ms=float(np.quantile(repeats,.95)),
            peak_allocated_mib=max(peaks),peak_reserved_mib=max(reserved),calls=observed,
            decode_seconds=decode_seconds))
        print(json.dumps(dict(dataset=args.dataset,mode=args.cost_mode,key=sample.key,median_ms=rows[-1]['median_ms'])),flush=True)
    flattened=[v for r in rows for v in r['milliseconds']]
    save(output,dict(status='complete',standalone_process=True,target_masks_loaded=False,
        dataset=args.dataset,mode=args.cost_mode,images=rows,metadata=metadata,
        initialization_and_text_seconds=init_seconds,gpu=torch.cuda.get_device_name(),
        median_ms=float(np.median(flattened)),p95_ms=float(np.quantile(flattened,.95)),
        mean_image_median_ms=float(np.mean([r['median_ms'] for r in rows])),
        peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows),
        peak_reserved_mib=max(r['peak_reserved_mib'] for r in rows),
        scope='Seven deterministic spread full inputs and7 warmed synchronized repeats. One candidate in this process; no other GPU process allowed. Host/other-GPU contention is possible. Not full-dataset mean or independent accuracy validation.',
        boundary='Decode/model loading/text encoding excluded and reported separately. Includes resizing/all encoding/semantic readout/alias aggregation/writeback/original-size restoration/CPU argmax output.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-root','suite-root','original-cache','dinov3-repo','checkpoint-dir','upstream-root','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--cost-mode',choices=('Retained','Uniform','VIPMatched','VIPOfficial','VIPOfficialProtocol'),required=True)
    p.add_argument('--samples',type=int,default=7);p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--device',default='cuda');p.add_argument('--mode',default='full')
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20260923);p.add_argument('--vdd-ontology',default='official')
    main(p.parse_args())
