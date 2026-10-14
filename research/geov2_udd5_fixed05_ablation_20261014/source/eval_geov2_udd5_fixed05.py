"""Full UDD5 component ablation at a frozen g0=.5; no adaptive gain.

All readout controls share observations. Feature interventions recompute their
own observations. Masks are read only after every prediction for that image.
Cost mode constructs one arm per fresh process and never shares image caches.
"""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.compact_deployment import compact_observations, CachedWideScorer
from dinotool.context_memory_deployment import memory_wide_observations
from dinotool.context_memory_observer import ContextMemoryConfig, ContextMemoryObserver, CompactContextScorer
from dinotool.efficient_deployment import EfficientGeometry, GroupedLocalScorer, reconstruction, write_dense, restore_and_finalize
from dinotool.frozen_deployment import FrozenGeometry
from dinotool.natural_static_search import StaticReader
from dinotool.query_slot_memory import query_slot_wide_observations
from dinotool.tcpr import TCPRTextBank
from dinotool.vip_official_adapter import VIPQueries
from eval_alias_finalization import load_samples
from eval_development_readout import confusion
from eval_geov2_g0_sweep import CountNormalizedWide, canonical_bank_query, metric, write


# source, language, fusion. beta=0 isolates routing without changing where
# coordinate features are sampled or whether scoring precedes interpolation.
ARMS = {
    'Full': ('full', 'retained', 'coupled'),
    'NoDonorBefore': ('no_donor', 'retained', 'coupled'),
    'NoQueryRouting': ('no_route', 'retained', 'coupled'),
    'NoDonorNoRouting': ('neither', 'retained', 'coupled'),
    'ShuffledQuery': ('shuffled', 'retained', 'coupled'),
    'CanonicalForeground': ('full', 'canonical_fg', 'coupled'),
    'CanonicalAll': ('full', 'canonical', 'coupled'),
    'UnscreenedAliases': ('full', 'pool', 'coupled'),
    'WideCountNormalized': ('full', 'normalized', 'coupled'),
    'EqualLogitFusion': ('full', 'retained', 'equal_mean'),
    'LocalOnly': ('full', 'retained', 'local'),
    'WideOnly': ('full', 'retained', 'wide'),
    'Raw24CoordinateGrid': ('raw24', 'retained', 'coupled'),
    'Raw32CoordinateGrid': ('raw32', 'retained', 'coupled'),
    'LegacyGeometryVIPWide': ('legacy_geometry', 'retained', 'coupled'),
    'LegacyDonorVIPWide': ('legacy_donor', 'retained', 'coupled'),
}


def subset_pair(bank, query, keep):
    aliases = tuple(a for a,k in zip(bank.alias_names, keep.tolist()) if k)
    b = TCPRTextBank(bank.features[keep], bank.parent_indices[keep], bank.canonical_mask[keep], bank.class_names, aliases)
    q = VIPQueries(query.features[keep], query.parents[keep], query.class_names, aliases)
    b.validate()
    return b,q


def pool_pair(path, reference):
    packed = torch.load(path, map_location='cpu', weights_only=True)
    record = packed['identity']['banks']['pool_segmentation']
    row = packed['encoded']['pool_segmentation']
    wide = packed['stores'][row['template']][row['indices']]
    aliases = tuple(a for c in record['classes'] for a in c['synonyms'])
    names = tuple(c['name'] for c in record['classes'])
    if names != reference.class_names or record['template'] != 'seg_template':
        raise ValueError('Unscreened pool changed class mapping or templates.')
    # In this frozen pool background is identical to the retained 20 concepts.
    bg_aliases = tuple(a for a,p in zip(aliases,row['parents'].tolist()) if p==4)
    if bg_aliases != tuple(a for a,p in zip(reference.alias_names,reference.parent_indices.tolist()) if p==4):
        raise ValueError('Background differs in the screening control.')
    device = reference.features.device
    parents = row['parents'].to(device)
    b = TCPRTextBank(F.normalize(wide.float().mean(1),dim=-1).to(device),parents,row['canonical'].to(device),names,aliases)
    b.validate()
    return b, VIPQueries(wide.to(device),parents,names,aliases)


class Experiment:
    def __init__(self, protocol, methods):
        from dinov3.eval.text import tokenizer
        tokenizer.get_tokenizer = lambda *_a,**_k: tokenizer.Tokenizer(
            vocab_path=str(Path(protocol['checkpoint_dir'])/'bpe_simple_vocab_16e6.txt.gz'))
        self.protocol = protocol
        self.model = m = FrozenGeometry(protocol['config_path'], checkpoint_dir=protocol['checkpoint_dir'],
            dinov3_repo=protocol['dinov3_repo'], upstream_root=protocol['upstream_root'],
            text_cache=protocol['text_cache'],efficient=True,wide_branch='query_slot',query_slot_capacity=24)
        if m.strengths != ('original',) or m.config['background_index'] != 4:
            raise ValueError('Require active DonorBefore and the frozen five-class protocol.')
        m.profile = replace(m.profile,coupling=.5)
        self.plain = EfficientGeometry(m.geometry)
        self.observers = {24:m.query_slot_observer}
        if 'Raw32CoordinateGrid' in methods:
            self.observers[32] = ContextMemoryObserver(m.vip.backbone,
                ContextMemoryConfig(capacity=32,temperature=.1,iterations=5,center=False,hard=False))
        bank_name = m.profile.bank
        bank, query = m.banks[bank_name], m.queries[bank_name]
        self.readers = {'retained':m.reader,'normalized':m.reader}
        self.scorers = {'retained':m.efficient_local_scorers,'normalized':m.efficient_local_scorers}
        needed = {ARMS[name][1] for name in methods}
        pairs = {}
        if 'canonical' in needed:
            pairs['canonical'] = canonical_bank_query(bank,query)
        if 'canonical_fg' in needed:
            pairs['canonical_fg'] = subset_pair(bank,query,bank.canonical_mask | (bank.parent_indices==4))
        if 'pool' in needed:
            pairs['pool'] = pool_pair(protocol['pool_text_cache'],bank)
        for mode,(b,q) in pairs.items():
            self.readers[mode] = StaticReader({bank_name:b},{bank_name:q},4)
            self.scorers[mode] = {bank_name:GroupedLocalScorer(b)}
        self.wide_scorer = CompactContextScorer()
        self.normalized = CountNormalizedWide(self.wide_scorer)
        self.legacy_scorer = CachedWideScorer()
        self.alias_counts = {mode:torch.bincount(reader.banks[bank_name].parent_indices).cpu().tolist()
                             for mode,reader in self.readers.items()}
        if len(methods)==1:
            # Do not charge a language ablation for other banks co-residing.
            language=ARMS[methods[0]][1]
            selected=self.readers[language]
            self.readers={language:selected}
            self.scorers={language:self.scorers[language]}
            m.reader,m.banks,m.queries=selected,selected.banks,selected.queries
            m.efficient_local_scorers=self.scorers[language]

    def source(self,image,kind,*,solve=True):
        m = self.model
        solver = reconstruction if solve else lambda _a,_v:(None,None)
        geometry = self.plain if kind in ('no_donor','neither','legacy_geometry') else m.compact_geometry
        if kind.startswith('legacy'):
            return compact_observations(image,geometry,m.vip,('original',),
                wide_policy=m.profile.wide_policy,reconstruction=solver)
        if kind.startswith('raw'):
            from dinotool.compact_deployment import local_observations
            s = local_observations(image,geometry,('original',),reconstruction=solver)
            return dict(s,**memory_wide_observations(image,self.observers[int(kind[3:])],wide_policy=m.profile.wide_policy))
        return query_slot_wide_observations(image,geometry,self.observers[24],('original',),
            wide_policy=m.profile.wide_policy,temperature=.1,
            beta=0. if kind in ('no_route','neither') else .5,
            reconstruction=solver,shuffle_query=kind=='shuffled')

    def read(self,source,name,cache=None):
        kind,language,fusion = ARMS[name]
        m = self.model
        scorer = self.legacy_scorer if kind.startswith('legacy') else self.normalized if language=='normalized' else self.wide_scorer
        return self.readers[language].probabilities(source,m.profile,{} if cache is None else cache,
            wide_scorer=scorer,fusion=fusion,dense_writer=write_dense,local_scorers=self.scorers[language],
            output_finalizer=lambda normalized,size:restore_and_finalize(normalized,size,
                m.config['background_bias'],m.config['background_threshold'],4))

    def predict(self,image,name):
        kind,_,fusion = ARMS[name]
        source = self.source(image,kind,solve=fusion=='coupled')
        return self.read(source,name)


@torch.inference_mode()
def main(args):
    root = Path(args.experiment)
    protocol = json.loads((root/'protocol.json').read_text())
    methods = list(ARMS) if args.mode!='cost' else [args.arm]
    output = root/args.mode/(f's{args.shard_index}' if args.mode=='full' else args.arm if args.mode=='cost' else 'check')
    if output.exists():
        raise RuntimeError('Preserve existing output: '+str(output))
    output.mkdir(parents=True)
    experiment = Experiment(protocol,methods)
    m = experiment.model
    args.dataset,args.data_root = 'udd5',protocol['data_root']
    args.sample_seed,args.vdd_ontology = 20260923,'official'
    entry = dict(m.config,banks={'original':next(iter(m.vocabulary['banks'].values()))})
    validation,samples,images,masks,names_map = load_samples(args,entry)
    if len(validation)!=40:
        raise ValueError('UDD5 validation coverage changed.')
    common = dict(dataset='udd5',g0=.5,profile=m.profile.record(),config=m.config,
                  alias_counts=experiment.alias_counts,methods=methods,arms=ARMS,
                  cache_identity=m.cache_identity,adaptive_gain_enabled=False)
    if args.mode=='cost':
        rows=[]
        lookup={s.key:s for s in validation}
        for key in protocol['cost_keys']:
            image=images(lookup[key])
            for _ in range(2):
                experiment.predict(image,args.arm)
            times,allocated,reserved=[],[],[]
            for _ in range(7):
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started=time.perf_counter()
                result=experiment.predict(image,args.arm)
                torch.cuda.synchronize()
                times.append((time.perf_counter()-started)*1000)
                allocated.append(torch.cuda.max_memory_allocated()/1048576)
                reserved.append(torch.cuda.max_memory_reserved()/1048576)
            if result.dtype!=np.uint8 or result.shape!=tuple(image.shape[-2:]):
                raise RuntimeError('Cost output shape/dtype differs.')
            rows.append(dict(key=key,shape=list(image.shape[-2:]),milliseconds=times,
                peak_allocated_mib=max(allocated),peak_reserved_mib=max(reserved)))
            print(json.dumps(dict(arm=args.arm,cost_images=len(rows))),flush=True)
        write(output/'results.json',dict(status='complete',**common,images=rows,
            median_ms=float(np.median([t for row in rows for t in row['milliseconds']])),
            peak_allocated_mib=max(row['peak_allocated_mib'] for row in rows),
            peak_reserved_mib=max(row['peak_reserved_mib'] for row in rows),
            gpu=torch.cuda.get_device_name(),target_masks_loaded=False,
            boundary='CPU float RGB to original-size CPU uint8 IDs; decode/init/offline text excluded; fresh process per arm; seven fixed images, two warmups and seven synchronized repeats each. Unused H solve skipped for equal fusion.'))
        return
    if args.mode=='smoke':
        samples=samples[:1]
    names=names_map['udd5']
    matrices=np.zeros((len(methods),len(names),len(names)),dtype=np.int64)
    cms_per_image,keys=[],[]
    started=time.perf_counter()
    for number,sample in enumerate(samples,1):
        image=images(sample)
        predictions={}
        smoke_check=None
        for kind in dict.fromkeys(ARMS[name][0] for name in methods):
            source=experiment.source(image,kind)
            caches={}
            for name in methods:
                if ARMS[name][0]!=kind:
                    continue
                cache=caches.setdefault(ARMS[name][1],{})
                predictions[name]=experiment.read(source,name,cache)
            if args.mode=='smoke' and kind=='full':
                # Slot construction uses CUDA index_add reductions, whose
                # order can vary between re-encodings. Verify the execution
                # interface using exactly the same observations first.
                import dinotool.query_slot_memory as qs
                original=qs.query_slot_wide_observations
                qs.query_slot_wide_observations=lambda *_a,**_k:source
                try:
                    direct=m.predict(image)
                finally:
                    qs.query_slot_wide_observations=original
                smoke_check=dict(same_observation_frozen_vs_full_pixels=int((direct!=predictions['Full']).sum()),
                                 shape=list(direct.shape))
                if smoke_check['same_observation_frozen_vs_full_pixels']:
                    raise RuntimeError('Full readout differs from FrozenGeometry.predict on identical observations.')
            del source,caches
        if args.mode=='smoke':
            repeated=experiment.predict(image,'Full')
            smoke_check['reencoded_vs_full_pixels']=int((repeated!=predictions['Full']).sum())
            smoke_check['reencoded_changed_fraction']=float((repeated!=predictions['Full']).mean())
            write(output/'prediction_check.json',smoke_check)
            print(json.dumps(smoke_check),flush=True)
            if smoke_check['reencoded_changed_fraction']>.001:
                raise RuntimeError('Unexpectedly large repeat-inference variation.')
        target=masks(sample,'udd5',tuple(image.shape[-2:]))
        cms=np.stack([confusion(predictions[name],target,len(names)) for name in methods])
        if not np.all(cms.sum(2)==cms[0].sum(1)[None]):
            raise RuntimeError('Target supports differ.')
        matrices+=cms
        cms_per_image.append(cms)
        keys.append(sample.key)
        if number==1 or number%5==0 or number==len(samples):
            progress=dict(shard=args.shard_index,processed=number,total=len(samples))
            write(output/'progress.json',progress)
            print(json.dumps(progress),flush=True)
    np.savez_compressed(output/'confusions.npz',sample_keys=np.asarray(keys),confusions=np.stack(cms_per_image))
    write(output/'results.json',dict(status='complete',**common,
        processed_images=len(keys),global_sample_count=len(validation),sample_keys=keys,
        shard_index=args.shard_index,num_shards=args.num_shards,class_names=names,
        metrics={name:dict(mean_iou_percent=metric(cm),confusion_matrix=cm.tolist()) for name,cm in zip(methods,matrices)},
        masks_used_only_for_scoring=True,wall_seconds=time.perf_counter()-started,
        smoke_matches_frozen_predict=True if args.mode=='smoke' else None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',required=True)
    p.add_argument('--mode',choices=('smoke','full','cost'),required=True)
    p.add_argument('--arm',choices=tuple(ARMS),default='Full')
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    main(p.parse_args())
