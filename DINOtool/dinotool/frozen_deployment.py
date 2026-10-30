"""Portable frozen model: explicit YAML, words and ordinary local text caches."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .config import CheckpointConfig
from .model import DINOTextSegmenter
from .tcpr import TCPRConfig,TCPRSegmenter,TCPRTextBank
from .geometry_execution import GeometryExecution
from .finite_vip_observer import FiniteVIPObserver
from .vip_official_adapter import VIPQueries,_load_upstream
from .natural_static_search import StaticProfile,StaticReader,projected
from .taxonomy_inference import TaxonomyInference
from .prompts import REMOTE_SENSING_TEMPLATES
from .compact_deployment import CompactGeometry,compact_observations,CachedWideScorer,cached_wide,offload_text_towers


class FrozenGeometry:
    """No search or labels; explicit frozen profile and selectable wide branch.

    ``wide_branch='vip'`` preserves the incumbent crop-based reader.  The
    compact ``raw32`` and ``query_slot`` branches are opt-in at this API and
    CLI level; changing the wide branch must never happen implicitly.
    """
    @torch.inference_mode()
    def __init__(self,config_path,*,checkpoint_dir,dinov3_repo,upstream_root,text_cache,device='cuda',compact=True,efficient=False,
                 wide_branch='vip', query_slot_capacity=24, writeback='coupled', shared_visual=False,
                 wide_alias_aggregation='logsumexp'):
        config_path=Path(config_path)
        self.config=yaml.safe_load(config_path.read_text(encoding='utf-8'))
        self.vocabulary=json.loads((config_path.parent/self.config['vocabulary']).read_text(encoding='utf-8'))
        c=self.config; self.compact=compact; self.efficient=efficient
        self.shared_visual=bool(shared_visual)
        if wide_alias_aggregation not in ('logsumexp', 'logmeanexp'):
            raise ValueError("wide_alias_aggregation must be 'logsumexp' or 'logmeanexp'.")
        self.wide_alias_aggregation=wide_alias_aggregation
        if self.shared_visual and (wide_branch!='query_slot' or not compact or not Path(text_cache).exists()):
            raise ValueError('Shared visual storage requires compact query_slot and an existing frozen text cache.')
        if writeback not in ('coupled', 'adaptive', 'equal_mean', 'local', 'wide', 'fixed_g2'):
            raise ValueError('Unknown writeback rule.')
        self.writeback = writeback
        if wide_branch not in ('vip', 'raw32', 'query_slot'):
            raise ValueError('wide_branch must be vip, raw32, or query_slot.')
        if query_slot_capacity < 1:
            raise ValueError('query_slot_capacity must be positive.')
        if wide_branch == 'query_slot' and not compact:
            raise ValueError('query_slot wide_branch requires compact=True.')
        if efficient and not compact:
            raise ValueError('Efficient execution requires the compact backend.')
        import subprocess
        commit=subprocess.check_output(['git','-C',str(upstream_root),'rev-parse','HEAD'],text=True).strip()
        if commit!=c['upstream']['vip_commit']:
            raise ValueError('Use the pinned VIP revision in the YAML.')
        root=Path(checkpoint_dir)
        checkpoints=CheckpointConfig(Path(dinov3_repo),root,
            root/'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth',
            root/'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
            root/'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',root/'bpe_simple_vocab_16e6.txt.gz')
        visual_only=compact and Path(text_cache).exists()
        self.geometry=GeometryExecution(TCPRSegmenter(DINOTextSegmenter(checkpoints,device=device,visual_only=visual_only),TCPRConfig(**c['geometry'])))
        if self.shared_visual:
            # QuerySlot only needs a visual backbone/head and its cached text
            # interface. Reuse the already loaded frozen local visual model;
            # never construct/cast a second visual model on CUDA.
            from types import SimpleNamespace
            self.vip=SimpleNamespace(backbone=self.geometry.backbone,device=self.geometry.device)
        else:
            self.vip=FiniteVIPObserver(DINOTextSegmenter(checkpoints,device=device,visual_only=visual_only),Path(upstream_root))
        from .model import checkpoint_manifest
        identity=dict(vocabulary=self.vocabulary,checkpoints=checkpoint_manifest(checkpoints),vip_commit=commit)
        cache=Path(text_cache)
        if cache.exists():
            packed=torch.load(cache,map_location='cpu',weights_only=True)
            if packed['identity']!=identity:
                raise ValueError('Text cache does not match vocabulary/checkpoints. Use a new cache filename.')
        else:
            packed=dict(identity=identity,encoded={})
            prompts=_load_upstream(Path(upstream_root)/'prompts/imagenet_template.py','frozen_deployment_prompts')
            for name,record in self.vocabulary['banks'].items():
                groups=tuple(tuple(row['synonyms']) for row in record['classes'])
                names=tuple(row['name'] for row in record['classes'])
                template=record['template']
                self.vip.templates=prompts.get_text_template('openai_imagenet_template' if template in ('ImageNet80','RS6') else template)
                query=self.vip.encode_queries(names,groups)
                if template=='RS6':
                    local,parents,canonical=self.geometry.backbone.encode_text_aliases(groups,templates=REMOTE_SENSING_TEMPLATES)
                else:
                    # CPU reduction matches the finalized variable-bank builder.
                    local=F.normalize(query.features.cpu().float().mean(1),dim=-1).to(device)
                    parents=query.parents
                    canonical=torch.zeros(len(query.aliases),dtype=torch.bool,device=device)
                    canonical[torch.tensor(np.cumsum([0]+[len(g) for g in groups[:-1]]),device=device)]=True
                packed['encoded'][name]=dict(local=local.cpu(),wide=query.features.cpu(),parents=parents.cpu(),canonical=canonical.cpu())
            residual=self.vocabulary.get('residual')
            if residual:
                self.vip.templates=prompts.get_text_template('openai_imagenet_template' if residual['template']=='ImageNet80' else residual['template'])
                groups=tuple((name,) for name in residual['names'])
                packed['residual']=self.vip.encode_queries(tuple(residual['names']),groups).features.cpu()
            cache.parent.mkdir(parents=True,exist_ok=True)
            torch.save(packed,cache.with_suffix('.tmp')); cache.with_suffix('.tmp').replace(cache)
        banks,queries={},{}
        for name,record in self.vocabulary['banks'].items():
            row=packed['encoded'][name]
            names=tuple(c['name'] for c in record['classes']);aliases=tuple(a for c in record['classes'] for a in c['synonyms'])
            parents=row['parents'].to(device)
            banks[name]=TCPRTextBank(row['local'].to(device),parents,row['canonical'].to(device),names,aliases)
            banks[name].validate()
            queries[name]=VIPQueries(row['wide'].to(device),parents,names,aliases)
        residual=packed.get('residual');residual=None if residual is None else residual.to(device)
        self.banks,self.queries=banks,queries
        self.profile=StaticProfile(**c['profile'])
        self.previous=(TaxonomyInference.for_finalization(self.geometry,self.vip,banks,queries,
            ordinary_alias_policy='uniform',family=c['family'],background=c['background_index'],residual_features=residual)
            if self.profile.bank=='__previous_final__' else None)
        if self.previous and writeback != 'coupled':
            raise ValueError('Alternate writeback requires an explicit StaticProfile.')
        self.reader=StaticReader(banks,queries,c['background_index'],residual)
        self.strengths=self.previous.strengths if self.previous else (self.profile.strength,)
        self.compact_geometry=CompactGeometry(self.geometry);self.wide_scorer=CachedWideScorer()
        self.wide_branch='vip'
        self.local_branch='geometry'
        if efficient:
            self.enable_efficient()
        if compact:
            offload_text_towers(self.geometry,self.vip,release=True)
        if wide_branch == 'raw32':
            self.enable_context_memory()
        elif wide_branch == 'query_slot':
            self.enable_query_slot(capacity=query_slot_capacity)
        self.cache_identity=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()

    def enable_efficient(self, *, grouped=True):
        from .efficient_deployment import EfficientGeometry, EfficientVIPObserver, GroupedWideScorer, GroupedLocalScorer
        if self.previous:
            raise ValueError('Efficient backend currently requires an explicit frozen StaticProfile.')
        self.efficient=True
        if hasattr(self, 'query_slot_observer'):
            from .response_geometry_deployment import EfficientResponseGeometry
            self.compact_geometry=EfficientResponseGeometry(self.geometry)
        else:
            self.compact_geometry=EfficientGeometry(self.geometry)
        if not self.shared_visual:
            self.vip.__class__=EfficientVIPObserver
        self.wide_scorer=GroupedWideScorer() if grouped else CachedWideScorer()
        self.efficient_local_scorers={k:GroupedLocalScorer(v) for k,v in self.banks.items()} if grouped else self.reader.scorers
        if hasattr(self,'context_memory'):
            from .context_memory_observer import CompactContextScorer
            from .view_disjoint_deployment import ViewDisjointScorer
            kwargs={'aggregation':self.wide_alias_aggregation}
            self.wide_scorer=ViewDisjointScorer(**kwargs) if getattr(self,'context_memory_donors','all')=='outside' else CompactContextScorer(**kwargs)
        elif hasattr(self, 'query_slot_observer'):
            from .context_memory_observer import CompactContextScorer
            self.wide_scorer=CompactContextScorer(aggregation=self.wide_alias_aggregation)

    def enable_context_memory(self, config=None, *, donor_view='all', context_depth='both', context_mode='memory'):
        """Select raw32 whole-image memory, optionally excluding own-view donors.

        Outside donors retain their original full-memory probability; removed
        probability returns to query self in both blocks or only the last.
        """
        from .context_memory_observer import ContextMemoryObserver, ContextMemoryConfig, CompactContextScorer
        if hasattr(self, 'query_slot_observer'):
            raise ValueError('Context memory and query-slot routing are mutually exclusive deployment modes.')
        if not self.compact or self.previous or self.reader.residual_features is not None:
            raise ValueError('Context memory requires compact StaticProfile without a protected residual reader.')
        if donor_view not in ('all','outside') or context_depth not in ('both','final'):
            raise ValueError('Use all/outside donor_view and both/final context_depth.')
        if context_mode not in ('memory','response_before'):
            raise ValueError('Use memory or response_before context_mode.')
        if context_mode == 'response_before' and (donor_view != 'all' or context_depth != 'both'):
            raise ValueError('response_before currently requires all donors and both blocks.')
        self.context_memory = ContextMemoryObserver(self.vip.backbone,
            ContextMemoryConfig(capacity=32, temperature=.1, iterations=5, center=False, hard=False)
            if config is None else config)
        self.context_memory_donors,self.context_memory_depth,self.context_memory_mode = donor_view,context_depth,context_mode
        self.wide_branch='raw32'
        self.local_branch='geometry'
        from .view_disjoint_deployment import ViewDisjointScorer
        kwargs={'aggregation':self.wide_alias_aggregation}
        self.wide_scorer = ViewDisjointScorer(**kwargs) if donor_view=='outside' else CompactContextScorer(**kwargs)
        return self

    def enable_query_slot(self, config=None, *, beta=.5, temperature=.1, capacity=24):
        """Use local Geometry tokens to route a single whole-image slot bank.

        The wide image is encoded once by the existing frozen context observer.
        Each local Geometry tile then chooses a different mixture of the shared
        semantic slots.  ``ResponseGeometry`` applies the tested DonorBefore
        local ordering for the original strength; numeric strength profiles
        retain the incumbent Geometry projection.
        """
        from .context_memory_observer import ContextMemoryConfig, ContextMemoryObserver, CompactContextScorer
        from .response_geometry_deployment import ResponseGeometry, EfficientResponseGeometry
        if hasattr(self, 'context_memory'):
            raise ValueError('Context memory and query-slot routing are mutually exclusive deployment modes.')
        if not self.compact or self.previous:
            raise ValueError('Query-slot routing requires a compact StaticProfile.')
        if not 0 <= beta <= 1 or temperature <= 0:
            raise ValueError('beta must be in [0,1] and temperature must be positive.')
        if config is None and capacity < 1:
            raise ValueError('capacity must be positive.')
        self.query_slot_observer = ContextMemoryObserver(
            self.vip.backbone,
            ContextMemoryConfig(capacity=int(capacity), temperature=.1, iterations=5, center=False, hard=False)
            if config is None else config,
        )
        self.query_slot_capacity = int(self.query_slot_observer.config.capacity)
        self.query_slot_beta = float(beta)
        self.query_slot_temperature = float(temperature)
        self.compact_geometry = (EfficientResponseGeometry if self.efficient else ResponseGeometry)(self.geometry)
        self.wide_scorer = CompactContextScorer(aggregation=self.wide_alias_aggregation)
        self.wide_branch='query_slot'
        self.local_branch='donor_before'
        return self

    @torch.inference_mode()
    def predict(self,image,*,protocol=None):
        import eval_development_readout as base
        from .development_readout import biased_probabilities
        reader=self.reader
        background=self.config['background_index']
        if protocol=='P':
            if self.config['dataset']!='loveda' or self.previous:
                raise ValueError('The independent P reader is only defined for LoveDA.')
            from eval_alias_finalization import foreground_banks
            if not hasattr(self,'foreground_reader'):
                bs,qs=foreground_banks(self.banks,self.queries)
                self.foreground_reader=StaticReader(bs,qs,None,None)
                from .efficient_deployment import GroupedLocalScorer
                self.foreground_local_scorers={k:GroupedLocalScorer(v) for k,v in bs.items()}
            reader=self.foreground_reader; background=None
        elif protocol not in (None,'D'):
            raise ValueError('Unknown prediction protocol.')
        if self.compact:
            from .efficient_deployment import reconstruction, write_dense, restore_and_finalize
            observation_function=compact_observations
            wide_observer=self.vip
            observation_options={}
            if hasattr(self, 'query_slot_observer'):
                from .query_slot_memory import query_slot_wide_observations
                observation_function = query_slot_wide_observations
                wide_observer = self.query_slot_observer
                observation_options.update(
                    temperature=self.query_slot_temperature,
                    beta=self.query_slot_beta,
                )
            elif hasattr(self,'context_memory'):
                from .context_memory_deployment import context_memory_observations
                observation_function=context_memory_observations
                wide_observer=self.context_memory
                if getattr(self,'context_memory_mode','memory') == 'response_before':
                    from .context_memory_deployment import response_before_observations
                    observation_function=response_before_observations
                if getattr(self,'context_memory_donors','all')=='outside':
                    from .view_disjoint_deployment import view_disjoint_observations
                    observation_function=view_disjoint_observations
                    observation_options['final_only']=self.context_memory_depth=='final'
            source=observation_function(image,self.compact_geometry,wide_observer,self.strengths,wide_policy=self.profile.wide_policy,
                reconstruction=reconstruction if self.efficient else None,**observation_options)
            probability=(self.previous.predict_observations(source,return_probability=True,wide_scorer=self.wide_scorer,return_prediction=False)[2]
                if self.previous else reader.probabilities(source,self.profile,{},wide_scorer=self.wide_scorer,
                    fusion=self.writeback, dense_writer=write_dense if self.efficient else None,
                    local_scorers=(self.foreground_local_scorers if protocol=='P' else self.efficient_local_scorers) if self.efficient else None,
                    output_finalizer=(lambda normalized,size:restore_and_finalize(normalized,size,self.config['background_bias'],
                        self.config['background_threshold'],background)) if self.efficient else None))
        else:
            old=base.projected;base.projected=projected
            try:source=base.observations(image,self.geometry,self.vip,self.strengths,wide_policy=self.profile.wide_policy)
            finally:base.projected=old
            probability=self.previous.predict_observations(source,return_probability=True)[2] if self.previous else reader.probabilities(source,self.profile,{},fusion=self.writeback)
        c=self.config
        if isinstance(probability,np.ndarray):return probability
        if probability.shape[0]>256:raise ValueError('PNG/byte prediction requires at most256 classes.')
        if self.efficient:
            from .efficient_deployment import finalize
            return finalize(probability,c['background_bias'],c['background_threshold'],background)
        probability=biased_probabilities(probability,c['background_bias'] if background is not None else 0.,background)
        confidence,prediction=probability.max(0)
        if background is not None:
            prediction=prediction.masked_fill(confidence<c['background_threshold'],background)
        return prediction.to(torch.uint8).cpu().numpy()
