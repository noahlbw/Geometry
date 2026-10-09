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
    """No search or labels; explicit frozen profile and optional compact execution."""
    @torch.inference_mode()
    def __init__(self,config_path,*,checkpoint_dir,dinov3_repo,upstream_root,text_cache,device='cuda',compact=True):
        config_path=Path(config_path)
        self.config=yaml.safe_load(config_path.read_text(encoding='utf-8'))
        self.vocabulary=json.loads((config_path.parent/self.config['vocabulary']).read_text(encoding='utf-8'))
        c=self.config; self.compact=compact
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
        self.reader=StaticReader(banks,queries,c['background_index'],residual)
        self.strengths=self.previous.strengths if self.previous else (self.profile.strength,)
        self.compact_geometry=CompactGeometry(self.geometry);self.wide_scorer=CachedWideScorer()
        if compact:
            offload_text_towers(self.geometry,self.vip,release=True)
        self.cache_identity=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()

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
            reader=self.foreground_reader; background=None
        elif protocol not in (None,'D'):
            raise ValueError('Unknown prediction protocol.')
        if self.compact:
            source=compact_observations(image,self.compact_geometry,self.vip,self.strengths,wide_policy=self.profile.wide_policy)
            probability=(self.previous.predict_observations(source,return_probability=True,wide_scorer=self.wide_scorer,return_prediction=False)[2]
                if self.previous else reader.probabilities(source,self.profile,{},wide_scorer=self.wide_scorer))
        else:
            old=base.projected;base.projected=projected
            try:source=base.observations(image,self.geometry,self.vip,self.strengths,wide_policy=self.profile.wide_policy)
            finally:base.projected=old
            probability=self.previous.predict_observations(source,return_probability=True)[2] if self.previous else reader.probabilities(source,self.profile,{})
        c=self.config
        if probability.shape[0]>256:raise ValueError('PNG/byte prediction requires at most256 classes.')
        probability=biased_probabilities(probability,c['background_bias'] if background is not None else 0.,background)
        confidence,prediction=probability.max(0)
        if background is not None:
            prediction=prediction.masked_fill(confidence<c['background_threshold'],background)
        return prediction.to(torch.uint8).cpu().numpy()
