"""Mask-free inference API for frozen taxonomy readout and optional soft weights."""
import torch
import torch.nn.functional as F
from .packed_alias_readout import PackedAliasReadout
from .taxonomy_readout import natural_profile,rs_profile,canonical_features
from .residual_rival_weight import ResidualRivalWeight
from .device_probability_accumulator import DeviceProbabilityAccumulator
from .inference import hann_blend_window
from .development_readout import coupled


class TaxonomyInference:
    """Uses already loaded frozen Geometry/VIP models and already encoded banks.

    Repository scripts must be on PYTHONPATH, as for the retained bounded model.
    Construction caches static alias metadata; predict never receives masks.
    """

    @classmethod
    def for_finalization(cls,geometry,vip,banks,queries,*,ordinary_alias_policy,family,
                         background=None,residual_features=None,local_background='retained'):
        """Compare one global simplification without changing the incumbent API.

        `uniform` disables only the extra foreground coverage-soft rule. Local
        LME, inherited image salience, and an explicitly supplied residual
        ontology/protected residual rule remain unchanged. This entry does not
        claim that uniform aggregation gives every word equal actual influence.
        """
        if ordinary_alias_policy not in ('retained','uniform'):
            raise ValueError('Declare retained or uniform ordinary alias policy.')
        model=cls.for_task(geometry,vip,banks,queries,family=family,background=background,
            residual_features=residual_features,local_background=local_background)
        if ordinary_alias_policy=='uniform':
            model.soft=False
        return model

    @classmethod
    def for_task(cls,geometry,vip,banks,queries,*,family,background=None,residual_features=None,local_background='retained'):
        """Selected task-conditioned candidate, distinct from historical defaults.

        No dataset-name dispatch: caller declares view family and supplies the
        taxonomy/optional residual concepts. Prior development provenance remains.
        """
        return cls(geometry,vip,banks,queries,family=family,background=background,
            soft=family=='remote_sensing',residual_features=residual_features,
            residual_mode='protected' if residual_features is not None else 'max',
            correction='frozen',wide_aggregation='inherited',read_policy='retained',
            wide_policy='natural_short336_cap672' if family=='natural' else 'long448',local_background=local_background)

    @torch.inference_mode()
    def __init__(self,geometry,vip,banks,queries,*,family,background=None,soft=True,residual_features=None,residual_mode='max',correction='frozen',wide_aggregation='inherited',wide_policy='long448',read_policy='retained',local_background='retained'):
        if local_background not in ('retained','union_max'):
            raise ValueError('Declared retained or union-max background aggregation required.')
        self.local_background=local_background
        if read_policy not in ('retained','natural_strength3'):
            raise ValueError('Declared retained or natural-strength3 read policy required.')
        self.read_policy=read_policy
        if wide_policy not in ('long448','natural_short336_cap672') or (wide_policy!='long448' and family!='natural'):
            raise ValueError('The short-edge policy is opt-in for declared natural tasks only.')
        self.wide_policy=wide_policy
        if wide_aggregation not in ('inherited','count_normalized','bounded_rival','semantic_mass'):
            raise ValueError('Declared inherited, count-normalized or bounded-rival aggregation required.')
        self.wide_aggregation=wide_aggregation
        if correction not in ('frozen','envelope','geometry_consistency','local_endpoint','wide_endpoint','equal_mean','two_witness'):
            raise ValueError('Declared frozen, envelope or consistency correction required.')
        self.correction=correction
        if family not in ('natural','remote_sensing'):
            raise ValueError('Declared natural/remote_sensing task required.')
        needed=('semantic_segmentation',) if family=='natural' else ('original_imagenet','focused20')
        if any(k not in banks or k not in queries for k in needed):
            raise ValueError('The frozen task candidate banks must be encoded first.')
        classes=[tuple(banks[k].class_names) for k in needed]
        if any(c!=classes[0] for c in classes):
            raise ValueError('Candidate banks must have identical class order.')
        if background is not None and not 0<=background<len(classes[0]):
            raise ValueError('Residual index outside semantic taxonomy.')
        if residual_mode not in ('max','protected') or (residual_mode=='protected' and residual_features is None):
            raise ValueError('Residual mode must be max or protected with encoded residual queries.')
        if residual_features is not None:
            if family!='natural' or background is None or soft:
                raise ValueError('Residual maximum requires natural taxonomy, background index and uniform foreground weights.')
            reference=banks['semantic_segmentation'].features
            if (residual_features.ndim!=3 or min(residual_features.shape[:2])==0
                    or residual_features.shape[-1]!=reference.shape[-1]
                    or residual_features.device!=reference.device
                    or not bool(torch.isfinite(residual_features).all())):
                raise ValueError('Residual features must be finite [query,template,dimension] on the bank device.')
        self.geometry,self.vip=geometry,vip
        self.banks,self.queries=banks,queries
        self.family,self.background,self.soft=family,background,soft
        self.residual_features=residual_features
        self.residual_text=None if residual_features is None else F.normalize(residual_features.float().mean(1),dim=-1)
        self.residual_mode=residual_mode
        self.residual_reader=ResidualRivalWeight(self.residual_text,canonical_features(banks['semantic_segmentation'])) if residual_mode=='protected' else None
        self.scorers={k:PackedAliasReadout(banks[k]) for k in needed}
        self.local_background_members={}
        if local_background=='union_max' and family=='natural' and background is not None and residual_features is None:
            for k in needed:
                members=torch.nonzero(banks[k].parent_indices==background,as_tuple=False).flatten()
                # Exact historical arithmetic retained for single-query controls.
                if len(members)>1:self.local_background_members[k]=members
        self.wide_normalizers={}
        if wide_aggregation=='count_normalized':
            from .count_normalized_wide import CountNormalizedWide
            self.wide_normalizers={k:CountNormalizedWide(banks[k].parent_indices,banks[k].class_count,
                keep_equal_count_gauge=residual_features is None) for k in needed}
        self.text={k:F.normalize(banks[k].features.float(),dim=-1) for k in needed}
        self.wide_rivals={}
        if wide_aggregation=='bounded_rival':
            from .bounded_wide_rival import BoundedWideRival
            self.wide_rivals={k:BoundedWideRival(queries[k],background) for k in needed}
        self.semantic_readers={}
        if wide_aggregation=='semantic_mass':
            from .semantic_mass_wide import SemanticMassWide
            self.semantic_readers={k:SemanticMassWide(queries[k],background) for k in needed}
        self.profile,self.decision=natural_profile(banks['semantic_segmentation']) if family=='natural' else (None,None)
        from .task_read_strength import select_read_strength
        self.profile,self.decision=select_read_strength(self.profile,self.decision,read_policy)
        self.strengths=(1.,3.,'original') if self.profile is None else (self.profile.strength,)
        self.blend=torch.from_numpy(hann_blend_window(512)).to(geometry.device)

    @torch.inference_mode()
    def predict(self,image,*,return_probability=False):
        import eval_development_readout as observation
        from eval_geometry_vip_reliability import sample_broad
        source=observation.observations(image,self.geometry,self.vip,self.strengths,wide_policy=self.wide_policy)
        return self.predict_observations(source,return_probability=return_probability)

    @torch.inference_mode()
    def predict_observations(self,source,*,return_probability=False):
        """Read already computed observations for paired frozen evaluations."""
        import eval_development_readout as observation
        from eval_geometry_vip_reliability import sample_broad
        profile,decision=rs_profile(source,self.banks,self.background) if self.profile is None else (self.profile,self.decision)
        bank=self.banks[profile.bank]
        broad=observation.wide_scores(source,self.queries[profile.bank],profile.tau,profile.tem)
        if self.wide_aggregation=='count_normalized':
            broad=self.wide_normalizers[profile.bank].apply(broad,profile.tau)
        old_broad=broad
        if self.wide_aggregation=='bounded_rival':
            broad=self.wide_rivals[profile.bank].score(source,self.queries[profile.bank],profile.tau,profile.tem)
        elif self.wide_aggregation=='semantic_mass':
            broad=self.semantic_readers[profile.bank].score(source,profile.tau,profile.tem)
        if self.residual_features is not None:
            from eval_residual_ontology import residual_wide
            broad=broad.clone()
            broad[self.background]=residual_wide(source,self.residual_features)['max'][0]
        h,w=source['size']
        text=self.text[profile.bank]
        scorer=self.scorers[profile.bank]
        with DeviceProbabilityAccumulator(bank.class_count,h,w,text.device) as accumulator:
            for tile in source['local']:
                alias=tile['features'][profile.strength].float()@text.T
                local=(scorer.uniform_reference(alias) if self.residual_reader is not None
                    else scorer.coverage(alias) if self.soft else scorer.uniform(alias))/profile.temperature
                if profile.bank in self.local_background_members:
                    local[:,self.background]=alias[:,self.local_background_members[profile.bank]].amax(-1)/profile.temperature
                top,left=tile['top'],tile['left']
                if self.residual_text is not None:
                    cosine=tile['features'][profile.strength].float()@self.residual_text.T
                    if self.residual_reader is None:
                        residual=cosine.amax(-1)
                    else:
                        previous=sample_broad(old_broad,top,left,h,w).reshape_as(local)
                        foreground=coupled(local,previous,tile['operator'],profile.coupling)
                        foreground[:,self.background]=-torch.inf
                        residual=self.residual_reader.score(cosine,foreground.argmax(-1))
                    local[:,self.background]=residual/profile.temperature
                wide=sample_broad(broad,top,left,h,w).reshape_as(local)
                if self.correction=='local_endpoint':
                    logits=local
                elif self.correction=='wide_endpoint':
                    logits=wide
                elif self.correction=='equal_mean':
                    logits=.5*(local.double()+wide.double())
                elif self.correction=='envelope':
                    from .evidence_envelope import envelope_coupled
                    logits=envelope_coupled(local,wide,tile['operator'],profile.coupling)
                elif self.correction=='geometry_consistency':
                    from .geometry_consistency_gain import consistency_coupled
                    logits=consistency_coupled(local,wide,tile['operator'],profile.coupling)
                elif self.correction=='two_witness':
                    from .two_witness_correction import two_witness_coupled
                    logits=two_witness_coupled(local,wide,tile['operator'],profile.coupling,self.background)
                else:
                    logits=coupled(local,wide,tile['operator'],profile.coupling)
                dense=F.interpolate(logits.T.reshape(1,bank.class_count,32,32),(512,512),mode='bilinear',align_corners=False)[0]
                ah,aw=min(512,h-top),min(512,w-left)
                accumulator.add(dense[:,:ah,:aw].softmax(0).float(),self.blend[:ah,:aw],left,top)
            probability=F.interpolate((accumulator.probabilities/accumulator.normalizer[None])[None],
                                      source['output_size'],mode='bilinear',align_corners=False)[0]
        if not bool(torch.isfinite(probability).all()):
            raise RuntimeError('Nonfinite inference probability.')
        prediction=probability.argmax(0).cpu().numpy()
        diagnostic=dict(profile=profile.record(),decision=decision,soft_alias_weights=self.soft,
            geometry_encodings=len(source['local']),wide_encodings=len(source['wide']),fine_forwards=0,
            alias_backend='cached-reference-groups-for-hard-residual-rival' if self.residual_reader is not None else 'precomputed-vectorized-variable-groups',target_masks_loaded=False)
        diagnostic['residual_queries']=0 if self.residual_text is None else len(self.residual_text)
        diagnostic['residual_mode']=None if self.residual_text is None else self.residual_mode
        diagnostic['correction']=self.correction
        diagnostic['wide_aggregation']=self.wide_aggregation
        diagnostic['wide_policy']=source.get('wide_policy','long448')
        diagnostic['read_policy']=self.read_policy
        diagnostic['local_background']=self.local_background
        diagnostic['local_background_union_applied']=profile.bank in self.local_background_members
        return (prediction,diagnostic,probability) if return_probability else (prediction,diagnostic)
