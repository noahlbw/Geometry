import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from dinotool.taxonomy_inference import TaxonomyInference


class TaxonomyInferenceTests(unittest.TestCase):
    def test_finalization_preserves_residual_and_selected_views(self):
        bank=SimpleNamespace(features=torch.eye(3),parent_indices=torch.arange(3),
            canonical_mask=torch.ones(3,dtype=torch.bool),class_count=3,class_names=('bg','a','b'))
        geometry=SimpleNamespace(device=torch.device('cpu'))
        residual=torch.tensor([[[1.,0.,0.]]])
        models=[TaxonomyInference.for_finalization(geometry,None,{'semantic_segmentation':bank},
            {'semantic_segmentation':None},ordinary_alias_policy=policy,family='natural',
            background=0,residual_features=residual) for policy in ('retained','uniform')]
        for model in models:
            self.assertFalse(model.soft)
            self.assertEqual(model.residual_mode,'protected')
            self.assertIs(model.residual_features,residual)
            self.assertEqual(model.wide_policy,'natural_short336_cap672')
        with self.assertRaises(ValueError):
            TaxonomyInference.for_finalization(geometry,None,{}, {},ordinary_alias_policy='auto',family='natural')

    def test_finalization_rs_changes_only_extra_weights(self):
        bank=SimpleNamespace(features=torch.eye(3),parent_indices=torch.arange(3),
            canonical_mask=torch.ones(3,dtype=torch.bool),class_count=3,class_names=('a','b','c'))
        args=(SimpleNamespace(device=torch.device('cpu')),None,
            {'original_imagenet':bank,'focused20':bank},{'original_imagenet':None,'focused20':None})
        retained=TaxonomyInference.for_task(*args,family='remote_sensing')
        simple=TaxonomyInference.for_finalization(*args,ordinary_alias_policy='uniform',family='remote_sensing')
        self.assertTrue(retained.soft)
        self.assertFalse(simple.soft)
        for attribute in ('wide_policy','correction','wide_aggregation','read_policy','strengths'):
            self.assertEqual(getattr(retained,attribute),getattr(simple,attribute))

    def test_union_background_only_changes_multiquery_background(self):
        bank=SimpleNamespace(features=torch.tensor([[1.,0.],[0.,1.],[.9,.1]]),
            parent_indices=torch.tensor([0,0,1]),canonical_mask=torch.tensor([True,False,True]),
            class_count=2,class_names=('background','object'))
        # Constant branch scores with zero transport isolate the local reduction.
        features=torch.tensor([[1.,0.]]).repeat(1024,1)
        source=dict(local=[dict(top=0,left=0,features={2.:features},operator=torch.zeros(1024,1024))],
            wide=[dict()],size=(32,32),output_size=(16,16))
        kwargs=dict(family='natural',background=0,soft=False,correction='local_endpoint')
        with patch('eval_development_readout.wide_scores',return_value=torch.zeros(2,32,32)):
            old=TaxonomyInference(SimpleNamespace(device=torch.device('cpu')),None,
                {'semantic_segmentation':bank},{'semantic_segmentation':None},**kwargs)
            changed=TaxonomyInference(SimpleNamespace(device=torch.device('cpu')),None,
                {'semantic_segmentation':bank},{'semantic_segmentation':None},local_background='union_max',**kwargs)
            # Rank profile strength depends on taxonomy; no encoder invoked here.
            source['local'][0]['features']['original']=features
            p0=old.predict_observations(source,return_probability=True)[2]
            _,d,p1=changed.predict_observations(source,return_probability=True)
        self.assertTrue(d['local_background_union_applied'])
        self.assertTrue(bool((p1[0]>=p0[0]).all()))
        self.assertTrue(bool((p1[0]>p0[0]).any()))

    def test_single_query_background_union_keeps_exact_probability(self):
        bank=SimpleNamespace(features=torch.eye(3),parent_indices=torch.arange(3),
            canonical_mask=torch.ones(3,dtype=torch.bool),class_count=3,class_names=('bg','a','b'))
        features=torch.tensor([[1.,0.,0.]]).repeat(1024,1)
        source=dict(local=[dict(top=0,left=0,features={2.:features},operator=torch.eye(1024)*.25)],
            wide=[dict()],size=(32,32),output_size=(16,16))
        with patch('eval_development_readout.wide_scores',return_value=torch.zeros(3,32,32)):
            models=[TaxonomyInference(SimpleNamespace(device=torch.device('cpu')),None,
                {'semantic_segmentation':bank},{'semantic_segmentation':None},family='natural',background=0,
                soft=False,local_background=mode) for mode in ('retained','union_max')]
            a=models[0].predict_observations(source,return_probability=True)[2]
            _,d,b=models[1].predict_observations(source,return_probability=True)
        self.assertFalse(d['local_background_union_applied'])
        self.assertTrue(torch.equal(a,b))

    def test_endpoints_share_observations_but_not_final_coupling(self):
        bank=SimpleNamespace(features=torch.eye(3),parent_indices=torch.arange(3),
            canonical_mask=torch.ones(3,dtype=torch.bool),class_count=3,class_names=('a','b','c'))
        features=torch.tensor([[1.,0.,0.]]).repeat(1024,1)
        source=dict(local=[dict(top=0,left=0,features={2.:features},operator=torch.eye(1024)*.25)],
            wide=[dict()],size=(32,32),output_size=(16,16))
        wide=torch.zeros(3,32,32);wide[1]=40
        with patch('eval_development_readout.wide_scores',return_value=wide):
            for mode,wanted in (('local_endpoint',0),('wide_endpoint',1)):
                model=TaxonomyInference(SimpleNamespace(device=torch.device('cpu')),None,
                    {'semantic_segmentation':bank},{'semantic_segmentation':None},
                    family='natural',soft=False,correction=mode)
                prediction,diagnostic=model.predict_observations(source)
                np.testing.assert_array_equal(prediction,np.full((16,16),wanted))
                self.assertEqual(diagnostic['fine_forwards'],0)

    def test_mask_free_api_and_uniform_alias_equivalence(self):
        bank=SimpleNamespace(features=torch.eye(3).repeat_interleave(2,0),
            parent_indices=torch.tensor([0,0,1,1,2,2]),canonical_mask=torch.tensor([True,False]*3),
            class_count=3,class_names=('a','b','c'))
        geometry=SimpleNamespace(device=torch.device('cpu'))
        features=torch.tensor([[1.,0.,0.]]).repeat(1024,1)
        source=dict(local=[dict(top=0,left=0,features={2.:features},operator=torch.eye(1024)*.25)],
            wide=[dict()],size=(32,32),output_size=(16,16))
        with patch('eval_development_readout.observations',return_value=source),patch('eval_development_readout.wide_scores',return_value=torch.zeros(3,32,32)):
            model=TaxonomyInference(geometry,None,{'semantic_segmentation':bank},{'semantic_segmentation':None},family='natural')
            selected,diagnostic=model.predict(torch.zeros(3,16,16))
            model.soft=False
            baseline,_=model.predict(torch.zeros(3,16,16))
        np.testing.assert_array_equal(selected,baseline)
        np.testing.assert_array_equal(selected,np.zeros((16,16),dtype=np.int64))
        self.assertFalse(diagnostic['target_masks_loaded'])
        self.assertEqual(diagnostic['fine_forwards'],0)

    def test_protected_mode_requires_residual_queries(self):
        bank=SimpleNamespace(features=torch.eye(3),parent_indices=torch.arange(3),
            canonical_mask=torch.ones(3,dtype=torch.bool),class_count=3,class_names=('a','b','c'))
        with self.assertRaises(ValueError):
            TaxonomyInference(SimpleNamespace(device=torch.device('cpu')),None,
                {'semantic_segmentation':bank},{'semantic_segmentation':None},
                family='natural',background=0,soft=False,residual_mode='protected')

    def test_missing_task_bank_is_rejected(self):
        with self.assertRaises(ValueError):
            TaxonomyInference(None,None,{}, {},family='natural')

    def test_residual_max_is_opt_in_and_reuses_observations(self):
        bank=SimpleNamespace(features=torch.eye(3).repeat_interleave(2,0),
            parent_indices=torch.tensor([0,0,1,1,2,2]),canonical_mask=torch.tensor([True,False]*3),
            class_count=3,class_names=('a','b','c'))
        geometry=SimpleNamespace(device=torch.device('cpu'))
        residual=torch.tensor([[[1.,0.,0.]]])
        args=(geometry,None,{'semantic_segmentation':bank},{'semantic_segmentation':None})
        with self.assertRaises(ValueError):
            TaxonomyInference(*args,family='natural',background=2,residual_features=residual)
        with self.assertRaises(ValueError):
            TaxonomyInference(*args,family='natural',soft=False,residual_features=residual)
        model=TaxonomyInference(*args,family='natural',background=2,soft=False,residual_features=residual)
        features=torch.tensor([[1.,0.,0.]]).repeat(1024,1)
        source=dict(local=[dict(top=0,left=0,features={2.:features},operator=torch.eye(1024)*.25)],
            wide=[dict()],size=(32,32),output_size=(16,16))
        with patch('eval_development_readout.observations',return_value=source) as observe,patch('eval_development_readout.wide_scores',return_value=torch.zeros(3,32,32)),patch('eval_residual_ontology.residual_wide',return_value={'max':torch.full((1,32,32),40.)}) as wide:
            prediction,diagnostic=model.predict(torch.zeros(3,16,16))
        np.testing.assert_array_equal(prediction,np.full((16,16),2,dtype=np.int64))
        observe.assert_called_once()
        wide.assert_called_once()
        self.assertEqual(diagnostic['residual_queries'],1)
        self.assertEqual(diagnostic['fine_forwards'],0)
        protected=TaxonomyInference(*args,family='natural',background=2,soft=False,
            residual_features=residual,residual_mode='protected')
        with patch('eval_development_readout.observations',return_value=source),patch('eval_development_readout.wide_scores',return_value=torch.zeros(3,32,32)),patch('eval_residual_ontology.residual_wide',return_value={'max':torch.full((1,32,32),40.)}):
            prediction,diagnostic=protected.predict(torch.zeros(3,16,16))
        np.testing.assert_array_equal(prediction,np.full((16,16),2,dtype=np.int64))
        self.assertEqual(diagnostic['residual_mode'],'protected')


if __name__=='__main__':
    unittest.main()
