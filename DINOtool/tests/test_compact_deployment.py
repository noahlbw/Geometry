import unittest
import torch
from types import SimpleNamespace
from dinotool.compact_deployment import CompactGeometry,offload_text_towers
from dinotool.geometry_execution import GeometryExecution
from dinotool.natural_static_search import projected
from test_geometry_execution import fixture
from unittest.mock import patch
from dinotool.taxonomy_inference import TaxonomyInference


class CompactDeploymentTests(unittest.TestCase):
    def test_deferred_prediction_keeps_probability(self):
        bank=SimpleNamespace(features=torch.eye(3),parent_indices=torch.arange(3),
            canonical_mask=torch.ones(3,dtype=torch.bool),class_count=3,class_names=('a','b','c'))
        source=dict(local=[dict(top=0,left=0,features={2.:torch.ones(1024,3)},operator=torch.zeros(1024,1024))],
            wide=[dict()],size=(32,32),output_size=(16,16))
        with patch('eval_development_readout.wide_scores',return_value=torch.zeros(3,32,32)):
            model=TaxonomyInference(SimpleNamespace(device=torch.device('cpu')),None,
                {'semantic_segmentation':bank},{'semantic_segmentation':None},family='natural',soft=False)
            prediction,_,expected=model.predict_observations(source,return_probability=True)
            deferred,_,actual=model.predict_observations(source,return_probability=True,return_prediction=False)
        self.assertIsNone(deferred)
        self.assertTrue(torch.equal(actual,expected))
        self.assertTrue((prediction==expected.argmax(0).numpy()).all())

    def test_text_release_preserves_visual_weights(self):
        models=[]
        for _ in range(2):
            model=torch.nn.Module()
            model.visual_model=torch.nn.Linear(3,4)
            model.text_model=torch.nn.Linear(5,6)
            models.append(model)
        before=[{k:v.clone() for k,v in m.visual_model.state_dict().items()} for m in models]
        wrappers=[SimpleNamespace(backbone=SimpleNamespace(model=m)) for m in models]
        offload_text_towers(*wrappers,release=True)
        for model,weights in zip(models,before):
            self.assertIsNone(model.text_model)
            self.assertTrue(all(torch.equal(value,weights[key]) for key,value in model.visual_model.state_dict().items()))

    def test_requested_heads_preserve_exact_frozen_outputs(self):
        torch.manual_seed(17)
        geometry=GeometryExecution(fixture())
        compact=CompactGeometry(geometry)
        image=torch.rand(1,3,32,32)
        reference=geometry.prepare_image(image)
        prepared=compact.prepare_image(image)
        self.assertTrue(torch.equal(prepared.geometry_patch_conditional,reference.geometry_patch_conditional))
        for strength in ('original',.5,1.,2.,3.,4.):
            expected=projected(geometry.backbone.model.visual_model.head,reference,strength)
            self.assertTrue(torch.equal(expected,compact.project(prepared,strength)),str(strength))


if __name__=='__main__':unittest.main()
