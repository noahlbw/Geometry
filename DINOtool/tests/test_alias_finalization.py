import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from dinotool.tcpr import TCPRTextBank
from dinotool.vip_official_adapter import VIPQueries
from eval_alias_finalization import foreground_banks,load_samples


class FinalizationProtocolTests(unittest.TestCase):
    def test_loveda_p_has_separate_queries_and_preserves_d(self):
        names=('background','building','road');aliases=('bg','other','building','roof','road')
        parents=torch.tensor([0,0,1,1,2]);canonical=torch.tensor([True,False,True,False,True])
        bank=TCPRTextBank(torch.eye(5),parents,canonical,names,aliases)
        query=VIPQueries(torch.eye(5)[:,None],parents,names,aliases)
        bs,qs=foreground_banks({'x':bank},{'x':query})
        self.assertEqual(bs['x'].class_names,('building','road'))
        self.assertEqual(bs['x'].alias_names,('building','roof','road'))
        self.assertEqual(bs['x'].parent_indices.tolist(),[0,0,1])
        self.assertTrue(torch.equal(qs['x'].features,query.features[2:]))
        self.assertEqual(bank.parent_indices.tolist(),[0,0,1,1,2])
        self.assertEqual(bank.class_count,3)

    def test_loveda_loaders_do_not_access_masks_before_inference(self):
        from dinotool.prompts import ClassSpec
        samples=[SimpleNamespace(key='a',image_path='a.png'),SimpleNamespace(key='b',image_path='b.png')]
        specs={p:[ClassSpec(n,(n,)) for n in names] for p,names in
            dict(P=('building','road'),D=('background','building','road')).items()}
        args=SimpleNamespace(dataset='loveda',mode='smoke',shard_index=0,num_shards=1)
        entry=dict(banks=dict(original=dict(classes=[dict(name=c.name,synonyms=[c.name]) for c in specs['D']])))
        masks=lambda *a:(_ for _ in ()).throw(RuntimeError('Mask access forbidden'))
        from unittest.mock import Mock
        loader=Mock(return_value='rgb')
        with patch('eval_alias_finalization.rs_protocol',return_value=(samples,specs,loader,masks)):
            full,chosen,load_image,_,names=load_samples(args,entry)
        self.assertEqual(full,samples);self.assertEqual(chosen,samples[:1])
        self.assertEqual(names['P'],('building','road'))
        self.assertEqual(load_image(chosen[0]),'rgb')
        loader.assert_called_once_with('a.png')


if __name__=='__main__':unittest.main()
