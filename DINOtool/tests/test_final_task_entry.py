import unittest
from types import SimpleNamespace
import torch
from dinotool.taxonomy_inference import TaxonomyInference


class FinalTaskEntryTests(unittest.TestCase):
    def objects(self,family):
        bank=SimpleNamespace(features=torch.eye(3).repeat_interleave(2,0),
            parent_indices=torch.tensor([0,0,1,1,2,2]),canonical_mask=torch.tensor([True,False]*3),
            class_count=3,class_names=('a','b','residual'))
        needed=('semantic_segmentation',) if family=='natural' else ('original_imagenet','focused20')
        return SimpleNamespace(device=torch.device('cpu')),{k:bank for k in needed},{k:None for k in needed}

    def test_natural_task_selects_single_scale_prior(self):
        g,b,q=self.objects('natural')
        m=TaxonomyInference.for_task(g,None,b,q,family='natural',background=2)
        self.assertEqual(m.wide_policy,'natural_short336_cap672')
        self.assertFalse(m.soft)
        self.assertEqual((m.correction,m.wide_aggregation,m.read_policy),('frozen','inherited','retained'))
        historical=TaxonomyInference(g,None,b,q,family='natural',soft=False)
        self.assertEqual(historical.wide_policy,'long448')
        self.assertEqual(m.profile,historical.profile)

    def test_remote_task_keeps_existing_selector(self):
        g,b,q=self.objects('remote_sensing')
        m=TaxonomyInference.for_task(g,None,b,q,family='remote_sensing',background=2)
        self.assertEqual(m.wide_policy,'long448');self.assertTrue(m.soft)
        self.assertIsNone(m.profile)
        self.assertEqual(m.strengths,(1.,3.,'original'))

    def test_residual_is_explicit_and_protected(self):
        g,b,q=self.objects('natural')
        residual=torch.tensor([[[1.,0.,0.]]])
        m=TaxonomyInference.for_task(g,None,b,q,family='natural',background=2,residual_features=residual)
        self.assertEqual(m.residual_mode,'protected');self.assertFalse(m.soft)
        with self.assertRaises(ValueError):
            TaxonomyInference.for_task(g,None,b,q,family='natural',residual_features=residual)

    def test_unknown_family_is_rejected(self):
        with self.assertRaises(ValueError):
            TaxonomyInference.for_task(None,None,{}, {},family='dataset-specific-winner')

    def test_union_policy_is_explicit_and_keeps_historical_default(self):
        g,b,q=self.objects('natural')
        old=TaxonomyInference.for_task(g,None,b,q,family='natural',background=2)
        new=TaxonomyInference.for_task(g,None,b,q,family='natural',background=2,local_background='union_max')
        self.assertEqual(old.local_background,'retained')
        self.assertFalse(old.local_background_members)
        self.assertEqual(new.local_background,'union_max')
        self.assertEqual(new.local_background_members['semantic_segmentation'].tolist(),[4,5])
        self.assertEqual(old.profile,new.profile)
        g,b,q=self.objects('remote_sensing')
        rs=TaxonomyInference.for_task(g,None,b,q,family='remote_sensing',background=2,local_background='union_max')
        self.assertFalse(rs.local_background_members)


if __name__=='__main__':unittest.main()
