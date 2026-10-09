import unittest
import torch
from dinotool.residual_ontology import complement_queries,residual_score


class ResidualTests(unittest.TestCase):
    def test_metadata_only_complement(self):
        names='\n'.join(f'{i}: name{i}' for i in range(1,460))
        queries,rows=complement_queries(names,list(range(60)))
        self.assertEqual(len(queries),401)
        self.assertEqual(queries[0],'background')
        self.assertEqual(rows[0],(60,'name60'))
        self.assertNotIn('name2',queries)

    def test_count_normalized_mean_and_disjunction(self):
        x=torch.tensor([[1.,2.,3.]])
        self.assertEqual(float(residual_score(x,'max',.07)),3.)
        self.assertLess(float(residual_score(x,'mean',.07)),3.)
        torch.testing.assert_close(residual_score(x.repeat(1,2),'mean',.07),residual_score(x,'mean',.07))

    def test_incomplete_metadata_rejected(self):
        with self.assertRaises(ValueError):
            complement_queries('1: aircraft',list(range(60)))


if __name__=='__main__':
    unittest.main()
