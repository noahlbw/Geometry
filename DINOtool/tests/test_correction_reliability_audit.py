import unittest
import numpy as np
from dinotool.correction_reliability_audit import exact_auc,observations,summarize


class ReliabilityTests(unittest.TestCase):
    def test_exact_auc_order_and_ties(self):
        self.assertEqual(exact_auc([1,2,3,4],[0,0,1,1]),1.)
        self.assertEqual(exact_auc([1,2,3,4],[1,1,0,0]),0.)
        self.assertEqual(exact_auc([1,1,1,1],[0,0,1,1]),.5)
        self.assertIsNone(exact_auc([1,2],[1,1]))

    def test_only_fixed_broken_and_ignore_labels(self):
        local=np.array([[[.9,.9,.9,.9]],[[.1,.1,.1,.1]]])
        candidate=1-local;target=np.array([[1,0,255,2]])
        row=observations(local,candidate,candidate,target,'LocalEndpoint')
        np.testing.assert_array_equal(row['true_class'],[1,0])
        np.testing.assert_array_equal(row['beneficial'],[True,False])
        self.assertEqual(row['valid_pixels'],2)
        summary=summarize({'LocalEndpoint':[row]},('bg','object'),0)['LocalEndpoint']
        self.assertEqual(summary['fixed'],1);self.assertEqual(summary['broken'],1)
        self.assertIsNone(summary['signals']['negative_base_confidence']['macro_class_auc'])

    def test_nonfinite_or_unmatched_inputs_rejected(self):
        with self.assertRaises(ValueError):exact_auc([np.nan],[True])
        with self.assertRaises(ValueError):observations(np.zeros((2,2,2)),np.zeros((2,2,2)),np.zeros((2,2,2)),np.zeros((1,1)),'LocalEndpoint')


if __name__=='__main__':unittest.main()
