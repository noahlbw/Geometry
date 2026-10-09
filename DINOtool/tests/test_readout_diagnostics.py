import unittest
import numpy as np
from dinotool.readout_diagnostics import transition_counts


class TransitionTests(unittest.TestCase):
    def test_partition_and_correctness(self):
        target=np.array([0,0,0,1,1,1,255])
        a=np.array([0,1,2,1,0,0,0]);b=np.array([1,0,1,1,1,0,2])
        result=transition_counts(target,a,b,3)
        np.testing.assert_array_equal(result,[[3,3,1,1,1,1,1],[3,1,1,0,0,1,2],[0,0,0,0,0,0,0]])

    def test_same_prediction_has_no_changes(self):
        target=np.array([[0,1],[1,255]]);p=np.zeros((2,2),int)
        result=transition_counts(target,p,p,2)
        self.assertEqual(int(result[:,1:5].sum()),0)
        self.assertEqual(int(result[:,0].sum()),3)

    def test_invalid_shapes_and_predictions(self):
        with self.assertRaises(ValueError):transition_counts([0],[0,1],[0],2)
        with self.assertRaises(ValueError):transition_counts([0],[2],[0],2)


if __name__=='__main__':unittest.main()
