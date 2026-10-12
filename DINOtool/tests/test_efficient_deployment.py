import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from dinotool.alias_action_capacity import reconstruction_operator
from dinotool.device_probability_accumulator import DeviceProbabilityAccumulator
from dinotool.development_readout import biased_probabilities
from dinotool.efficient_deployment import GroupedLocalScorer, reconstruction, restore_and_finalize, write_dense
from dinotool.packed_alias_readout import PackedAliasReadout


class EfficientDeploymentTests(unittest.TestCase):
    def test_variable_groups_preserve_local_reduction(self):
        torch.manual_seed(17)
        counts=[1,3,2,3,1,2]
        bank=SimpleNamespace(class_count=len(counts),features=torch.randn(sum(counts),8),
            parent_indices=torch.tensor([c for c,n in enumerate(counts) for _ in range(n)]))
        alias=torch.randn(41,sum(counts))
        self.assertTrue(torch.equal(PackedAliasReadout(bank).uniform_reference(alias),
            GroupedLocalScorer(bank).uniform_reference(alias)))

    def test_reconstruction_keeps_valid_mask_and_double_solve(self):
        relation=torch.randn(16,16).softmax(-1)
        valid=torch.arange(16)%3!=0
        actual,_=reconstruction(relation,valid)
        expected,_=reconstruction_operator(relation,valid)
        self.assertTrue(torch.equal(actual,expected))

    def test_dense_slab_softmax_preserves_probabilities(self):
        torch.manual_seed(3)
        logits=torch.randn(1024,5,dtype=torch.float64)
        blend=torch.rand(512,512)
        dense=F.interpolate(logits.T.reshape(1,5,32,32),(512,512),mode='bilinear',align_corners=False)[0]
        expected=dense[:,:477,:490].softmax(0).float()*blend[None,:477,:490]
        with DeviceProbabilityAccumulator(5,477,490,'cpu') as accumulator:
            write_dense(accumulator,logits,5,477,490,blend,0,0)
            self.assertTrue(torch.equal(accumulator.probabilities,expected))

    def test_streamed_restore_calibration_and_tie_order(self):
        torch.manual_seed(7)
        for classes,background,bias in ((33,None,0.),(21,0,-.25),(60,0,0.)):
            normalized=torch.rand(classes,37,43).softmax(0)
            normalized[16]=normalized[0]  # Exact cross-chunk tie: first class wins.
            size=(137,173)
            restored=F.interpolate(normalized[None],size,mode='bilinear',align_corners=False)[0]
            calibrated=biased_probabilities(restored,bias,background)
            confidence,expected=calibrated.max(0)
            if background is not None:expected=expected.masked_fill(confidence<.05,background)
            actual=restore_and_finalize(normalized,size,bias,.05,background)
            self.assertTrue(np.array_equal(actual,expected.numpy()))


if __name__=='__main__':unittest.main()
