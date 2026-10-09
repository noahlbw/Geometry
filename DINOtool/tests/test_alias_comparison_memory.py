import unittest
import torch

from dinotool.alias_comparison_memory import LazyMargins,sampled_observations,chunked_risk,execution
from dinotool.matched_contribution_alias import contribution_margins
from dinotool.native_query_alias import native_risk
from dinotool.one_sided_alias_stream import fixed_slot_stream
from dinotool.rival_alias_fast import CachedCrop,sampled_cached


class MemoryEquivalence(unittest.TestCase):
    def test_exact_interpolation_risk_and_writer(self):
        for dtype in (torch.float32,torch.float64):
            with self.subTest(dtype=dtype):self.check_dtype(dtype)

    def check_dtype(self,dtype):
        generator=torch.Generator().manual_seed(8308)
        classes,slots,tokens,queries=7,20,51,73
        evidence=torch.randn(tokens,classes,slots,generator=generator,dtype=dtype)*17
        coordinates=torch.zeros(queries,2,dtype=dtype)
        members=torch.arange(classes*slots).reshape(classes,slots)
        parents=torch.arange(classes).repeat_interleave(slots)
        canonical=members[:,0]
        valid=torch.ones(queries,dtype=torch.bool);valid[-3:]=False
        ordinary=[];lazy=[]
        for _ in range(3):
            ids=torch.randint(tokens,(queries,4),generator=generator)
            weights=torch.rand(queries,4,generator=generator,dtype=dtype)
            weights/=weights.sum(-1,keepdim=True)
            e=evidence+torch.randn(tokens,classes,slots,generator=generator,dtype=dtype)
            full=contribution_margins(e)
            compressed=LazyMargins(e,1.)
            self.assertTrue(torch.equal(full[ids],compressed[ids]))
            ordinary.append(CachedCrop(e,full,ids,weights))
            lazy.append(CachedCrop(e,compressed,ids,weights))
        wide=sampled_cached(ordinary,coordinates)
        self.assertTrue(torch.equal(wide,sampled_observations(lazy,coordinates)))
        local=torch.randn(wide.shape,generator=generator,dtype=wide.dtype)*11
        risk=native_risk(wide,local,local,parents,canonical,valid)
        self.assertTrue(torch.equal(risk,chunked_risk(wide,local,local,parents,canonical,valid)))
        a,_=fixed_slot_stream(ordinary,members,risk,valid)
        b,_=fixed_slot_stream(lazy,members,risk,valid)
        self.assertTrue(torch.equal(a,b))

    def test_globals_restored_on_exception(self):
        from dinotool import shared_local_alias as light
        original=light.sampled_cached
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with execution():
                self.assertIs(light.sampled_cached,sampled_observations)
                raise RuntimeError('intentional')
        self.assertIs(light.sampled_cached,original)


if __name__=='__main__':unittest.main()
