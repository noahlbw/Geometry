import unittest
from unittest.mock import patch
import torch
from dinotool.inference import tile_starts
from dinotool.natural_wide_resolution import target_size,replace_wide


class NaturalWideTests(unittest.TestCase):
    def test_production_observer_encodes_only_one_wide_view(self):
        from types import SimpleNamespace
        import eval_development_readout as base
        prepared=SimpleNamespace(geometry_patch_conditional=torch.eye(1024)[None])
        geometry=SimpleNamespace(device=torch.device('cpu'),prepare_image=lambda crop:prepared,
            backbone=SimpleNamespace(_autocast=lambda:__import__('contextlib').nullcontext(),model=SimpleNamespace(visual_model=SimpleNamespace(head=None))))
        count=[]
        vip=SimpleNamespace(crop_patch_features=lambda crop:count.append(crop.shape) or torch.ones(1,441,3))
        with patch.object(base,'resize_geometry',return_value=torch.zeros(3,32,32)),patch.object(base,'geometry_windows',return_value=[(0,0)]),patch.object(base,'reconstruction_operator',return_value=(torch.eye(1024),0.)),patch.object(base,'projected',return_value=torch.ones(1,1024,3)):
            source=base.observations(torch.zeros(3,400,600),geometry,vip,(2.,),wide_policy='natural_short336_cap672')
        self.assertEqual(len(count),3)
        self.assertEqual(len(source['wide']),3)
        self.assertEqual(len(source['local']),1)
        self.assertEqual(source['wide_size'],(336,504))

    def test_short_edge_or_long_cap(self):
        for h,w in ((500,500),(333,500),(500,333),(100,5000),(5000,100),(1,1)):
            nh,nw=target_size(h,w)
            self.assertLessEqual(max(nh,nw),672)
            self.assertLessEqual(min(nh,nw),336)
            if max(h,w)/min(h,w)<=2:
                self.assertEqual(min(nh,nw),336)
            count=len(tile_starts(nh,336,224))*len(tile_starts(nw,336,224))
            self.assertLessEqual(count,4)

    def test_all_aspect_ratios_and_overlap_semantics(self):
        self.assertEqual(tile_starts(672,336,224),[0,112,224,336])
        for short in (1,32,333,512,2048):
            for long in (short,short+1,short*2,short*4,short*100):
                h,w=target_size(short,long)
                self.assertLessEqual(len(tile_starts(h,336,224))*len(tile_starts(w,336,224)),4)

    def test_local_features_reused_and_no_geometry_encoding(self):
        from types import SimpleNamespace
        vip=SimpleNamespace(crop_patch_features=lambda crop:torch.ones(1,441,3))
        local=[object()]
        source=dict(local=local,wide=[object()],size=(896,896),output_size=(400,600),wide_size=(299,448))
        out=replace_wide(source,torch.zeros(3,400,600),vip)
        self.assertIs(out['local'],local)
        self.assertEqual(out['size'],source['size'])
        self.assertEqual(out['output_size'],source['output_size'])
        self.assertEqual(out['wide_size'],(336,504))
        self.assertEqual(len(out['wide']),3)


if __name__=='__main__':unittest.main()
