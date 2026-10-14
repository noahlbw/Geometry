import torch
import torch.nn.functional as F
from torch import nn

from dinotool.geov2 import (AliasScorer, anchored_correction, restore_ids,
                           adaptive_writeback_gain, sample_wide, structure,
                           structure_writeback, value_head, Geov2Config)


def test_anchored_solver_matches_original_weighted_objective():
    torch.manual_seed(9)
    a, s, degree = structure(torch.randn(1,12,8), 3, 4)
    local, wide = torch.randn(12,5), torch.randn(12,5)
    result = anchored_correction(local,wide,s,degree,1.,2)
    d, a64 = torch.diag(degree.double()), a.double()
    penalty = a64.T @ d @ a64
    exact = torch.linalg.solve(d+penalty, d@local.double()+penalty@wide.double())
    torch.testing.assert_close(result.double(),exact,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(anchored_correction(local,local,s,degree),local)
    spectrum = torch.linalg.eigvalsh(s.double())
    assert spectrum.min() > -1e-6 and spectrum.max() <= 1+1e-6


def test_two_stage_writer_identity_fixed_gain_and_adaptive_bounds():
    torch.manual_seed(11)
    _, symmetric, degree = structure(torch.randn(1, 9, 6), 3, 3)
    local, wide = torch.randn(9, 4), torch.randn(9, 4)

    target_cfg = Geov2Config(gate_mode='target', solve_chunk=2)
    target, optimum, gain = structure_writeback(local, wide, symmetric, degree, target_cfg)
    torch.testing.assert_close(target, optimum)
    torch.testing.assert_close(gain, torch.ones(9))

    fixed_cfg = Geov2Config(gate_mode='fixed', writeback_gain=2., solve_chunk=2)
    twice, optimum2, fixed_gain = structure_writeback(local, wide, symmetric, degree, fixed_cfg)
    torch.testing.assert_close(optimum2, optimum)
    torch.testing.assert_close(twice, local + 2. * (optimum - local))
    torch.testing.assert_close(fixed_gain, torch.full((9,), 2.))

    adaptive = adaptive_writeback_gain(local, wide, optimum)
    assert adaptive.shape == (9,)
    assert bool(torch.isfinite(adaptive).all())
    assert float(adaptive.min()) >= .5 - 1e-6
    assert float(adaptive.max()) <= 2. + 1e-6


def test_adaptive_writer_is_exact_noop_when_views_agree():
    torch.manual_seed(12)
    _, symmetric, degree = structure(torch.randn(1, 9, 5), 3, 3)
    local = torch.randn(9, 3)
    result, target, gain = structure_writeback(
        local, local.clone(), symmetric, degree, Geov2Config(gate_mode='adaptive'))
    torch.testing.assert_close(target, local)
    torch.testing.assert_close(result, local)
    torch.testing.assert_close(gain, torch.ones(9))


def test_margin_control_can_amplify_an_unsupported_correction():
    local = torch.tensor([[1., .9], [2., .1]])
    wide = torch.tensor([[2., .1], [1., .9]])
    # No structural displacement: full gate must stay at one even when the
    # wide margin is larger. Removing support exposes that distinction.
    target = local.clone()
    full = adaptive_writeback_gain(local, wide, target)
    margin = adaptive_writeback_gain(local, wide, target, variant='margin')
    torch.testing.assert_close(full, torch.ones(2))
    assert margin[0] > 1 and margin[1] < 1
    support = adaptive_writeback_gain(local, wide, local + .5*(wide-local), variant='support')
    torch.testing.assert_close(support, torch.full((2,), 1.375), atol=1e-6, rtol=1e-6)


def test_alias_mean_is_count_normalized_and_background_is_union():
    text = torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
    query = torch.tensor([[1.,0.],[0.,1.]])
    ordinary = AliasScorer(text,torch.tensor([0,0,1]),('a','b'))(query)
    torch.testing.assert_close(ordinary,query)
    union = AliasScorer(text,torch.tensor([0,0,0]),('background',),(0,))(query)
    torch.testing.assert_close(union,torch.ones(2,1))


def test_chunk_restore_and_wide_sampling_match_dense_bilinear():
    torch.manual_seed(2)
    scores = torch.randn(5,3,4)
    expected = F.interpolate(scores[None],(48,64),mode='bilinear',align_corners=False)[0].argmax(0)
    actual = restore_ids(scores,(48,64),(48,64),(48,64),7,2)
    assert torch.equal(torch.from_numpy(actual),expected)
    sampled = sample_wide(scores,0,0,(3,4),(48,64),(48,64),(48,64))
    torch.testing.assert_close(sampled,scores.flatten(1).T)
    tied = restore_ids(torch.zeros(5,3,4),(17,19),(37,51),(48,64),7,2)
    assert not tied.any()


def test_value_only_head_matches_full_qkv_patch_only_read():
    torch.manual_seed(4)
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv=nn.Linear(8,24)
            self.proj=nn.Linear(8,8)
            self.proj_drop=nn.Identity()
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1=nn.LayerNorm(8); self.norm2=nn.LayerNorm(8)
            self.attn=Attention(); self.ls1=nn.Identity(); self.ls2=nn.Identity()
            self.mlp=nn.Sequential(nn.Linear(8,16),nn.GELU(),nn.Linear(16,8))
    head=nn.Module()
    head.blocks=nn.ModuleList([Block(),Block()])
    head.ln_final=nn.LayerNorm(8); head.linear_projection=nn.Linear(8,8)
    raw=torch.randn(1,6,8); relation=torch.randn(6,6).softmax(-1)
    reference=raw
    for block in head.blocks:
        v=block.attn.qkv(block.norm1(reference)).chunk(3,-1)[2]
        reference=reference+block.attn.proj(relation@v)
        reference=reference+block.mlp(block.norm2(reference))
    reference=F.normalize(head.linear_projection(head.ln_final(reference)),dim=-1)
    torch.testing.assert_close(value_head(head,raw,relation),reference)
