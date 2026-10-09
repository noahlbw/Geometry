from types import SimpleNamespace
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from dinotool.cross_view_value_innovation import (
    conditioned_value_read, cross_view_prior, matched_value_displacement,
    normalize_prior, read_value_innovation, ValueInnovationConfig)
from dinotool.tcpr import _intervened_head


def tiny_head_and_prepared():
    torch.manual_seed(22)
    blocks = []
    for _ in range(2):
        attention = SimpleNamespace(qkv=nn.Linear(8, 24), num_heads=2, scale=.5,
                                     proj=nn.Linear(8, 8), proj_drop=nn.Identity())
        blocks.append(SimpleNamespace(attn=attention, norm1=nn.LayerNorm(8), norm2=nn.LayerNorm(8),
                                       ls1=nn.Identity(), ls2=nn.Identity(), mlp=nn.Linear(8, 8)))
    head = SimpleNamespace(blocks=blocks, ln_final=nn.LayerNorm(8), linear_projection=nn.Linear(8, 6))
    tokens = torch.randn(1, 6, 8)
    relation = torch.rand(1, 4, 4).softmax(-1)
    projected = _intervened_head(head, tokens, relation, 2, 1, 2, "preserve")
    prepared = SimpleNamespace(backbone_tokens=tokens, raw_patch_tokens=tokens[:, 2:],
        geometry_patch_conditional=relation, geometry_projected=F.normalize(projected[:, 2:].float(), dim=-1),
        prefix_tokens=2, block_index=1)
    return head, prepared


class CrossViewValueTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.query = torch.randn(1, 2, 3, 4)
        self.key = torch.randn(1, 2, 5, 4)
        self.value = torch.randn(1, 2, 5, 4)
        self.prior = torch.rand(1, 3, 5).softmax(-1)

    def test_matches_explicit_conditioned_attention(self):
        read, log_z, _ = conditioned_value_read(self.query, self.key, self.value, self.prior, .5)
        joint = .5*(self.query @ self.key.transpose(-1, -2))+self.prior.log()[:, None]
        torch.testing.assert_close(read, joint.softmax(-1) @ self.value)
        torch.testing.assert_close(log_z, joint.logsumexp(-1, keepdim=True))

    def test_duplicate_value_read_has_exact_zero_innovation(self):
        delta, fraction = matched_value_displacement(self.query, self.key, self.value, self.prior,
            self.key, self.value, self.prior, .5)
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        self.assertTrue(torch.equal(fraction, torch.full_like(fraction, .5)))

    def test_no_context_is_neutral_not_negative_evidence(self):
        delta, fraction = matched_value_displacement(self.query, self.key, self.value, self.prior,
            self.key, self.value, torch.zeros_like(self.prior), .5)
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        self.assertTrue(torch.equal(fraction, torch.zeros_like(fraction)))

    def test_common_value_vector_cancels(self):
        context = self.value.flip(2)
        first, _ = matched_value_displacement(self.query, self.key, self.value, self.prior,
            self.key, context, self.prior, .5)
        common = torch.tensor([.4, -.7, .2, .8])
        second, _ = matched_value_displacement(self.query, self.key, self.value+common, self.prior,
            self.key, context+common, self.prior, .5)
        torch.testing.assert_close(first, second, atol=2e-7, rtol=1e-6)

    def test_duplicate_donors_do_not_change_view_evidence(self):
        read, log_z, _ = conditioned_value_read(self.query, self.key, self.value, self.prior, .5)
        expanded, expanded_z, _ = conditioned_value_read(self.query,
            self.key.repeat_interleave(2, 2), self.value.repeat_interleave(2, 2),
            self.prior.repeat_interleave(2, 2)/2, .5)
        torch.testing.assert_close(read, expanded, atol=2e-7, rtol=1e-6)
        torch.testing.assert_close(log_z, expanded_z, atol=2e-7, rtol=1e-6)

    def test_padding_excluded_before_normalizing_prior(self):
        prior = normalize_prior(torch.ones(1, 2, 3), torch.tensor([[True, False, True]]))
        self.assertTrue(torch.equal(prior, torch.tensor([[[.5, 0., .5], [.5, 0., .5]]])))

    def test_empty_cross_view_support_finite_and_zero(self):
        relation = cross_view_prior(torch.randn(1, 3, 4), torch.randn(1, 2, 4),
            torch.randn(3, 2), torch.randn(2, 2), torch.zeros(1, 2, dtype=torch.bool))
        self.assertTrue(torch.equal(relation, torch.zeros_like(relation)))

    def test_duplicate_view_replays_both_geometry_blocks_exactly(self):
        head, prepared = tiny_head_and_prepared()
        valid = torch.ones(1, 4, dtype=torch.bool)
        feature, diag = read_value_innovation(head, prepared, prepared, valid, valid,
            prepared.geometry_patch_conditional, ValueInnovationConfig(query_chunk=2))
        self.assertTrue(torch.equal(feature, prepared.geometry_projected))
        self.assertEqual(diag["context_geometry_replay_max_error"], 0.)
        self.assertEqual(diag["prefix_displacement_max_error"], 0.)

    def test_absent_context_replays_geometry_exactly(self):
        head, prepared = tiny_head_and_prepared()
        valid = torch.ones(1, 4, dtype=torch.bool)
        feature, diag = read_value_innovation(head, prepared, prepared, valid, ~valid,
            prepared.geometry_patch_conditional)
        self.assertTrue(torch.equal(feature, prepared.geometry_projected))
        self.assertEqual(diag["mean_context_fraction"], 0.)

    def test_shuffle_cannot_move_padding_into_real_donors(self):
        head, prepared = tiny_head_and_prepared()
        valid = torch.tensor([[True, True, False, False]])
        with self.assertRaises(ValueError):
            read_value_innovation(head, prepared, prepared, valid, valid, prepared.geometry_patch_conditional,
                context_permutation=torch.tensor([2, 1, 0, 3]))

    def test_invalid_fine_queries_receive_no_direct_displacement(self):
        head, prepared = tiny_head_and_prepared()
        valid = torch.tensor([[True, False, True, False]])
        _, diag = read_value_innovation(head, prepared, prepared, valid, torch.ones_like(valid),
            prepared.geometry_patch_conditional.flip(-1))
        self.assertEqual(diag["invalid_query_displacement_max_error"], 0.)
        self.assertEqual(diag["prefix_displacement_max_error"], 0.)


if __name__ == "__main__":
    unittest.main()
