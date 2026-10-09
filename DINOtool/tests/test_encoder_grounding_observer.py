import unittest

import torch

from dinotool.encoder_grounding_observer import align_encoder_aliases, native_class_scores, couple_encoder_scores


class EncoderGroundingTests(unittest.TestCase):
    def test_phrase_means_use_exact_spans_not_sentence_tokens(self):
        logits = torch.tensor([[0., 1., -1., 20.]]*4)
        masks = torch.tensor([[True, False, False, False], [False, True, True, False]])
        probabilities, observed = align_encoder_aliases(logits, masks, [(2, 2)], torch.ones(4, dtype=torch.bool), 2)
        torch.testing.assert_close(probabilities, torch.full((4, 2), .5), atol=1e-7, rtol=0)
        self.assertTrue(bool(observed.all()))

    def test_missing_positions_are_not_zeros_in_spatial_mean(self):
        logits = torch.tensor([[0.], [float("nan")], [0.], [0.]])
        valid = torch.tensor([True, False, True, True])
        values, observed = align_encoder_aliases(logits, torch.ones(1, 1, dtype=torch.bool), [(2, 2)], valid, 4)
        torch.testing.assert_close(values[observed], torch.full_like(values[observed], .5), atol=0, rtol=0)
        self.assertTrue(torch.equal(values[~observed], torch.zeros_like(values[~observed])))

    def test_scale_pooling_is_equal_not_native_pixel_count_weighted(self):
        logits = torch.cat((torch.zeros(4, 1), torch.full((1, 1), 2.)))
        values, _ = align_encoder_aliases(logits, torch.ones(1, 1, dtype=torch.bool), [(2, 2), (1, 1)],
                                         torch.ones(5, dtype=torch.bool), 2)
        expected = (.5+torch.tensor(2.).sigmoid())/2
        torch.testing.assert_close(values, torch.full_like(values, expected), atol=1e-7, rtol=0)

    def test_unused_negative_infinity_tokens_are_ignored(self):
        logits = torch.tensor([[0., -float("inf")]])
        values, _ = align_encoder_aliases(logits, torch.tensor([[True, False]]), [(1, 1)],
                                         torch.ones(1, dtype=torch.bool), 1)
        self.assertEqual(float(values[0, 0]), .5)

    def test_nonfinite_used_valid_token_fails(self):
        with self.assertRaises(ValueError):
            align_encoder_aliases(torch.tensor([[float("nan")]]), torch.ones(1, 1, dtype=torch.bool),
                                  [(1, 1)], torch.ones(1, dtype=torch.bool), 1)

    def test_unknown_observation_returns_local_exactly(self):
        local = torch.randn(3, 2)
        scores, observed = native_class_scores(torch.zeros(3, 2), [0, 1], local,
                                             torch.ones(3, dtype=torch.bool))
        self.assertTrue(torch.equal(scores, local))
        self.assertFalse(bool(observed.any()))

    def test_class_softmax_matches_native_probability_normalization(self):
        probability = torch.tensor([[.2, .4, .6, .8]])
        scores, _ = native_class_scores(probability, [0, 0, 1, 1], torch.zeros(1, 2),
                                       torch.ones(1, dtype=torch.bool))
        torch.testing.assert_close((scores/.07).softmax(-1), torch.tensor([[.3, .7]]), atol=1e-7, rtol=0)

    def test_native_level_size_mismatch_fails(self):
        with self.assertRaises(ValueError):
            align_encoder_aliases(torch.zeros(4, 1), torch.ones(1, 1, dtype=torch.bool), [(1, 1)],
                                  torch.ones(4, dtype=torch.bool), 2)

    def test_missing_observation_stays_exact_despite_neighbor_residual(self):
        local = torch.tensor([[[.2, .4], [.3, .5]]])
        observation = torch.tensor([[[2., -1.], [.3, .5]]])
        relation = torch.full((1, 2, 2), .5)
        result, _ = couple_encoder_scores(local, observation, relation,
            torch.ones(1, 2, dtype=torch.bool), torch.tensor([[True, False]]))
        self.assertTrue(torch.equal(result[0, 1], local[0, 1]))
        self.assertFalse(torch.equal(result[0, 0], local[0, 0]))

    def test_zero_innovation_is_exact_identity(self):
        local = torch.randn(1, 3, 2)
        result, _ = couple_encoder_scores(local, local, torch.eye(3)[None],
            torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool))
        self.assertTrue(torch.equal(result, local))


if __name__ == "__main__":
    unittest.main()
