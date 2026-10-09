import unittest

import torch

from dinotool.geometry_support_erasure import (
    balanced_support_scores, erase_supported_rgb, reconstruct_support_scores,
    semantic_score_influence,
)


class SupportErasureTests(unittest.TestCase):
    def test_inverse_matches_dense_system(self):
        torch.manual_seed(9)
        local, operator, observation = torch.randn(7, 3), torch.rand(4, 7), torch.randn(4, 3)
        result = reconstruct_support_scores(local, operator, observation)
        expected = torch.linalg.solve(torch.eye(7)+operator.T @ operator,
                                      local+operator.T @ observation)
        torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-6)

    def test_no_new_measurement_is_exact_identity(self):
        torch.manual_seed(10)
        local, operator = torch.randn(9, 4), torch.rand(3, 9)
        result = reconstruct_support_scores(local, operator, operator @ local)
        torch.testing.assert_close(result, local, rtol=0, atol=0)

    def test_unobserved_positions_do_not_change(self):
        local = torch.randn(5, 2)
        operator = torch.tensor([[.3, .7, 0., 0., 0.]])
        result = reconstruct_support_scores(local, operator, torch.randn(1, 2))
        torch.testing.assert_close(result[2:], local[2:], rtol=0, atol=0)

    def test_erasure_preserves_complement_and_input(self):
        rgb = torch.rand(1, 3, 4, 4)
        original = rgb.clone()
        mask = torch.tensor([[1., 0.], [0., 0.]])
        result = erase_supported_rgb(rgb, mask, torch.ones(2, 2, dtype=torch.bool))
        torch.testing.assert_close(result[:, :, 2:], rgb[:, :, 2:], rtol=0, atol=0)
        torch.testing.assert_close(result[:, :, :, 2:], rgb[:, :, :, 2:], rtol=0, atol=0)
        torch.testing.assert_close(result[:, :, :2, :2], rgb.mean((-2, -1), keepdim=True).expand(-1, -1, 2, 2))
        self.assertTrue(torch.equal(rgb, original))

    def test_padding_is_not_a_replacement_donor(self):
        rgb = torch.zeros(1, 3, 4, 4)
        rgb[:, :, :2, :2] = .8
        valid = torch.tensor([[True, False], [False, False]])
        result = erase_supported_rgb(rgb, torch.ones(2, 2), valid)
        torch.testing.assert_close(result, rgb, rtol=0, atol=0)

    def test_negative_operator_rejected(self):
        with self.assertRaises(ValueError):
            reconstruct_support_scores(torch.zeros(3, 2), -torch.ones(1, 3), torch.zeros(1, 2))

    def test_balanced_response_is_independent_of_uniform_support_size(self):
        for size in (1, 4, 16, 81):
            local = torch.tensor([[.2, .3]]).expand(size, -1)
            operator = torch.full((1, size), 1./size)
            observation = torch.tensor([[.4, .1]])
            result = balanced_support_scores(local, operator, observation)
            torch.testing.assert_close(result, torch.tensor([[.3, .2]]).expand(size, -1), atol=1e-6, rtol=1e-6)

    def test_balanced_no_measurement_change_is_identity(self):
        torch.manual_seed(11)
        local, operator = torch.randn(12, 3), torch.rand(4, 12)
        result = balanced_support_scores(local, operator, operator @ local)
        torch.testing.assert_close(result, local, atol=0, rtol=0)

    def test_zero_score_gain_is_exact_identity(self):
        local, operator = torch.randn(12, 3), torch.rand(4, 12).softmax(-1)
        result = semantic_score_influence(local, operator, torch.zeros(4, 3), torch.ones(12, dtype=torch.bool))
        torch.testing.assert_close(result, local, atol=0, rtol=0)

    def test_area_scaled_influence_is_invariant_to_support_size(self):
        for size in (4, 16, 64):
            local = torch.zeros(64, 2)
            operator = torch.zeros(1, 64)
            operator[:, :size] = 1./size
            measured_gain = torch.tensor([[.1, -.1]])*(size/64)
            result = semantic_score_influence(local, operator, measured_gain, torch.ones(64, dtype=torch.bool))
            torch.testing.assert_close(result[:size], torch.tensor([[.05, -.05]]).expand(size, -1), atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(result[size:], local[size:], atol=0, rtol=0)

    def test_common_class_gain_does_not_change_competition(self):
        local = torch.tensor([[.1, .4, .2]]).expand(8, -1)
        result = semantic_score_influence(local, torch.full((1, 8), 1./8), torch.full((1, 3), .1),
                                          torch.ones(8, dtype=torch.bool))
        self.assertTrue(torch.equal(local.argmax(-1), result.argmax(-1)))

    def test_class_permutation_equivariance(self):
        local, operator, gain = torch.randn(10, 3), torch.rand(2, 10).softmax(-1), torch.randn(2, 3)
        valid, permutation = torch.ones(10, dtype=torch.bool), torch.tensor([2, 0, 1])
        original = semantic_score_influence(local, operator, gain, valid)
        permuted = semantic_score_influence(local[:, permutation], operator, gain[:, permutation], valid)
        torch.testing.assert_close(permuted, original[:, permutation])


if __name__ == "__main__":
    unittest.main()
