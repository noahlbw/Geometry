import unittest

import torch
import torch.nn.functional as F

from dinotool.geometry_residual_metric import (
    metric_alias_scores, residual_covariance, spatial_relation, text_span, valid_relation)


class ResidualMetricTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(32)
        self.features = F.normalize(torch.randn(1, 5, 8), dim=-1)
        self.text = F.normalize(torch.randn(4, 8), dim=-1)
        self.basis, _ = text_span({"test": self.text})
        self.valid = torch.ones(1, 5, dtype=torch.bool)
        self.relation = torch.rand(1, 5, 5).softmax(-1)

    def test_text_basis_preserves_all_alias_directions(self):
        torch.testing.assert_close(self.text @ self.basis @ self.basis.T, self.text, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(self.basis.T @ self.basis, torch.eye(4), atol=2e-6, rtol=2e-6)

    def test_identity_geometry_has_exact_zero_covariance(self):
        covariance = residual_covariance(self.features @ self.basis, torch.eye(5)[None], self.valid)
        self.assertTrue(torch.equal(covariance, torch.zeros_like(covariance)))

    def test_zero_covariance_exactly_recovers_original_scores(self):
        scores, _ = metric_alias_scores(self.features, self.text, self.basis, torch.zeros(1, 4, 4), self.valid)
        self.assertTrue(torch.equal(scores, self.features @ self.text.T))

    def test_covariance_is_positive_semidefinite(self):
        covariance = residual_covariance(self.features @ self.basis, self.relation, self.valid)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(covariance).min()), -1e-7)

    def test_uniform_geometry_recovers_centered_covariance(self):
        projected = self.features @ self.basis
        covariance = residual_covariance(projected, torch.ones_like(self.relation), self.valid)
        centered = projected-projected.mean(1, keepdim=True)
        torch.testing.assert_close(covariance, centered.transpose(-1, -2) @ centered/5)

    def test_trace_normalization_bounds_metric(self):
        covariance = residual_covariance(self.features @ self.basis, self.relation, self.valid)
        metric = torch.eye(4)[None]+covariance/covariance.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]
        eigenvalues = torch.linalg.eigvalsh(metric)
        self.assertGreaterEqual(float(eigenvalues.min()), 1.-1e-6)
        self.assertLessEqual(float(eigenvalues.max()), 2.+1e-6)

    def test_scores_equal_explicit_full_space_metric(self):
        covariance = residual_covariance(self.features @ self.basis, self.relation, self.valid)
        scores, _ = metric_alias_scores(self.features, self.text, self.basis, covariance, self.valid)
        inverse = torch.linalg.inv(torch.eye(4)+covariance[0]/covariance[0].trace())
        metric = torch.eye(8)+self.basis @ (inverse-torch.eye(4)) @ self.basis.T
        numerator = self.features @ metric @ self.text.T
        fy = ((self.features @ metric)*self.features).sum(-1).sqrt()[..., None]
        ft = ((self.text @ metric)*self.text).sum(-1).sqrt()[None, None]
        torch.testing.assert_close(scores, numerator/(fy*ft), atol=3e-6, rtol=3e-6)

    def test_padding_does_not_enter_residuals(self):
        valid = torch.tensor([[True, True, True, False, False]])
        projected = self.features @ self.basis
        first = residual_covariance(projected, self.relation, valid)
        changed = projected.clone()
        changed[:, 3:] = 10000
        torch.testing.assert_close(first, residual_covariance(changed, self.relation, valid), rtol=0, atol=0)

    def test_invalid_queries_keep_exact_original_scores(self):
        valid = torch.tensor([[True, True, True, False, False]])
        covariance = residual_covariance(self.features @ self.basis, self.relation, valid)
        scores, _ = metric_alias_scores(self.features, self.text, self.basis, covariance, valid)
        self.assertTrue(torch.equal(scores[:, 3:], (self.features @ self.text.T)[:, 3:]))

    def test_absent_geometry_rows_are_identity_not_absence_evidence(self):
        weights = valid_relation(torch.zeros_like(self.relation), self.valid)
        self.assertTrue(torch.equal(weights, torch.eye(5)[None]))

    def test_joint_patch_permutation_preserves_measurement(self):
        projected = self.features @ self.basis
        permutation = torch.tensor([2, 0, 4, 1, 3])
        first = residual_covariance(projected, self.relation, self.valid)
        second = residual_covariance(projected[:, permutation],
            self.relation[:, permutation][:, :, permutation], self.valid[:, permutation])
        torch.testing.assert_close(first, second, atol=1e-7, rtol=2e-6)

    def test_basis_rotation_does_not_change_readout(self):
        rotation, _ = torch.linalg.qr(torch.randn(4, 4))
        covariance = residual_covariance(self.features @ self.basis, self.relation, self.valid)
        first, _ = metric_alias_scores(self.features, self.text, self.basis, covariance, self.valid)
        second, _ = metric_alias_scores(self.features, self.text, self.basis @ rotation,
            rotation.T @ covariance @ rotation, self.valid)
        torch.testing.assert_close(first, second, atol=3e-6, rtol=3e-6)

    def test_constant_descriptors_have_no_residual(self):
        projected = torch.ones(1, 5, 4)
        covariance = residual_covariance(projected, torch.ones_like(self.relation), self.valid)
        self.assertTrue(torch.equal(covariance, torch.zeros_like(covariance)))

    def test_constant_descriptor_roundoff_is_not_amplified(self):
        features = self.features[:, :1].expand(-1, 5, -1)
        covariance = residual_covariance(features @ self.basis, self.relation, self.valid)
        scores, _ = metric_alias_scores(features, self.text, self.basis, covariance, self.valid)
        self.assertTrue(torch.equal(scores, features @ self.text.T))

    def test_empty_validity_is_exact_neutral(self):
        valid = torch.zeros_like(self.valid)
        covariance = residual_covariance(self.features @ self.basis, self.relation, valid)
        scores, _ = metric_alias_scores(self.features, self.text, self.basis, covariance, valid)
        self.assertTrue(torch.equal(scores, self.features @ self.text.T))

    def test_repeated_aliases_do_not_change_text_span_metric(self):
        basis, _ = text_span({"test": self.text.repeat_interleave(2, 0)})
        first_cov = residual_covariance(self.features @ self.basis, self.relation, self.valid)
        second_cov = residual_covariance(self.features @ basis, self.relation, self.valid)
        first, _ = metric_alias_scores(self.features, self.text, self.basis, first_cov, self.valid)
        second, _ = metric_alias_scores(self.features, self.text, basis, second_cov, self.valid)
        torch.testing.assert_close(first, second, atol=3e-6, rtol=3e-6)

    def test_invalid_geometry_and_lattice_rejected(self):
        with self.assertRaises(ValueError):
            valid_relation(-self.relation, self.valid)
        with self.assertRaises(ValueError):
            spatial_relation(5, self.features.device)


if __name__ == "__main__":
    unittest.main()
