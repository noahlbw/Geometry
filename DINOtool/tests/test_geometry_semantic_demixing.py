import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.optimize import nnls
import torch

from dinotool.geometry_semantic_demixing import (
    DemixConfig, JointSemanticDecoder, read_semantic_demixing, reconstruction_objective)


class SemanticDemixingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(16)
        self.text = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
        self.parents = torch.tensor([0, 0, 1, 1, 2, 2])
        self.config = DemixConfig(maximum_iterations=4096, kkt_tolerance=1e-6)
        self.decoder = JointSemanticDecoder(self.text, self.parents, 3, self.config)
        self.y = torch.randn(4, 8)
        self.g = torch.randn(4, 4).softmax(-1)

    def test_local_agrees_with_scipy_nnls(self):
        actual, diagnostics = self.decoder.solve(self.y)
        design = np.concatenate((self.text.numpy().T, np.sqrt(self.config.ridge)*np.eye(6)))
        expected = np.stack([nnls(design, np.concatenate((row.numpy(), np.zeros(6))))[0] for row in self.y])
        np.testing.assert_allclose(actual.numpy(), expected, atol=2e-5, rtol=2e-5)
        self.assertLessEqual(diagnostics["solver_kkt_relative"], self.config.kkt_tolerance)

    def test_coupled_agrees_with_dense_reference(self):
        identity = np.eye(4)
        observation = np.concatenate((identity, self.g.numpy()), axis=0)/np.sqrt(2)
        # Flatten patch-major coefficients and observed descriptors consistently.
        design = np.kron(observation, self.text.numpy().T)
        design = np.concatenate((design, np.sqrt(self.config.ridge)*np.eye(24)), axis=0)
        target = np.concatenate(((observation@self.y.numpy()).reshape(-1), np.zeros(24)))
        expected = nnls(design, target, maxiter=10000)[0].reshape(4, 6)
        actual, diagnostics = self.decoder.solve(self.y, self.g)
        np.testing.assert_allclose(actual.numpy(), expected, atol=8e-5, rtol=8e-5)
        self.assertLessEqual(diagnostics["final_objective"], diagnostics["initial_objective"])

    def test_identity_relation_exactly_recovers_text_control(self):
        first, _ = self.decoder.solve(self.y)
        second, _ = self.decoder.solve(self.y, torch.eye(4))
        self.assertTrue(torch.equal(first, second))

    def test_zero_information_is_exact_neutral_scores(self):
        scores, _ = self.decoder.decode(torch.zeros_like(self.y), self.g, torch.ones(4, dtype=torch.bool))
        self.assertTrue(torch.equal(scores, torch.zeros(4, 3)))

    def test_padding_is_excluded_and_unchanged(self):
        valid = torch.tensor([True, True, False, False])
        scores, _ = self.decoder.decode(self.y, self.g, valid)
        perturbed = self.y.clone()
        perturbed[~valid] = 99
        second, _ = self.decoder.decode(perturbed, self.g, valid)
        torch.testing.assert_close(scores[valid], second[valid], atol=0, rtol=0)
        baseline, _ = self.decoder.decode(self.y, self.g, torch.zeros(4, dtype=torch.bool))
        self.assertTrue(torch.equal(scores[~valid], baseline[~valid]))

    def test_geometry_measures_residual_not_label_agreement(self):
        x = torch.tensor([[1., 0, 0, 0, 0, 0], [0, 0, 1., 0, 0, 0],
                          [0, 0, 0, 0, 1., 0], [1., 0, 0, 0, 0, 0]])
        y = x @ self.text
        local = reconstruction_objective(y, x, self.text, None, self.config.ridge)
        coupled = reconstruction_objective(y, x, self.text, self.g, self.config.ridge)
        self.assertTrue(torch.equal(local, coupled))

    def test_duplicate_aliases_across_classes_are_not_falsely_identified(self):
        text = torch.tensor([[1., 0], [1., 0]])
        decoder = JointSemanticDecoder(text, torch.tensor([0, 1]), 2, self.config)
        x, _ = decoder.solve(text[:1])
        torch.testing.assert_close(x[:, 0], x[:, 1], atol=1e-6, rtol=1e-6)

    def test_orthogonal_single_alias_scores_are_positive_cosines(self):
        decoder = JointSemanticDecoder(torch.eye(3), torch.arange(3), 3, self.config)
        y = torch.tensor([[.8, -.2, .3]])
        scores, _ = decoder.decode(y, None, torch.tensor([True]))
        torch.testing.assert_close(scores, y.clamp_min(0)/(1+self.config.ridge), atol=1e-6, rtol=1e-6)

    def test_inputs_are_immutable(self):
        before = (self.y.clone(), self.g.clone(), self.text.clone(), self.parents.clone())
        self.decoder.decode(self.y, self.g, torch.ones(4, dtype=torch.bool))
        for actual, original in zip((self.y, self.g, self.text, self.parents), before):
            self.assertTrue(torch.equal(actual, original))

    def test_malformed_input_is_rejected(self):
        for bad in (torch.zeros(4, 4), self.g*float("nan"), -self.g):
            with self.assertRaises(ValueError):
                self.decoder.solve(self.y, bad)
        with self.assertRaises(ValueError):
            JointSemanticDecoder(self.text, self.parents, 4)
        with self.assertRaises(ValueError):
            DemixConfig(ridge=-1).validate()

    def test_correlated_dictionary_and_concentrated_support_converge(self):
        common = torch.randn(1, 32)
        text = torch.nn.functional.normalize(common+.15*torch.randn(40, 32), dim=-1)
        decoder = JointSemanticDecoder(text, torch.arange(40)//10, 4)
        features = torch.nn.functional.normalize(torch.randn(12, 32)+common, dim=-1)
        relation = torch.zeros(12, 12)
        relation[:, 0] = .9
        relation += .1/12
        coefficients, stats = decoder.solve(features, relation)
        self.assertTrue(bool(torch.isfinite(coefficients).all()))
        self.assertLessEqual(stats["solver_kkt_relative"], decoder.config.kkt_tolerance)
        self.assertLess(stats["solver_backtracks"], 8)

    def test_readout_passes_unnormalized_native_patches_to_matched_controls(self):
        tokens = 3*torch.randn(1, 6, 8)
        aligned = torch.nn.functional.normalize(self.y, dim=-1)[None]
        prepared = SimpleNamespace(backbone_tokens=tokens, prefix_tokens=2, block_index=1,
            geometry_patch_conditional=self.g[None], geometry_projected=aligned)
        backbone = SimpleNamespace(_autocast=nullcontext,
            model=SimpleNamespace(visual_model=SimpleNamespace(head=object())))
        bank = SimpleNamespace(features=self.text, parent_indices=self.parents, class_count=3)
        def control(head, original, raw, *args):
            self.assertTrue(torch.equal(raw, tokens[:, 2:]))
            return aligned, {}
        with patch("dinotool.geometry_semantic_demixing.run_head", side_effect=control) as mocked:
            read_semantic_demixing(SimpleNamespace(backbone=backbone), prepared, {"a": bank},
                {"a": self.text}, torch.ones(1, 4, dtype=torch.bool), {"a": self.decoder})
        self.assertEqual(mocked.call_count, 2)
        self.assertTrue(torch.equal(prepared.backbone_tokens, tokens))


if __name__ == "__main__":
    unittest.main()
