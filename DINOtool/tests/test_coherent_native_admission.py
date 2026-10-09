import math
import unittest
from dataclasses import replace

import torch

from dinotool.coherent_native_admission import (CONFIG, PRIMARY, ALIAS_SHUFFLES,
    native_risk, source_controls, coherent_observation, hard_observation, action_controls)
from dinotool.stratified_soft_alias import WideCrop


class CoherentNativeAdmissionTest(unittest.TestCase):
    def fixture(self):
        field = torch.tensor([[[1., 3.], [3., 1.]], [[3., 1.], [1., 3.]], [[0., 0.], [0., 0.]]], dtype=torch.float64)
        known = torch.ones(2, 2, dtype=torch.bool)
        assignments, parents = torch.tensor([0, 1, 0, 1]), torch.tensor([0, 0, 1, 1])
        valid, members = torch.tensor([True, True, False]), torch.tensor([[0, 1], [2, 3]])
        return field, known, assignments, parents, valid, members

    def writer_fixture(self):
        crop = WideCrop(torch.tensor([[4., 0., 0., 3.]]).expand(441, -1).clone(),
            torch.zeros(4), 0, 0, 336, 336)
        return [crop], torch.ones(336, 336), torch.tensor([[8., 8.], [24., 24.], [40., 40.]]), (336, 336)

    def test_unknown_and_invalid_are_neutral(self):
        f, k, a, p, v, _ = self.fixture()
        self.assertEqual(float(native_risk(f, torch.zeros_like(k), a, p, v).abs().max()), 0.)
        k[:, 0] = False
        self.assertEqual(float(native_risk(f, k, a, p, v).abs().max()), 0.)
        self.assertEqual(float(native_risk(f, torch.ones_like(k), a, p, v)[~v].abs().max()), 0.)

    def test_native_margin_offset_and_canonical_behavior(self):
        f, k, a, p, v, _ = self.fixture()
        risk = native_risk(f, k, a, p, v)
        self.assertAlmostEqual(float(risk[0, 0]), 1-math.exp(-2))
        self.assertEqual(float(risk[0, 1]), 0.)
        torch.testing.assert_close(risk, native_risk(f+17., k, a, p, v), atol=0, rtol=0)

    def test_source_controls_and_unprotected_spectrum(self):
        f, k, a, p, v, members = self.fixture()
        canonical = torch.tensor([0, 2])
        output = source_controls(torch.zeros(3, 4, 2), f, k, a, p, canonical,
            v, members, f[:, :1], k[:1])
        self.assertGreater(float(output[PRIMARY][:, canonical].max()), 0.)
        self.assertEqual(float(output['CoherentProtected_Exact'][:, canonical].abs().max()), 0.)
        for name in ALIAS_SHUFFLES:
            torch.testing.assert_close(output[name][:, members].sort(-1).values,
                output[PRIMARY][:, members].sort(-1).values, atol=0, rtol=0)

    def test_zero_risk_identity_and_uniform_response_identity(self):
        _, _, _, _, v, members = self.fixture()
        args = self.writer_fixture()
        delta, _, _ = coherent_observation(*args, members, torch.zeros(3, 4), v)
        self.assertEqual(float(delta.abs().max()), 0.)
        args[0][0].alias_logits[:] = 1.
        delta, _, _ = coherent_observation(*args, members, torch.ones(3, 4), v)
        self.assertEqual(float(delta.abs().max()), 0.)

    def test_soft_bounds_coherence_and_chunk_parity(self):
        _, _, _, _, v, members = self.fixture()
        args = self.writer_fixture()
        risk = torch.ones(3, 4)
        delta, cap, _ = coherent_observation(*args, members, risk, v)
        self.assertTrue(bool(((delta <= 0) & (delta >= -cap-1e-12)).all()))
        self.assertLessEqual(float(-delta.min()), math.log(2))
        margins = delta[:, :, None]-delta[:, None]
        self.assertEqual(float((margins[:, 0, 1]+margins[:, 1, 0]).abs().max()), 0.)
        other, _, _ = coherent_observation(*args, members, risk, v, replace(CONFIG, query_chunk=1))
        torch.testing.assert_close(delta, other, atol=0, rtol=0)
        self.assertEqual(float(delta[~v].abs().max()), 0.)

    def test_hard_empty_class_and_zero_risk_are_neutral(self):
        _, _, _, _, v, members = self.fixture()
        args = self.writer_fixture()
        for risk in (torch.ones(3, 4), torch.zeros(3, 4)):
            delta, _ = hard_observation(*args, members, risk, v)
            self.assertEqual(float(delta.abs().max()), 0.)

    def test_action_controls_keep_each_class_budget(self):
        _, _, _, _, v, _ = self.fixture()
        action = torch.tensor([[-.2, -.3], [-.1, -.5], [0., 0.]], dtype=torch.float64)
        for delta in action_controls(action, v).values():
            torch.testing.assert_close(delta.sum(0), action.sum(0), atol=1e-15, rtol=0)
            self.assertEqual(float(delta[~v].abs().max()), 0.)

    def test_wrong_shapes_and_unbounded_risk_fail(self):
        f, k, a, p, v, members = self.fixture()
        with self.assertRaises(ValueError):
            native_risk(f, k[:, :1], a, p, v)
        with self.assertRaises(ValueError):
            coherent_observation(*self.writer_fixture(), members, torch.ones(3, 4)*2, v)


if __name__ == '__main__':
    unittest.main()
