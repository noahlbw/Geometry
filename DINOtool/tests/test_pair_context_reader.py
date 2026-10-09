import json
import subprocess
import sys
import unittest

import numpy as np
import torch

from dinotool.pair_context_reader import pair_risk, pair_observation, reconcile
from dinotool.stratified_soft_alias import WideCrop
from dinotool.target_context_alias import intervention_retention, observation_delta


class PairContextReaderTest(unittest.TestCase):
    def observations(self):
        full = torch.tensor([[.8, .9, .3, .4], [.1, .3, .9, .8]])
        kept = torch.tensor([[[.1, .2, .9, .4], [.6, .1, .1, .3]],
                             [[.2, .3, .8, .5], [.5, .2, .2, .2]]])
        removed = torch.zeros_like(kept)
        return full, kept, removed, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 2]), torch.ones(2, dtype=torch.bool)

    def test_max_rival_exactly_recovers_old_source(self):
        args = self.observations()
        risk = pair_risk(*args)
        old, _ = intervention_retention(*args)
        torch.testing.assert_close(1-risk.amax(-1), old, atol=0, rtol=0)

    def test_canonical_unknown_and_self_pairs_are_inert(self):
        args = list(self.observations())
        args[-1][1] = False
        risk = pair_risk(*args)
        self.assertEqual(float(risk[:, args[-2]].abs().max()), 0)
        self.assertEqual(float(risk[1].abs().max()), 0)
        for a, c in enumerate(args[-3]):
            self.assertEqual(float(risk[:, a, c].abs().max()), 0)

    def test_two_fill_rule_is_conservative(self):
        args = self.observations()
        torch.testing.assert_close(pair_risk(*args), torch.minimum(pair_risk(*args, fill=0), pair_risk(*args, fill=1)))

    def test_pair_writer_matches_existing_writer_for_each_rival(self):
        generator = torch.Generator().manual_seed(71)
        members = torch.arange(6).reshape(3, 2)
        crop = WideCrop(torch.randn(441, 6, generator=generator), torch.randn(6, generator=generator), 0, 0, 336, 336)
        coordinates = torch.tensor([[8., 8.], [120., 120.], [248., 248.]])
        valid, count = torch.tensor([True, True, False]), torch.ones(336, 336)
        risk = torch.rand(3, 6, 3, generator=generator)
        margins, weight, directed = pair_observation([crop], count, coordinates, (512, 512), members, risk, valid)
        for d in range(3):
            old = observation_delta([crop], count, coordinates, (512, 512), members, 1-risk[:, :, d], valid)
            torch.testing.assert_close(directed[:, :, d], old, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(margins, -margins.transpose(-1, -2), atol=0, rtol=0)
        torch.testing.assert_close(weight, weight.transpose(-1, -2), atol=0, rtol=0)

    def test_pair_writer_identity_is_exact(self):
        members = torch.arange(4).reshape(2, 2)
        crop = WideCrop(torch.ones(441, 4), torch.zeros(4), 0, 0, 336, 336)
        for risk in (torch.zeros(1, 4, 2), torch.ones(1, 4, 2)):
            _, _, directed = pair_observation([crop], torch.ones(336, 336), torch.tensor([[8., 8.]]),
                (512, 512), members, risk, torch.ones(1, dtype=torch.bool))
            self.assertEqual(float(directed.abs().max()), 0)

    def fields(self, n=7, classes=4):
        generator = torch.Generator().manual_seed(19)
        e = torch.randn(n, classes, classes, generator=generator, dtype=torch.float64)
        e = e-e.transpose(-1, -2)
        w = torch.rand(n, classes, classes, generator=generator, dtype=torch.float64)
        w = .5*(w+w.transpose(-1, -2))
        cap = torch.rand(n, classes, generator=generator, dtype=torch.float64)+.1
        return e, w, cap, torch.ones(n, dtype=torch.bool)

    def test_solution_matches_standard_bounded_least_squares(self):
        e, w, cap, valid = self.fields()
        result, diag = reconcile(e, w, cap, valid)
        # Keep SciPy and torch's OpenMP runtimes isolated on Windows.
        code = '''
import json, sys
import numpy as np
from scipy.optimize import lsq_linear
e, w, cap = [np.asarray(a) for a in json.load(sys.stdin)]
classes, answers = cap.shape[-1], []
for i in range(len(cap)):
    rows, values = [np.eye(classes)], [np.zeros(classes)]
    for c in range(classes):
        for d in range(c+1, classes):
            row = np.zeros((1, classes))
            row[0, c], row[0, d] = np.sqrt(w[i, c, d]), -np.sqrt(w[i, c, d])
            rows.append(row)
            values.append(np.array([np.sqrt(w[i, c, d])*e[i, c, d]]))
    oracle = lsq_linear(np.concatenate(rows), np.concatenate(values), bounds=(-cap[i], np.zeros(classes)), tol=1e-13)
    assert oracle.success
    answers.append(oracle.x.tolist())
print(json.dumps(answers))
'''
        process = subprocess.run([sys.executable, '-c', code], input=json.dumps([a.tolist() for a in (e, w, cap)]),
                                 capture_output=True, text=True, check=True)
        np.testing.assert_allclose(result.numpy(), json.loads(process.stdout), atol=1e-8, rtol=0)
        self.assertLessEqual(diag['projected_kkt_max_error'], 1e-10)

    def test_two_class_closed_form_and_capacity(self):
        e = torch.tensor([[[0., -2.], [2., 0.]]], dtype=torch.float64)
        w = torch.ones_like(e)
        for bound, expected in ((3., -1.), (.2, -.2)):
            x, _ = reconcile(e, w, torch.full((1, 2), bound, dtype=torch.float64), torch.ones(1, dtype=torch.bool))
            torch.testing.assert_close(x, torch.tensor([[expected, 0.]], dtype=torch.float64), atol=1e-12, rtol=0)

    def test_zero_evidence_exactly_recovers_identity(self):
        e, w, cap, valid = self.fields()
        for margin, weight in ((torch.zeros_like(e), w), (e, torch.zeros_like(w))):
            x, _ = reconcile(margin, weight, cap, valid)
            self.assertEqual(float(x.abs().max()), 0)

    def test_invalid_and_zero_capacity_are_inert(self):
        e, w, cap, valid = self.fields()
        valid[2] = False
        cap[1, 0] = 0
        x, _ = reconcile(e, w, cap, valid)
        self.assertEqual(float(x[2].abs().max()), 0)
        self.assertEqual(float(x[1, 0]), 0)
        self.assertTrue(bool((x <= 0).all() and (x >= -cap).all()))

    def test_class_permutation_and_objective(self):
        e, w, cap, valid = self.fields()
        permutation = torch.tensor([2, 0, 3, 1])
        x, diag = reconcile(e, w, cap, valid)
        changed, _ = reconcile(e[:, permutation][:, :, permutation], w[:, permutation][:, :, permutation], cap[:, permutation], valid)
        torch.testing.assert_close(changed, x[:, permutation], atol=1e-9, rtol=0)
        self.assertLessEqual(diag['objective'], diag['identity_objective']+1e-10)

    def test_invalid_inputs_fail(self):
        e, w, cap, valid = self.fields()
        with self.assertRaises(ValueError):
            reconcile(torch.ones_like(e), w, cap, valid)
        with self.assertRaises(ValueError):
            reconcile(e, -w, cap, valid)


if __name__ == '__main__':
    unittest.main()
