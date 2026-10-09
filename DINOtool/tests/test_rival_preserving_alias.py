import json
import subprocess
import sys
import unittest

import numpy as np
import torch

from dinotool.rival_preserving_alias import competitive_potential, directional_controls


class RivalPreservingAliasTest(unittest.TestCase):
    def fields(self):
        generator = torch.Generator().manual_seed(81)
        cap = torch.rand(9, 4, generator=generator, dtype=torch.float64)+.1
        directed = -torch.rand(9, 4, 4, generator=generator, dtype=torch.float64)*cap[..., None]
        directed[:, torch.arange(4), torch.arange(4)] = 0
        valid = torch.ones(9, dtype=torch.bool)
        valid[-1] = False
        directed[-1] = 0
        return directed, cap, valid

    def test_zero_actions_preserve_original_exactly(self):
        directed, _, valid = self.fields()
        v, e, _ = competitive_potential(torch.zeros_like(directed), valid)
        self.assertEqual(float(v.abs().max()), 0)
        self.assertEqual(float(e.abs().max()), 0)
        base, operator = torch.randn(9, 4).double(), torch.randn(9, 9).double()
        torch.testing.assert_close(base+operator @ v, base, atol=0, rtol=0)

    def test_two_class_margin_has_no_risk_weight_shrinkage(self):
        a = torch.tensor([[[0., -.8], [0., 0.]]], dtype=torch.float64)
        v, _, _ = competitive_potential(a, torch.ones(1, dtype=torch.bool))
        torch.testing.assert_close(v, torch.tensor([[-.4, .4]], dtype=torch.float64), atol=0, rtol=0)

    def test_matches_standard_least_squares(self):
        directed, _, valid = self.fields()
        v, e, _ = competitive_potential(directed, valid)
        design, targets = [], []
        for c in range(4):
            for d in range(c+1, 4):
                row = np.zeros(4)
                row[c], row[d] = 1, -1
                design.append(row)
                targets.append(e[:, c, d].numpy())
        code = 'import json,sys,numpy as np; a,b=json.load(sys.stdin); print(json.dumps(np.linalg.lstsq(np.asarray(a),np.asarray(b),rcond=None)[0].T.tolist()))'
        process = subprocess.run([sys.executable, '-c', code], input=json.dumps([np.asarray(design).tolist(), np.asarray(targets).tolist()]),
            capture_output=True, text=True, check=True)
        oracle = json.loads(process.stdout)
        np.testing.assert_allclose(v.numpy(), oracle, atol=1e-12, rtol=0)

    def test_cycle_component_is_discarded(self):
        a = torch.zeros(1, 3, 3, dtype=torch.float64)
        a[0, 0, 1] = a[0, 1, 2] = a[0, 2, 0] = -1
        v, _, stats = competitive_potential(a, torch.ones(1, dtype=torch.bool))
        self.assertEqual(float(v.abs().max()), 0)
        self.assertGreater(stats['discarded_cycle_energy'], 0)

    def test_common_class_suppression_cancels(self):
        a = torch.full((2, 4, 4), -.3, dtype=torch.float64)
        a[:, torch.arange(4), torch.arange(4)] = 0
        v, _, _ = competitive_potential(a, torch.ones(2, dtype=torch.bool))
        self.assertEqual(float(v.abs().max()), 0)

    def test_rival_independent_actions_recover_centered_old_action(self):
        action = -torch.tensor([[.1, .3, .8]], dtype=torch.float64)
        directed = action[..., None].expand(1, 3, 3).clone()
        directed[:, torch.arange(3), torch.arange(3)] = 0
        v, _, _ = competitive_potential(directed, torch.ones(1, dtype=torch.bool))
        torch.testing.assert_close(v, action-action.mean(-1, keepdim=True), atol=1e-15, rtol=0)

    def test_unrelated_final_margin_need_not_be_invariant(self):
        directed = torch.zeros(1, 3, 3, dtype=torch.float64)
        directed[0, 0, 1] = -.9
        v, e, _ = competitive_potential(directed, torch.ones(1, dtype=torch.bool))
        self.assertEqual(float(e[0, 0, 2]), 0)
        self.assertAlmostEqual(float(v[0, 0]-v[0, 2]), -.3)

    def test_class_permutation_and_energy_identity(self):
        directed, _, valid = self.fields()
        permutation = torch.tensor([2, 0, 3, 1])
        v, _, stats = competitive_potential(directed, valid)
        changed, _, _ = competitive_potential(directed[:, permutation][:, :, permutation], valid)
        torch.testing.assert_close(changed, v[:, permutation], atol=1e-12, rtol=0)
        self.assertAlmostEqual(stats['requested_margin_energy'],
            stats['discarded_cycle_energy']+stats['realized_margin_energy'], places=12)

    def test_potential_bounds_follow_directed_capacities(self):
        directed, cap, valid = self.fields()
        v, _, _ = competitive_potential(directed, valid)
        lower, upper = -3*cap/4, (cap.sum(-1, keepdim=True)-cap)/4
        self.assertTrue(bool(((v >= lower) & (v <= upper)).all()))

    def test_controls_preserve_pair_budgets_caps_and_shuffle_spectra(self):
        directed, cap, valid = self.fields()
        a, bounds, known = directed.numpy(), cap.numpy(), valid.numpy()
        controls = directional_controls(a, bounds, known)
        for name, changed in controls.items():
            np.testing.assert_allclose(changed.sum(0), a.sum(0), atol=1e-12, rtol=0)
            self.assertTrue((changed >= -bounds[:, :, None]).all())
            self.assertTrue((changed <= 0).all())
            self.assertFalse(changed[~known].any())
            v, _, _ = competitive_potential(torch.from_numpy(changed), valid)
            base, _, _ = competitive_potential(directed, valid)
            torch.testing.assert_close(v.sum(0), base.sum(0), atol=1e-12, rtol=0)
            if name.startswith('DirectionalShuffle'):
                np.testing.assert_array_equal(np.sort(changed[known], axis=0), np.sort(a[known], axis=0))

    def test_controls_are_deterministic_and_zero_safe(self):
        directed, cap, valid = self.fields()
        first = directional_controls(directed.numpy(), cap.numpy(), valid.numpy())
        second = directional_controls(directed.numpy(), cap.numpy(), valid.numpy())
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        zero = directional_controls(np.zeros_like(directed.numpy()), cap.numpy(), valid.numpy())
        self.assertFalse(any(a.any() for a in zero.values()))

    def test_invalid_directed_inputs_fail(self):
        directed, _, valid = self.fields()
        for changed in (directed.abs(), directed+torch.eye(4)[None]*-.1):
            with self.assertRaises(ValueError):
                competitive_potential(changed, valid)


if __name__ == '__main__':
    unittest.main()
