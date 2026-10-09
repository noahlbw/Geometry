import unittest

import torch

from dinotool.rival_fine_support import (PRIMARY, NEW_METHODS, support_weights,
    support_control, support_scores)
from dinotool.rival_competition_admission import retained_competition_scores
from test_rival_alias_fast import RivalAliasFastTest


class FineSupportTests(unittest.TestCase):
    def test_weight_uses_fine_support_beyond_old_risk(self):
        native = torch.tensor([[[0., 0.], [-3., 3.], [3., -3.], [0., 0.]]])
        risk = torch.zeros_like(native)
        result = support_weights(native, risk, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 3]), torch.tensor([True]))
        self.assertEqual(float(result[:, [0, 3]].min()), 1.)
        self.assertGreater(float(result[0, 1, 1]), .5)
        self.assertLess(float(result[0, 2, 0]), .99)
        native[0, 1, 1] = -3.
        changed = support_weights(native, risk, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 3]), torch.tensor([True]))
        self.assertLess(float(changed[0, 1, 1]), .5)

    def test_controls_preserve_fixed_hard_support_and_mass(self):
        args = RivalAliasFastTest().fixture()
        risk = retained_competition_scores(*args)[1]
        weights = support_weights(torch.randn_like(risk), risk, args[12], args[11], args[9])
        for seed in (None, 1, 2, 3):
            result = support_control(weights, risk, args[10], args[11], seed)
            self.assertTrue(torch.equal(result[:, args[11]], weights[:, args[11]]))
            self.assertTrue(torch.equal(result[risk > 0], weights[risk > 0]))
            torch.testing.assert_close(result[:, args[10]].sum(2), weights[:, args[10]].sum(2), atol=1e-12, rtol=0)
            if seed is not None:
                self.assertTrue(torch.equal(result[:, args[10]].sort(2).values, weights[:, args[10]].sort(2).values))

    def test_replay_and_primary_singleton_are_exact(self):
        for k in (20, 40):
            args = RivalAliasFastTest().fixture(k=k)
            values, _ = support_scores(*args)
            old = retained_competition_scores(*args)[0]
            for method in ('NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact'):
                self.assertTrue(torch.equal(values[method], old[method]))
            solo = support_scores(*args, methods=(PRIMARY,))[0][PRIMARY]
            self.assertTrue(torch.equal(solo, values[PRIMARY]))

    def test_invalid_rows_create_no_action(self):
        args = list(RivalAliasFastTest().fixture())
        args[9].fill_(False)
        values, _ = support_scores(*args)
        self.assertTrue(all(torch.equal(values[m], values['NoAdmission_Exact']) for m in NEW_METHODS))


if __name__ == '__main__':
    unittest.main()
