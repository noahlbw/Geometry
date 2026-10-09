import unittest

import torch

from dinotool.contrastive_reversal_alias import reversal_risk


class ContrastiveReversalAliasTest(unittest.TestCase):
    def inputs(self):
        full = torch.tensor([[.4, .6, .3, .2]], dtype=torch.float64)
        kept = torch.tensor([[[.2, .1, .5, .4]], [[.2, .1, .5, .4]]], dtype=torch.float64)
        removed = torch.tensor([[[.1, .6, .2, .2]], [[.1, .6, .2, .2]]], dtype=torch.float64)
        return [full, kept, removed, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 2]), torch.ones(1, dtype=torch.bool)]

    def test_background_exclusive_advantage_is_attenuated(self):
        risk = reversal_risk(*self.inputs())['reversal']
        self.assertAlmostEqual(float(risk[0, 1, 1]), .5)

    def test_target_supported_alias_stays_unknown(self):
        args = self.inputs()
        args[1][:, 0, 1] = .8
        self.assertEqual(float(reversal_risk(*args)['reversal'][0, 1, 1]), 0)

    def test_all_three_signs_required(self):
        for tensor, index, value in ((0, (0, 1), .2), (1, (slice(None), 0, 1), .6), (2, (slice(None), 0, 1), .1)):
            args = self.inputs()
            args[tensor][index] = value
            self.assertEqual(float(reversal_risk(*args)['reversal'][0, 1, 1]), 0)

    def test_fill_disagreement_abstains(self):
        args = self.inputs()
        args[1][1, 0, 1] = .8
        self.assertGreater(float(reversal_risk(*args, fill=0)['reversal'][0, 1, 1]), 0)
        self.assertEqual(float(reversal_risk(*args)['reversal'][0, 1, 1]), 0)

    def test_canonical_invalid_and_self_are_inert(self):
        args = self.inputs()
        for values in reversal_risk(*args).values():
            self.assertEqual(float(values[:, args[4]].abs().max()), 0)
            for a, c in enumerate(args[3]):
                self.assertEqual(float(values[:, a, c].abs().max()), 0)
        args[-1][:] = False
        for values in reversal_risk(*args).values():
            self.assertEqual(float(values.abs().max()), 0)

    def test_view_specific_common_mode_bias_cancels(self):
        args = self.inputs()
        base = reversal_risk(*args)
        args[0] += 3
        args[1] += torch.tensor([7., -4.], dtype=torch.float64)[:, None, None]
        args[2] += torch.tensor([-6., 8.], dtype=torch.float64)[:, None, None]
        changed = reversal_risk(*args)
        for name in base:
            torch.testing.assert_close(changed[name], base[name], atol=1e-12, rtol=0)

    def test_rival_specificity_and_alias_permutation(self):
        args = self.inputs()
        permutation = torch.tensor([0, 3, 2, 1])
        base = reversal_risk(*args)
        args[:4] = [args[0][:, permutation], args[1][:, :, permutation], args[2][:, :, permutation], args[3][permutation]]
        changed = reversal_risk(*args)
        for name in base:
            torch.testing.assert_close(changed[name], base[name][:, permutation], atol=0, rtol=0)

    def test_equal_or_nonfinite_observations(self):
        args = self.inputs()
        args[:3] = [torch.zeros_like(a) for a in args[:3]]
        for risk in reversal_risk(*args).values():
            self.assertEqual(float(risk.abs().max()), 0)
        args[0][0, 1] = float('nan')
        with self.assertRaises(ValueError):
            reversal_risk(*args)


if __name__ == '__main__':
    unittest.main()
