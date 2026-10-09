import math
import unittest

import torch

from dinotool.rival_alias_fast import CachedCrop
from dinotool.rival_mass_admission import (BASE, METHODS, CANDIDATES, CONTROLS, SHUFFLES,
    rejected_mass, mass_actions, mass_admission_scores)
from dinotool.rival_competition_admission import posterior_potential, retained_competition_scores
from test_rival_alias_fast import RivalAliasFastTest


class RivalMassAdmissionTest(unittest.TestCase):
    def test_mass_uses_actual_alias_responsibility_not_deletion_count(self):
        crop = CachedCrop(torch.tensor([[[0., math.log(3)], [math.log(5), 0.]]]), None,
                          torch.tensor([[0]]), torch.ones(1, 1))
        risk = torch.zeros(1, 4, 2)
        risk[0, 1, 1], risk[0, 3, 0] = .2, .7
        mass = rejected_mass([crop], risk, torch.arange(4).reshape(2, 2), torch.tensor([True]))
        self.assertAlmostEqual(float(mass[0, 0, 1]), .75, places=7)
        self.assertAlmostEqual(float(mass[0, 1, 0]), 1/6, places=7)
        self.assertEqual(float(mass.diagonal(dim1=-2, dim2=-1).abs().max()), 0.)

    def test_zero_and_invalid_mass_is_zero(self):
        crop = CachedCrop(torch.zeros(1, 2, 2), None, torch.tensor([[0], [0]]), torch.ones(2, 1))
        risk = torch.ones(2, 4, 2)
        mass = rejected_mass([crop], risk, torch.arange(4).reshape(2, 2), torch.tensor([True, False]))
        self.assertEqual(float(mass[1].abs().max()), 0.)
        self.assertTrue(torch.equal(mass[0], torch.ones(2, 2, dtype=torch.float64)))

    def test_common_innovation_offset_does_not_change_pair_actions(self):
        torch.manual_seed(31)
        mass = torch.rand(7, 4, 4, dtype=torch.float64)
        innovation = torch.randn(7, 4, dtype=torch.float64)
        p = torch.randn(7, 4, dtype=torch.float64).softmax(-1)
        valid = torch.ones(7, dtype=torch.bool)
        offset = torch.randn(7, 1, dtype=torch.float64)
        for kind in ('directed', 'pair'):
            a, b = [mass_actions(mass, z, p, valid, kind) for z in (innovation, innovation+offset)]
            self.assertTrue(torch.allclose(a, b, atol=1e-14, rtol=0))
            self.assertTrue(torch.equal(a, -a.transpose(-1, -2)))

    def test_unit_mass_recovers_full_fine_innovation_up_to_gauge(self):
        torch.manual_seed(32)
        z = torch.randn(5, 4, dtype=torch.float64)
        p = torch.randn(5, 4, dtype=torch.float64).softmax(-1)
        valid = torch.ones(5, dtype=torch.bool)
        mass = torch.ones(5, 4, 4, dtype=torch.float64)
        for kind in ('directed', 'pair'):
            action = mass_actions(mass, z, p, valid, kind)
            u, _ = posterior_potential(action, p, valid)
            self.assertTrue(torch.allclose(u, z-z.mean(-1, keepdim=True), atol=1e-14, rtol=0))

    def test_class_permutation_equivariance(self):
        torch.manual_seed(33)
        mass, z = torch.rand(3, 4, 4, dtype=torch.float64), torch.randn(3, 4, dtype=torch.float64)
        p, valid = z.softmax(-1), torch.ones(3, dtype=torch.bool)
        order = torch.tensor([2, 0, 3, 1])
        for kind in ('directed', 'pair'):
            a = mass_actions(mass, z, p, valid, kind)
            b = mass_actions(mass[:, order][:, :, order], z[:, order], p[:, order], valid, kind)
            self.assertTrue(torch.allclose(a[:, order][:, :, order], b, atol=1e-14, rtol=0))

    def test_base_five_scores_and_risk_are_bitwise_unchanged(self):
        args = RivalAliasFastTest().fixture()
        old, risk, _ = retained_competition_scores(*args, methods=BASE)
        new, changed, _ = mass_admission_scores(*args)
        self.assertTrue(torch.equal(risk, changed))
        for m in BASE: self.assertTrue(torch.equal(old[m], new[m]), m)

    def test_standalone_candidate_scores_are_bitwise_identical(self):
        args = RivalAliasFastTest().fixture()
        full = mass_admission_scores(*args)[0]
        for m in CANDIDATES:
            isolated = mass_admission_scores(*args, methods=(m,))[0]
            self.assertTrue(torch.equal(full[m], isolated[m]), m)

    def test_zero_risk_restores_baseline_for_every_new_endpoint(self):
        args = list(RivalAliasFastTest().fixture())
        for crop in (*args[3], *args[5]):
            crop.alias_logits.zero_()
            crop.salience.zero_()
        baseline = args[0].double()+args[1].double()@(args[2].double()-args[0].double())
        values, risk, _ = mass_admission_scores(*args)
        self.assertEqual(float(risk.abs().max()), 0.)
        for m in (*CANDIDATES, *CONTROLS, *SHUFFLES): self.assertTrue(torch.equal(values[m], baseline), m)


if __name__ == '__main__': unittest.main()
