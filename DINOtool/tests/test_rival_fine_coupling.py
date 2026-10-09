import unittest

import torch

from dinotool.rival_fine_coupling import PRIMARY, WRITES, write_admitted_pairs


class RivalFineCouplingTest(unittest.TestCase):
    def fixture(self):
        generator = torch.Generator().manual_seed(20261003)
        local = torch.randn(4, 3, generator=generator, dtype=torch.float64)
        broad = torch.randn(4, 3, generator=generator, dtype=torch.float64)
        operator = torch.rand(4, 4, generator=generator, dtype=torch.float64)
        baseline = local+operator @ (broad-local)
        observations = {name: broad.clone() for name in WRITES.values()}
        observations['ObserverAll20'] = broad
        return baseline, local, operator, observations

    def test_no_admission_is_original_coupling_exactly(self):
        args = self.fixture()
        values = write_admitted_pairs(*args)
        for name in WRITES:
            torch.testing.assert_close(values[name], args[0], atol=0, rtol=0)

    def test_equals_single_admitted_observation_reconstruction(self):
        baseline, local, operator, observations = self.fixture()
        observations['ObserverFinePairHard'] += .2
        value = write_admitted_pairs(baseline, local, operator, observations)[PRIMARY]
        expected = local+operator @ (observations['ObserverFinePairHard']-local)
        torch.testing.assert_close(value, expected, atol=1e-12, rtol=0)

    def test_no_geometry_writeback_is_local_baseline(self):
        baseline, local, operator, observations = self.fixture()
        value = write_admitted_pairs(local, local, torch.zeros_like(operator), observations)[PRIMARY]
        torch.testing.assert_close(value, local, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
