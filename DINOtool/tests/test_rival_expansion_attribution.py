import unittest

import torch

from dinotool.rival_expansion_attribution import centered_restorations, common_count_shift


class ExpansionAttributionTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(20261005)
        self.old = torch.randn(11, 4, generator=generator, dtype=torch.float64)
        self.expanded = torch.randn(11, 4, generator=generator, dtype=torch.float64)

    def test_common_gauge_does_not_change_interventions(self):
        base = centered_restorations(self.old, self.expanded)
        shifted = centered_restorations(self.old+7., self.expanded-13.)
        for key in base:
            torch.testing.assert_close(base[key], shifted[key], atol=4e-15, rtol=0)
            self.assertTrue(torch.equal(base[key].argmax(-1), shifted[key].argmax(-1)))

    def test_restoration_factorial_endpoints(self):
        values = centered_restorations(self.old, self.expanded)
        for c in range(4):
            others = torch.arange(4) != c
            own = values['Own20_Rivals40__'+str(c)]
            rivals = values['Own40_Rivals20__'+str(c)]
            self.assertTrue(torch.equal(own[:, c], values['All20'][:, c]))
            self.assertTrue(torch.equal(own[:, others], values['All40'][:, others]))
            self.assertTrue(torch.equal(rivals[:, c], values['All40'][:, c]))
            self.assertTrue(torch.equal(rivals[:, others], values['All20'][:, others]))

    def test_identity_and_common_count_offset(self):
        shift = common_count_shift(torch.eye(11, dtype=torch.float64))
        self.assertTrue(torch.equal((self.expanded-shift).argmax(-1), self.expanded.argmax(-1)))
        values = centered_restorations(self.old, self.old)
        self.assertTrue(all(torch.equal(v, values['All20']) for v in values.values()))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            centered_restorations(self.old, self.expanded[:, :3])
        with self.assertRaises(ValueError):
            common_count_shift(torch.eye(3), 0)


if __name__ == '__main__':
    unittest.main()
