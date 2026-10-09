import unittest
import torch
from dinotool.semantic_mass_wide import semantic_mass_score, lexical_groups


class SemanticMassTests(unittest.TestCase):
    def setUp(self):
        self.raw = torch.tensor([[.1, .8, -.2], [.9, .2, .4]], dtype=torch.float64)
        self.salience = torch.tensor([.2, .7, -.1], dtype=torch.float64)

    def test_singleton_replays_inherited(self):
        for tau in (1., 2.2, 3.):
            expected = (tau*self.raw*(3*(self.salience/.3).softmax(0))).logsumexp(-1)/tau
            actual = semantic_mass_score(self.raw, self.salience, [0, 1, 2], 3, tau, .3)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_one_alias_duplicate_invariant(self):
        for i in range(3):
            raw = torch.cat((self.raw, self.raw[:, i:i+1].repeat(1, 4)), -1)
            salience = torch.cat((self.salience, self.salience[i:i+1].repeat(4)))
            for tau in (1., 3.):
                expected = semantic_mass_score(self.raw, self.salience, [0, 1, 2], 3, tau)
                actual = semantic_mass_score(raw, salience, [0, 1, 2]+[i]*4, 3, tau)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_entire_group_duplicate_invariant(self):
        base = semantic_mass_score(self.raw, self.salience, [0, 1, 2], 3)
        other = semantic_mass_score(self.raw.repeat(1, 3), self.salience.repeat(3), [0, 1, 2]*3, 3)
        torch.testing.assert_close(base, other)

    def test_new_concept_not_discarded(self):
        base = semantic_mass_score(self.raw, self.salience, [0, 1, 2], 3)
        other = semantic_mass_score(torch.cat((self.raw, torch.full((2, 1), 2.)), -1),
            torch.cat((self.salience, torch.tensor([1.]))), [0, 1, 2, 3], 3)
        self.assertTrue(bool((other > base).all()))

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            semantic_mass_score(self.raw, self.salience, [0], 3)
        with self.assertRaises(ValueError):
            semantic_mass_score(self.raw, self.salience, [0, 1, 2], 0)

    def test_lexical_boundaries(self):
        self.assertEqual(lexical_groups(['car', 'cars', 'truck', 'parked car']), [0, 0, 1, 2])
        self.assertEqual(lexical_groups(['wall', 'roof', 'building']), [0, 1, 2])
        self.assertEqual(lexical_groups(['car', 'cars'], residual=True), [0, 1])


if __name__ == '__main__':
    unittest.main()
