import unittest

import torch

from dinotool.geometry_generative_observation import (
    conditional_noise_error, reduce_alias_errors, supported_errors, timestep_grid, window_seed,
)


class GenerativeObservationTests(unittest.TestCase):
    def test_midpoint_schedule_is_distinct_and_in_bounds(self):
        steps = timestep_grid(1000, 32)
        self.assertEqual(len(steps.unique()), 32)
        self.assertEqual((int(steps[0]), int(steps[-1])), (15, 984))
        with self.assertRaises(ValueError):
            timestep_grid(3, 4)

    def test_noise_is_shared_across_competing_explanations(self):
        noise = torch.arange(16.).reshape(1, 4, 2, 2)
        predictions = torch.cat((noise+1, noise-1, noise+2))
        error = conditional_noise_error(predictions, noise)
        torch.testing.assert_close(error[:, 0, 0], torch.tensor([1., 1., 4.]))

    def test_all_aliases_have_uniform_weight(self):
        errors = torch.tensor([1., 3., 9., 11.]).reshape(4, 1, 1)
        result = reduce_alias_errors(errors, torch.tensor([0, 0, 1, 1]), 2)
        torch.testing.assert_close(result[:, 0, 0], torch.tensor([2., 10.]))
        duplicated = reduce_alias_errors(errors.repeat(2, 1, 1), torch.tensor([0, 0, 1, 1]*2), 2)
        torch.testing.assert_close(duplicated, result)

    def test_identity_geometry_preserves_local_error(self):
        errors = torch.tensor([[1., 2.], [3., 4.]])
        result = supported_errors(errors, torch.eye(2), torch.tensor([True, True]))
        torch.testing.assert_close(result, errors)

    def test_invalid_donors_cannot_supply_evidence(self):
        errors = torch.tensor([[1., 2.], [1000., 0.]])
        result = supported_errors(errors, torch.ones(2, 2), torch.tensor([True, False]))
        torch.testing.assert_close(result, errors)

    def test_class_permutation_is_equivariant(self):
        errors = torch.rand(4, 3)
        relation = torch.rand(4, 4)
        valid = torch.tensor([True]*4)
        order = torch.tensor([2, 0, 1])
        torch.testing.assert_close(supported_errors(errors[:, order], relation, valid),
                                   supported_errors(errors, relation, valid)[:, order])

    def test_observation_seed_is_key_specific_and_reproducible(self):
        self.assertEqual(window_seed("image0/0/0", 4), window_seed("image0/0/0", 4))
        self.assertNotEqual(window_seed("image0/0/0", 4), window_seed("image0/0/1", 4))


if __name__ == "__main__":
    unittest.main()
