import unittest

import torch

from dinotool.grounded_mask_observer import masked_box_support
from dinotool.geometry_localized_likelihood import class_localization_density


class GroundedMaskSupportTests(unittest.TestCase):
    def setUp(self):
        self.valid = torch.ones(4, dtype=torch.bool)

    def test_full_window_does_not_become_an_arbitrary_object_mask(self):
        boxes = torch.tensor([[0., 0., 1., 1.]])
        observed = masked_box_support(boxes, torch.tensor([[1., 0., 0., 0.]]), self.valid)
        self.assertTrue(torch.equal(observed, torch.ones(1, 4)))

    def test_observer_mask_cannot_activate_outside_original_box(self):
        boxes = torch.tensor([[0., 0., .5, .5]])
        observed = masked_box_support(boxes, torch.ones(1, 4), self.valid)
        self.assertTrue(torch.equal(observed, torch.tensor([[1., 0., 0., 0.]])))

    def test_same_support_recovers_original_box_density_exactly(self):
        row = {"boxes": torch.tensor([[0., 0., .5, 1.]]), "alias_scores": torch.tensor([[.6, .7]]),
               "alias_indices": [0, 1]}
        old, _ = class_localization_density([row], [0, 1], 2, self.valid)
        row["supports"] = torch.tensor([[1., 0., 1., 0.]])
        new, _ = class_localization_density([row], [0, 1], 2, self.valid)
        self.assertTrue(torch.equal(old, new))

    def test_identity_geometry_recovers_mask_only_control_exactly(self):
        row = {"boxes": torch.tensor([[0., 0., .5, 1.]]), "alias_scores": torch.tensor([[.6, .7]]),
               "alias_indices": [0, 1], "supports": torch.tensor([[.1, 0., .9, 0.]])}
        mask, _ = class_localization_density([row], [0, 1], 2, self.valid)
        geo, _ = class_localization_density([row], [0, 1], 2, self.valid, torch.eye(4))
        self.assertTrue(torch.equal(mask, geo))

    def test_invalid_mask_support_fails_even_for_inactive_query(self):
        row = {"boxes": torch.tensor([[0., 0., .5, 1.]]), "alias_scores": torch.zeros(1, 2),
               "alias_indices": [0, 1], "supports": torch.tensor([[1., -1., 0., 0.]])}
        with self.assertRaises(ValueError):
            class_localization_density([row], [0, 1], 2, self.valid)

    def test_empty_mask_has_no_localization_update(self):
        row = {"boxes": torch.tensor([[0., 0., .5, 1.]]), "alias_scores": torch.tensor([[.6, .7]]),
               "alias_indices": [0, 1], "supports": torch.zeros(1, 4)}
        density, _ = class_localization_density([row], [0, 1], 2, self.valid, torch.ones(4, 4))
        self.assertTrue(torch.equal(density, torch.ones(4, 2)))


if __name__ == "__main__":
    unittest.main()
