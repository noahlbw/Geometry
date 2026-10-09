import unittest

import torch

from dinotool.geometry_language_observation import (
    LanguageObservationConfig, consensus_predictions, foveal_extent, mark_image, option_prompt, probe_indices,
)


class LanguageObservationTests(unittest.TestCase):
    def test_probe_sampling_is_unique_valid_and_includes_margin_extrema(self):
        scores = torch.tensor([[0., 1.], [0., 2.], [0., 3.], [0., 4.], [0., 5.]])
        valid = torch.tensor([True, True, False, True, True])
        self.assertEqual(probe_indices(scores, valid, 3).tolist(), [0, 3, 4])

    def test_empty_probe_set(self):
        self.assertEqual(probe_indices(torch.ones(5, 2), torch.zeros(5, dtype=torch.bool)).numel(), 0)

    def test_self_support_uses_minimum_crop(self):
        relation = torch.eye(1024)
        self.assertEqual(foveal_extent(relation, torch.ones(1024, dtype=torch.bool), 500, (32, 32)), 64)

    def test_padding_donors_do_not_expand_crop(self):
        relation = torch.eye(1024)
        relation[0, -1] = 100
        valid = torch.ones(1024, dtype=torch.bool)
        valid[-1] = False
        self.assertEqual(foveal_extent(relation, valid, 0, (32, 32)), 64)

    def test_wide_support_is_bounded(self):
        relation = torch.ones(1024, 1024)
        self.assertEqual(foveal_extent(relation, torch.ones(1024, dtype=torch.bool), 0, (32, 32)), 512)

    def test_unknown_or_disagreement_retains_geometry(self):
        original = torch.tensor([0, 1, 0])
        choices = torch.tensor([[[1, 2, 0], [1, 2, 1]], [[1, 2, 0], [1, 2, 0]]])
        result, accepted = consensus_predictions(original, choices, 2)
        self.assertEqual(result.tolist(), [1, 1, 0])
        self.assertEqual(accepted.tolist(), [True, False, False])

    def test_prompt_keeps_all_aliases_and_reorders_ownership(self):
        names = ("surface", "object")
        aliases = [f"word_{i}" for i in range(40)]
        prompt = option_prompt(names, aliases, [0]*20+[1]*20, (1, 0))
        self.assertIn("A. object", prompt)
        self.assertIn("B. surface", prompt)
        for alias in aliases:
            self.assertIn(alias, prompt)
        with self.assertRaises(ValueError):
            option_prompt(names, aliases[:-1], [0]*20+[1]*19, (0, 1))

    def test_hollow_marker_preserves_target_center(self):
        rgb = torch.ones(3, 32, 32)*.25
        image = mark_image(rgb, (16, 16))
        self.assertEqual(image.getpixel((16, 16)), (64, 64, 64))


if __name__ == "__main__":
    unittest.main()
