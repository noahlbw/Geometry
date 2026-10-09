import unittest

import torch

from dinotool.geometry_localized_likelihood import (
    class_localization_density, apply_localized_likelihood)


def row(boxes, scores, indices=(0, 1)):
    return {"boxes": torch.tensor(boxes).reshape(-1, 4),
            "alias_scores": torch.tensor(scores).reshape(-1, len(indices)), "alias_indices": indices}


class LocalizedLikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.valid = torch.ones(4, dtype=torch.bool)

    def test_empty_detections_are_exactly_neutral(self):
        density, _ = class_localization_density([row([], [])], [0, 1], 2, self.valid)
        scores = torch.randn(4, 2)
        self.assertTrue(torch.equal(apply_localized_likelihood(scores, density), scores))

    def test_full_window_response_is_neutral_even_when_confident(self):
        rows = [row([[0., 0., 1., 1.]], [[.95, .5]])]
        geometry = torch.tensor([[1., 2., 3., 4.]]*4).softmax(-1)
        density, _ = class_localization_density(rows, [0, 1], 2, self.valid, geometry)
        self.assertTrue(torch.equal(density, torch.ones_like(density)))

    def test_identity_exactly_recovers_box_control(self):
        rows = [row([[0., 0., .2, .5]], [[.9, .1]])]
        control, _ = class_localization_density(rows, [0, 1], 2, self.valid)
        actual, _ = class_localization_density(rows, [0, 1], 2, self.valid, torch.eye(4))
        self.assertTrue(torch.equal(control, actual))

    def test_many_uniform_responses_are_bit_exact_neutral(self):
        rows = [row([[0., 0., 1., 1.]]*900, torch.rand(900, 2).tolist())]
        density, _ = class_localization_density(rows, [0, 1], 2, self.valid, torch.rand(4, 4))
        self.assertTrue(torch.equal(density, torch.ones_like(density)))

    def test_alias_and_class_spatial_mean_remains_one(self):
        rows = [row([[0., 0., .5, .5], [.5, 0., 1., 1.]], [[.9, .1], [.2, .7]])]
        density, _ = class_localization_density(rows, [0, 1], 2, self.valid, torch.rand(4, 4))
        torch.testing.assert_close(density.mean(0), torch.ones(2), atol=2e-7, rtol=0)
        self.assertGreater(float(density[0, 0]), 1.)
        self.assertLess(float(density[3, 0]), 1.)

    def test_alias_without_detections_supplies_no_absence_veto(self):
        density, _ = class_localization_density([row([[0., 0., .5, .5]], [[.9, .1]])], [0, 1], 2, self.valid)
        self.assertTrue(torch.equal(density[:, 1], torch.ones(4)))

    def test_padding_scores_and_donors_are_excluded(self):
        valid = torch.tensor([True, True, False, False])
        rows = [row([[0., 0., 1., .5]], [[.9, .7]])]
        first, _ = class_localization_density(rows, [0, 1], 2, valid, torch.ones(4, 4))
        geometry = torch.ones(4, 4)
        geometry[:, ~valid] = 900
        second, _ = class_localization_density(rows, [0, 1], 2, valid, geometry)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(second[~valid], torch.ones(2, 2)))

    def test_duplicate_alias_strings_can_remain_distinct_parents(self):
        density, _ = class_localization_density([row([[0., 0., .5, .5]], [[.6, .6]])], [0, 1], 2, self.valid)
        self.assertTrue(torch.equal(density[:, 0], density[:, 1]))

    def test_duplicate_missing_alias_id_fails(self):
        with self.assertRaises(ValueError):
            class_localization_density([row([[0., 0., .5, .5]], [[.6, .6]], (0, 0))], [0, 1], 2, self.valid)

    def test_bayes_update_matches_normalized_posterior_product(self):
        scores = torch.randn(4, 2)
        density = torch.tensor([[2., .5]]*4)
        actual = (apply_localized_likelihood(scores, density)/.07).softmax(-1)
        product = (scores/.07).softmax(-1)*density
        expected = product/product.sum(-1, keepdim=True)
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
