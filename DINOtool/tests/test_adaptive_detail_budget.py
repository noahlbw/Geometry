import unittest

import torch

from dinotool.adaptive_detail_budget import detail_rgb, replace_witness, select_detail
from dinotool.bounded_alias_reuse import bounded_scores, contender_union
from dinotool.sparse_alias_reuse import SparseCrop, alias_layout, profile_aliases


class AdaptiveDetailBudgetTests(unittest.TestCase):
    def test_physical_crop_batch_one_and_unbatched_inputs_are_identical(self):
        rgb = torch.rand(3, 512, 512)
        for position in ((0, 0, 256, 256), (256, 256, 256, 256), (0, 0, 100, 171)):
            self.assertTrue(torch.equal(detail_rgb(rgb, position), detail_rgb(rgb[None], position)))
            self.assertEqual(detail_rgb(rgb, position).shape, (3, 512, 512))
        with self.assertRaises(ValueError):
            detail_rgb(rgb[None].expand(2, -1, -1, -1), (0, 0, 256, 256))

    def fixture(self):
        coordinates = torch.tensor([[64., 64.], [64., 320.], [320., 64.], [320., 320.]])
        valid = torch.ones(4, dtype=torch.bool)
        local = torch.tensor([[3., 0., 0.]]).expand(4, -1).clone()
        broad = local.clone()
        return local, broad, torch.eye(4, dtype=torch.float64), coordinates, valid

    def test_no_disagreement_or_no_propagation_uses_zero_budget(self):
        local, broad, operator, coordinates, valid = self.fixture()
        self.assertIsNone(select_detail([local], [broad], operator, coordinates, valid, 512, 512)[0])
        broad[0] = torch.tensor([0., 3., 0.])
        operator.zero_()
        self.assertIsNone(select_detail([local], [broad], operator, coordinates, valid, 512, 512)[0])

    def test_selection_tracks_disagreement_and_geometry_leverage(self):
        local, broad, operator, coordinates, valid = self.fixture()
        broad[:] = torch.tensor([0., 3., 0.])
        operator[2, 2] = 4.
        chosen, _ = select_detail([local], [broad], operator, coordinates, valid, 512, 512)
        self.assertEqual(chosen, (256, 0, 256, 256))
        valid[2] = False
        self.assertEqual(select_detail([local], [broad], operator, coordinates, valid, 512, 512)[0], (0, 0, 256, 256))

    def test_partial_dimensions_use_actual_image_and_stable_ties(self):
        local, broad, operator, coordinates, valid = self.fixture()
        broad[:] = torch.tensor([0., 3., 0.])
        valid[2:] = False
        self.assertEqual(select_detail([local], [broad], operator, coordinates, valid, 100, 400)[0], (0, 0, 100, 256))

    def test_detail_replacement_preserves_unobserved_invalid_and_padding(self):
        parents = torch.tensor([0, 0, 1, 2, 2, 2])
        layout = alias_layout(parents, torch.tensor([0, 2, 3]), 3)
        native = profile_aliases(torch.randn(4, 6), torch.randn(6), layout)
        detail = profile_aliases(torch.randn(3, 6), torch.randn(6), layout)
        crop = SparseCrop(detail, torch.tensor([[0, 1], [1, 2], [0, 0], [0, 0]]),
                          torch.tensor([[.5, .5], [.2, .8], [0., 0.], [1., 0.]]))
        valid = torch.tensor([True, True, True, False])
        mixed, covered = replace_witness(native, crop, layout, valid)
        self.assertEqual(covered.tolist(), [True, True, False, False])
        self.assertTrue(torch.equal(mixed[~covered], native[~covered]))
        self.assertTrue(bool(torch.isneginf(mixed[:, ~layout.valid]).all()))
        self.assertTrue(bool(torch.isfinite(mixed[:, layout.valid]).all()))

    def test_witness_union_max4_includes_new_class_outside_baseline(self):
        base = torch.tensor([[4., 3., 2., 1., 0.], [4., 3., 2., 1., 0.]])
        source = base.flip(-1)
        pairs, active = contender_union(base, base, base, witness_classes=source)
        self.assertEqual(pairs.shape, (2, 4))
        self.assertEqual(set(pairs[0, active[0]].tolist()), {0, 1, 3, 4})


if __name__ == '__main__':
    unittest.main()
