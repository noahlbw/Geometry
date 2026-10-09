import unittest

import torch

from dinotool.geometry_role_compatibility import role_attended, role_relation
from dinotool.tcpr import _geometry_attended


class RoleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.native = torch.randn(2, 3, 8, 8).softmax(-1)
        self.geometry = torch.rand(2, 6, 6).softmax(-1)
        self.valid = torch.ones(2, 6, dtype=torch.bool)

    def test_uninformative_roles_exactly_recover_geometry(self):
        native = self.native[..., :1, :].expand_as(self.native).clone()
        relation, info = role_relation(native, self.geometry, self.valid, 2)
        expected = self.geometry[:, None].expand_as(relation)
        self.assertTrue(torch.equal(expected, relation))
        self.assertEqual(info["relation_displacement"], 0.)

    def test_equal_donor_roles_do_not_impose_global_donor_budgets(self):
        native = torch.zeros_like(self.native)
        native[..., 0] = 1.
        relation, _ = role_relation(native, self.geometry, self.valid, 2)
        self.assertTrue(torch.equal(relation, self.geometry[:, None].expand_as(relation)))

    def test_valid_relation_mass_and_nonnegativity(self):
        relation, info = role_relation(self.native, self.geometry, self.valid, 2)
        self.assertTrue(bool((relation >= 0).all()))
        torch.testing.assert_close(relation.sum(-1), self.geometry.sum(-1)[:, None].expand(2, 3, 6), atol=1e-6, rtol=1e-6)
        self.assertLess(info["row_mass_error"], 1e-6)

    def test_padding_edges_remain_exactly_unchanged(self):
        self.valid[:, -1] = False
        relation, info = role_relation(self.native, self.geometry, self.valid, 2)
        base = self.geometry[:, None].expand_as(relation)
        self.assertTrue(torch.equal(relation[..., -1, :], base[..., -1, :]))
        self.assertTrue(torch.equal(relation[..., :, -1], base[..., :, -1]))
        self.assertEqual(info["invalid_edge_error"], 0.)

    def test_all_invalid_support_does_not_create_nan_or_displacement(self):
        self.valid[:] = False
        relation, _ = role_relation(self.native, self.geometry, self.valid, 2)
        self.assertTrue(torch.equal(relation, self.geometry[:, None].expand_as(relation)))

    def test_donor_profile_permutation_does_not_change_pair_compatibility(self):
        order = torch.randperm(8)
        relation, _ = role_relation(self.native, self.geometry, self.valid, 2)
        changed, _ = role_relation(self.native[..., order], self.geometry, self.valid, 2)
        torch.testing.assert_close(relation, changed, atol=2e-6, rtol=2e-6)

    def test_patch_permutation_equivariance(self):
        order = torch.tensor([4, 0, 3, 5, 1, 2])
        all_order = torch.cat((torch.arange(2), order+2))
        relation, _ = role_relation(self.native, self.geometry, self.valid, 2)
        changed, _ = role_relation(self.native[..., all_order, :][..., all_order],
            self.geometry[:, order][:, :, order], self.valid[:, order], 2)
        torch.testing.assert_close(changed, relation[..., order, :][..., order], atol=2e-6, rtol=2e-6)

    def test_zero_intervention_replays_actual_attention_exactly(self):
        values = torch.randn(2, 3, 8, 4)
        result, _ = role_attended(self.native, self.geometry, values, self.valid, 2, mode="geometry")
        expected = _geometry_attended(self.native, self.geometry, values, 2, "preserve")
        self.assertTrue(torch.equal(result, expected))

    def test_prefix_and_invalid_queries_are_not_written(self):
        self.valid[:, -1] = False
        values = torch.randn(2, 3, 8, 4)
        result, info = role_attended(self.native, self.geometry, values, self.valid, 2)
        expected = _geometry_attended(self.native, self.geometry, values, 2, "preserve")
        self.assertTrue(torch.equal(result[..., :2, :], expected[..., :2, :]))
        self.assertTrue(torch.equal(result[..., -1, :], expected[..., -1, :]))
        self.assertEqual(info["invalid_query_displacement_max_error"], 0.)

    def test_displacement_is_neutral_to_a_common_patch_value(self):
        values = torch.randn(2, 3, 8, 4)
        shifted = values.clone(); shifted[..., 2:, :] += 4.
        old, _ = role_attended(self.native, self.geometry, values, self.valid, 2)
        new, _ = role_attended(self.native, self.geometry, shifted, self.valid, 2)
        old_base = _geometry_attended(self.native, self.geometry, values, 2, "preserve")
        new_base = _geometry_attended(self.native, self.geometry, shifted, 2, "preserve")
        torch.testing.assert_close(new-new_base, old-old_base, atol=3e-6, rtol=3e-6)

    def test_negative_or_nonfinite_profiles_rejected(self):
        native = self.native.clone(); native[..., 0, 0] = -1.
        with self.assertRaises(ValueError):
            role_relation(native, self.geometry, self.valid, 2)
        with self.assertRaises(ValueError):
            role_relation(self.native, self.geometry*float("nan"), self.valid, 2)


if __name__ == "__main__":
    unittest.main()
