import unittest

import torch

from dinotool.geometry_donor_budget import (
    DonorBudgetConfig, budget_attended, donor_reference, marginal_error, project_donor_flow)
from dinotool.tcpr import _geometry_attended


class DonorBudgetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)

    def test_feasible_flow_is_exact_identity(self):
        flow = torch.rand(2, 3, 7, 7)
        result, info = project_donor_flow(flow, flow.sum(-2))
        self.assertTrue(torch.equal(result, flow))
        self.assertEqual(info["iterations"], 0.)

    def test_projection_retains_query_mass_and_donor_budget(self):
        flow = torch.rand(2, 3, 7, 7)
        columns = torch.rand(2, 3, 7)
        columns *= flow.sum((-1, -2), keepdim=False)[..., None]/columns.sum(-1, keepdim=True)
        result, info = project_donor_flow(flow, columns)
        self.assertLessEqual(marginal_error(result, flow.sum(-1), columns), 1e-5)
        self.assertGreater(info["flow_kl"], 0.)

    def test_zero_rows_and_columns_never_create_mass(self):
        flow = torch.ones(4, 4); flow[-1] = 0
        columns = torch.tensor([6., 4., 2., 0.])
        result, _ = project_donor_flow(flow, columns)
        self.assertTrue(bool((result[-1] == 0).all()))
        self.assertTrue(bool((result[:, -1] == 0).all()))

    def test_incompatible_totals_and_infeasible_edges_rejected(self):
        with self.assertRaises(ValueError):
            project_donor_flow(torch.ones(3, 3), torch.ones(3))
        with self.assertRaises(RuntimeError):
            project_donor_flow(torch.eye(2), torch.tensor([1.5, .5]), DonorBudgetConfig(maximum_iterations=4))

    def test_eligible_edge_cross_ratio_preserved(self):
        flow = torch.rand(5, 5)+.1
        columns = torch.arange(1, 6).float(); columns *= flow.sum()/columns.sum()
        result, _ = project_donor_flow(flow, columns)
        old = flow[0, 0]*flow[1, 1]/(flow[0, 1]*flow[1, 0])
        new = result[0, 0]*result[1, 1]/(result[0, 1]*result[1, 0])
        torch.testing.assert_close(old, new)

    def test_row_and_donor_permutation_equivariant(self):
        flow = torch.rand(5, 5)
        columns = torch.arange(1, 6).float(); columns *= flow.sum()/columns.sum()
        p, q = torch.tensor([2, 0, 4, 1, 3]), torch.tensor([4, 2, 0, 3, 1])
        original, _ = project_donor_flow(flow, columns)
        changed, _ = project_donor_flow(flow[p][:, q], columns[q])
        torch.testing.assert_close(changed, original[p][:, q], atol=1e-5, rtol=1e-5)

    def test_global_value_sum_matches_prescribed_donor_mass(self):
        flow, values = torch.rand(8, 8), torch.randn(8, 6)
        columns = torch.rand(8); columns *= flow.sum()/columns.sum()
        result, _ = project_donor_flow(flow, columns)
        torch.testing.assert_close((result @ values).sum(0), columns @ values, atol=2e-5, rtol=2e-5)

    def test_geometry_budget_exactly_replays_original_attention(self):
        native = torch.rand(1, 2, 9, 9).softmax(-1)
        geometry = torch.rand(1, 7, 7).softmax(-1)
        values = torch.randn(1, 2, 9, 4)
        valid = torch.ones(1, 7, dtype=torch.bool)
        original = _geometry_attended(native, geometry, values, 2, "preserve")
        result, _ = budget_attended(native, geometry, values, valid, 2, "geometry", DonorBudgetConfig())
        self.assertTrue(torch.equal(original, result))

    def test_native_budget_leaves_prefix_and_invalid_queries_exactly_unchanged(self):
        native = torch.rand(1, 2, 9, 9).softmax(-1)
        geometry = torch.rand(1, 7, 7).softmax(-1)
        values = torch.randn(1, 2, 9, 4)
        valid = torch.ones(1, 7, dtype=torch.bool); valid[:, -1] = False
        original = _geometry_attended(native, geometry, values, 2, "preserve")
        result, info = budget_attended(native, geometry, values, valid, 2, "native", DonorBudgetConfig())
        self.assertTrue(torch.equal(result[..., :2, :], original[..., :2, :]))
        self.assertTrue(torch.equal(result[..., -1, :], original[..., -1, :]))
        self.assertEqual(info["invalid_query_displacement_max_error"], 0.)
        reference, columns = donor_reference(native, geometry, valid, 2, "native")
        self.assertTrue(bool((reference[..., -1, :] == 0).all()))
        self.assertTrue(bool((reference[..., :, -1] == 0).all()))
        self.assertTrue(bool((columns[..., -1] == 0).all()))


if __name__ == "__main__":
    unittest.main()
