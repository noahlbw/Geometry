import unittest

import torch
import torch.nn.functional as F

from dinotool.query_slot_memory import query_slot_features, slot_coordinate_prototypes, slot_prototypes


class QuerySlotMemoryTests(unittest.TestCase):
    def test_slot_prototypes_are_normalized_and_mass_weighted(self):
        raw = F.normalize(torch.tensor([[[1., 0.], [1., 0.], [0., 1.]]]), dim=-1)
        assignment = torch.tensor([[[1., 0.], [1., 0.], [0., 1.]]])
        values = torch.tensor([[[2., 0.], [4., 0.], [0., 3.]]])
        centers, slots = slot_prototypes(raw, assignment, values)
        torch.testing.assert_close(centers, torch.tensor([[[1., 0.], [0., 1.]]]))
        torch.testing.assert_close(slots, torch.tensor([[[1., 0.], [0., 1.]]]))

    def test_query_routing_uses_local_query_and_residual_baseline(self):
        centers = torch.tensor([[[1., 0.], [0., 1.]]])
        values = centers.clone()
        query = torch.tensor([[[1., 0.], [0., 1.]]])
        baseline = torch.zeros_like(query)
        routed = query_slot_features(query, centers, values, baseline, temperature=.01, beta=1.)
        torch.testing.assert_close(routed, values, atol=1e-3, rtol=1e-3)
        half = query_slot_features(query, centers, values, baseline, temperature=.01, beta=.5)
        torch.testing.assert_close(half, values, atol=1e-3, rtol=1e-3)
        swapped = query_slot_features(query.flip(1), centers, values, baseline, temperature=.01, beta=1.)
        self.assertGreater(float((routed - swapped).abs().sum()), 1.)

    def test_query_routing_rejects_invalid_controls(self):
        q = torch.randn(1, 2, 3)
        c = torch.randn(1, 2, 3)
        with self.assertRaises(ValueError):
            query_slot_features(q, c, c, temperature=0.)
        with self.assertRaises(ValueError):
            query_slot_features(q, c, c, beta=1.1)
        with self.assertRaises(ValueError):
            query_slot_features(q, c, c, routing='unknown')
        with self.assertRaises(ValueError):
            query_slot_features(q, c, c, spatial_weight=.1)
        with self.assertRaises(ValueError):
            query_slot_features(q, c, c, secondary_query=q)
        with self.assertRaises(ValueError):
            query_slot_features(q, c, c, mass_weight=.1)

    def test_slot_coordinates_follow_soft_assignment_mass(self):
        assignment = torch.eye(4)[None]
        coordinates = slot_coordinate_prototypes(assignment, (2, 2))
        expected = torch.tensor([[[-.5, -.5], [.5, -.5], [-.5, .5], [.5, .5]]])
        torch.testing.assert_close(coordinates, expected)

    def test_confidence_gates_fall_back_to_baseline_for_ambiguous_queries(self):
        centers = torch.tensor([[[1., 0.], [0., 1.]]])
        values = torch.tensor([[[0., 1.], [1., 0.]]])
        query = torch.tensor([[[1., 1.], [1., 0.]]])
        baseline = torch.tensor([[[1., 0.], [1., 0.]]])
        entropy = query_slot_features(query, centers, values, baseline,
                                      temperature=.1, beta=1., routing='entropy')
        maximum = query_slot_features(query, centers, values, baseline,
                                      temperature=.1, beta=1., routing='max')
        self.assertLess(float((entropy[0, 0] - baseline[0, 0]).norm()),
                        float((entropy[0, 1] - baseline[0, 1]).norm()))
        self.assertLess(float((maximum[0, 0] - baseline[0, 0]).norm()),
                        float((maximum[0, 1] - baseline[0, 1]).norm()))

    def test_hybrid_query_interpolates_two_key_spaces(self):
        centers = torch.tensor([[[1., 0.], [0., 1.]]])
        values = torch.tensor([[[1., 0.], [0., 1.]]])
        raw_query = torch.tensor([[[1., 0.]]])
        semantic_query = torch.tensor([[[0., 1.]]])
        routed = query_slot_features(raw_query, centers, values, temperature=.01,
                                     secondary_query=semantic_query, secondary_centers=centers,
                                     hybrid_weight=0.)
        torch.testing.assert_close(routed, values[:, 1:2], atol=1e-3, rtol=1e-3)

    def test_mass_prior_changes_routing_only_when_enabled(self):
        centers = torch.tensor([[[1., 0.], [0., 1.]]])
        values = torch.tensor([[[1., 0.], [0., 1.]]])
        query = torch.tensor([[[1., 0.]]])
        unweighted = query_slot_features(query, centers, values, temperature=.1)
        weighted = query_slot_features(query, centers, values, temperature=.1,
                                       slot_mass=torch.tensor([[1., 100.]]), mass_weight=1.)
        self.assertGreater(float((unweighted-weighted).abs().sum()), 1e-3)

    def test_residual_transport_keeps_coordinate_baseline(self):
        centers = torch.tensor([[[1., 0.], [0., 1.]]])
        values = torch.tensor([[[1., 0.], [0., 1.]]])
        query = torch.tensor([[[1., 0.], [0., 1.]]])
        baseline = torch.tensor([[[0., 1.], [1., 0.]]])
        anchor = torch.zeros_like(values)
        residual = query_slot_features(query, centers, values, baseline,
                                       temperature=.01, beta=1., transport='global_residual',
                                       residual_anchor=anchor)
        torch.testing.assert_close(residual, torch.tensor([[[1., 1.], [1., 1.]]])/2**.5,
                                   atol=1e-3, rtol=1e-3)

    def test_residual_transport_requires_anchor(self):
        q = torch.randn(1, 2, 3)
        with self.assertRaises(ValueError):
            query_slot_features(q, q, q, q, transport='global_residual')


if __name__ == '__main__':
    unittest.main()
