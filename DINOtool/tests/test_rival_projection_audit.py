import unittest

import torch

from dinotool.rival_competition_admission import retained_competition_scores, project_actions
from dinotool.rival_matched_support import semantic_rivals
from dinotool.rival_projection_audit import (PRIMARY, CLASS_UNPROJECTED, UNPROJECTED,
    screened_target, projection_statistics, target_audit_scores)
from dinotool.rival_survivor_redistribution import PRIMARY as SURVIVOR, redistribution_scores
from test_rival_alias_fast import RivalAliasFastTest


class ProjectionAuditTests(unittest.TestCase):
    def fixture(self, k=20):
        inputs = RivalAliasFastTest().fixture(k=k)
        torch.manual_seed(44)
        matches = semantic_rivals(torch.randn(inputs[10].numel(), 2, 12), inputs[10])
        return inputs, matches

    def test_screened_target_is_conditional_class_difference(self):
        original = torch.tensor([[[0., 2.], [-2., 0.]]], dtype=torch.float64)
        directed = torch.tensor([[[0., -.7], [.2, 0.]]], dtype=torch.float64)
        actual = screened_target(original, directed)
        self.assertEqual(float(actual[0, 0, 1]), 1.1)
        self.assertTrue(torch.equal(actual, -actual.transpose(-1, -2)))
        self.assertTrue(torch.equal(screened_target(original, torch.zeros_like(directed)), original))

    def test_projection_can_block_or_limit_and_screening_can_change_both(self):
        requested = torch.tensor([[[0., 2., -3.], [-2., 0., 4.], [3., -4., 0.]]], dtype=torch.float64)
        target = torch.tensor([[[0., -1., -1.], [1., 0., 8.], [1., -8., 0.]]], dtype=torch.float64)
        alternate = -target
        old, new = project_actions(requested, target), project_actions(requested, alternate)
        stats = projection_statistics(requested, old, new, torch.full((1, 3), 1/3), torch.tensor([True]))
        self.assertEqual(stats['active_pairs'], 3)
        self.assertEqual(stats['old_zeroed_pairs'], 1)
        self.assertEqual(stats['screened_zeroed_pairs'], 2)
        self.assertEqual(stats['old_only_zeroed_pairs'], 1)
        self.assertEqual(stats['requested_absolute_mass'], 9.)
        self.assertEqual(stats['old_absolute_mass'], 5.)

    def test_existing_endpoints_and_unprojected_class_control(self):
        for k in (20, 40):
            inputs, matches = self.fixture(k)
            values, diag = target_audit_scores(*inputs, matches=matches)
            old = retained_competition_scores(*inputs)[0]
            for m in ('NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact'):
                self.assertTrue(torch.equal(values[m], old[m]))
            previous = redistribution_scores(*inputs, matches=matches, methods=(SURVIVOR,))[0][SURVIVOR]
            self.assertTrue(torch.equal(values[SURVIVOR], previous))
            self.assertTrue(torch.equal(values[CLASS_UNPROJECTED], old['FineRivalPosterior_Exact']))
            self.assertEqual(diag['original_survivor_score_max_error'], 0.)
            self.assertTrue(torch.equal(values[PRIMARY], target_audit_scores(*inputs, matches=matches, methods=(PRIMARY,))[0][PRIMARY]))
            self.assertTrue(torch.equal(values[UNPROJECTED], target_audit_scores(*inputs, matches=matches, methods=(UNPROJECTED,))[0][UNPROJECTED]))

    def test_invalid_queries_and_target_validation(self):
        inputs, matches = self.fixture()
        inputs = list(inputs)
        inputs[9].fill_(False)
        values, _ = target_audit_scores(*inputs, matches=matches)
        self.assertTrue(all(torch.equal(v, values['NoAdmission_Exact']) for v in values.values()))
        with self.assertRaisesRegex(ValueError, 'antisymmetric'):
            screened_target(torch.ones(1, 2, 2), torch.zeros(1, 2, 2))


if __name__ == '__main__':
    unittest.main()
