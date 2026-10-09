import unittest

import torch

from dinotool.rival_competition_admission import retained_competition_scores
from test_rival_alias_fast import RivalAliasFastTest


class ProjectedCandidateSubsetTests(unittest.TestCase):
    def test_endpoint_subset_does_not_change_candidate_or_original_scores(self):
        args = RivalAliasFastTest().fixture()
        methods = ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact')
        all_values, all_risk, _ = retained_competition_scores(*args, methods=(*methods, 'FineBudgetOnly_Exact'))
        full_values, full_risk, _ = retained_competition_scores(*args, methods=methods)
        single_values, single_risk, _ = retained_competition_scores(*args, methods=('FineRivalProjected_Exact',))
        self.assertTrue(torch.equal(all_risk, full_risk))
        self.assertTrue(torch.equal(all_risk, single_risk))
        for method in methods:
            self.assertTrue(torch.equal(all_values[method], full_values[method]), method)
        self.assertTrue(torch.equal(all_values['FineRivalProjected_Exact'], single_values['FineRivalProjected_Exact']))


if __name__ == '__main__':
    unittest.main()
