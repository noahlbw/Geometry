import unittest

import torch

from dinotool.rival_alias_stress import evidence_audit
from dinotool.rival_competition_admission import retained_competition_scores
from dinotool.rival_fine_full import retained_scores
from test_rival_alias_fast import RivalAliasFastTest


class RivalStressTests(unittest.TestCase):
    def test_variable_count_original_reader_replays_exactly(self):
        for k in (20, 30, 40):
            args = RivalAliasFastTest().fixture(classes=3, k=k)
            original, risk, _ = retained_scores(*args)
            actual, actual_risk, _ = retained_competition_scores(*args,
                methods=('NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact'))
            self.assertTrue(torch.equal(risk, actual_risk))
            for method in ('NoAdmission_Exact', 'RivalFineHard_Exact'):
                self.assertTrue(torch.equal(original[method], actual[method]))

    def test_audit_excludes_self_canonical_and_unknown_comparisons(self):
        shape = (2, 4, 2)
        flags = torch.ones(shape, dtype=torch.bool)
        parents, canonical, known = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 2]), torch.tensor([True, False])
        result = evidence_audit(flags, flags, ~flags, parents, canonical, known, query_target=torch.tensor([1, 0]))
        self.assertEqual(result['audited_comparisons'], 2)
        self.assertEqual(result['uncontradicted_advantage_comparisons'], 2)
        self.assertEqual(result['target_audit']['wrong_parent_comparisons'], 1)
        self.assertEqual(result['target_audit']['wrong_parent_uncontradicted_advantage'], 1)

    def test_subset_and_no_broad_advantage_have_no_fake_fraction(self):
        zeros = torch.zeros(1, 4, 2, dtype=torch.bool)
        parents, canonical = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 2])
        actual = evidence_audit(zeros, ~zeros, zeros, parents, canonical, torch.tensor([True]),
            alias_subset=torch.tensor([True, True, False, False]), query_target=torch.tensor([-1]))
        self.assertEqual(actual['audited_comparisons'], 1)
        self.assertIsNone(actual['uncontradicted_given_broad_fraction'])
        self.assertEqual(actual['target_audit']['wrong_parent_comparisons'], 0)


if __name__ == '__main__':
    unittest.main()
