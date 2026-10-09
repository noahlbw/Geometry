import unittest

import torch

from dinotool.fine_responsibility_reader import METHODS, PRIMARY, PROJECTED, responsibility_scores
from dinotool.fine_responsibility_audit import audit_scores
from dinotool.rival_competition_admission import retained_competition_scores
from test_rival_alias_fast import RivalAliasFastTest


class RequestedResponsibilityReaderTests(unittest.TestCase):
    def test_every_endpoint_matches_retained_and_singleton_on_nonzero_support(self):
        for classes, k in ((3, 20), (5, 40), (12, 20)):
            inputs = RivalAliasFastTest().fixture(classes, k)
            values, risk, diagnostics = responsibility_scores(*inputs)
            self.assertTrue(bool((risk > 0).any()))
            older = retained_competition_scores(*inputs, methods=METHODS[:4])[0]
            previous = audit_scores(*inputs, methods=METHODS[1:])[0]
            for name in METHODS:
                expected = older[name] if name in older else previous[name]
                self.assertTrue(torch.equal(values[name], expected), name)
                chosen = responsibility_scores(*inputs, methods=(name,))[0][name]
                self.assertTrue(torch.equal(values[name], chosen), name)
            self.assertEqual(diagnostics['canonical_risk_max'], 0.)

    def test_invalid_queries_are_exact_identity_and_bad_methods_reject(self):
        inputs = list(RivalAliasFastTest().fixture())
        inputs[9].fill_(False)
        values, risk, _ = responsibility_scores(*inputs)
        self.assertEqual(float(risk.abs().max()), 0.)
        self.assertTrue(torch.equal(values[PRIMARY], values['NoAdmission_Exact']))
        self.assertTrue(torch.equal(values[PROJECTED], values['NoAdmission_Exact']))
        for methods in ((), ('undeclared',)):
            with self.assertRaises(ValueError):
                responsibility_scores(*inputs, methods=methods)


if __name__ == '__main__':
    unittest.main()
