import unittest

import torch

from audit_localized_likelihood_signs import signed_scores


class SignCounterfactualTests(unittest.TestCase):
    def test_negative_only_update_has_exact_neutral_positive_counterfactual(self):
        base = torch.tensor([[.1, .2]])
        positive, negative = signed_scores(base, base-.03)
        self.assertTrue(torch.equal(positive, base))
        torch.testing.assert_close(negative, base-.03, atol=0, rtol=0)

    def test_positive_only_update_has_exact_neutral_negative_counterfactual(self):
        base = torch.tensor([[.1, .2]])
        positive, negative = signed_scores(base, base+.03)
        self.assertTrue(torch.equal(negative, base))
        torch.testing.assert_close(positive, base+.03, atol=0, rtol=0)

    def test_split_retains_source_signs(self):
        base = torch.zeros(1, 3)
        positive, negative = signed_scores(base, torch.tensor([[.3, -.2, 0.]]))
        torch.testing.assert_close(positive, torch.tensor([[.3, 0., 0.]]), atol=0, rtol=0)
        torch.testing.assert_close(negative, torch.tensor([[0., -.2, 0.]]), atol=0, rtol=0)

    def test_positive_update_does_not_guarantee_correct_class_retention(self):
        base = torch.tensor([[.6, .5]])
        positive, _ = signed_scores(base, torch.tensor([[.6, .8]]))
        self.assertNotEqual(int(base.argmax()), int(positive.argmax()))


if __name__ == "__main__":
    unittest.main()
