import unittest

import torch

from dinotool.class_attachment_admission import PRIMARY, ALIAS_SHUFFLES, source_controls


class ClassAttachmentAdmissionTest(unittest.TestCase):
    def fixture(self):
        field = torch.tensor([[[1., 4.]], [[4., 1.]], [[0., 0.]]], dtype=torch.float64)
        known = torch.ones(1, 2, dtype=torch.bool)
        parents, canonical = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 2])
        conflict = torch.tensor([[0., 0.], [0., .6], [0., 0.], [.7, 0.]])
        valid = torch.tensor([True, True, False])
        members = torch.tensor([[0, 1], [2, 3]])
        return (torch.zeros(3, 4, 2), field, known, torch.zeros(4, dtype=torch.long),
            parents, canonical, conflict, valid, members, field, known)

    def test_no_semantic_conflict_is_exact_class_base(self):
        args = list(self.fixture())
        args[6] = torch.zeros_like(args[6])
        output, base, attachment = source_controls(*args)
        torch.testing.assert_close(output[PRIMARY], base, atol=0, rtol=0)
        self.assertEqual(float(attachment.abs().max()), 0.)

    def test_attachment_requires_observed_rival_and_keeps_canonical_base(self):
        args = self.fixture()
        output, base, attachment = source_controls(*args)
        self.assertGreater(float(attachment[0, 1]), 0.)
        self.assertEqual(float(attachment[1, 1]), 0.)
        torch.testing.assert_close(output[PRIMARY][:, args[5]], base[:, args[5]], atol=0, rtol=0)
        self.assertGreater(float(base[:, args[5]].max()), 0.)

    def test_unknown_reference_and_invalid_are_neutral(self):
        args = list(self.fixture())
        args[2] = torch.zeros_like(args[2])
        args[-1] = torch.zeros_like(args[-1])
        output, _, _ = source_controls(*args)
        self.assertEqual(float(output[PRIMARY].abs().max()), 0.)

    def test_alias_shuffle_preserves_extra_attachment_distribution(self):
        args = list(self.fixture())
        args[0] = torch.zeros(3, 6, 2)
        args[3] = torch.zeros(6, dtype=torch.long)
        args[4] = torch.tensor([0, 0, 0, 1, 1, 1])
        args[5] = torch.tensor([0, 3])
        args[6] = torch.tensor([[0., 0.], [0., .6], [0., .2], [0., 0.], [.7, 0.], [.3, 0.]])
        args[8] = torch.tensor([[0, 1, 2], [3, 4, 5]])
        output, base, attachment = source_controls(*args)
        expected = attachment[:, args[8]].sort(-1).values
        for name in ALIAS_SHUFFLES:
            recovered = (output[name]-base)/(1-base).clamp_min(1e-30)
            torch.testing.assert_close(recovered[:, args[8]].sort(-1).values, expected, atol=1e-14, rtol=0)
        self.assertTrue(any(not torch.equal(output[name], output[PRIMARY]) for name in ALIAS_SHUFFLES))

    def test_all_sources_bounded_and_inputs_unchanged(self):
        args = self.fixture()
        field = args[1].clone()
        output, _, _ = source_controls(*args)
        torch.testing.assert_close(args[1], field, atol=0, rtol=0)
        for risk in output.values():
            self.assertTrue(bool(((risk >= 0) & (risk <= 1)).all()))
            self.assertEqual(float(risk[~args[7]].abs().max()), 0.)


if __name__ == '__main__':
    unittest.main()
