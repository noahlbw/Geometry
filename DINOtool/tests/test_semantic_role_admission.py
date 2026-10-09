import unittest

import torch

from dinotool.semantic_role_admission import (PRIMARY, ALIAS_SHUFFLES, role_prompt,
    semantic_conflict, source_controls)


class SemanticRoleAdmissionTest(unittest.TestCase):
    def test_roles_restoration_and_unknown_neutrality(self):
        logits = torch.zeros(2, 4, 6)
        logits[:, 0, 5] = 10.
        logits[:, 1, 1] = 10.
        logits[:, 2, 2] = 10.
        logits[0, 3, 5], logits[1, 3, 4] = 10., 10.
        risk = semantic_conflict(logits, torch.tensor([0, 0, 1, 1]), 2)
        self.assertGreater(float(risk[0, 1]), .99)
        self.assertEqual(float(risk[1:].abs().max()), 0.)
        self.assertEqual(float(risk[0, 0]), 0.)

    def test_order_prompt_keeps_exact_meaning(self):
        prompt = role_prompt(('wall', 'roof'), 0, 'roof', tuple(range(6)))
        reverse = role_prompt(('wall', 'roof'), 0, 'roof', tuple(reversed(range(6))))
        self.assertIn('F. Directly denotes the competing category "roof"', prompt)
        self.assertIn('A. Directly denotes the competing category "roof"', reverse)
        self.assertIn('Shared material', prompt)

    def fixture(self):
        members = torch.tensor([[0, 1, 2], [3, 4, 5]])
        parents = torch.tensor([0, 0, 0, 1, 1, 1])
        valid = torch.tensor([True, True, False])
        field = torch.tensor([[[1., 4.]], [[4., 1.]], [[0., 0.]]], dtype=torch.float64)
        conflict = torch.tensor([[0., 0.], [0., .8], [0., .3], [0., 0.], [.7, 0.], [.2, 0.]], dtype=torch.float64)
        base = torch.full((3, 6), .2, dtype=torch.float64).masked_fill(~valid[:, None], 0.)
        return (base, conflict, field, torch.ones(1, 2, dtype=torch.bool), torch.zeros(6, dtype=torch.long), parents, valid, members)

    def test_semantic_unknown_is_exact_visual_base(self):
        args = list(self.fixture())
        args[1] = torch.zeros_like(args[1])
        output, _ = source_controls(*args)
        torch.testing.assert_close(output[PRIMARY], args[0], atol=0, rtol=0)

    def test_semantic_wrong_parent_does_not_need_visual_agreement(self):
        args = self.fixture()
        output, semantic = source_controls(*args)
        self.assertEqual(float(semantic[0, 1]), float(semantic[1, 1]))
        self.assertGreater(float(output[PRIMARY][1, 1]), float(output['SemanticRoleVisualSupported_Exact'][1, 1]))
        for risk in output.values():
            self.assertTrue(bool(((risk >= 0) & (risk <= 1)).all()))
            self.assertEqual(float(risk[~args[6]].abs().max()), 0.)

    def test_alias_controls_keep_semantic_spectrum(self):
        args = self.fixture()
        output, semantic = source_controls(*args)
        for name in ALIAS_SHUFFLES:
            recovered = (output[name]-args[0])/(1-args[0])
            torch.testing.assert_close(recovered[:, args[7]].sort(-1).values,
                semantic[:, args[7]].sort(-1).values, atol=1e-14, rtol=0)

    def test_writer_identity_and_bound(self):
        from dinotool.coherent_native_admission import union_risk
        base = self.fixture()[0]
        torch.testing.assert_close(union_risk(base, torch.zeros_like(base)), base, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            semantic_conflict(torch.full((2, 6, 6), float('nan')), self.fixture()[5], 2)


if __name__ == '__main__':
    unittest.main()
