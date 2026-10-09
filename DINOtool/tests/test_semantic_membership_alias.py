import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dinotool import semantic_membership_alias as current
from dinotool.stratified_soft_alias import WideCrop


class SemanticMembershipTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6638)
        self.table = torch.rand(3, 4, 3, dtype=torch.float64)
        self.pairs = torch.tensor([[0, 1]]*8)
        self.canonical = torch.tensor([0, 1, 2])
        self.valid = torch.tensor([True]*7+[False])

    def weights(self, table=None, pairs=None, **kwargs):
        return current.membership_weights(self.table if table is None else table,
            self.pairs if pairs is None else pairs, self.canonical, self.valid, **kwargs)

    def test_matches_explicit_own_rival_unknown_posterior(self):
        actual, _ = self.weights()
        for q in range(7):
            for side in range(2):
                c, r = self.pairs[q, side], self.pairs[q, 1-side]
                for a in range(4):
                    own, rival = self.table[c, a, c], self.table[c, a, r]
                    unknown = (1-own)*(1-rival)
                    expected = 1. if a == self.canonical[c] else float((own+unknown)/(own+rival+unknown))
                    self.assertAlmostEqual(float(actual[q, side, a]), max(expected, 1e-6), places=12)

    def test_exclusive_shared_wrong_and_unknown_have_distinct_roles(self):
        table = torch.zeros_like(self.table)
        table[0, 1, 0] = 1.
        table[0, 2, 0] = table[0, 2, 1] = 1.
        table[0, 3, 1] = 1.
        weights, _ = self.weights(table)
        torch.testing.assert_close(weights[0, 0], torch.tensor([1., 1., .5, 1e-6], dtype=torch.float64))
        neutral, _ = self.weights(torch.zeros_like(table))
        self.assertTrue(torch.equal(neutral, torch.ones_like(neutral)))

    def test_same_word_is_conditional_on_rival_not_permanently_deleted(self):
        table = self.table.clone()
        table[0, 1] = torch.tensor([.8, 1., 0.])
        ab = self.weights(table)[0]
        ac = self.weights(table, torch.tensor([[0, 2]]*8))[0]
        self.assertLess(float(ab[0, 0, 1]), .5)
        self.assertEqual(float(ac[0, 0, 1]), 1.)

    def test_canonical_padding_and_self_competition_protected(self):
        weights, _ = self.weights()
        self.assertTrue(torch.equal(weights.gather(-1, self.canonical[self.pairs][..., None]), torch.ones(8, 2, 1, dtype=torch.float64)))
        self.assertTrue(torch.equal(weights[~self.valid], torch.ones_like(weights[~self.valid])))
        weights, _ = self.weights(pairs=torch.zeros_like(self.pairs))
        self.assertTrue(torch.equal(weights, torch.ones_like(weights)))

    def fixture(self):
        members = torch.arange(12).reshape(3, 4)
        crop = WideCrop(torch.randn(4, 12), torch.randn(12), 0, 0, 2, 2, 2, 2)
        coordinates = torch.tensor([[.25, .25], [.75, .25], [1.25, .25], [1.75, .25],
                                    [.25, 1.25], [.75, 1.25], [1.25, 1.25], [1.75, 1.25]])
        return (torch.randn(8, 3), torch.diag(self.valid.double()*.5), self.table,
                [crop], torch.ones(2, 2), coordinates, (2, 2), members,
                self.pairs, self.canonical, self.valid)

    def test_singletons_replay_simultaneous_execution(self):
        args = self.fixture()
        combined, _ = current.scores(*args)
        for name in current.METHODS[3:]:
            single, _ = current.scores(*args, methods=(name,))
            self.assertTrue(torch.equal(combined[name], single[name]), name)

    def test_matched_fields_preserve_norm_gauge_and_padding(self):
        args = self.fixture()
        values, info = current.scores(*args)
        base = args[0].double()
        primary = values[current.PRIMARY]-base
        for name in (current.MATCHED_POOLED, current.MATCHED_CLASS_MEAN, *current.MATCHED_SHUFFLES,
                     current.MATCHED_RIVAL_COLLAPSED, current.SHUFFLED_WRITE, current.DIRECT):
            delta = values[name]-base
            torch.testing.assert_close(delta[self.valid].norm(), primary[self.valid].norm(), atol=1e-12, rtol=0)
            torch.testing.assert_close(delta.sum(-1), torch.zeros(8, dtype=torch.float64), atol=1e-12, rtol=0)
            self.assertEqual(float(delta[~self.valid].abs().max()), 0.)
        self.assertEqual(info['matched_unmatchable'], 0.)

    def test_unknown_recovers_baseline_and_primary_skips_controls(self):
        args = list(self.fixture())
        args[2] = torch.zeros_like(self.table)
        with patch('dinotool.semantic_membership_alias.weight_controls', side_effect=AssertionError), \
             patch('dinotool.semantic_membership_alias.permuted_operator', side_effect=AssertionError), \
             patch('dinotool.semantic_membership_alias.match_previous', side_effect=AssertionError):
            values, _ = current.scores(*args, methods=(current.PRIMARY,))
        self.assertTrue(torch.equal(values[current.PRIMARY], args[0].double()))

    def test_source_cache_rejects_changed_or_nonfinite_provenance(self):
        data = dict(model=current.MODEL_ID, revision=current.REVISION, template=current.TEMPLATE,
            target_images_loaded=False, target_masks_loaded=False, setup={},
            entries=[dict(pair=['car', 'vehicle'], probabilities=[.1, .8, .1])])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'source.json'
            path.write_text(json.dumps(data))
            source = current.SemanticSource(path)
            self.assertEqual(source.entries['car', 'vehicle'], .8)
            data['target_masks_loaded'] = True
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                current.SemanticSource(path)

    def test_invalid_posteriors_or_unfrozen_methods_rejected(self):
        for invalid in (torch.full_like(self.table, torch.nan), torch.full_like(self.table, 1.1)):
            with self.assertRaises(ValueError):
                self.weights(invalid)
        with self.assertRaises(ValueError):
            current.scores(*self.fixture(), methods=('Unfrozen',))


if __name__ == '__main__':
    unittest.main()
