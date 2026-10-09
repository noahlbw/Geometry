import copy
from types import SimpleNamespace
import unittest

import torch

from dinotool.one_sided_alias_stress import SCENARIOS, validate_variants


class VocabularyValidationTest(unittest.TestCase):
    def setUp(self):
        self.bank = SimpleNamespace(class_names=('a', 'b'), class_count=2)
        self.banks = {'p': self.bank}
        self.vocabulary = {s: {'p': [dict(name=c, synonyms=[c]+[c+str(i) for i in range(1,
            int(s[1:]) if s.startswith('k') else 20)]) for c in ('a', 'b')]} for s in SCENARIOS}
        self.variants = {}
        for s in SCENARIOS:
            classes = self.vocabulary[s]['p']
            n = len(classes[0]['synonyms'])
            q = SimpleNamespace(class_names=('a', 'b'), aliases=tuple(a for c in classes for a in c['synonyms']),
                                parents=torch.arange(2).repeat_interleave(n))
            self.variants[s] = self.banks, {'p': q}

    def test_exact_nested_and_fixed_local(self):
        validate_variants(self.banks, self.variants, self.vocabulary)

    def test_reject_local_bank_replacement(self):
        self.variants['k30'] = {'p': copy.copy(self.bank)}, self.variants['k30'][1]
        with self.assertRaises(ValueError):
            validate_variants(self.banks, self.variants, self.vocabulary)

    def test_reject_nonnested_pool(self):
        self.vocabulary['k30']['p'][0]['synonyms'][1] = 'changed'
        self.variants['k30'][1]['p'].aliases = tuple(a for c in self.vocabulary['k30']['p'] for a in c['synonyms'])
        with self.assertRaises(ValueError):
            validate_variants(self.banks, self.variants, self.vocabulary)

    def test_reject_missing_scenario(self):
        self.variants.pop('paraphrase')
        with self.assertRaises(ValueError):
            validate_variants(self.banks, self.variants, self.vocabulary)

    def test_reject_parent_mismatch(self):
        self.variants['k40'][1]['p'].parents[0] = 1
        with self.assertRaises(ValueError):
            validate_variants(self.banks, self.variants, self.vocabulary)


if __name__ == '__main__':
    unittest.main()
