import unittest

import torch

from dinotool.rival_alias_count import COUNTS, METHODS, canonical_indices, count_predictions, nested_classes, parse_additions
from dinotool.stratified_soft_alias import WideCrop


class RivalAliasCountTest(unittest.TestCase):
    def test_canonical_lookup_stays_within_parent_class(self):
        aliases = ('road', 'roof', 'roof', 'road')
        parents = torch.tensor([0, 0, 1, 1])
        indices = canonical_indices(('road', 'roof'), aliases, parents)
        self.assertEqual(indices.tolist(), [0, 2])
        with self.assertRaises(ValueError):
            canonical_indices(('road', 'roof'), aliases, torch.zeros(4, dtype=torch.long))

    def test_nested_prefix_and_no_semantic_filter(self):
        base = [{'name': 'road', 'synonyms': ['road']+['road '+str(i) for i in range(19)]}]
        extra = ['roof', 'building']+['candidate '+str(i) for i in range(18)]
        for count in COUNTS:
            result = nested_classes(base, [extra], count)
            self.assertEqual(result[0]['synonyms'][:20], base[0]['synonyms'])
            self.assertEqual(len(result[0]['synonyms']), count)
        self.assertEqual(nested_classes(base, [extra], 30)[0]['synonyms'][20], 'roof')

    def test_parse_is_structural_only_and_preserves_order(self):
        result = parse_additions('```json\n["ROAD", "roof", " painted_wall ", "roof"]\n```', ['road'])
        self.assertEqual(result, ['roof', 'painted wall'])
        with self.assertRaises(ValueError):
            parse_additions('["valid", 4]', [])

    def test_variable_count_identity_and_canonical_protection(self):
        for count in COUNTS:
            members = torch.arange(2*count).reshape(2, count)
            canonical, parents = members[:, 0], torch.arange(2).repeat_interleave(count)
            evidence = torch.zeros(1024, 2*count)
            crop = WideCrop(evidence, torch.zeros(2*count), 0, 0, 512, 512, 32, 512)
            coords = torch.tensor([[8., 8.], [504., 504.]])
            valid = torch.tensor([True, True])
            local, broad = torch.zeros(2, 2), torch.zeros(2, 2)
            values, pair, stats = count_predictions(local, torch.eye(2), broad, [crop],
                torch.ones(512, 512), [crop], torch.ones(512, 512), coords, valid,
                members, canonical, parents, (512, 512))
            self.assertEqual(set(values), set(METHODS))
            self.assertEqual(pair.shape, (2, 2*count, 2))
            self.assertEqual(float(pair.abs().max()), 0.)
            self.assertEqual(stats['retained_count_min'], count)
            for value in values.values():
                self.assertEqual(float(value.abs().max()), 0.)


if __name__ == '__main__':
    unittest.main()
