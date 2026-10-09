import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from build_qualified_sense_vocab import build


def source(dataset, names, groups):
    return dict(dataset=dataset, class_names=names, aliases_by_class=groups,
        alias_counts=list(map(len, groups)), implementation='frozen-base',
        source_identity=dict(classes=names, groups=groups), target_masks_loaded=False,
        image_data_loaded=False, target_label_tuning=False)


class QualifiedSenseVocabularyTest(unittest.TestCase):
    def test_qualified_query_replaces_not_merely_reorders_ambiguous_words(self):
        row = build(source('coco_object81', ['mouse', 'orange'],
            [['computer mouse', 'mouse', 'desktop mouse'], ['orange fruit', 'orange', 'citrus orange']]))
        self.assertNotIn('mouse', row['aliases_by_class'][0])
        self.assertNotIn('orange', row['aliases_by_class'][1])
        self.assertEqual(row['alias_counts'], [2, 2])
        self.assertEqual(len(row['changed_query_sets']), 2)

    def test_ade_light_sense_and_glass_are_qualified(self):
        row = build(source('ade150', ['light', 'glass'],
            [['light source', 'light', 'skylight', 'fanlight', 'lighting fixture'], ['drinking glass', 'glass']]))
        self.assertEqual(row['aliases_by_class'], [['light source', 'lighting fixture'], ['drinking glass']])

    def test_ade_specific_senses_do_not_leak_into_other_domains(self):
        row = build(source('context59', ['glass', 'light'], [['glass'], ['light']]))
        self.assertEqual(row['aliases_by_class'], [['glass'], ['light']])

    def test_legal_subtypes_remain_without_quota(self):
        row = build(source('ade150', ['animal', 'unknown'], [['animal', 'dog', 'cat', 'horse'], ['unknown']]))
        self.assertEqual(row['alias_counts'], [4, 1])
        self.assertIsNone(row['quota'])
        self.assertFalse(row['main_model_changed'])

    def test_person_has_explicit_noun_head_and_codes_are_removed(self):
        row = build(source('coco_stuff171', ['person', 'wall-brick'],
            [['person', 'people in shirt', 'human in jeans'], ['brick wall', 'wall-brick', 'wall made of bricks']]))
        self.assertIn('person wearing a shirt', row['aliases_by_class'][0])
        self.assertIn('person wearing jeans', row['aliases_by_class'][0])
        self.assertNotIn('wall-brick', row['aliases_by_class'][1])

    def test_reject_label_conditioned_input(self):
        row = source('voc20', ['bicycle'], [['bicycle', 'bike']])
        row['target_masks_loaded'] = True
        with self.assertRaises(ValueError):
            build(row)


if __name__ == '__main__':
    unittest.main()
