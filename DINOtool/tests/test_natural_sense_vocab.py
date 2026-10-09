import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from build_natural_sense_vocab import build


def source(names, groups):
    return dict(status='complete', target_masks_loaded=False, target_label_tuning=False,
                identity=dict(classes=names, groups=groups), selected_counts=list(map(len, groups)),
                chosen=dict(template='seg_template', tau=5., tem=.3, prob_thd=0.))


class NaturalSenseVocabularyTest(unittest.TestCase):
    def test_annotations_stay_fixed_and_senses_are_corrected(self):
        row = source(['base', 'step', 'apparel'],
                     [['base', 'stall', 'stand', 'sales booth'],
                      ['step', 'pedestal', 'plinth', 'footstall'],
                      ['apparel', 'clothes closet', 'clothespress']])
        result = build(row, 'ade150', ['base, pedestal, stand', 'step, stair', 'apparel, clothes'])
        self.assertEqual(result['class_names'], row['identity']['classes'])
        self.assertEqual([g[0] for g in result['aliases_by_class']], ['pedestal', 'stair step', 'clothing'])
        self.assertNotIn('pedestal', result['aliases_by_class'][1])
        self.assertIn('pedestal', result['aliases_by_class'][0])
        self.assertNotIn('clothes closet', result['aliases_by_class'][2])
        self.assertFalse(result['target_masks_loaded'])
        self.assertFalse(result['image_data_loaded'])
        self.assertIsNone(result['quota'])

    def test_singletons_and_subtypes_are_not_forced_to_twenty(self):
        row = source(['unknown object', 'animal'], [['unknown object'], ['animal', 'dog', 'cat', 'horse']])
        result = build(row, 'ade150', ['unknown object', 'animal, creature'])
        self.assertEqual(result['alias_counts'], [1, 4])
        self.assertEqual(result['aliases_by_class'][1], ['animal', 'dog', 'cat', 'horse'])

    def test_sense_extensions_are_not_applied_to_wrong_domain(self):
        row = source(['glass', 'mouse'], [['glass'], ['mouse', 'computer mouse']])
        result = build(row, 'coco_stuff171')
        self.assertEqual(result['aliases_by_class'][0], ['glass'])
        self.assertEqual(result['aliases_by_class'][1][0], 'computer mouse')

    def test_empty_and_duplicate_descriptions_are_removed(self):
        row = source(['floor-marble'], [['floor-marble', '', 'floor marble']])
        result = build(row, 'coco_stuff171')
        self.assertEqual(result['alias_counts'], [2])
        self.assertEqual(result['aliases_by_class'][0][0], 'marble floor')

    def test_split_parenthetical_queries_are_not_kept(self):
        row = source(['textile-other'], [['textile-other', 'textile fabric',
                                         'home textile (bedsheet', 'tablecloth)']])
        group = build(row, 'coco_stuff171')['aliases_by_class'][0]
        self.assertEqual(group[0], 'textile fabric')
        self.assertIn('bed sheet', group)
        self.assertIn('tablecloth', group)
        self.assertNotIn('tablecloth)', group)

    def test_metadata_alignment_and_masks_are_checked(self):
        row = source(['base'], [['base']])
        with self.assertRaises(ValueError):
            build(row, 'ade150', ['booth, stall'])
        row['target_masks_loaded'] = True
        with self.assertRaises(ValueError):
            build(row, 'ade150', ['base, pedestal'])


if __name__ == '__main__':
    unittest.main()
