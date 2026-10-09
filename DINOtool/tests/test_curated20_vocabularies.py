"""Metadata-only checks for the frozen lexical experiment."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from build_curated20_vocabularies import make_groups, natural_roots, RS_ROOTS


class CuratedVocabularyTests(unittest.TestCase):
    def test_exact_twenty_without_rival_duplicate(self):
        groups, _, removed = make_groups(['car', 'truck'],
                                        [('car', 'automobile', 'truck'), ('truck', 'lorry')])
        self.assertEqual([len(g) for g in groups], [20, 20])
        self.assertNotIn('truck', groups[0])
        self.assertTrue(any(r['reason'] == 'other_scored_category' for r in removed))

    def test_safe_roots_keep_class_identity(self):
        self.assertNotIn('roof', RS_ROOTS['wall'])
        self.assertNotIn('truck', RS_ROOTS['car'])
        self.assertNotIn('wet area', RS_ROOTS['water'])
        self.assertNotIn('linear pavement', RS_ROOTS['road'])

    def test_polysemy_and_materials(self):
        self.assertEqual(natural_roots('base', 'ade150')[0], 'pedestal')
        self.assertEqual(natural_roots('step', 'ade150')[0], 'stair step')
        self.assertEqual(natural_roots('orange', 'coco_object81')[0], 'orange fruit')
        self.assertEqual(natural_roots('floor-wood', 'coco_stuff171')[0], 'wooden floor')

    def test_shared_synonym_is_not_owned_twice(self):
        groups, safe, _ = make_groups(['cabinet', 'cupboard'],
                                     [('cabinet', 'storage unit'), ('cupboard', 'storage unit')])
        self.assertEqual(safe, [['cabinet'], ['cupboard']])
        self.assertTrue(all(len(g) == len(set(g)) == 20 for g in groups))


if __name__ == '__main__':
    unittest.main()
