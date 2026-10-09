from types import SimpleNamespace
import unittest

from dinotool.qualified_sense_words import paired_inputs, rename_predictions, METHODS


class QualifiedSenseWordsTest(unittest.TestCase):
    def test_words_do_not_change_profiles(self):
        old, new = (SimpleNamespace(class_names=('background', 'mouse')) for _ in range(2))
        queries, profiles = paired_inputs(new, old, .1)
        self.assertIs(queries['default'], new)
        self.assertIs(queries['frozen'], old)
        self.assertEqual(profiles['default'], profiles['frozen'])
        self.assertEqual(profiles['default'], dict(template='seg_template', tau=1., tem=1., prob_thd=.1))

    def test_foreground_only_rejection_is_not_added(self):
        query = SimpleNamespace(class_names=('mouse', 'cat'))
        with self.assertRaises(ValueError):
            paired_inputs(query, query, .1)
        paired_inputs(query, query, 0.)

    def test_class_order_is_fixed(self):
        with self.assertRaises(ValueError):
            paired_inputs(SimpleNamespace(class_names=('cat', 'mouse')),
                          SimpleNamespace(class_names=('mouse', 'cat')), 0.)

    def test_names_do_not_confuse_baseline_with_new_words(self):
        result = rename_predictions(dict(Sense_Words=1, Sense_Default=2,
                                         Sense_Words_NoThreshold=3, Sense_Default_NoThreshold=4))
        self.assertEqual(tuple(result), METHODS)
        self.assertEqual(result['V2_Words'], 1)
        self.assertEqual(result['Qualified_Words'], 2)
        with self.assertRaises(ValueError):
            rename_predictions({'Source': 1})


if __name__ == '__main__':
    unittest.main()
