import unittest
from types import SimpleNamespace

import numpy as np
import torch

from dinotool.natural_static_search import StaticProfile, StaticReader, stage_profiles, rank, THRESHOLDS
from dinotool.tcpr import TCPRTextBank
from dinotool.development_readout import Profile
import eval_development_readout as base


class StaticSearchTests(unittest.TestCase):
    def test_retained_configuration_equivalence(self):
        torch.manual_seed(17)
        features = torch.nn.functional.normalize(torch.randn(6, 4), dim=-1)
        bank = TCPRTextBank(features, torch.tensor([0, 0, 1, 2, 2, 2]), torch.tensor([1, 0, 1, 1, 0, 0], dtype=torch.bool),
            ('a', 'b', 'c'), ('a', 'aa', 'b', 'c', 'cc', 'ccc'))
        local = torch.randn(1024, 4)
        source = dict(size=(512, 512), output_size=(512, 512), local=[dict(top=0, left=0,
            features={2.: local}, operator=torch.eye(1024, dtype=torch.float64) * .2)])
        broad = torch.randn(3, 21, 21)
        reader = StaticReader({'original': bank}, {'original': SimpleNamespace()})
        actual = reader.probabilities(source, StaticProfile('original'), {('wide', 'original', 1., 1.): broad})
        expected = base.probabilities(source, Profile(), bank, broad, {})
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_menus_keep_incumbent(self):
        incumbent = StaticProfile('original', strength=3., coupling=.5)
        for stage in ('words', 'readout', 'calibration', 'revisit_words'):
            candidates = stage_profiles(stage, incumbent, ('original', 'official'), 0, True)
            self.assertEqual(candidates[0], incumbent)
            self.assertEqual(len(candidates), len(set(candidates)))

    def test_stable_development_tie(self):
        histogram = np.zeros((2, 1, len(THRESHOLDS) + 1, 2, 2), np.int64)
        histogram[:, 0, -1] = np.eye(2, dtype=np.int64)
        candidates = (StaticProfile('incumbent'), StaticProfile('other'))
        choice = rank(histogram, candidates, None)[0]
        self.assertEqual(choice['profile']['bank'], 'incumbent')
        self.assertEqual(choice['development_miou'], 100.)


if __name__ == '__main__':
    unittest.main()
