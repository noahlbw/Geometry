import unittest

import torch

from dinotool.natural_rival_reader import retained_scores_chunked
from dinotool.rival_fine_full import retained_scores, METHODS
from dinotool.stratified_soft_alias import WideCrop


class NaturalRivalReaderTest(unittest.TestCase):
    def test_chunked_equations_match_retained_reader(self):
        torch.manual_seed(901)
        classes, queries = 3, 19
        members = torch.arange(classes*20).reshape(classes, 20)
        parents = torch.arange(classes).repeat_interleave(20)
        coords = torch.rand(queries, 2)*512
        valid = torch.ones(queries, dtype=torch.bool)
        valid[-1] = False
        wide = WideCrop(torch.randn(1024, classes*20), torch.randn(classes*20), 0, 0, 512, 512, 32, 512)
        fine = WideCrop(torch.randn(1024, classes*20), torch.randn(classes*20), 0, 0, 512, 512, 32, 512)
        local, broad = torch.randn(queries, classes), torch.randn(queries, classes)
        operator = torch.rand(queries, queries).double()
        operator /= operator.sum(-1, keepdim=True)
        args = (local, operator, broad, [wide], torch.ones(512, 512), [fine], torch.ones(512, 512),
                coords, coords, valid, members, members[:, 0], parents, (512, 512))
        expected, old_pair, old_stats = retained_scores(*args)
        actual, pair, stats = retained_scores_chunked(*args, query_chunk=7)
        self.assertTrue(torch.equal(pair, old_pair))
        for method in METHODS:
            torch.testing.assert_close(actual[method], expected[method], atol=1e-12, rtol=0)
            self.assertTrue(torch.equal(actual[method].argmax(-1), expected[method].argmax(-1)))
        for key in ('retained_count_mean', 'deleted_alias_rival_fraction', 'canonical_risk_max'):
            self.assertEqual(stats[key], old_stats[key])


if __name__ == '__main__':
    unittest.main()
