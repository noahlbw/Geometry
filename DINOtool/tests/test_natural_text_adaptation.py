import unittest
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from dinotool.calibrated_competitive_alias import profiled_logits
from dinotool.matched_contribution_alias import contribution_margins
from dinotool.native_alias_noise import hard_pair_observation
from dinotool.natural_text_adaptation import semantic_pool, choose_aliases, background_threshold, class_logits
from dinotool.natural_variable_alias_reader import alias_groups, profile, margins, hard_observation, retained_variable_scores
from dinotool.natural_rival_reader import retained_scores_chunked
from dinotool.stratified_soft_alias import WideCrop


class NaturalAdaptationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)

    def test_pool_has_variable_counts_and_rejects_wrong_parent(self):
        groups, removed = semantic_pool(('background', 'chair', 'armchair'),
                                       (('sky', 'chair', 'ground'), ('chair', 'armchair', 'seat'), ('armchair',)))
        self.assertEqual(groups[0], ('background', 'sky', 'ground'))
        self.assertNotIn('armchair', groups[1])
        self.assertEqual(groups[2][0], 'armchair')
        self.assertTrue(any(r['reason'] == 'other_scored_class_canonical' for r in removed))
        self.assertGreater(len(set(map(len, groups))), 1)

    def test_absent_aliases_are_not_forced_out(self):
        parents = torch.tensor([0, 0, 1])
        selected, details = choose_aliases([], parents, torch.tensor([0, 2]), ('a', 'b'))
        self.assertEqual(selected.tolist(), [0, 1, 2])
        self.assertTrue(all(d['retained'] for d in details))

    def test_repeated_negative_alias_is_removed_but_canonical_survives(self):
        rows = [dict(geo=torch.tensor([[3., -3., 0.], [0., 3., 3.]]),
                     wide=torch.tensor([[3., -3., 0.], [0., 3., 3.]]),
                     labels=torch.tensor([0, 1]), quality=torch.ones(2)) for _ in range(8)]
        selected, _ = choose_aliases(rows, torch.tensor([0, 0, 1]), torch.tensor([0, 2]), ('a', 'b'))
        self.assertEqual(selected.tolist(), [0, 2])

    def test_threshold_falls_between_separable_witness_groups(self):
        threshold, _ = background_threshold([.1, .2, .8, .9], [True, True, False, False], [1]*4)
        self.assertGreater(threshold, .2)
        self.assertLessEqual(threshold, .8)
        self.assertEqual(background_threshold([.8], [False], [1])[0], 0.)

    def test_singleton_class_is_finite(self):
        score = class_logits(torch.randn(5, 4), torch.randn(4), torch.tensor([0, 1, 1, 1]), 2)
        self.assertTrue(torch.isfinite(score).all())

    def test_equal_count_profile_and_margin_are_exact(self):
        parents = torch.arange(3).repeat_interleave(20)
        groups = alias_groups(parents, 3)
        members = torch.stack(groups)
        crop = WideCrop(torch.randn(441, 60), torch.randn(60), 0, 0, 336, 336)
        torch.testing.assert_close(profile(crop, groups)[:, members], profiled_logits(crop, members))
        torch.testing.assert_close(margins(crop, groups), contribution_margins(profiled_logits(crop, members)))

    def test_equal_count_hard_observation_matches_original(self):
        parents = torch.arange(3).repeat_interleave(20)
        groups = alias_groups(parents, 3)
        crop = WideCrop(torch.randn(441, 60), torch.randn(60), 0, 0, 336, 336)
        coords = torch.tensor([[8., 8.], [100., 100.]])
        count = torch.ones(336, 336)
        risk = torch.rand(2, 60, 3)
        risk[:, [0, 20, 40]] = 0
        risk.scatter_(-1, parents[None, :, None].expand(2, -1, 1), 0.)
        valid = torch.ones(2, dtype=torch.bool)
        a = hard_pair_observation([crop], count, coords, (336, 336), torch.stack(groups), risk, valid)
        b = hard_observation([crop], count, coords, (336, 336), groups, risk, valid, 1.)
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)

    def test_complete_equal_count_reader_matches_original(self):
        parents = torch.arange(3).repeat_interleave(20)
        crop = WideCrop(torch.randn(441, 60), torch.randn(60), 0, 0, 336, 336)
        fine = WideCrop(torch.randn(1024, 60), torch.randn(60), 0, 0, 512, 512, 32, 512)
        coords = torch.tensor([[8., 8.], [100., 100.]])
        valid = torch.ones(2, dtype=torch.bool)
        local, broad, operator = torch.randn(2, 3), torch.randn(2, 3), torch.eye(2)*.25
        old, _, _ = retained_scores_chunked(local, operator, broad, [crop], torch.ones(336, 336),
            [fine], torch.ones(512, 512), coords, coords, valid, torch.stack(alias_groups(parents, 3)),
            torch.tensor([0, 20, 40]), parents, (336, 336))
        new, _ = retained_variable_scores(local, operator, broad, [crop], torch.ones(336, 336),
            [fine], torch.ones(512, 512), coords, coords, valid, parents, torch.tensor([0, 20, 40]), (336, 336))
        for method in old:
            torch.testing.assert_close(old[method], new[method], atol=1e-12, rtol=1e-12)

    def test_ragged_hard_observation_keeps_singleton_and_is_finite(self):
        groups = alias_groups(torch.tensor([0, 1, 1, 1]), 2)
        crop = WideCrop(torch.randn(441, 4), torch.randn(4), 0, 0, 336, 336)
        risk = torch.zeros(2, 4, 2)
        risk[:, 2, 0] = 1
        out = hard_observation([crop], torch.ones(336, 336), torch.tensor([[8., 8.], [40., 40.]]),
                               (336, 336), groups, risk, torch.ones(2, dtype=torch.bool), 1.)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue((out[:, 0] == 0).all())

    def test_reject_predictions_are_false_negatives_not_ignored_ground_truth(self):
        from eval_geometry_vip_reliability import summary
        metric = summary(np.array([[2, 0, 2], [0, 4, 0]]), ('a', 'b'), 0)
        self.assertEqual(metric['mean_iou_percent'], 75.)
        self.assertEqual(metric['per_class'][0]['iou_percent'], 50.)
        self.assertEqual(sum(map(sum, metric['confusion_matrix'])), 8)

    def test_pilot_gate_accepts_saved_string_paths_and_rectangular_confusions(self):
        from run_natural_text_adaptation import pilot_decision
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'merged.json'
            row = dict(signature=dict(classes={'test': ['a', 'b']}),
                       metrics={'test': {m: dict(confusion_matrix=cm) for m, cm in {
                           'Geometry_Pool': [[2, 1], [1, 2]], 'RivalFine_Pool': [[2, 1], [1, 2]],
                           'Adaptive_RivalFine': [[3, 0], [0, 3]],
                           'VIP_Official_Finite': [[2, 0, 1], [0, 3, 0]]}.items()}})
            path.write_text(json.dumps(row))
            result = pilot_decision({'test': str(path)})
            self.assertTrue(result['passed'])
            self.assertEqual(result['values']['test']['Adaptive_RivalFine'], 100.)


if __name__ == '__main__':
    unittest.main()
