import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_qualified_sense_ready import baseline_ready, waiting_queue, yield_to_upstream


class QualifiedSenseReadyTest(unittest.TestCase):
    def fixture(self, root):
        for relative in ('vocabularies/voc20.json', 'text_cache/voc20/seg_template.pt',
                         'full/voc20/s0/results.json', 'full/voc20/s0/per_image_confusions.npz'):
            path = root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        row = dict(status='complete', processed_images=1449, total_images=1449,
            coverage_verified=True, per_image_confusions_verified=True, signature=dict(num_shards=1))
        (root/'full/voc20/merged.json').write_text(json.dumps(row))
        return row

    def test_completed_protocol_does_not_wait_for_unrelated_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            self.assertTrue(baseline_ready(root, 'voc20', 1449, 1))
            self.assertFalse(baseline_ready(root, 'context59', 5105, 2))

    def test_missing_confusions_or_incomplete_coverage_blocks_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self.fixture(root)
            (root/'full/voc20/s0/per_image_confusions.npz').unlink()
            self.assertFalse(baseline_ready(root, 'voc20', 1449, 1))
            (root/'full/voc20/s0/per_image_confusions.npz').touch()
            row['processed_images'] = 1448
            (root/'full/voc20/merged.json').write_text(json.dumps(row))
            self.assertFalse(baseline_ready(root, 'voc20', 1449, 1))

    def test_only_idle_unstarted_waiting_controller_can_transition(self):
        state = dict(status='waiting_for_preceding_suite', active=[], failures={})
        protocol = dict(implementation='natural-fixed-reader-qualified-queries-v1-20261004',
            physical_gpus=[4, 5, 6, 7], main_model_changed=False, no_target_label_tuning=True)
        waiting_queue(state, protocol)
        state['active'] = [dict(session='existing_worker')]
        with self.assertRaisesRegex(RuntimeError, 'unstarted'):
            waiting_queue(state, protocol)
        state['active'] = []
        protocol['physical_gpus'] = [0, 1, 2, 3]
        with self.assertRaisesRegex(RuntimeError, 'waiting queue'):
            waiting_queue(state, protocol)

    def test_existing_upstream_dispatch_has_priority_over_new_tail_work(self):
        context = dict(status='waiting_for_preceding_suite', pending=['context59'])
        self.assertTrue(yield_to_upstream(dict(status='running', pending=['ade150']), context))
        self.assertFalse(yield_to_upstream(dict(status='running', pending=[]), context))
        self.assertTrue(yield_to_upstream(dict(status='complete', pending=[]), context))
        self.assertTrue(yield_to_upstream(dict(status='complete', pending=[]), dict(status='running', pending=['full'])))
        self.assertFalse(yield_to_upstream(dict(status='complete', pending=[]), dict(status='running', pending=[])))


if __name__ == '__main__':
    unittest.main()
