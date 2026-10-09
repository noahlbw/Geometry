import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import natural_count_calibration_experiment as manager


class CountCollectionTests(unittest.TestCase):
    def setUp(self):
        self.root = manager.ROOT + '/full/ade150'
        self.merged = dict(
            status='complete', processed_images=1, total_images=1,
            coverage_verified=True, per_image_confusions_verified=True,
            signature=dict(num_shards=1, sample_keys=['sample']),
            metrics={'ade150': {'Source': {'mean_iou_percent': 1.0}}},
        )
        self.files = {
            self.root + '/s0/results.json': dict(
                status='complete', processed_images=1, total_images=1,
                signature=dict(num_shards=1, sample_keys=['sample']),
                source_equivalence_first_image_verified=True,
            ),
            self.root + '/merged.json': copy.deepcopy(self.merged),
        }

    def run_collection(self, execute):
        sftp = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            with (patch.object(manager, 'LOCAL', Path(directory)),
                  patch.object(manager, 'remote_json', side_effect=lambda _, path: self.files.get(path)),
                  patch.object(manager, 'execute', execute),
                  patch.object(manager, 'report'),
                  contextlib.redirect_stdout(io.StringIO())):
                manager.collect(MagicMock(), sftp, 'ade150')
        return sftp

    def test_verified_merge_is_reused_without_new_command(self):
        execute = MagicMock()
        sftp = self.run_collection(execute)
        execute.assert_not_called()
        self.assertTrue(any(call.args[0] == self.root + '/merged.json'
                            for call in sftp.get.call_args_list))

    def test_unverified_existing_merge_is_not_replaced(self):
        self.files[self.root + '/merged.json']['coverage_verified'] = False
        execute = MagicMock()
        with self.assertRaisesRegex(RuntimeError, 'no overwrite'):
            self.run_collection(execute)
        execute.assert_not_called()

    def test_missing_merge_still_runs_the_existing_verifier(self):
        del self.files[self.root + '/merged.json']

        def finish_merge(*_):
            self.files[self.root + '/merged.json'] = copy.deepcopy(self.merged)

        execute = MagicMock(side_effect=finish_merge)
        self.run_collection(execute)
        execute.assert_called_once()
        self.assertIn('verify,merge_results', execute.call_args.args[1])


if __name__ == '__main__':
    unittest.main()
