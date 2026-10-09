import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from merge_support_conditioned_shards import merge_supported
from eval_support_conditioned_geometry import changed_pixel_counts


def paired_changes_exclude_ignored_and_partition_correctness():
    import numpy as np
    result = changed_pixel_counts(np.array([0, 0, 1, 0, 1]),
                                  np.array([1, 1, 0, 1, 0]),
                                  np.array([1, 0, -1, 2, 1]), 3)
    assert result["valid"] == 4 and result["changed"] == 4
    assert result["beneficial"] == 1 and result["harmful"] == 2
    assert result["wrong_to_wrong"] == 1
    assert result["beneficial_by_true_class"] == [0, 1, 0]
    assert result["harmful_by_true_class"] == [1, 1, 0]


def coverage_verified_merge_sums_paired_class_counts(tmp_path):
    digest = lambda keys: hashlib.sha256("\n".join(keys).encode()).hexdigest()
    folders = []
    for index, key in enumerate(("a", "b")):
        folder = tmp_path / f"s{index}"
        folder.mkdir()
        folders.append(str(folder))
        signature = {"implementation": "support-test", "dataset": "potsdam",
                     "methods": ["Multiscale", "SupportConditioned"],
                     "classes": {"potsdam": ["a", "b"]}, "gear": {}, "competitive": None,
                     "vocabulary": {}, "checkpoints": {}, "global_sample_count": 2,
                     "global_sample_keys_sha256": digest(["a", "b"]), "num_shards": 2,
                     "shard_index": index, "sample_keys": [key], "sample_keys_sha256": digest([key]),
                     "config": {"shard_index": index, "output_dir": str(folder)}}
        shard = {"status": "complete", "processed_images": 1, "total_images": 1,
                 "signature": signature, "wall_seconds": 1, "peak_cuda_memory_mb": 1,
                 "metrics": {"potsdam": {method: {"confusion_matrix": [[1, 0], [0, 1]],
                                                  "ignored_pixels": 0}
                                         for method in signature["methods"]}},
                 "diagnostics": {"potsdam": {"tiles": 1, "mean_abs_class_delta": .1}},
                 "changes": {"potsdam": {"valid": 2, "changed": 2, "beneficial": 1,
                                          "harmful": 1, "wrong_to_wrong": 0,
                                          "beneficial_by_true_class": [1, 0],
                                          "harmful_by_true_class": [0, 1]}}}
        (folder / "results.json").write_text(json.dumps(shard))
    result = merge_supported(folders[::-1], str(tmp_path / "merged.json"))
    assert result["coverage_verified"] and result["processed_images"] == 2
    assert result["changes"]["potsdam"]["beneficial_by_true_class"] == [2, 0]
    assert result["changes"]["potsdam"]["harmful"] == 2
    invalid = json.loads((tmp_path / "s0/results.json").read_text())
    invalid["changes"]["potsdam"]["beneficial_by_true_class"] = [0, 0]
    (tmp_path / "s0/results.json").write_text(json.dumps(invalid))
    try:
        merge_supported(folders, str(tmp_path / "invalid.json"))
    except ValueError as error:
        assert "Per-class" in str(error)
    else:
        raise AssertionError("Invalid class counts were accepted.")
    assert not (tmp_path / "invalid.json").exists()


class SupportMergeTests(unittest.TestCase):
    def test_paired_changes(self):
        paired_changes_exclude_ignored_and_partition_correctness()

    def test_full_merge_and_invalid_class_counts(self):
        with TemporaryDirectory() as folder:
            coverage_verified_merge_sums_paired_class_counts(Path(folder))


if __name__ == "__main__":
    unittest.main()
