import hashlib
import json
from pathlib import Path
import sys

import pytest

from dinotool.local_readout_audit import ARMS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from merge_local_readout_2x2_shards import main


def test_merge_requires_complete_unique_coverage_and_sums_intersections(tmp_path):
    digest = lambda keys: hashlib.sha256("\n".join(keys).encode()).hexdigest()
    keys = ["a", "b"]
    folders = []
    for index, key in enumerate(keys):
        folder = tmp_path / f"s{index}"
        folder.mkdir()
        folders.append(str(folder))
        patterns = [0] * 16
        patterns[15 if index else 0] = 1
        matrix = [[1, 0], [0, 0]] if index == 0 else [[0, 0], [1, 0]]
        signature = {
            "implementation": "test", "dataset": "vdd", "arms": list(ARMS),
            "classes": {"vdd": ["one", "two"]}, "readouts": {}, "vocabulary": {},
            "checkpoints": {}, "global_sample_count": 2,
            "global_sample_keys_sha256": digest(keys), "num_shards": 2,
            "shard_index": index, "sample_keys": [key], "sample_keys_sha256": digest([key]),
            "config": {"output_dir": str(folder), "shard_index": index, "tile_size": 512},
        }
        shard = {
            "status": "complete", "processed_images": 1, "total_images": 1,
            "metrics": {"vdd": {arm: {"confusion_matrix": matrix, "ignored_pixels": 0}
                                for arm in ARMS}},
            "error_intersections": {"vdd": {
                "valid_pixels": 1, "error_patterns": patterns,
                "error_patterns_by_true_class": [patterns if index == 0 else [0] * 16,
                                                 patterns if index == 1 else [0] * 16],
                "pairwise": {},
            }},
            "diagnostics": {"tiles": 1, "sparse_active_fraction": 0.5},
            "wall_seconds": 1, "peak_cuda_memory_mb": 1,
            "signature": signature,
        }
        (folder / "results.json").write_text(json.dumps(shard), encoding="utf-8")
    merged = main(folders[::-1], tmp_path / "merged.json")
    assert merged["coverage_verified"] and merged["processed_images"] == 2
    assert merged["error_intersections"]["vdd"]["error_patterns"][15] == 1
    assert merged["error_intersections"]["vdd"]["oracle_any_correct_pixel_accuracy_percent"] == 50
    shard = json.loads((tmp_path / "s1/results.json").read_text())
    shard["signature"]["sample_keys"] = ["a"]
    (tmp_path / "s1/results.json").write_text(json.dumps(shard))
    with pytest.raises(ValueError):
        main(folders, tmp_path / "invalid.json")
