from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from merge_grounded_context_shards import main


def test_external_merge_exact_counts_and_rejects_changed_protocol():
    digest = lambda keys: hashlib.sha256("\n".join(keys).encode()).hexdigest()
    methods = ["G_all20_uniform", "GroundedContext"]
    keys = ["a", "b", "c"]
    names = ["vegetation", "other"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        inputs = []
        for index in range(2):
            folder = root / f"s{index}"
            folder.mkdir()
            inputs.append(str(folder))
            sample_keys = keys[index::2]
            metric = {"confusion_matrix": [[index + 1, 1], [0, 2]], "ignored_pixels": 0}
            data = {
                "status": "complete", "processed_images": len(sample_keys), "total_images": len(sample_keys),
                "metrics": {"udd5": {m: metric for m in methods}},
                "changes_vs_geometry": {"udd5": {"GroundedContext": {
                    "valid": 4, "changed": 0, "beneficial": 0, "harmful": 0, "wrong_to_wrong": 0}}},
                "diagnostics": {"udd5": {"tiles": len(sample_keys), "solver_error": 0.0}},
                "wall_seconds": 1.0, "peak_cuda_memory_mb": 1.0,
                "forward_counts": {"local": len(sample_keys), "context": len(sample_keys)},
                "signature": {
                    "implementation": "test", "dataset": "udd5", "methods": methods,
                    "classes": {"udd5": names}, "grounded": {}, "contextual_phrase": {}, "tcpr": {},
                    "vocabulary": {}, "checkpoints": {}, "global_sample_count": 3,
                    "global_sample_keys_sha256": digest(keys), "num_shards": 2, "shard_index": index,
                    "sample_keys": sample_keys, "sample_keys_sha256": digest(sample_keys),
                    "config": {"shard_index": index, "output_dir": str(folder), "tile_size": 512},
                },
            }
            (folder / "results.json").write_text(json.dumps(data))
        result = main(inputs[::-1], root / "merged.json")
        assert result["coverage_verified"] and result["processed_images"] == 3
        assert result["metrics"]["udd5"][methods[0]]["confusion_matrix"] == [[3, 2], [0, 4]]
        assert result["metrics"]["udd5"][methods[0]]["non_residual_mean_iou_percent"] == 60.0
        original = json.loads((root / "s1/results.json").read_text())
        for modification in ("config", "keys"):
            changed = copy.deepcopy(original)
            if modification == "config":
                changed["signature"]["config"]["tile_size"] = 1024
            else:
                changed["signature"]["sample_keys"] = ["a"]
            (root / "s1/results.json").write_text(json.dumps(changed))
            try:
                main(inputs, root / "invalid.json")
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid shard configuration or coverage was accepted")
