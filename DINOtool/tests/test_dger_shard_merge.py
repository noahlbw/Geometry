from __future__ import annotations

from argparse import Namespace
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from merge_dger_loveda_shards import main


def test_merge_checks_actual_coverage_counts_and_sums_confusion():
    global_keys = ["a", "b", "c"]
    digest = lambda keys: hashlib.sha256("\n".join(keys).encode()).hexdigest()
    shards = []
    for index in range(2):
        keys = global_keys[index::2]
        metric = {"per_class": [{"name": "first"}, {"name": "second"}],
                  "confusion_matrix": [[index + 1, 1], [0, 2]], "ignored_pixels": 0}
        shards.append({
            "status": "complete", "processed_images": len(keys), "total_images": len(keys),
            "metrics": {p: {"G_all20_uniform": metric} for p in ("P", "D")},
            "diagnostics": {p: {"tiles": len(keys), "score": float(index)} for p in ("P", "D")},
            "candidate_audit": {p: {"geometry_errors": index + 1} for p in ("P", "D")},
            "wall_seconds": 2.0, "peak_cuda_memory_mb": 10.0,
            "signature": {
                "num_shards": 2, "shard_index": index,
                "sample_keys": keys, "sample_keys_sha256": digest(keys),
                "global_sample_count": 3, "global_sample_keys_sha256": digest(global_keys),
                "methods": ["G_all20_uniform"], "checkpoints": {}, "vocabulary": {},
                "dger": {}, "tcpr": {}, "implementation": "test",
                "config": {"tile_size": 512, "shard_index": index, "output_dir": f"s{index}"},
            },
        })
    with tempfile.TemporaryDirectory() as directory:
        paths = [Path(directory) / f"s{i}.json" for i in range(2)]
        args = Namespace(inputs=[str(p) for p in paths[::-1]], output=str(Path(directory) / "merged.json"))

        def run(items):
            for path, item in zip(paths, items):
                path.write_text(json.dumps(item))
            with redirect_stdout(io.StringIO()):
                return main(args)

        result = run(shards)
        assert result["coverage_verified"] and result["processed_images"] == 3
        assert result["metrics"]["P"]["G_all20_uniform"]["confusion_matrix"] == [[3, 2], [0, 4]]
        assert result["candidate_audit"]["P"]["geometry_errors"] == 3
        for mutation in ("coverage", "counts", "configuration"):
            wrong = deepcopy(shards)
            if mutation == "coverage":
                wrong[0]["signature"]["sample_keys"][1] = "unrelated"
                wrong[0]["signature"]["sample_keys_sha256"] = digest(wrong[0]["signature"]["sample_keys"])
            elif mutation == "counts":
                wrong[0]["processed_images"] = 1
            else:
                wrong[0]["signature"]["config"]["tile_size"] = 1008
            try:
                run(wrong)
            except ValueError:
                continue
            raise AssertionError(f"Merge accepted inconsistent {mutation}.")
