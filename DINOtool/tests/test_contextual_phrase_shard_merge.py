from __future__ import annotations

from argparse import Namespace
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from merge_contextual_phrase_shards import main


def test_contextual_phrase_merge_checks_coverage_and_sums_confusion() -> None:
    global_keys = ["a", "b", "c"]
    digest = lambda keys: hashlib.sha256("\n".join(keys).encode()).hexdigest()
    methods = ["G_all20_uniform", "ContextualPhrase"]
    shards = []
    for index in range(2):
        keys = global_keys[index::2]
        metrics = {}
        for protocol in ("P", "D"):
            metrics[protocol] = {}
            for method in methods:
                metrics[protocol][method] = {
                    "per_class": [{"name": "first"}, {"name": "second"}],
                    "confusion_matrix": [[index + 1, 1], [0, 2]],
                    "ignored_pixels": 0,
                }
        change = {field: 1 for field in ("valid", "changed", "beneficial", "harmful", "wrong_to_wrong")}
        shards.append({
            "status": "complete",
            "processed_images": len(keys),
            "total_images": len(keys),
            "metrics": metrics,
            "changes_vs_geometry": {
                protocol: {"ContextualPhrase": change} for protocol in ("P", "D")
            },
            "diagnostics": {protocol: {"tiles": len(keys), "score": float(index)} for protocol in ("P", "D")},
            "wall_seconds": 2.0,
            "peak_cuda_memory_mb": 10.0,
            "signature": {
                "num_shards": 2,
                "shard_index": index,
                "sample_keys": keys,
                "sample_keys_sha256": digest(keys),
                "global_sample_count": 3,
                "global_sample_keys_sha256": digest(global_keys),
                "methods": methods,
                "checkpoints": {},
                "vocabulary": {},
                "contextual_phrase": {},
                "tcpr": {},
                "implementation": "test",
                "config": {"tile_size": 512, "shard_index": index, "output_dir": f"s{index}"},
            },
        })
    with tempfile.TemporaryDirectory() as directory:
        paths = [Path(directory) / f"s{i}.json" for i in range(2)]
        for path, item in zip(paths, shards):
            path.write_text(json.dumps(item))
        args = Namespace(inputs=[str(path) for path in paths[::-1]], output=str(Path(directory) / "merged.json"))
        with redirect_stdout(io.StringIO()):
            result = main(args)
        assert result["coverage_verified"] and result["processed_images"] == 3
        assert result["metrics"]["P"]["G_all20_uniform"]["confusion_matrix"] == [[3, 2], [0, 4]]
