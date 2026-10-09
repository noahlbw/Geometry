#!/usr/bin/env python3
"""Verify DeepGlobe raw-download integrity without reading target labels or running a model.

This is deliberately a metadata-only gate. A successful run proves that the
five expected raw Parquet files agree with their download checksum manifest
and have a common Parquet schema; it does not approve a taxonomy, split, or
evaluation protocol.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


EXPECTED_SHARDS = tuple(f"train-{index:05d}-of-00005.parquet" for index in range(5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", required=True, type=Path)
    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
        help="New metadata-only audit artifact. It must be outside --target-root.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_checksum_manifest(path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            digest, filename = line.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Malformed SHA256SUMS line {line_number}: {raw_line!r}") from exc
        filename = filename.lstrip("*")
        if Path(filename).name != filename or filename in expected:
            raise ValueError(f"Unsafe or duplicate SHA256SUMS filename on line {line_number}: {filename!r}")
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            raise ValueError(f"Invalid SHA-256 digest on line {line_number}: {digest!r}")
        expected[filename] = digest.lower()
    if set(expected) != set(EXPECTED_SHARDS):
        raise ValueError(
            "SHA256SUMS must contain exactly the five expected shard names; "
            f"got {sorted(expected)}."
        )
    return expected


def parquet_metadata(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on evaluation image
        raise RuntimeError("pyarrow is required for the metadata-only DeepGlobe audit.") from exc

    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.metadata
    if metadata is None:
        raise ValueError(f"No Parquet metadata found in {path}")
    # Do not call read(), read_row_group(), or iterate batches here: labels and
    # images remain unread until a separately registered structural audit.
    return {
        "rows": int(metadata.num_rows),
        "row_groups": int(metadata.num_row_groups),
        "created_by": metadata.created_by,
        "format_version": str(metadata.format_version),
        "schema": str(parquet_file.schema_arrow),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}.", suffix=".tmp"
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    root = args.target_root.resolve()
    output = args.output_json.resolve()
    if output == root or root in output.parents:
        raise ValueError("--output-json must be outside --target-root to keep raw target data immutable.")
    complete_marker = root / "DOWNLOAD_COMPLETE"
    checksum_path = root / "raw" / "SHA256SUMS"
    raw_root = root / "raw"
    if not complete_marker.is_file():
        raise FileNotFoundError(f"Target download is not complete: missing {complete_marker}")
    if not raw_root.is_dir() or not checksum_path.is_file():
        raise FileNotFoundError(f"Missing raw directory or SHA256SUMS under {root}")

    expected = parse_checksum_manifest(checksum_path)
    shard_records: list[dict[str, Any]] = []
    common_schema: str | None = None
    for name in EXPECTED_SHARDS:
        shard = raw_root / name
        if not shard.is_file():
            raise FileNotFoundError(f"Missing expected DeepGlobe shard: {shard}")
        actual_digest = sha256(shard)
        if actual_digest != expected[name]:
            raise ValueError(f"SHA-256 mismatch for {shard}: {actual_digest} != {expected[name]}")
        metadata = parquet_metadata(shard)
        if common_schema is None:
            common_schema = metadata["schema"]
        elif metadata["schema"] != common_schema:
            raise ValueError(f"Parquet schema mismatch in {shard}")
        shard_records.append(
            {
                "name": name,
                "bytes": shard.stat().st_size,
                "sha256": actual_digest,
                "metadata": metadata,
            }
        )

    audit = {
        "status": "METADATA_ONLY_COMPLETE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target_root": str(root),
        "download_complete_sha256": sha256(complete_marker),
        "checksum_manifest_sha256": sha256(checksum_path),
        "shards": shard_records,
        "total_rows": sum(record["metadata"]["rows"] for record in shard_records),
        "script_sha256": sha256(Path(__file__).resolve()),
        "restriction": (
            "No Parquet rows, images, masks, labels, taxonomy, target scores, or model predictions were read. "
            "A separate structural/taxonomy audit remains required before evaluation."
        ),
    }
    write_json_atomic(output, audit)
    print(json.dumps({"status": audit["status"], "output_json": str(output), "total_rows": audit["total_rows"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"DeepGlobe metadata audit failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
