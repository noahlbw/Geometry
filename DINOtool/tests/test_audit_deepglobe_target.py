from __future__ import annotations

from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import audit_deepglobe_target as audit


def _valid_manifest() -> str:
    digest = "a" * 64
    return "\n".join(f"{digest}  {name}" for name in audit.EXPECTED_SHARDS) + "\n"


def test_checksum_manifest_requires_exact_expected_set(tmp_path: Path) -> None:
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(_valid_manifest(), encoding="utf-8")
    parsed = audit.parse_checksum_manifest(manifest)
    assert tuple(parsed) == audit.EXPECTED_SHARDS

    manifest.write_text(_valid_manifest().replace(audit.EXPECTED_SHARDS[-1], "unexpected.parquet"), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly the five expected"):
        audit.parse_checksum_manifest(manifest)


def test_raw_target_cannot_receive_audit_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "target"
    target.mkdir()
    nested_output = target / "metadata.json"
    monkeypatch.setattr(sys, "argv", ["audit", "--target-root", str(target), "--output-json", str(nested_output)])
    with pytest.raises(ValueError, match="outside --target-root"):
        audit.main()
