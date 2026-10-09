"""Download one pinned public frozen encoder for the region-observation study."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


REPO = "google/siglip2-base-patch16-256"
REVISION = "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir
    snapshot_download(REPO, revision=REVISION, local_dir=root,
                      allow_patterns=["*.json", "*.model", "model.safetensors", "README.md"],
                      max_workers=4)
    checkpoint = root / "model.safetensors"
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    manifest = {"repository": REPO, "revision": REVISION,
                "weights_sha256": digest.hexdigest(), "weights_bytes": checkpoint.stat().st_size,
                "frozen": True}
    (root / "region_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
