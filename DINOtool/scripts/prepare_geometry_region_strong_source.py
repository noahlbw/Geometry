"""Download the pinned public observer, keeping its source card and provenance."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from dinotool.geometry_region_strong import REPOSITORY, REVISION, SOURCE


def main(args):
    root = args.output_dir
    if (root/"region_source_manifest.json").exists():
        raise ValueError("Refusing to replace an already prepared frozen source.")
    snapshot_download(REPOSITORY, revision=REVISION, local_dir=root,
        allow_patterns=["*.json", "*.model", "model.safetensors", "README.md"], max_workers=4)
    config = json.loads((root/"config.json").read_text())
    vision = config["vision_config"]
    if vision["image_size"] != 384 or vision["patch_size"] != 14:
        raise ValueError("Unexpected pinned source geometry.")
    checkpoint = root/"model.safetensors"
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    manifest = {"repository": REPOSITORY, "revision": REVISION,
        "weights_sha256": digest.hexdigest(), "weights_bytes": checkpoint.stat().st_size,
        "frozen": True, "vision_config": vision,
        "training_data_source": "Pinned author model card: WebLI",
        "provenance_limit": "WebLI contents are not independently audited for evaluation-image overlap.",
        "fairness": "Additional independent frozen encoder; not a matched single-backbone VIP comparison."}
    (root/"region_source_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SOURCE)
    main(parser.parse_args())
