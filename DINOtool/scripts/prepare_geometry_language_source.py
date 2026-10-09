"""Prepare a pinned, public frozen VLM for Geometry observation diagnostics."""
import argparse
import json
from pathlib import Path

import requests
from huggingface_hub import snapshot_download


REPOSITORY = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
SOURCE = Path("/data/test/code/ovss_cafe_ped_v2_20260919/ckpt/Qwen25VL7B_GeometryDiagnostic_20261002")


def main(output):
    manifest = output / "language_source_manifest.json"
    if manifest.exists():
        raise ValueError("Preserve an already prepared source.")
    response = requests.get(f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true", timeout=40)
    response.raise_for_status()
    metadata = response.json()
    if metadata["sha"] != REVISION or metadata.get("gated"):
        raise ValueError("Require the pinned public ungated source.")
    files = [item for item in metadata["siblings"] if item["rfilename"].endswith(
        (".json", ".safetensors", ".txt", ".md")) or item["rfilename"] == "LICENSE"]
    snapshot_download(REPOSITORY, revision=REVISION, local_dir=output, token=False,
                      allow_patterns=[item["rfilename"] for item in files], max_workers=4)
    records = []
    for item in files:
        path = output / item["rfilename"]
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError(f"Downloaded file size differs: {item['rfilename']}")
        records.append({"path": item["rfilename"], "bytes": item["size"],
                        "lfs_sha256": item.get("lfs", {}).get("sha256")})
    config = json.loads((output / "config.json").read_text())
    if config["model_type"] != "qwen2_5_vl":
        raise ValueError("Unexpected model architecture.")
    row = {"repository": REPOSITORY, "revision": REVISION, "files": records,
           "frozen": True, "trust_remote_code": False, "public_gated": False,
           "purpose": "local position-conditional class observation, no target training",
           "fairness": "Additional instruction-tuned VLM with pretrained localization supervision. "
                       "Not a matched single-backbone VIP comparison or adaptation-free claim.",
           "provenance_limit": "Pretraining/evaluation overlap has not been independently audited.",
           "license_note": "See the pinned author LICENSE and README; no gated-license acceptance.",
           "config": config}
    manifest.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps({"repository": REPOSITORY, "revision": REVISION,
                      "downloaded_bytes": sum(item["bytes"] for item in records),
                      "manifest": str(manifest)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SOURCE)
    main(parser.parse_args().output_dir)
