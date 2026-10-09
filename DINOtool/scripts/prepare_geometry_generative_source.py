"""Prepare pinned public weights and author protocol for a bounded observer test."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from huggingface_hub import snapshot_download


REPOSITORY = "CompVis/stable-diffusion-v1-4"
REVISION = "133a221b8aa7292a167afc5127cb63fb5005638b"
AUTHOR_COMMIT = "e9f772d78b75976112d88a1002c496f6ef0cb27e"
FILES = ["model_index.json", "README.md", "scheduler/scheduler_config.json",
         "tokenizer/merges.txt", "tokenizer/vocab.json", "tokenizer/tokenizer_config.json",
         "tokenizer/special_tokens_map.json", "text_encoder/config.json",
         "text_encoder/model.fp16.safetensors", "unet/config.json",
         "unet/diffusion_pytorch_model.fp16.safetensors", "vae/config.json",
         "vae/diffusion_pytorch_model.safetensors"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(part)
    return digest.hexdigest()


def main(output):
    manifest = output/"generative_source_manifest.json"
    if manifest.exists():
        raise ValueError("Preserve an existing completed source.")
    snapshot_download(REPOSITORY, revision=REVISION, local_dir=output,
                      allow_patterns=FILES, max_workers=4)
    records = []
    for name in FILES:
        path = output/name
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    public = output/"author_protocol"
    public.mkdir(exist_ok=True)
    author_files = []
    for name in ("README.md", "diffusion/models.py", "eval_prob_adaptive.py"):
        url = f"https://raw.githubusercontent.com/diffusion-classifier/diffusion-classifier/{AUTHOR_COMMIT}/{name}"
        content = urllib.request.urlopen(url, timeout=40).read()
        path = public/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        author_files.append({"path": name, "url": url, "sha256": sha256(path)})
    row = {"repository": REPOSITORY, "revision": REVISION, "files": records,
           "author_repository": "diffusion-classifier/diffusion-classifier",
           "author_commit": AUTHOR_COMMIT, "author_files": author_files,
           "frozen": True, "purpose": "epsilon prediction compatibility; no image generation",
           "license_note": "Stable Diffusion v1-4 CreativeML OpenRAIL-M; see pinned model README.",
           "protocol_boundary": "Regional loss maps/all20 uniform aliases are our diagnostic adaptation, "
                                "not a reproduction of the complete official adaptive classifier."}
    manifest.write_text(json.dumps(row, indent=2)+"\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args().output_dir)
