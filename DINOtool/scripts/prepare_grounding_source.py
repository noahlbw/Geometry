"""Fetch only the pinned public safetensors detector, never executable repo code."""
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


REPO = "IDEA-Research/grounding-dino-tiny"
REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
SOURCE = Path("/data/test/code/ovss_cafe_ped_v2_20260919/ckpt/GroundingDINO_tiny_a2bb814")


if __name__ == "__main__":
    snapshot_download(REPO, revision=REVISION, local_dir=SOURCE, token=False,
                      allow_patterns=["*.json", "vocab.txt", "model.safetensors", "README.md"])
    processor = AutoProcessor.from_pretrained(SOURCE, local_files_only=True, trust_remote_code=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        SOURCE, local_files_only=True, trust_remote_code=False, use_safetensors=True)
    model.eval().requires_grad_(False)
    manifest = {"repo": REPO, "revision": REVISION, "safetensors_only": True,
        "trust_remote_code": False, "source": str(SOURCE), "status": "ready",
        "maximum_text_length": model.config.max_text_len,
        "tokenizer_fast": processor.tokenizer.is_fast,
        "parameters": sum(p.numel() for p in model.parameters()),
        "supervision_note": "Additional pretrained detection supervision; not a matched single-encoder system."}
    if not processor.tokenizer.is_fast:
        raise RuntimeError("Need fast-tokenizer offsets for exact alias spans.")
    (SOURCE/"source_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest), flush=True)
