"""Fetch only pinned public SAM2 safetensors/configs, never repository code."""
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import Sam2Model, Sam2Processor

from dinotool.grounded_mask_observer import REPO, REVISION


SOURCE = Path("/data/test/code/ovss_cafe_ped_v2_20260919/ckpt/SAM21_tiny_de431c4")


if __name__ == "__main__":
    snapshot_download(REPO, revision=REVISION, local_dir=SOURCE, token=False,
        allow_patterns=["config.json", "preprocessor_config.json", "processor_config.json", "model.safetensors"])
    processor = Sam2Processor.from_pretrained(SOURCE, local_files_only=True, trust_remote_code=False)
    model, loading = Sam2Model.from_pretrained(SOURCE, local_files_only=True,
        trust_remote_code=False, use_safetensors=True, output_loading_info=True)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError("SAM2 checkpoint does not exactly match the image-segmentation model.")
    model.eval().requires_grad_(False)
    manifest = {"repo": REPO, "revision": REVISION, "source": str(SOURCE), "status": "ready",
        "safetensors_only": True, "trust_remote_code": False,
        "parameters": sum(p.numel() for p in model.parameters()),
        "exact_checkpoint_loading": True,
        "loading_info": {key: len(loading.get(key, [])) for key in
                         ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")},
        "supervision": "additional frozen segmentation pretraining; SAM2 is not our innovation"}
    (SOURCE/"source_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest), flush=True)
