#!/usr/bin/env python3
"""Measure learned region diversity on source images without target labels."""
import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from cafedino_locked_loveda import build_model, image_tensor
from dinotool.cafe_region_assembly import CafeRegionAssembly, config_from_checkpoint
from dinotool.coco_stuff import COCO_CAFE41_NAMES, samples_from_manifest
from dinotool.ov_train import _write_json_atomic


def describe(membership):
    m = membership.float().reshape(-1, membership.shape[-2], membership.shape[-1])
    columns = F.normalize(m, dim=1)
    similarity = columns.transpose(1, 2) @ columns
    k = m.shape[-1]
    off_diagonal = ~torch.eye(k, dtype=torch.bool, device=m.device)
    singular = torch.linalg.svdvals(m)
    energy = singular.square()
    coverage = energy.cumsum(-1) / energy.sum(-1, keepdim=True).clamp_min(1e-12)
    return dict(mean_mask_cosine=float(similarity[:, off_diagonal].mean()),
                energy_rank95=float(((coverage < .95).sum(-1)+1).float().mean()),
                mean_pixel_max_membership=float(m.max(-1).values.mean()),
                mean_spatial_std=float(m.std(dim=1).mean()))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "official-root", "base-checkpoint", "bpe-path", "coco-root", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    cafe, _, _ = build_model(argparse.Namespace(official_root=args.official_root, checkpoint=args.base_checkpoint,
        bpe_path=args.bpe_path, device=str(device), window_size=224, model_mode="eval"))
    payload = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = CafeRegionAssembly(cafe, config_from_checkpoint(payload)).to(device).eval().requires_grad_(False)
    model.load_adapted_state_dict(payload["adapted_state"])
    text = cafe.build_text_embeddings([list(COCO_CAFE41_NAMES)]).detach()
    samples = samples_from_manifest(Path(args.coco_root) / "manifests/val2017_source_dev.json")
    samples = [samples[i] for i in torch.linspace(0, len(samples)-1, 8).long().tolist()]
    records = []
    for sample in samples:
        image_path = Path(args.coco_root) / "images/val2017" / sample.image_path.name
        image = image_tensor(image_path, 224, device)
        changes = []
        def observe(module, inputs, output):
            before, after = inputs[3].float(), output[0].float()
            changes.append(float((after-before).norm()/before.norm().clamp_min(1e-8)))
        handles = [stage.register_forward_hook(observe) for stage in model.assembly]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(image, text, return_regions=True)
        for handle in handles:
            handle.remove()
        # Only aggregate numerical diagnostics; do not retain GPU trace tensors.
        parents = [describe(torch.cat([r["parent_membership"] for r in stage if r["round"] == 1]))
                   for stage in output["region_traces"]]
        record = dict(key=sample.key, proposal=describe(output["proposal_logits"].float().softmax(-1)),
                      final_children=describe(output["final_membership"]), parents=parents,
                      stage_relative_cost_changes=changes)
        records.append(record)
        print(json.dumps(record), flush=True)
        del output
    _write_json_atomic(Path(args.output), dict(weights=args.weights, checkpoint_step=payload.get("step"), images=8, dataset="COCO source-dev",
                                             uses_target_labels=False, records=records))


if __name__ == "__main__":
    main()
