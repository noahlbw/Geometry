#!/usr/bin/env python3
"""COCO-Stuff-only training for DINO query-conditioned reconstruction.

This entrypoint never discovers OEM, LoveDA, ADE20K, UDD5, or another target
dataset. Checkpoint selection is performed only on the locked 2,500-image
COCO-Stuff source-development manifest.
"""
from train_cafe_coco import main


if __name__ == "__main__":
    main(defaults={
        "arm": "query_reconstruction",
        "epochs": 8,
        "max_updates": 20896,
        "batch_size": 2,
        "accum_steps": 4,
        "lr": 1e-4,
        "cafe_lr": 2e-5,
        "warmup_steps": 500,
        "seed": 20260926,
    })
