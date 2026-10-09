#!/usr/bin/env python3
"""COCO-only training for DINO SupportReobservation v1."""
from train_cafe_coco import main


if __name__ == "__main__":
    main(defaults={
        "arm": "support_reobservation",
        "epochs": 8,
        "max_updates": 20896,
        "crop_size": 448,
        "batch_size": 1,
        "accum_steps": 8,
        "lr": 1e-4,
        "cafe_lr": 2e-5,
        "warmup_steps": 500,
        "kd_weight": 0.0,
        "seed": 20260927,
    })
