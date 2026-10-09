#!/usr/bin/env python3
"""CAFe-RS stage A, using the existing locked source-only training engine."""
from train_cafe_coco import main

if __name__ == "__main__":
    main({"arm": "rs", "content_dim": 256, "epochs": 8, "seed": 20260913,
          "stages": 3, "modes": 4, "warmup_steps": 500})
