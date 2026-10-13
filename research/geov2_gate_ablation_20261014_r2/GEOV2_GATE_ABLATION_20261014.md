# Geov2 structure target and adaptive gate ablation (2026-10-14)

All rows use the same 16-image development screen, backbone, aliases, background rule and 24-slot observer.

| Dataset | VIP-wide endpoint | Raw32 | QuerySlot | DonorBefore | Fixed writer g=1 | Z* target | Fixed g=2 | Adaptive gate | Canonical only | Equal fusion | Shuffled query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vdd | 62.2716 | 60.9890 | 61.3613 | 62.2255 | 61.4631 | 61.4631 | 59.8281 | 61.6014 | 37.8676 | 61.1622 | 55.5703 |
| potsdam | 52.2582 | 51.4833 | 50.9320 | 52.2582 | 50.9296 | 50.9296 | 52.6104 | 50.9271 | 41.9422 | 52.3331 | 48.2705 |
| ade150 | 20.9862 | 20.6131 | 20.7421 | 21.2813 | 20.9202 | 20.9202 | 23.2339 | 20.7692 | 21.5046 | 22.6201 | 20.1257 |
| voc20 | 88.0115 | 96.0814 | 97.6089 | 88.0115 | 97.6090 | 97.6090 | 78.6806 | 97.4065 | 81.5719 | 97.3835 | 88.9360 |

| Equal-domain macro | 55.8819 | 57.2917 | 57.6611 | 55.9441 | 57.7305 | 57.7305 | 53.5883 | 57.6761 | 45.7216 | 58.3747 | 53.2256 |

## What is selected

The final writer is `Z = L + g(x)(Z* - L)`. `Z*` is computed by the same structure-anchored solve used by the fixed writer; the gate is bounded to `[0.5, 2.0]` and uses no labels. `Geov2_FixedG2`, `Geov2_Target`, and `Geov2_EqualFusion` are controls.

## External VIP reference

| Dataset | Full-validation VIP mIoU |
|---|---:|
| vdd | 52.5765 |
| potsdam | 44.8995 |
| ade150 | 29.4630 |
| voc20 | 92.5051 |

This screen is a mechanism/ablation screen. It does not replace the frozen full-protocol results.
