# Frozen Patch-Only2: Same-Field Coupling Controls

UDD5 full40; eight deterministic evenly spaced complete images per other domain. This96-image development audit is not another full eight-domain benchmark. Original20 words, local896/wide448 RGB inputs and frozen heads are unchanged. All score controls reuse identical observations with no added encoder or alias rule. No coefficients or class/domain routes are fitted.

| Domain/protocol | Geometry | Original coupled | Frozen patch2 | Local2 | Wide lifted | Equal mean | Permuted H | Patch2 - mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 42.7217 | 46.3491 | 46.8468 | 44.3779 | 43.7690 | 46.2692 | 39.9591 | +0.5776 |
| potsdam/potsdam | 45.6466 | 49.7562 | 51.5168 | 50.3739 | 48.5208 | 50.3618 | 46.9197 | +1.1550 |
| udd5/udd5 | 47.1219 | 47.0096 | 47.0786 | 47.5057 | 44.5471 | 46.6700 | 37.7300 | +0.4086 |
| oem/oem | 31.4297 | 25.7311 | 25.2530 | 34.2396 | 22.4286 | 25.7763 | 22.5129 | -0.5233 |
| loveda/P | 70.8274 | 67.0933 | 67.2104 | 75.5096 | 60.7151 | 66.9756 | 61.8859 | +0.2348 |
| loveda/D | 46.6369 | 41.0700 | 39.4705 | 43.6800 | 37.5573 | 39.7846 | 37.3995 | -0.3141 |
| vaihingen/vaihingen | 46.0831 | 48.3789 | 49.1468 | 47.6029 | 45.3231 | 47.3370 | 43.6264 | +1.8098 |
| landcoverai/landcoverai | 57.9352 | 75.1447 | 76.2441 | 60.0647 | 69.8945 | 74.0803 | 76.8194 | +2.1638 |
| flair1/flair1 | 39.8935 | 37.2481 | 37.6036 | 41.6452 | 34.0349 | 37.3591 | 31.5861 | +0.2445 |

| Arm | Equal-domain mean, LoveDA D once |
| --- | ---: |
| Geometry | 44.6836 |
| NoAdmission_Exact | 46.3360 |
| Geometry_PatchOnly2Coupled | 46.6450 |
| PatchOnly2_Local | 46.1862 |
| WideOnly_Lifted | 43.2594 |
| MeanLogit_PatchOnly2 | 45.9548 |
| PermutedH_PatchOnly2 | 42.0691 |

Coupling beats equal mean on 6/8 diagnostic domains; mean delta +0.6902pp. All96 unique complete images and frozen primary per-image confusion endpoints match the full suite. Permutation preserves valid-token operator spectrum and padding, not spatial correspondence. It does not change the original observations. These controls are not automatically promoted as candidates.

The ridge solve H=(I+W^T W)^(-1)W^T W and score correction Z=L+H(B-L) use familiar linear algebra. Do not call the solver, twofold read strength or inherited VIP observer mathematical innovations. This panel can support a coupled-use mechanism, not priority or untouched generalization.
