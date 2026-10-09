# Frozen Patch-Only2: Same-Field Coupling Controls

All20092 complete images on eight domains. This is a full same-field module ablation. Original20 words, local896/wide448 RGB inputs and frozen heads are unchanged. All score controls reuse identical observations with no added encoder or alias rule. No coefficients or class/domain routes are fitted.

| Domain/protocol | Geometry | Original coupled | Frozen patch2 | Local2 | Wide lifted | Equal mean | Permuted H | Patch2 - mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 46.6010 | 53.8090 | 53.4600 | 44.7173 | 50.6077 | 52.2349 | 49.8451 | +1.2251 |
| potsdam/potsdam | 40.3779 | 43.4301 | 45.9339 | 46.6666 | 42.8479 | 45.1555 | 39.7255 | +0.7784 |
| udd5/udd5 | 47.1219 | 47.0096 | 47.0786 | 47.5057 | 44.5471 | 46.6700 | 37.7300 | +0.4086 |
| oem/oem | 43.9751 | 37.2104 | 36.8400 | 44.4697 | 32.0789 | 36.7048 | 32.4239 | +0.1352 |
| loveda/P | 66.5715 | 62.5150 | 61.4752 | 68.5332 | 57.1042 | 61.5772 | 55.3993 | -0.1020 |
| loveda/D | 42.8650 | 40.0210 | 38.8353 | 40.7647 | 36.3922 | 38.7903 | 36.2012 | +0.0450 |
| vaihingen/vaihingen | 48.2368 | 50.7949 | 51.4282 | 50.1940 | 47.1621 | 49.6408 | 47.3399 | +1.7874 |
| landcoverai/landcoverai | 57.0988 | 63.6099 | 65.1069 | 59.3124 | 61.0262 | 63.2893 | 65.5873 | +1.8176 |
| flair1/flair1 | 40.5208 | 42.5038 | 42.4363 | 42.7315 | 38.7164 | 41.6849 | 40.4645 | +0.7514 |

| Arm | Equal-domain mean, LoveDA D once |
| --- | ---: |
| Geometry | 45.8497 |
| NoAdmission_Exact | 47.2986 |
| Geometry_PatchOnly2Coupled | 47.6399 |
| PatchOnly2_Local | 47.0452 |
| WideOnly_Lifted | 44.1723 |
| MeanLogit_PatchOnly2 | 46.7713 |
| PermutedH_PatchOnly2 | 43.6647 |

Coupling beats equal mean on 8/8 full domains; mean delta +0.8686pp. All20092 unique complete images and frozen primary per-image confusion endpoints match the full suite. Permutation preserves valid-token operator spectrum and padding, not spatial correspondence. It does not change the original observations. These controls are not automatically promoted as candidates.

The ridge solve H=(I+W^T W)^(-1)W^T W and score correction Z=L+H(B-L) use familiar linear algebra. Do not call the solver, twofold read strength or inherited VIP observer mathematical innovations. This panel can support a coupled-use mechanism, not priority or untouched generalization.
