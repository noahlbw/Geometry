# Supported Positive Alias: Frozen Joint-Read Pilot

Same96 developed complete inputs, fixed20 words and checkpoints. Original Geometry G/H, patch-only2 local and bounded896/448 sources unchanged. Up to16 fine views read resized896 RGB, not original-image physical8. Equal positive fine/wide observation plus Geometry-supported conditional wide hard intervention. No labels used by the model. Inherited VIP observers attributed.

| Domain/protocol | SupportedPositive_Hard | SupportedPositive_ObservationMean | SupportedPositive_ClassMean | SupportedPositive_AliasShuffle0 | SupportedPositive_AliasShuffle1 | SupportedPositive_AliasShuffle2 | SupportedPositive_NoGeometryRisk | SupportedPositive_WideOnly | SupportedPositive_ShuffledWrite | Primary-mean pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 49.7588 | 49.5849 | 49.5920 | 49.6084 | 49.5302 | 49.5755 | 49.7970 | 47.4120 | 49.6824 | +0.1739 |
| potsdam/potsdam | 54.8968 | 54.8216 | 54.8259 | 54.8437 | 54.8293 | 54.9009 | 54.9119 | 52.4613 | 54.7717 | +0.0752 |
| udd5/udd5 | 51.0549 | 50.7777 | 50.7793 | 50.5124 | 50.7550 | 50.7213 | 51.2136 | 48.0663 | 50.9123 | +0.2772 |
| oem/oem | 29.6748 | 29.5309 | 29.5237 | 29.4919 | 29.4815 | 29.5984 | 29.8628 | 26.0905 | 29.5384 | +0.1439 |
| loveda/P | 75.2554 | 75.2590 | 75.2841 | 75.5063 | 75.3606 | 75.3436 | 75.2137 | 70.3478 | 75.4051 | -0.0036 |
| loveda/D | 46.8765 | 45.5946 | 45.6650 | 45.8213 | 45.8260 | 45.4135 | 47.0139 | 41.9359 | 46.6758 | +1.2819 |
| vaihingen/vaihingen | 51.7859 | 51.5130 | 51.4827 | 51.4778 | 51.5592 | 51.5498 | 51.9034 | 50.2182 | 51.6236 | +0.2729 |
| landcoverai/landcoverai | 77.9686 | 77.9179 | 77.9167 | 77.9280 | 77.9417 | 77.9047 | 77.9643 | 76.5092 | 77.9819 | +0.0507 |
| flair1/flair1 | 38.5774 | 38.7829 | 38.8673 | 38.7995 | 38.8840 | 38.8078 | 38.6442 | 37.2238 | 38.5039 | -0.2055 |

Eight-domain means, LoveDA D once:

| Method | mIoU |
| --- | ---: |
| Geometry | 44.6836 |
| NoAdmission_Exact | 46.3360 |
| Geometry_PatchOnly2Coupled | 46.6450 |
| SupportedPositive_Hard | 50.0742 |
| SupportedPositive_ObservationMean | 49.8154 |
| SupportedPositive_ClassMean | 49.8316 |
| SupportedPositive_AliasShuffle0 | 49.8104 |
| SupportedPositive_AliasShuffle1 | 49.8509 |
| SupportedPositive_AliasShuffle2 | 49.8090 |
| SupportedPositive_NoGeometryRisk | 50.1639 |
| SupportedPositive_WideOnly | 47.4896 |
| SupportedPositive_ShuffledWrite | 49.9613 |

## Whole-Image Warm Timing

Not acquired: the frozen Geometry-risk attribution check fails. Previous exact-execution profile times are not measurements of this new joint reader. Real graph setup is retained in mask-free smoke outputs.

## Frozen Advancement Checks

```json
{
  "means": {
    "Geometry": 44.683575,
    "NoAdmission_Exact": 46.3359625,
    "Geometry_PatchOnly2Coupled": 46.645025000000004,
    "SupportedPositive_Hard": 50.0742125,
    "SupportedPositive_ObservationMean": 49.8154375,
    "SupportedPositive_ClassMean": 49.831575,
    "SupportedPositive_AliasShuffle0": 49.810375,
    "SupportedPositive_AliasShuffle1": 49.8508625,
    "SupportedPositive_AliasShuffle2": 49.8089875,
    "SupportedPositive_NoGeometryRisk": 50.1638875,
    "SupportedPositive_WideOnly": 47.48965,
    "SupportedPositive_ShuffledWrite": 49.96125000000001
  },
  "speed_ms": {},
  "unique_images": 96,
  "domain_wins": 7,
  "primary_minus_same_information_pp": 0.258775,
  "original_and_same_information_per_image_endpoints_verified": true,
  "accuracy_mechanism_passed": false,
  "gate": {
    "passed": false,
    "checks": {
      "mean_gain": true,
      "domain_wins": true,
      "worst_protocol_loss": true,
      "above_class_mean": true,
      "above_identity_null": true,
      "geometry_risk_attribution": false
    }
  }
}
```

Twelve CPU tests and VDD/Potsdam mask-free original/positive/singleton smokes precede labels. If accuracy/mechanism fails, do not launch further timing/stress/full rollout or promote a control. No labels fitted, no post-result parameter/routing changes. Developed inputs, not independent validation. Earlier failed gates and retained full model remain unchanged.

## Class Coverage And Competition

| Domain/protocol | Class | Same-info IoU | Primary IoU | Delta TP | Delta FP |
| --- | --- | ---: | ---: | ---: | ---: |
| vdd/vdd | other | 20.9792 | 20.9814 | +1659 | +4823 |
| vdd/vdd | wall | 54.7467 | 55.3461 | +22876 | -7994 |
| vdd/vdd | road | 16.4903 | 16.4485 | +5494 | +56483 |
| vdd/vdd | vegetation | 48.5575 | 48.6296 | +18305 | -4196 |
| vdd/vdd | vehicle | 26.8392 | 27.1147 | -154 | -10565 |
| vdd/vdd | roof | 87.1274 | 87.3352 | +4129 | -68540 |
| vdd/vdd | water | 92.3539 | 92.4557 | +92 | -22412 |
| potsdam/potsdam | impervious surface | 77.2062 | 77.1913 | -1794 | -1626 |
| potsdam/potsdam | building | 83.9723 | 83.8306 | -799 | +1238 |
| potsdam/potsdam | low vegetation | 64.6930 | 64.9100 | +7277 | +3480 |
| potsdam/potsdam | tree | 62.2148 | 62.4654 | +1340 | -3624 |
| potsdam/potsdam | car | 33.0103 | 32.9775 | +10 | +534 |
| potsdam/potsdam | clutter | 7.8333 | 8.0057 | +81 | -6117 |
| udd5/udd5 | vegetation | 72.3957 | 72.7013 | +463742 | +74230 |
| udd5/udd5 | building | 83.4492 | 83.6947 | -14933 | -616430 |
| udd5/udd5 | road | 43.9514 | 44.2104 | +317162 | +288815 |
| udd5/udd5 | vehicle | 19.2408 | 19.7048 | -8570 | -406058 |
| udd5/udd5 | other | 34.8514 | 34.9635 | +67472 | -165430 |
| oem/oem | bareland | 0.0000 | 0.0000 | +0 | +6905 |
| oem/oem | rangeland | 6.7226 | 6.6157 | -1454 | +165 |
| oem/oem | developed space | 22.6743 | 22.6544 | -11637 | -48362 |
| oem/oem | road | 29.2393 | 29.4012 | +1257 | +434 |
| oem/oem | tree | 25.1199 | 25.7872 | +9599 | +7352 |
| oem/oem | water | 17.1844 | 17.2118 | -117 | -891 |
| oem/oem | agriculture land | 72.6534 | 72.6374 | -897 | -1021 |
| oem/oem | building | 62.6536 | 63.0904 | +24099 | +14568 |
| loveda/P | building | 89.6217 | 89.6507 | -195 | -334 |
| loveda/P | road | 84.6982 | 85.0929 | -143 | -4447 |
| loveda/P | water | 71.5268 | 72.8622 | +19368 | +583 |
| loveda/P | barren | 70.8717 | 66.9727 | +49 | +12281 |
| loveda/P | tree | 62.7904 | 63.7490 | +14947 | +6655 |
| loveda/P | farm | 72.0453 | 73.2047 | -3972 | -44792 |
| loveda/D | background | 11.5187 | 16.3862 | +144068 | +48866 |
| loveda/D | building | 48.0033 | 48.2800 | -215 | -4279 |
| loveda/D | road | 65.5973 | 66.0433 | -203 | -8321 |
| loveda/D | water | 67.1491 | 67.7242 | +8759 | +157 |
| loveda/D | barren | 26.3807 | 26.4117 | +1018 | +3206 |
| loveda/D | tree | 51.0505 | 51.4993 | +11978 | +11932 |
| loveda/D | farm | 49.4626 | 51.7911 | -19047 | -197919 |
| vaihingen/vaihingen | impervious surface | 63.5007 | 63.9143 | +15435 | +4695 |
| vaihingen/vaihingen | building | 69.5201 | 70.0283 | -419 | -22241 |
| vaihingen/vaihingen | low vegetation | 29.3648 | 29.2521 | -2656 | -3222 |
| vaihingen/vaihingen | tree | 68.0361 | 68.1049 | +7599 | +9065 |
| vaihingen/vaihingen | car | 27.1435 | 27.6300 | -42 | -8214 |
| landcoverai/landcoverai | background | 85.0344 | 85.0676 | -224 | -557 |
| landcoverai/landcoverai | building | N/A | N/A | +0 | +0 |
| landcoverai/landcoverai | woodland | 92.7416 | 92.7037 | +563 | +1040 |
| landcoverai/landcoverai | water | 93.6191 | 93.7968 | -33 | -710 |
| landcoverai/landcoverai | road | 40.2765 | 40.3064 | -12 | -67 |
| flair1/flair1 | building | 58.8079 | 59.2202 | +11 | -1169 |
| flair1/flair1 | pervious surface | 27.6578 | 27.8501 | +116 | +7 |
| flair1/flair1 | impervious surface | 69.3510 | 69.4772 | +19 | -733 |
| flair1/flair1 | bare soil | 12.6674 | 11.6632 | +2 | +11487 |
| flair1/flair1 | water | 95.7515 | 95.7210 | -19 | -2 |
| flair1/flair1 | coniferous | 3.1536 | 2.8337 | -24 | -45 |
| flair1/flair1 | deciduous | 66.8952 | 66.8178 | +24 | +386 |
| flair1/flair1 | brushwood | 7.0324 | 7.4792 | +553 | -1955 |
| flair1/flair1 | vineyard | 55.9338 | 55.9432 | -3 | -40 |
| flair1/flair1 | herbaceous vegetation | 35.8534 | 35.2213 | -3951 | +526 |
| flair1/flair1 | agricultural land | 32.2904 | 30.7018 | -8197 | +195 |
| flair1/flair1 | plowed land | 0.0000 | 0.0000 | +0 | +2812 |
