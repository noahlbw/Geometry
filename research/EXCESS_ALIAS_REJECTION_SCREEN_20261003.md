# Contextual Alias Excess Rejection: Verified Results

## Outcome Interpretation

The isolated successor is implemented and complete, but neither predeclared advancement route passes. Preserve the original coupled model. Clean mean46.057795 versus46.058780% (-0.000985pp); wrong-parent mean45.318669 versus45.315359% (+0.003310pp); paraphrase mean45.981510 versus45.982989% (-0.001479pp). One clean shuffle is higher than the primary. On wrong-parent words, text-only45.482304% and hard deletion45.434836% both outperform the primary. Near-baseline stability is mostly weak intervention, not demonstrated useful screening.

The mask-free source decomposition isolates a bottleneck. On the sampled clean512 windows, requiring an absolute wide-versus-local cosine gain retains only6.24% of responsibility-weighted candidate rejection on VDD,2.86% on Potsdam, and3.79% on OEM. This does not show that all pre-gain candidates are correct; it shows where this source loses activity. Removing that condition is a future hypothesis, not a tested fix.

Absolute alias response increase is not necessary for increased class competition: local A=.50/B=.49 and wide A=.48/B=.20 grows A's margin from.01 to.28 although A's score falls. Same text templates do not make responses from two different visual readers semantically calibrated. The next source needs a class-relative/action-relevant criterion, not a retrospective scalar gain boost.

Semantic wrong-parent assignment is not synonymous with damaging predictions. Against the matched first8 clean coupling, the constructed replacement loses5.280052pp on VDD and0.583379pp on Potsdam, but improves OEM0.815290pp, Vaihingen1.238387pp, LandCover.ai1.217502pp, FLAIR-12.660894pp, and UDD50.235547pp. Replacement also removes original aliases and changes salience/class competition. These observations cannot identify which removed individual word was harmful or certify arbitrary-LLM robustness.

The text-only control improves full UDD5 from49.211686 to50.713894%, but its clean domain mean falls to45.884504% and its paraphrase mean to44.985799%. Do not select it for UDD5 alone. The current failures concern these fixed controllers, not all possible alias control. The retained Geometry/coupling and all historical outputs remain unchanged; no full rollout follows.

Clean96 complete images: UDD5 full40, seven other domains8 each. Each constructed stress scenario uses64 first8 images/domain. Local original20 anchor is fixed; only contextual20 words change. Frozen weights, image-only rejection, no target-label parameter fitting. Corrected IRRG Vaihingen, LandCover.ai substitution. Development screen, not independent full-dataset validation. Stress is not fresh LLM generation.

Eleven mathematical tests and mask-free GPU smoke passed. Complete unique coverage, exact three historical controls, confusion sums, transition endpoints and unchanged stress Geometry independently checked.

## clean

| Dataset/protocol | Geometry | BroadVIP | Anchored_VIP | ExcessReject_Coupled | TextOnlyReject_Coupled | HardDelete_Coupled | ShuffledReject0_Coupled | ShuffledReject1_Coupled | ShuffledReject2_Coupled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 31.946180 | 51.120279 | 57.001172 | 57.000026 | 57.204657 | 56.935543 | 57.000692 | 57.001021 | 57.000737 |
| potsdam/potsdam | 40.352836 | 40.504811 | 41.780594 | 41.770089 | 41.813204 | 41.331058 | 41.771895 | 41.771906 | 41.775187 |
| udd5/udd5 | 50.555301 | 45.281382 | 49.211686 | 49.214344 | 50.713894 | 49.205132 | 49.212240 | 49.215543 | 49.211949 |
| oem/oem | 39.354327 | 28.118106 | 31.953271 | 31.960512 | 33.343481 | 32.035359 | 31.957443 | 31.955484 | 31.954438 |
| loveda/P | 62.825368 | 56.377642 | 62.486347 | 62.485220 | 62.129595 | 62.446551 | 62.485878 | 62.486347 | 62.485375 |
| loveda/D | 38.455828 | 33.286100 | 36.559979 | 36.559691 | 35.572740 | 36.544633 | 36.560223 | 36.560145 | 36.559970 |
| vaihingen/vaihingen | 50.232464 | 47.630560 | 51.449095 | 51.440584 | 50.458047 | 51.172133 | 51.442787 | 51.443523 | 51.443591 |
| landcoverai/landcoverai | 60.904853 | 65.863798 | 66.906020 | 66.911688 | 65.609437 | 67.030866 | 66.909354 | 66.910043 | 66.910184 |
| flair1/flair1 | 38.842764 | 29.706744 | 33.608426 | 33.605426 | 32.360572 | 33.264719 | 33.605870 | 33.607237 | 33.605912 |
| Equal-domain mean | 43.830569 | 42.688973 | 46.058780 | 46.057795 | 45.884504 | 45.939931 | 46.057563 | 46.058113 | 46.057746 |

## wrong_parent

| Dataset/protocol | Geometry | BroadVIP | Anchored_VIP | ExcessReject_Coupled | TextOnlyReject_Coupled | HardDelete_Coupled | ShuffledReject0_Coupled | ShuffledReject1_Coupled | ShuffledReject2_Coupled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 31.946180 | 45.611161 | 51.721120 | 51.723356 | 52.524301 | 51.700068 | 51.721122 | 51.722534 | 51.721256 |
| potsdam/potsdam | 40.352836 | 36.761350 | 41.197215 | 41.204070 | 42.711365 | 41.544574 | 41.191220 | 41.195611 | 41.197039 |
| udd5/udd5 | 46.426488 | 38.606434 | 44.247061 | 44.280306 | 44.368185 | 44.917506 | 44.254468 | 44.256118 | 44.252472 |
| oem/oem | 39.354327 | 27.019790 | 32.768560 | 32.777543 | 34.520109 | 32.864215 | 32.773844 | 32.771361 | 32.770224 |
| loveda/P | 62.825368 | 53.757891 | 64.135523 | 64.136654 | 61.341851 | 63.968297 | 64.135190 | 64.135630 | 64.137409 |
| loveda/D | 38.455828 | 32.119870 | 35.508589 | 35.507982 | 34.868307 | 35.467015 | 35.510055 | 35.510105 | 35.508472 |
| vaihingen/vaihingen | 50.232464 | 46.551467 | 52.687483 | 52.668081 | 52.678838 | 52.644435 | 52.676560 | 52.669770 | 52.672871 |
| landcoverai/landcoverai | 60.904853 | 64.268221 | 68.123522 | 68.124572 | 66.854308 | 68.174616 | 68.125272 | 68.125052 | 68.124038 |
| flair1/flair1 | 38.842764 | 27.954073 | 36.269320 | 36.263442 | 35.333016 | 36.166256 | 36.265791 | 36.265847 | 36.265315 |
| Equal-domain mean | 43.314467 | 39.861546 | 45.315359 | 45.318669 | 45.482304 | 45.434836 | 45.314792 | 45.314550 | 45.313961 |

## paraphrase

| Dataset/protocol | Geometry | BroadVIP | Anchored_VIP | ExcessReject_Coupled | TextOnlyReject_Coupled | HardDelete_Coupled | ShuffledReject0_Coupled | ShuffledReject1_Coupled | ShuffledReject2_Coupled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 31.946180 | 54.478831 | 58.746982 | 58.743684 | 59.045212 | 58.832052 | 58.745616 | 58.746529 | 58.746909 |
| potsdam/potsdam | 40.352836 | 41.797707 | 43.019466 | 43.010538 | 42.491977 | 42.702238 | 43.010641 | 43.010667 | 43.013038 |
| udd5/udd5 | 46.426488 | 41.539204 | 45.332993 | 45.331489 | 43.011156 | 44.677052 | 45.332959 | 45.339000 | 45.333370 |
| oem/oem | 39.354327 | 29.423111 | 33.449625 | 33.457294 | 34.777333 | 33.429829 | 33.455220 | 33.453247 | 33.452408 |
| loveda/P | 62.825368 | 56.896162 | 62.135176 | 62.135183 | 61.826940 | 62.163292 | 62.135183 | 62.135197 | 62.135176 |
| loveda/D | 38.455828 | 32.561407 | 35.740139 | 35.740644 | 34.811123 | 35.745515 | 35.740548 | 35.740512 | 35.740330 |
| vaihingen/vaihingen | 50.232464 | 48.074988 | 51.415021 | 51.408456 | 50.409386 | 51.226044 | 51.410261 | 51.410606 | 51.409859 |
| landcoverai/landcoverai | 60.904853 | 64.077500 | 65.629553 | 65.632991 | 62.136947 | 65.673848 | 65.632650 | 65.632246 | 65.632368 |
| flair1/flair1 | 38.842764 | 30.821211 | 34.530135 | 34.526986 | 33.203257 | 34.233657 | 34.528242 | 34.528933 | 34.527422 |
| Equal-domain mean | 43.314467 | 42.846745 | 45.982989 | 45.981510 | 44.985799 | 45.815029 | 45.982017 | 45.982717 | 45.981963 |

## Decision

```json
{
  "promising": false,
  "clean_gain_route": false,
  "robustness_route": false,
  "wins": {
    "clean": 3,
    "wrong_parent": 5,
    "paraphrase": 3
  },
  "worst_protocol_delta": {
    "clean": -0.010505421698546513,
    "wrong_parent": -0.01940182836562343,
    "paraphrase": -0.008928265381598521
  },
  "gate": {
    "clean_gain_route": "mean>=+0.1pp, wins>=5/8, worst>=-1pp, beat text and each shuffle",
    "robustness_route": "clean mean>=-0.1pp and worst>=-0.5pp; wrong-parent gain>=0.5pp and wins>=5/8; beat text/hard/each shuffle; paraphrase mean>=-0.1pp and worst>=-0.5pp",
    "stress_scope": "first8 complete images per domain, contextual vocabulary only; local20 fixed",
    "automatic_full_rollout": false,
    "independent_validation": false
  }
}
```

## Matched First8 Vocabulary Effect

The clean main run has40 UDD5 images, but each stress run has8. Use these matched first8 baseline scores when measuring vocabulary effects. Wrong-parent assignment is a semantic construction, not a guarantee of harmful predictions. Both perturbations also remove four original aliases.

| Dataset/protocol | Clean first8 coupled | Wrong-parent coupled | Delta pp | Paraphrase coupled | Delta pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 57.001172 | 51.721120 | -5.280052 | 58.746982 | +1.745810 |
| potsdam/potsdam | 41.780594 | 41.197215 | -0.583379 | 43.019466 | +1.238872 |
| udd5/udd5 | 44.011514 | 44.247061 | +0.235547 | 45.332993 | +1.321479 |
| oem/oem | 31.953271 | 32.768560 | +0.815290 | 33.449625 | +1.496354 |
| loveda/P | 62.486347 | 64.135523 | +1.649176 | 62.135176 | -0.351170 |
| loveda/D | 36.559979 | 35.508589 | -1.051390 | 35.740139 | -0.819840 |
| vaihingen/vaihingen | 51.449095 | 52.687483 | +1.238387 | 51.415021 | -0.034074 |
| landcoverai/landcoverai | 66.906020 | 68.123522 | +1.217502 | 65.629553 | -1.276468 |
| flair1/flair1 | 33.608426 | 36.269320 | +2.660894 | 34.530135 | +0.921709 |

## Direct Corrections Versus Matched Coupling

| Dataset/protocol | Scenario | Beneficial | Harmful |
| --- | --- | ---: | ---: |
| vdd/vdd | clean | 335 | 431 |
| vdd/vdd | wrong_parent | 2779 | 975 |
| vdd/vdd | paraphrase | 1046 | 1417 |
| potsdam/potsdam | clean | 772 | 1053 |
| potsdam/potsdam | wrong_parent | 2583 | 1100 |
| potsdam/potsdam | paraphrase | 712 | 984 |
| udd5/udd5 | clean | 23238 | 14742 |
| udd5/udd5 | wrong_parent | 37812 | 3217 |
| udd5/udd5 | paraphrase | 1387 | 1771 |
| oem/oem | clean | 2011 | 65 |
| oem/oem | wrong_parent | 2721 | 172 |
| oem/oem | paraphrase | 1966 | 18 |
| loveda/P | clean | 0 | 4 |
| loveda/D | clean | 9 | 23 |
| loveda/P | wrong_parent | 13 | 16 |
| loveda/D | wrong_parent | 47 | 60 |
| loveda/P | paraphrase | 0 | 1 |
| loveda/D | paraphrase | 9 | 20 |
| vaihingen/vaihingen | clean | 79 | 663 |
| vaihingen/vaihingen | wrong_parent | 287 | 1772 |
| vaihingen/vaihingen | paraphrase | 99 | 604 |
| landcoverai/landcoverai | clean | 145 | 40 |
| landcoverai/landcoverai | wrong_parent | 136 | 68 |
| landcoverai/landcoverai | paraphrase | 108 | 36 |
| flair1/flair1 | clean | 86 | 163 |
| flair1/flair1 | wrong_parent | 136 | 177 |
| flair1/flair1 | paraphrase | 76 | 163 |

## vdd/vdd/clean Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| other | 53.4471 | 53.4473 | +0.0002 | 72.8293 | 66.7588 | 24.6717 |
| wall | 57.7209 | 57.7128 | -0.0081 | 68.4307 | 78.6543 | 2.2406 |
| road | 25.5594 | 25.5593 | -0.0001 | 25.7547 | 97.1164 | 5.2011 |
| vegetation | 60.0299 | 60.0306 | +0.0007 | 96.8773 | 61.2152 | 13.6483 |
| vehicle | 25.3443 | 25.3444 | +0.0001 | 25.5376 | 97.1009 | 0.8983 |
| roof | 84.4472 | 84.4464 | -0.0008 | 84.8273 | 99.4710 | 32.6278 |
| water | 92.4595 | 92.4595 | +0.0000 | 94.6528 | 97.5551 | 20.7123 |

non_residual_mean_iou_percent: {"Geometry": 33.0381, "BroadVIP": 51.5822, "Anchored_VIP": 57.5935, "ExcessReject_Coupled": 57.5921, "TextOnlyReject_Coupled": 57.7519, "HardDelete_Coupled": 57.5033, "ShuffledReject0_Coupled": 57.593, "ShuffledReject1_Coupled": 57.5933, "ShuffledReject2_Coupled": 57.593}

Diagnostics:
```json
{
  "tiles": 88.0,
  "supported_patch_fraction": 0.9997170188210226,
  "effective_support": 73.62826486609198,
  "mean_rejection": 4.379098424708771e-05,
  "rejected_alias_fraction": 0.005020805457969764,
  "positive_gain_fraction": 0.10779511092545148,
  "mean_score_suppression": 1.9800959891981876e-05,
  "shuffle_spectrum_error": 0.0
}
```

## vdd/vdd/wrong_parent Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| other | 46.5267 | 46.5287 | +0.0020 | 67.8767 | 59.6676 | 23.6600 |
| wall | 48.4329 | 48.4293 | -0.0036 | 58.0884 | 74.4407 | 2.4981 |
| road | 21.9441 | 21.9441 | +0.0000 | 22.0371 | 98.1129 | 6.1409 |
| vegetation | 51.2580 | 51.2628 | +0.0048 | 97.5672 | 51.9265 | 11.4955 |
| vehicle | 22.7869 | 22.7899 | +0.0030 | 22.8836 | 98.2358 | 1.0142 |
| roof | 81.7931 | 81.7923 | -0.0008 | 82.2131 | 99.3781 | 33.6339 |
| water | 89.3061 | 89.3164 | +0.0103 | 91.1584 | 97.7878 | 21.5576 |

non_residual_mean_iou_percent: {"Geometry": 33.0381, "BroadVIP": 46.0923, "Anchored_VIP": 52.5869, "ExcessReject_Coupled": 52.5891, "TextOnlyReject_Coupled": 53.4132, "HardDelete_Coupled": 52.5483, "ShuffledReject0_Coupled": 52.5867, "ShuffledReject1_Coupled": 52.5884, "ShuffledReject2_Coupled": 52.5868}

Diagnostics:
```json
{
  "tiles": 88.0,
  "supported_patch_fraction": 0.9997170188210226,
  "effective_support": 73.62826486609198,
  "mean_rejection": 0.0005579607982421198,
  "rejected_alias_fraction": 0.019494292023894074,
  "positive_gain_fraction": 0.10726020119406961,
  "mean_score_suppression": 0.0013190306663929839,
  "shuffle_spectrum_error": 0.0
}
```

## vdd/vdd/paraphrase Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| other | 58.0258 | 58.0258 | +0.0000 | 77.8267 | 69.5186 | 24.0419 |
| wall | 57.0772 | 57.0558 | -0.0214 | 61.6128 | 88.5245 | 2.8008 |
| road | 27.1252 | 27.1251 | -0.0001 | 27.5393 | 94.7472 | 4.7454 |
| vegetation | 66.5223 | 66.5226 | +0.0003 | 95.5274 | 68.6611 | 15.5247 |
| vehicle | 22.7339 | 22.7333 | -0.0006 | 22.8610 | 97.6013 | 1.0086 |
| roof | 87.2315 | 87.2302 | -0.0013 | 88.0966 | 98.8852 | 31.2320 |
| water | 92.5129 | 92.5129 | +0.0000 | 94.8295 | 97.4273 | 20.6466 |

non_residual_mean_iou_percent: {"Geometry": 33.0381, "BroadVIP": 54.6448, "Anchored_VIP": 58.8672, "ExcessReject_Coupled": 58.8633, "TextOnlyReject_Coupled": 59.1704, "HardDelete_Coupled": 58.9544, "ShuffledReject0_Coupled": 58.8656, "ShuffledReject1_Coupled": 58.8666, "ShuffledReject2_Coupled": 58.867}

Diagnostics:
```json
{
  "tiles": 88.0,
  "supported_patch_fraction": 0.9997170188210226,
  "effective_support": 73.62826486609198,
  "mean_rejection": 4.7734503634462985e-05,
  "rejected_alias_fraction": 0.0050384026069145715,
  "positive_gain_fraction": 0.10725725842760754,
  "mean_score_suppression": 3.6757449033082575e-05,
  "shuffle_spectrum_error": 0.0
}
```

## potsdam/potsdam/clean Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| impervious surface | 62.1577 | 62.1353 | -0.0224 | 73.6649 | 79.8790 | 39.2651 |
| building | 75.5787 | 75.5390 | -0.0397 | 76.9803 | 97.5814 | 15.4683 |
| low vegetation | 21.2306 | 21.2450 | +0.0144 | 82.1313 | 22.2746 | 5.2023 |
| tree | 64.8712 | 64.8853 | +0.0141 | 92.4909 | 68.4935 | 17.7356 |
| car | 24.2128 | 24.1896 | -0.0232 | 24.2978 | 98.1928 | 9.4673 |
| clutter | 2.6325 | 2.6263 | -0.0062 | 3.7753 | 7.9434 | 12.8614 |

Diagnostics:
```json
{
  "tiles": 9.0,
  "supported_patch_fraction": 0.9995252821180557,
  "effective_support": 51.81056420008341,
  "mean_rejection": 0.0004426559001229988,
  "rejected_alias_fraction": 0.01948649088541667,
  "positive_gain_fraction": 0.09708884910300926,
  "mean_score_suppression": 0.00032339954216006374,
  "shuffle_spectrum_error": 0.0
}
```

## potsdam/potsdam/wrong_parent Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| impervious surface | 67.7099 | 67.7017 | -0.0082 | 76.1083 | 85.9734 | 40.9041 |
| building | 79.6578 | 79.6004 | -0.0574 | 82.0977 | 96.3191 | 14.3165 |
| low vegetation | 17.7584 | 17.7727 | +0.0143 | 52.5337 | 21.1726 | 7.7309 |
| tree | 48.3774 | 48.4812 | +0.1038 | 95.2480 | 49.6829 | 12.4924 |
| car | 31.3451 | 31.3304 | -0.0147 | 31.5249 | 98.0690 | 7.2877 |
| clutter | 2.3346 | 2.3381 | +0.0035 | 3.0934 | 8.7389 | 17.2684 |

Diagnostics:
```json
{
  "tiles": 9.0,
  "supported_patch_fraction": 0.9995252821180557,
  "effective_support": 51.81056420008341,
  "mean_rejection": 0.0007151931752189367,
  "rejected_alias_fraction": 0.03019431785300926,
  "positive_gain_fraction": 0.09243650083188656,
  "mean_score_suppression": 0.0006626622329484271,
  "shuffle_spectrum_error": 0.0
}
```

## potsdam/potsdam/paraphrase Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| impervious surface | 60.3011 | 60.2800 | -0.0211 | 73.7628 | 76.7325 | 37.6683 |
| building | 75.1521 | 75.1130 | -0.0391 | 76.5496 | 97.5624 | 15.5523 |
| low vegetation | 31.9766 | 31.9953 | +0.0187 | 75.7949 | 35.6365 | 9.0187 |
| tree | 66.0613 | 66.0715 | +0.0102 | 92.0051 | 70.0960 | 18.2464 |
| car | 21.3632 | 21.3453 | -0.0179 | 21.4155 | 98.4878 | 10.7738 |
| clutter | 3.2624 | 3.2581 | -0.0043 | 5.3620 | 7.6669 | 8.7404 |

Diagnostics:
```json
{
  "tiles": 9.0,
  "supported_patch_fraction": 0.9995252821180557,
  "effective_support": 51.81056420008341,
  "mean_rejection": 0.00044175780833568385,
  "rejected_alias_fraction": 0.018941130461516204,
  "positive_gain_fraction": 0.09328375922309026,
  "mean_score_suppression": 0.0002751156858403647,
  "shuffle_spectrum_error": 0.0
}
```

## udd5/udd5/clean Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vegetation | 68.3846 | 68.3852 | +0.0006 | 97.4279 | 69.6425 | 21.1719 |
| building | 82.5264 | 82.5256 | -0.0008 | 83.6772 | 98.3597 | 46.1471 |
| road | 39.8083 | 39.8234 | +0.0151 | 66.8700 | 49.6119 | 9.9492 |
| vehicle | 20.4213 | 20.4227 | +0.0014 | 20.9506 | 89.0166 | 3.4012 |
| other | 34.9177 | 34.9148 | -0.0029 | 48.5202 | 55.4595 | 19.3305 |

non_residual_mean_iou_percent: {"Geometry": 55.548, "BroadVIP": 48.2785, "Anchored_VIP": 52.7852, "ExcessReject_Coupled": 52.7892, "TextOnlyReject_Coupled": 54.6831, "HardDelete_Coupled": 52.9253, "ShuffledReject0_Coupled": 52.7864, "ShuffledReject1_Coupled": 52.7904, "ShuffledReject2_Coupled": 52.7861}

Diagnostics:
```json
{
  "tiles": 80.8,
  "supported_patch_fraction": 0.9996760512843273,
  "effective_support": 70.09110437472661,
  "mean_rejection": 0.00018122389459138946,
  "rejected_alias_fraction": 0.013111646247632575,
  "positive_gain_fraction": 0.16759005274917144,
  "mean_score_suppression": 0.00016381348905375424,
  "shuffle_spectrum_error": 0.0
}
```

## udd5/udd5/wrong_parent Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vegetation | 58.9800 | 58.9989 | +0.0189 | 94.4961 | 61.0985 | 6.8854 |
| building | 88.0831 | 88.0532 | -0.0299 | 89.2143 | 98.5434 | 58.8244 |
| road | 44.7522 | 44.7393 | -0.0129 | 65.7814 | 58.3095 | 11.8361 |
| vehicle | 14.5226 | 14.6258 | +0.1032 | 14.6844 | 97.3458 | 17.4331 |
| other | 14.8973 | 14.9843 | +0.0870 | 65.2334 | 16.2848 | 5.0210 |

non_residual_mean_iou_percent: {"Geometry": 53.2477, "BroadVIP": 44.862, "Anchored_VIP": 51.5845, "ExcessReject_Coupled": 51.6043, "TextOnlyReject_Coupled": 51.7703, "HardDelete_Coupled": 51.9583, "ShuffledReject0_Coupled": 51.5903, "ShuffledReject1_Coupled": 51.5908, "ShuffledReject2_Coupled": 51.5842}

Diagnostics:
```json
{
  "tiles": 66.0,
  "supported_patch_fraction": 0.9997910008285984,
  "effective_support": 53.49134463613683,
  "mean_rejection": 0.002446916537458795,
  "rejected_alias_fraction": 0.045906982421875006,
  "positive_gain_fraction": 0.2556836862275094,
  "mean_score_suppression": 0.006451166357989065,
  "shuffle_spectrum_error": 0.0
}
```

## udd5/udd5/paraphrase Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vegetation | 61.7859 | 61.7861 | +0.0002 | 87.8958 | 67.5321 | 8.1819 |
| building | 86.5745 | 86.5750 | +0.0005 | 87.6450 | 98.6095 | 59.9178 |
| road | 38.1524 | 38.1535 | +0.0011 | 72.3742 | 44.6572 | 8.2391 |
| vehicle | 15.9054 | 15.9040 | -0.0014 | 16.0281 | 95.3567 | 15.6452 |
| other | 24.2467 | 24.2388 | -0.0079 | 68.4621 | 27.2855 | 8.0159 |

non_residual_mean_iou_percent: {"Geometry": 53.2477, "BroadVIP": 46.4877, "Anchored_VIP": 50.6046, "ExcessReject_Coupled": 50.6047, "TextOnlyReject_Coupled": 50.0757, "HardDelete_Coupled": 50.1951, "ShuffledReject0_Coupled": 50.6061, "ShuffledReject1_Coupled": 50.6109, "ShuffledReject2_Coupled": 50.6055}

Diagnostics:
```json
{
  "tiles": 66.0,
  "supported_patch_fraction": 0.9997910008285984,
  "effective_support": 53.49134463613683,
  "mean_rejection": 0.00025762188765039544,
  "rejected_alias_fraction": 0.02163222804214015,
  "positive_gain_fraction": 0.2578265195904356,
  "mean_score_suppression": 0.00014473345388421745,
  "shuffle_spectrum_error": 0.0
}
```

## oem/oem/clean Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bareland | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | 11.6258 |
| rangeland | 37.5959 | 37.5955 | -0.0004 | 60.8674 | 49.5792 | 15.5953 |
| developed space | 24.0531 | 24.1117 | +0.0586 | 36.5590 | 41.4584 | 25.4010 |
| road | 42.1205 | 42.1215 | +0.0010 | 53.7347 | 66.0900 | 5.6120 |
| tree | 38.7562 | 38.7578 | +0.0016 | 92.7490 | 39.9689 | 10.9586 |
| water | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.0000 |
| agriculture land | 68.8612 | 68.8581 | -0.0031 | 70.3760 | 96.9628 | 23.0907 |
| building | 44.2393 | 44.2394 | +0.0001 | 77.0440 | 50.9563 | 7.7167 |

Diagnostics:
```json
{
  "tiles": 8.375,
  "supported_patch_fraction": 0.9997422960069444,
  "effective_support": 70.06865766313342,
  "mean_rejection": 0.0007127657803999278,
  "rejected_alias_fraction": 0.02487252553304037,
  "positive_gain_fraction": 0.1277941385904948,
  "mean_score_suppression": 0.0003167764828263407,
  "shuffle_spectrum_error": 0.0
}
```

## oem/oem/wrong_parent Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bareland | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | 10.3822 |
| rangeland | 37.6962 | 37.6949 | -0.0013 | 61.0117 | 49.6561 | 15.5825 |
| developed space | 27.0193 | 27.0917 | +0.0724 | 40.4948 | 45.0101 | 24.8968 |
| road | 42.4163 | 42.4167 | +0.0004 | 56.4442 | 63.0556 | 5.0973 |
| tree | 36.0699 | 36.0727 | +0.0028 | 93.5916 | 36.9863 | 10.0495 |
| water | 0.0591 | 0.0497 | -0.0094 | 0.0752 | 0.1466 | 0.0693 |
| agriculture land | 67.9182 | 67.9202 | +0.0020 | 69.4122 | 96.9324 | 23.4040 |
| building | 50.9694 | 50.9745 | +0.0051 | 71.2155 | 64.2023 | 10.5184 |

Diagnostics:
```json
{
  "tiles": 8.375,
  "supported_patch_fraction": 0.9997422960069444,
  "effective_support": 70.06865766313342,
  "mean_rejection": 0.0011331190096219264,
  "rejected_alias_fraction": 0.03857288360595703,
  "positive_gain_fraction": 0.12421351538764105,
  "mean_score_suppression": 0.0016292489089693165,
  "shuffle_spectrum_error": 0.0
}
```

## oem/oem/paraphrase Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bareland | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | 13.3856 |
| rangeland | 40.9611 | 40.9637 | +0.0026 | 62.2619 | 54.4940 | 16.7573 |
| developed space | 20.9277 | 20.9879 | +0.0602 | 35.4140 | 34.0031 | 21.5068 |
| road | 42.4323 | 42.4331 | +0.0008 | 54.9414 | 65.0817 | 5.4050 |
| tree | 40.4105 | 40.4113 | +0.0008 | 92.6725 | 41.7451 | 11.4550 |
| water | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.0001 |
| agriculture land | 71.2025 | 71.2000 | -0.0025 | 72.8997 | 96.8291 | 22.2606 |
| building | 51.6630 | 51.6624 | -0.0006 | 77.1255 | 61.0108 | 9.2295 |

Diagnostics:
```json
{
  "tiles": 8.375,
  "supported_patch_fraction": 0.9997422960069444,
  "effective_support": 70.06865766313342,
  "mean_rejection": 0.0006842114428637716,
  "rejected_alias_fraction": 0.024046346876356334,
  "positive_gain_fraction": 0.12289892832438151,
  "mean_score_suppression": 0.0002537808928772348,
  "shuffle_spectrum_error": 0.0
}
```

## loveda/P/clean Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| building | 79.5953 | 79.5895 | -0.0058 | 96.0563 | 82.2781 | 0.5600 |
| road | 68.0988 | 68.0986 | -0.0002 | 69.1301 | 97.8560 | 10.0794 |
| water | 79.3816 | 79.3816 | +0.0000 | 85.6142 | 91.5997 | 22.3534 |
| barren | 23.8204 | 23.8198 | -0.0006 | 74.5717 | 25.9256 | 2.5412 |
| tree | 36.7931 | 36.7929 | -0.0002 | 99.0340 | 36.9255 | 4.1267 |
| farm | 87.2288 | 87.2288 | +0.0000 | 87.4778 | 99.6747 | 60.3394 |

Diagnostics:
```json
{
  "tiles": 9.0,
  "supported_patch_fraction": 0.9999864366319444,
  "effective_support": 71.99516412946912,
  "mean_rejection": 4.2141993934045015e-05,
  "rejected_alias_fraction": 0.0048698142722800935,
  "positive_gain_fraction": 0.07970422815393517,
  "mean_score_suppression": 2.764377821674655e-05,
  "shuffle_spectrum_error": 0.0
}
```

## loveda/D/clean Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| background | 6.4379 | 6.4375 | -0.0004 | 78.1638 | 6.5554 | 3.7435 |
| building | 36.3815 | 36.3802 | -0.0013 | 39.8747 | 80.5874 | 0.7315 |
| road | 56.4727 | 56.4722 | -0.0005 | 57.1819 | 97.8495 | 6.7459 |
| water | 62.7476 | 62.7479 | +0.0003 | 66.6177 | 91.5269 | 15.8921 |
| barren | 20.0167 | 20.0157 | -0.0010 | 48.3372 | 25.4630 | 2.1317 |
| tree | 31.2898 | 31.2907 | +0.0009 | 87.2022 | 32.7968 | 2.3046 |
| farm | 42.5737 | 42.5735 | -0.0002 | 42.6506 | 99.5774 | 68.4506 |

foreground_mean_iou_percent: {"Geometry": 39.1654, "BroadVIP": 38.5206, "Anchored_VIP": 41.5803, "ExcessReject_Coupled": 41.5801, "TextOnlyReject_Coupled": 41.3009, "HardDelete_Coupled": 41.5839, "ShuffledReject0_Coupled": 41.5806, "ShuffledReject1_Coupled": 41.5805, "ShuffledReject2_Coupled": 41.5803}

Diagnostics:
```json
{
  "tiles": 9.0,
  "supported_patch_fraction": 0.9999864366319444,
  "effective_support": 71.99516412946912,
  "mean_rejection": 7.021143117324553e-05,
  "rejected_alias_fraction": 0.006317623077876984,
  "positive_gain_fraction": 0.07711646670386903,
  "mean_score_suppression": 4.9539040904592526e-05,
  "shuffle_spectrum_error": 0.0
}
```

## loveda/P/wrong_parent Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| building | 77.7277 | 77.7351 | +0.0074 | 93.9163 | 81.8571 | 0.5698 |
| road | 68.2445 | 68.2455 | +0.0010 | 69.3823 | 97.6554 | 10.0221 |
| water | 78.5576 | 78.5578 | +0.0002 | 83.8911 | 92.5133 | 23.0401 |
| barren | 22.4151 | 22.4133 | -0.0018 | 77.3448 | 23.9882 | 2.2670 |
| tree | 48.1733 | 48.1740 | +0.0007 | 97.9583 | 48.6626 | 5.4981 |
| farm | 89.6949 | 89.6942 | -0.0007 | 90.0108 | 99.6094 | 58.6029 |

Diagnostics:
```json
{
  "tiles": 9.0,
  "supported_patch_fraction": 0.9999864366319444,
  "effective_support": 71.99516412946912,
  "mean_rejection": 0.0002496376262368156,
  "rejected_alias_fraction": 0.014233850549768518,
  "positive_gain_fraction": 0.07607184516059029,
  "mean_score_suppression": 0.00043252938532489793,
  "shuffle_spectrum_error": 0.0
}
```

## loveda/D/wrong_parent Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| background | 2.6641 | 2.6635 | -0.0006 | 69.6973 | 2.6947 | 1.7257 |
| building | 35.0162 | 35.0105 | -0.0057 | 39.3437 | 76.0700 | 0.6998 |
| road | 56.4705 | 56.4701 | -0.0004 | 57.2419 | 97.6682 | 6.7264 |
| water | 60.6151 | 60.6153 | +0.0002 | 63.7683 | 92.4581 | 16.7712 |
| barren | 18.8353 | 18.8341 | -0.0012 | 52.3320 | 22.7342 | 1.7580 |
| tree | 31.4643 | 31.4681 | +0.0038 | 51.5937 | 44.6508 | 5.3030 |
| farm | 43.4947 | 43.4942 | -0.0005 | 43.5713 | 99.5948 | 67.0159 |

foreground_mean_iou_percent: {"Geometry": 39.1654, "BroadVIP": 37.2943, "Anchored_VIP": 40.9827, "ExcessReject_Coupled": 40.9821, "TextOnlyReject_Coupled": 40.5816, "HardDelete_Coupled": 40.9508, "ShuffledReject0_Coupled": 40.9843, "ShuffledReject1_Coupled": 40.9842, "ShuffledReject2_Coupled": 40.9825}

Diagnostics:
```json
{
  "tiles": 9.0,
  "supported_patch_fraction": 0.9999864366319444,
  "effective_support": 71.99516412946912,
  "mean_rejection": 0.0002571297825939144,
  "rejected_alias_fraction": 0.014375426277281745,
  "positive_gain_fraction": 0.07417486281622024,
  "mean_score_suppression": 0.00037870298715544687,
  "shuffle_spectrum_error": 0.0
}
```

## loveda/P/paraphrase Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| building | 80.9933 | 80.9933 | +0.0000 | 94.9067 | 84.6737 | 0.5833 |
| road | 65.9572 | 65.9572 | +0.0000 | 66.8349 | 98.0477 | 10.4459 |
| water | 79.4922 | 79.4924 | +0.0002 | 86.1416 | 91.1492 | 22.1073 |
| barren | 23.4773 | 23.4773 | +0.0000 | 75.2906 | 25.4373 | 2.4695 |
| tree | 35.7407 | 35.7407 | +0.0000 | 99.1841 | 35.8461 | 4.0000 |
| farm | 87.1503 | 87.1501 | -0.0002 | 87.3987 | 99.6748 | 60.3940 |

Diagnostics:
```json
{
  "tiles": 9.0,
  "supported_patch_fraction": 0.9999864366319444,
  "effective_support": 71.99516412946912,
  "mean_rejection": 4.2105399646701424e-05,
  "rejected_alias_fraction": 0.00509926124855324,
  "positive_gain_fraction": 0.076580810546875,
  "mean_score_suppression": 2.29083903164494e-05,
  "shuffle_spectrum_error": 0.0
}
```

## loveda/D/paraphrase Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| background | 4.8635 | 4.8630 | -0.0005 | 78.4845 | 4.9287 | 2.8031 |
| building | 35.8849 | 35.8877 | +0.0028 | 38.5023 | 84.0890 | 0.7905 |
| road | 52.2186 | 52.2184 | -0.0002 | 52.7682 | 98.0437 | 7.3247 |
| water | 63.7886 | 63.7888 | +0.0002 | 68.0359 | 91.0862 | 15.4860 |
| barren | 19.5208 | 19.5211 | +0.0003 | 46.7764 | 25.0951 | 2.1711 |
| tree | 31.6801 | 31.6813 | +0.0012 | 85.9097 | 33.4177 | 2.3835 |
| farm | 42.2245 | 42.2243 | -0.0002 | 42.2958 | 99.6010 | 69.0412 |

foreground_mean_iou_percent: {"Geometry": 39.1654, "BroadVIP": 37.7805, "Anchored_VIP": 40.8862, "ExcessReject_Coupled": 40.8869, "TextOnlyReject_Coupled": 40.4504, "HardDelete_Coupled": 40.909, "ShuffledReject0_Coupled": 40.8868, "ShuffledReject1_Coupled": 40.8867, "ShuffledReject2_Coupled": 40.8865}

Diagnostics:
```json
{
  "tiles": 9.0,
  "supported_patch_fraction": 0.9999864366319444,
  "effective_support": 71.99516412946912,
  "mean_rejection": 6.761541732148978e-05,
  "rejected_alias_fraction": 0.006529405381944445,
  "positive_gain_fraction": 0.07406180245535715,
  "mean_score_suppression": 4.238739725399402e-05,
  "shuffle_spectrum_error": 0.0
}
```

## vaihingen/vaihingen/clean Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| impervious surface | 60.0516 | 60.0366 | -0.0150 | 74.5935 | 75.4688 | 30.9751 |
| building | 68.2387 | 68.2292 | -0.0095 | 68.4565 | 99.5158 | 31.3323 |
| low vegetation | 35.3912 | 35.3922 | +0.0010 | 96.4486 | 35.8595 | 9.4626 |
| tree | 71.8529 | 71.8531 | +0.0002 | 81.8849 | 85.4334 | 21.8036 |
| car | 21.7111 | 21.6919 | -0.0192 | 21.9356 | 95.1276 | 6.4263 |

Diagnostics:
```json
{
  "tiles": 9.0,
  "supported_patch_fraction": 0.9994981553819444,
  "effective_support": 50.40161763297187,
  "mean_rejection": 0.00030039247509396067,
  "rejected_alias_fraction": 0.01560791015625,
  "positive_gain_fraction": 0.09381781684027779,
  "mean_score_suppression": 0.00013918441593148144,
  "shuffle_spectrum_error": 0.0
}
```

## vaihingen/vaihingen/wrong_parent Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| impervious surface | 61.6065 | 61.5623 | -0.0442 | 74.8401 | 77.6284 | 31.7566 |
| building | 73.7301 | 73.6892 | -0.0409 | 74.3531 | 98.8027 | 28.6408 |
| low vegetation | 40.7081 | 40.7017 | -0.0064 | 86.1974 | 43.5394 | 12.8556 |
| tree | 69.9116 | 69.9166 | +0.0050 | 87.5869 | 77.6065 | 18.5167 |
| car | 17.4810 | 17.4707 | -0.0103 | 17.5501 | 97.4755 | 8.2304 |

Diagnostics:
```json
{
  "tiles": 9.0,
  "supported_patch_fraction": 0.9994981553819444,
  "effective_support": 50.40161763297187,
  "mean_rejection": 0.0005113679007950446,
  "rejected_alias_fraction": 0.02473809136284722,
  "positive_gain_fraction": 0.08887193467881943,
  "mean_score_suppression": 0.0004279530619239975,
  "shuffle_spectrum_error": 0.0
}
```

## vaihingen/vaihingen/paraphrase Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| impervious surface | 58.4498 | 58.4365 | -0.0133 | 77.0580 | 70.7447 | 28.1076 |
| building | 68.8035 | 68.7956 | -0.0079 | 69.0628 | 99.4407 | 31.0338 |
| low vegetation | 39.9640 | 39.9667 | +0.0027 | 93.9468 | 41.0231 | 11.1135 |
| tree | 71.7391 | 71.7388 | -0.0003 | 81.4971 | 85.6965 | 21.9748 |
| car | 18.1186 | 18.1047 | -0.0139 | 18.2527 | 95.7118 | 7.7704 |

Diagnostics:
```json
{
  "tiles": 9.0,
  "supported_patch_fraction": 0.9994981553819444,
  "effective_support": 50.40161763297187,
  "mean_rejection": 0.0003034446513710362,
  "rejected_alias_fraction": 0.015073242187499999,
  "positive_gain_fraction": 0.0891094292534722,
  "mean_score_suppression": 0.00014087585409501774,
  "shuffle_spectrum_error": 0.0
}
```

## landcoverai/landcoverai/clean Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| background | 87.4477 | 87.4535 | +0.0058 | 93.8645 | 92.7559 | 66.9597 |
| building | 44.4672 | 44.4669 | -0.0003 | 44.8108 | 98.3036 | 3.3237 |
| woodland | 78.6026 | 78.6281 | +0.0255 | 94.9510 | 82.0590 | 18.5003 |
| water | 97.6178 | 97.6173 | -0.0005 | 97.6173 | 100.0000 | 8.4732 |
| road | 26.3947 | 26.3926 | -0.0021 | 28.8503 | 75.5990 | 2.7431 |

foreground_mean_iou_percent: {"Geometry": 55.4788, "BroadVIP": 60.8275, "Anchored_VIP": 61.7706, "ExcessReject_Coupled": 61.7762, "TextOnlyReject_Coupled": 60.4074, "HardDelete_Coupled": 61.8907, "ShuffledReject0_Coupled": 61.774, "ShuffledReject1_Coupled": 61.7746, "ShuffledReject2_Coupled": 61.7748}

Diagnostics:
```json
{
  "tiles": 1.0,
  "supported_patch_fraction": 0.999755859375,
  "effective_support": 82.70246696472168,
  "mean_rejection": 7.828264875570312e-05,
  "rejected_alias_fraction": 0.0046057128906250005,
  "positive_gain_fraction": 0.070223388671875,
  "mean_score_suppression": 5.526358543264109e-05,
  "shuffle_spectrum_error": 0.0
}
```

## landcoverai/landcoverai/wrong_parent Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| background | 88.0128 | 88.0163 | +0.0035 | 94.2396 | 93.0209 | 66.8838 |
| building | 50.7657 | 50.7483 | -0.0174 | 51.2274 | 98.1903 | 2.9040 |
| woodland | 79.5889 | 79.6113 | +0.0224 | 94.8533 | 83.2055 | 18.7781 |
| water | 97.5136 | 97.5125 | -0.0011 | 97.5125 | 100.0000 | 8.4823 |
| road | 24.7367 | 24.7345 | -0.0022 | 26.8621 | 75.7447 | 2.9519 |

foreground_mean_iou_percent: {"Geometry": 55.4788, "BroadVIP": 58.7178, "Anchored_VIP": 63.1512, "ExcessReject_Coupled": 63.1516, "TextOnlyReject_Coupled": 61.7169, "HardDelete_Coupled": 63.1844, "ShuffledReject0_Coupled": 63.1526, "ShuffledReject1_Coupled": 63.1522, "ShuffledReject2_Coupled": 63.1511}

Diagnostics:
```json
{
  "tiles": 1.0,
  "supported_patch_fraction": 0.999755859375,
  "effective_support": 82.70246696472168,
  "mean_rejection": 0.00022145092654682233,
  "rejected_alias_fraction": 0.011505126953125,
  "positive_gain_fraction": 0.06895263671874999,
  "mean_score_suppression": 0.0003275301126564045,
  "shuffle_spectrum_error": 0.0
}
```

## landcoverai/landcoverai/paraphrase Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| background | 86.7775 | 86.7814 | +0.0039 | 94.4653 | 91.4301 | 65.5828 |
| building | 39.1418 | 39.1393 | -0.0025 | 39.3062 | 98.9268 | 3.8131 |
| woodland | 79.8202 | 79.8391 | +0.0189 | 93.9385 | 84.1756 | 19.1820 |
| water | 97.4402 | 97.4396 | -0.0006 | 97.4396 | 100.0000 | 8.4887 |
| road | 24.9682 | 24.9656 | -0.0026 | 27.1075 | 75.9588 | 2.9334 |

foreground_mean_iou_percent: {"Geometry": 55.4788, "BroadVIP": 58.8765, "Anchored_VIP": 60.3426, "ExcessReject_Coupled": 60.3459, "TextOnlyReject_Coupled": 57.1293, "HardDelete_Coupled": 60.3844, "ShuffledReject0_Coupled": 60.3455, "ShuffledReject1_Coupled": 60.3452, "ShuffledReject2_Coupled": 60.3453}

Diagnostics:
```json
{
  "tiles": 1.0,
  "supported_patch_fraction": 0.999755859375,
  "effective_support": 82.70246696472168,
  "mean_rejection": 7.954963228257839e-05,
  "rejected_alias_fraction": 0.0046435546875,
  "positive_gain_fraction": 0.067437744140625,
  "mean_score_suppression": 4.9442395417997884e-05,
  "shuffle_spectrum_error": 0.0
}
```

## flair1/flair1/clean Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| building | 56.7755 | 56.7569 | -0.0186 | 58.0160 | 96.3170 | 11.9193 |
| pervious surface | 47.2756 | 47.2717 | -0.0039 | 91.6988 | 49.3851 | 9.2415 |
| impervious surface | 51.9858 | 51.9724 | -0.0134 | 59.7740 | 79.9277 | 21.9060 |
| bare soil | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | 7.1489 |
| water | 61.4565 | 61.4588 | +0.0023 | 63.2508 | 95.5932 | 6.7043 |
| coniferous | 43.5282 | 43.5262 | -0.0020 | 65.7160 | 56.3136 | 0.4817 |
| deciduous | 59.5237 | 59.5304 | +0.0067 | 78.7961 | 70.8859 | 15.3129 |
| brushwood | 12.3287 | 12.3331 | +0.0044 | 26.2561 | 18.8693 | 3.5347 |
| vineyard | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.0016 |
| herbaceous vegetation | 57.4725 | 57.4673 | -0.0052 | 90.8263 | 61.0084 | 21.5181 |
| agricultural land | 12.9546 | 12.9485 | -0.0061 | 13.9067 | 65.2691 | 1.4311 |
| plowed land | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.7998 |

Diagnostics:
```json
{
  "tiles": 1.0,
  "supported_patch_fraction": 0.9998779296875,
  "effective_support": 60.644423484802246,
  "mean_rejection": 0.0002969221889240241,
  "rejected_alias_fraction": 0.027691141764322916,
  "positive_gain_fraction": 0.10399576822916666,
  "mean_score_suppression": 0.00021858282696977463,
  "shuffle_spectrum_error": 0.0
}
```

## flair1/flair1/wrong_parent Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| building | 52.0630 | 52.0282 | -0.0348 | 52.3552 | 98.8140 | 13.5505 |
| pervious surface | 47.5920 | 47.5919 | -0.0001 | 87.2482 | 51.1498 | 10.0600 |
| impervious surface | 52.0153 | 52.0015 | -0.0138 | 61.8760 | 76.5177 | 20.2590 |
| bare soil | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | 5.9158 |
| water | 65.6557 | 65.6565 | +0.0008 | 68.4278 | 94.1898 | 6.1061 |
| coniferous | 34.5846 | 34.5583 | -0.0263 | 46.1346 | 57.9345 | 0.7059 |
| deciduous | 53.4475 | 53.4590 | +0.0115 | 80.6398 | 61.3305 | 12.9458 |
| brushwood | 17.9649 | 17.9693 | +0.0044 | 28.2003 | 33.1238 | 5.7771 |
| vineyard | -- | -- | -- | 0.0000 | 0.0000 | 0.0000 |
| herbaceous vegetation | 61.2727 | 61.2765 | +0.0038 | 89.7667 | 65.8784 | 23.5101 |
| agricultural land | 14.3668 | 14.3566 | -0.0102 | 15.8928 | 59.7622 | 1.1466 |
| plowed land | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.0231 |

Diagnostics:
```json
{
  "tiles": 1.0,
  "supported_patch_fraction": 0.9998779296875,
  "effective_support": 60.644423484802246,
  "mean_rejection": 0.0003783368946036111,
  "rejected_alias_fraction": 0.03399200439453125,
  "positive_gain_fraction": 0.10134531656901043,
  "mean_score_suppression": 0.0003753103616001378,
  "shuffle_spectrum_error": 0.0
}
```

## flair1/flair1/paraphrase Classes

| Class | Original coupling IoU | Primary IoU | Delta pp | Primary precision % | Primary recall % | Primary area % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| building | 55.1554 | 55.1359 | -0.0195 | 56.7325 | 95.1437 | 12.0405 |
| pervious surface | 49.3192 | 49.3120 | -0.0072 | 94.5238 | 50.7623 | 9.2153 |
| impervious surface | 52.3579 | 52.3468 | -0.0111 | 60.2865 | 79.8983 | 21.7118 |
| bare soil | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | 6.8685 |
| water | 65.3852 | 65.3857 | +0.0005 | 67.6432 | 95.1437 | 6.2395 |
| coniferous | 43.9217 | 43.9159 | -0.0058 | 66.2558 | 56.5682 | 0.4799 |
| deciduous | 61.2313 | 61.2363 | +0.0050 | 78.1461 | 73.8899 | 16.0946 |
| brushwood | 14.7571 | 14.7605 | +0.0034 | 31.6680 | 21.6587 | 3.3638 |
| vineyard | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.0044 |
| herbaceous vegetation | 58.9987 | 58.9962 | -0.0025 | 90.5465 | 62.8685 | 22.2428 |
| agricultural land | 13.2352 | 13.2345 | -0.0007 | 14.2650 | 64.6902 | 1.3828 |
| plowed land | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.3561 |

Diagnostics:
```json
{
  "tiles": 1.0,
  "supported_patch_fraction": 0.9998779296875,
  "effective_support": 60.644423484802246,
  "mean_rejection": 0.00028499992837775306,
  "rejected_alias_fraction": 0.027090454101562498,
  "positive_gain_fraction": 0.10129140218098959,
  "mean_score_suppression": 0.00018768350753362029,
  "shuffle_spectrum_error": 0.0
}
```

## Shared Cost

Suite wall seconds: 1176.5485

| Dataset | Worker seconds | Peak allocated MiB |
| --- | ---: | ---: |
| vdd | 546.3533 | 6187.8120 |
| potsdam | 57.8683 | 5571.1118 |
| udd5 | 1070.5284 | 5702.7944 |
| oem | 64.3604 | 5601.8433 |
| loveda | 116.6896 | 5662.6450 |
| vaihingen | 58.3789 | 5558.6519 |
| landcoverai | 8.7699 | 5458.1870 |
| flair1 | 10.5141 | 5547.7969 |

Shared multi-arm/multi-vocabulary cost, not standalone primary latency. No automatic full rollout.

## Mask-Free Source Decomposition

Eight fixed top-left512 windows each for VDD, Potsdam and OEM. Same frozen rule and config, no masks loaded or parameters changed. This decomposes activity, not semantic correctness or full-image performance.

| Dataset/protocol | Scenario | Text conflict fraction | Text/Geometry positive fraction | Gain positive fraction | Mean joint before gain | Mean risk after gain | Responsibility gain survival |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | clean | 0.085714 | 0.041023 | 0.113348 | 0.001069 | 0.000049 | 0.062363 |
| vdd/vdd | wrong_parent | 0.264286 | 0.180451 | 0.113121 | 0.016635 | 0.000666 | 0.047082 |
| vdd/vdd | paraphrase | 0.100000 | 0.041836 | 0.112755 | 0.001181 | 0.000049 | 0.050953 |
| potsdam/potsdam | clean | 0.283333 | 0.219053 | 0.106077 | 0.024122 | 0.000482 | 0.028648 |
| potsdam/potsdam | wrong_parent | 0.433333 | 0.343260 | 0.102185 | 0.038363 | 0.000776 | 0.024987 |
| potsdam/potsdam | paraphrase | 0.283333 | 0.225340 | 0.102082 | 0.025039 | 0.000484 | 0.026985 |
| oem/oem | clean | 0.218750 | 0.175018 | 0.117336 | 0.015993 | 0.000560 | 0.037853 |
| oem/oem | wrong_parent | 0.381250 | 0.306049 | 0.114276 | 0.032026 | 0.000918 | 0.029747 |
| oem/oem | paraphrase | 0.218750 | 0.174512 | 0.113360 | 0.016024 | 0.000543 | 0.035570 |
