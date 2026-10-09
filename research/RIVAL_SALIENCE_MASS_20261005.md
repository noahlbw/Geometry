# Retained observer salience-mass normalization

Same64 developed top-left512 windows with full-image wide context. NOT full datasets or independent validation. Geometry/finite VIP wide/fine/all20/risk decisions/reconstruction frozen; five earlier scores and per-image endpoints exact. Scores persist before masks; LoveDA D once, P separate; common scored-class support.

Replace log(K/remaining)/beta only by negative log surviving observer salience-prior mass/beta. Fine priors are original query-weighted fine crop mixtures; wide priors are original wide crop salience distributions. Uniform prior recovers count normalization; zero risk exact identity. No new forward or parameter. Salience is already used inside profiled logits and stays there: this compensation experiment is not a new probability model or correctness calibration of the original reader.

| Dataset/protocol | Geometry | NoAdmission_Exact | RivalFineHard_Exact | FineRivalProjected_Exact | FineBudgetOnly_Exact | FineSalienceMass_Projected | WideSalienceMass_Projected | HardFixedMass_Projected | FineSalienceMass_Projected__ClassMean | FineSalienceMass_Projected__HardStrength | FineSalienceMass_Projected__AliasShuffle0 | FineSalienceMass_Projected__AliasShuffle1 | FineSalienceMass_Projected__AliasShuffle2 | WideSalienceMass_Projected__ClassMean | WideSalienceMass_Projected__HardStrength | WideSalienceMass_Projected__AliasShuffle0 | WideSalienceMass_Projected__AliasShuffle1 | WideSalienceMass_Projected__AliasShuffle2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 38.4695 | 53.7470 | 54.3441 | 54.1183 | 54.1333 | 54.1245 | 54.1959 | 49.4830 | 54.1103 | 54.1336 | 54.1208 | 54.0984 | 54.0946 | 54.1154 | 54.1865 | 54.1313 | 54.1418 | 54.1127 |
| potsdam/potsdam | 35.7035 | 38.9203 | 40.6173 | 41.1231 | 40.9475 | 41.0931 | 41.0752 | 43.9302 | 41.1209 | 41.0934 | 41.1012 | 41.1110 | 41.1263 | 41.1213 | 41.0858 | 41.1006 | 41.1116 | 41.1236 |
| udd5/udd5 | 30.5010 | 28.1758 | 34.0394 | 34.6174 | 34.6906 | 34.6094 | 34.5088 | 36.4177 | 34.6222 | 34.6124 | 34.6362 | 34.6165 | 34.6249 | 34.6188 | 34.5272 | 34.6722 | 34.6108 | 34.6262 |
| oem/oem | 39.8205 | 39.0232 | 39.7906 | 40.4484 | 40.1611 | 40.4087 | 40.3963 | 42.1255 | 40.4496 | 40.4179 | 40.4690 | 40.4377 | 40.4303 | 40.4493 | 40.4119 | 40.4730 | 40.4303 | 40.4298 |
| loveda/P | 49.5399 | 50.7066 | 52.5879 | 52.6436 | 52.0588 | 52.4539 | 52.3732 | 45.7072 | 52.6455 | 52.4609 | 52.5147 | 52.6523 | 52.6856 | 52.6474 | 52.4324 | 52.6105 | 52.5345 | 52.6499 |
| loveda/D | 33.8779 | 30.7360 | 37.2156 | 37.6023 | 37.2513 | 37.5759 | 37.5354 | 38.9757 | 37.5997 | 37.5500 | 37.5962 | 37.5848 | 37.5954 | 37.6060 | 37.5520 | 37.6060 | 37.6109 | 37.6266 |
| vaihingen/vaihingen | 49.3651 | 51.8270 | 52.9446 | 53.3059 | 53.3440 | 53.2827 | 53.2765 | 55.6349 | 53.3085 | 53.2862 | 53.3110 | 53.3105 | 53.3050 | 53.3077 | 53.2777 | 53.3070 | 53.3065 | 53.2980 |
| landcoverai/landcoverai | 60.9049 | 66.9060 | 67.4834 | 67.7836 | 67.6095 | 67.7683 | 67.7663 | 68.2753 | 67.7842 | 67.7679 | 67.7831 | 67.7867 | 67.7800 | 67.7832 | 67.7654 | 67.7805 | 67.7870 | 67.7766 |
| flair1/flair1 | 35.6059 | 33.6084 | 34.0624 | 34.3507 | 34.4546 | 34.3389 | 34.3345 | 35.1708 | 34.3518 | 34.3395 | 34.3517 | 34.3528 | 34.3489 | 34.3518 | 34.3368 | 34.3516 | 34.3540 | 34.3500 |
| Eight-domain mean | 40.5310 | 42.8679 | 45.0622 | 45.4187 | 45.3240 | 45.4002 | 45.3861 | 46.2516 | 45.4184 | 45.4001 | 45.4212 | 45.4123 | 45.4132 | 45.4192 | 45.3929 | 45.4278 | 45.4191 | 45.4179 |

## Predeclared Gates

```json
{
  "FineSalienceMass_Projected": {
    "mean_gain_vs_projected_pp": -0.01853891971594379,
    "main_domain_wins": 1,
    "worst_protocol_gain_pp": -0.18970205856270184,
    "gain_vs_own_class_mean_pp": -0.01821595242311247,
    "gain_vs_own_strength_matched_pp": 6.0808813493906655e-05,
    "gain_vs_own_alias_null_mean_pp": -0.015368361512010154,
    "gain_vs_unsafe_fixed_mass_pp": -0.8514421372005003,
    "accuracy_gate": false,
    "mechanism_gate": false
  },
  "WideSalienceMass_Projected": {
    "mean_gain_vs_projected_pp": -0.0326122287472117,
    "main_domain_wins": 1,
    "worst_protocol_gain_pp": -0.27045560108756206,
    "gain_vs_own_class_mean_pp": -0.03308378446713789,
    "gain_vs_own_strength_matched_pp": -0.006805484407365725,
    "gain_vs_own_alias_null_mean_pp": -0.03550442342021398,
    "gain_vs_unsafe_fixed_mass_pp": -0.8655154462317682,
    "accuracy_gate": false,
    "mechanism_gate": false
  }
}
```

Each candidate has its own matched prior-identity null/class-mean/strength controls. Prior spectra and risk decisions are fixed under alias null, but surviving prior mass changes by design. Class mean protects canonical mass, not each rival surviving mass. Fixed-mass hard is an explicitly stability-failing diagnostic, not a model candidate or gate target. No automatic full20092 rollout or retained-model replacement.

## Label-Free Prior Diagnostics

| Dataset/protocol | Prior | Normalizer change vs count | Larger compensation fraction | Prior entropy/log K |
| --- | --- | ---: | ---: | ---: |
| vdd/vdd | FineSalienceMass_Projected | -0.000024 | 0.473342 | 0.999952 |
| vdd/vdd | WideSalienceMass_Projected | 0.001609 | 0.628688 | 0.999938 |
| potsdam/potsdam | FineSalienceMass_Projected | -0.000062 | 0.480758 | 0.999941 |
| potsdam/potsdam | WideSalienceMass_Projected | 0.001302 | 0.694039 | 0.999933 |
| udd5/udd5 | FineSalienceMass_Projected | -0.001021 | 0.331519 | 0.999926 |
| udd5/udd5 | WideSalienceMass_Projected | 0.003451 | 0.838551 | 0.999873 |
| oem/oem | FineSalienceMass_Projected | 0.000287 | 0.550061 | 0.999920 |
| oem/oem | WideSalienceMass_Projected | 0.001038 | 0.631247 | 0.999900 |
| loveda/P | FineSalienceMass_Projected | -0.000200 | 0.462297 | 0.999935 |
| loveda/P | WideSalienceMass_Projected | 0.002840 | 0.734735 | 0.999893 |
| loveda/D | FineSalienceMass_Projected | -0.000214 | 0.470881 | 0.999935 |
| loveda/D | WideSalienceMass_Projected | 0.002969 | 0.741779 | 0.999893 |
| vaihingen/vaihingen | FineSalienceMass_Projected | 0.000096 | 0.504741 | 0.999942 |
| vaihingen/vaihingen | WideSalienceMass_Projected | 0.001378 | 0.711340 | 0.999941 |
| landcoverai/landcoverai | FineSalienceMass_Projected | -0.000387 | 0.482979 | 0.999944 |
| landcoverai/landcoverai | WideSalienceMass_Projected | 0.000448 | 0.583029 | 0.999931 |
| flair1/flair1 | FineSalienceMass_Projected | 0.000314 | 0.560091 | 0.999919 |
| flair1/flair1 | WideSalienceMass_Projected | 0.000921 | 0.634807 | 0.999913 |

Uniform/count adjustment errors and class-mean mass errors<=1e-10; prior shuffle spectra exact and canonical risk0. Entropy and fractions are diagnostic averages, not confidence estimates.

## Independent Window-Context Cost

Warmed synchronized alternating three repetitions, resident memory included; loading/text encoding/GT excluded. Candidate timing includes small CPU score-equality checks absent for hard; do not claim sub-percent differences. All singleton scores match all-arm scores bitwise in four real-model checks per candidate/domain. Singleton paths do not execute other writers or diagnostic controls; fine views remain.

| Dataset | Cached hard s | Fine-mass s | Wide-mass s |
| --- | ---: | ---: | ---: |
| vdd | 0.320734 | 0.333486 | 0.331796 |
| potsdam | 0.284328 | 0.301050 | 0.301455 |
| udd5 | 0.282612 | 0.294732 | 0.293238 |
| oem | 0.283402 | 0.304707 | 0.305878 |
| loveda | 0.322134 | 0.362627 | 0.365793 |
| vaihingen | 0.271179 | 0.288636 | 0.289198 |
| landcoverai | 0.272342 | 0.292248 | 0.291235 |
| flair1 | 0.285364 | 0.304406 | 0.305664 |

| Dataset | Cached hard MiB | Fine-mass MiB | Wide-mass MiB |
| --- | ---: | ---: | ---: |
| vdd | 5566.273 | 5566.273 | 5566.273 |
| potsdam | 5564.263 | 5564.263 | 5564.263 |
| udd5 | 5557.144 | 5557.144 | 5557.144 |
| oem | 5574.023 | 5574.023 | 5574.023 |
| loveda | 5595.028 | 5595.028 | 5595.028 |
| vaihingen | 5559.950 | 5559.950 | 5559.950 |
| landcoverai | 5559.950 | 5559.950 | 5559.950 |
| flair1 | 5797.988 | 5797.988 | 5797.988 |

## Per-Class Outcomes

| Dataset/protocol | Method | Class | IoU | Delta vs projected hard | Precision | Recall | Area |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | Geometry | other | 12.2851 | -42.7749 | 70.4934 | 12.9511 | 9.3800 |
| vdd/vdd | Geometry | wall | 19.8085 | -33.3364 | 20.3032 | 89.0471 | 34.8906 |
| vdd/vdd | Geometry | road | 24.9744 | 3.3126 | 29.4846 | 62.0157 | 6.9810 |
| vdd/vdd | Geometry | vegetation | 69.7492 | 15.4815 | 89.0479 | 76.2942 | 19.7359 |
| vdd/vdd | Geometry | vehicle | 3.8880 | -34.3717 | 3.8880 | 100.0000 | 9.2105 |
| vdd/vdd | Geometry | roof | 63.9827 | -25.8988 | 64.6650 | 98.3777 | 9.1667 |
| vdd/vdd | Geometry | water | 74.5988 | 8.0463 | 75.8754 | 97.7943 | 10.6353 |
| vdd/vdd | NoAdmission_Exact | other | 59.7766 | 4.7166 | 81.1541 | 69.4120 | 43.6684 |
| vdd/vdd | NoAdmission_Exact | wall | 59.6816 | 6.5367 | 65.8225 | 86.4811 | 10.4520 |
| vdd/vdd | NoAdmission_Exact | road | 23.2250 | 1.5632 | 23.3759 | 97.2962 | 13.8146 |
| vdd/vdd | NoAdmission_Exact | vegetation | 44.1426 | -10.1251 | 94.9106 | 45.2128 | 10.9733 |
| vdd/vdd | NoAdmission_Exact | vehicle | 50.3084 | 12.0487 | 50.3286 | 99.9201 | 0.7110 |
| vdd/vdd | NoAdmission_Exact | roof | 86.2607 | -3.6208 | 86.7114 | 99.4009 | 6.9072 |
| vdd/vdd | NoAdmission_Exact | water | 52.8339 | -13.7186 | 55.7407 | 91.0164 | 13.4736 |
| vdd/vdd | RivalFineHard_Exact | other | 57.2425 | 2.1825 | 81.0538 | 66.0849 | 41.6267 |
| vdd/vdd | RivalFineHard_Exact | wall | 58.4390 | 5.2941 | 63.3706 | 88.2481 | 11.0782 |
| vdd/vdd | RivalFineHard_Exact | road | 21.3171 | -0.3447 | 21.3725 | 98.7975 | 15.3427 |
| vdd/vdd | RivalFineHard_Exact | vegetation | 50.4296 | -3.8381 | 95.0722 | 51.7831 | 12.5465 |
| vdd/vdd | RivalFineHard_Exact | vehicle | 42.1815 | 3.9218 | 42.1815 | 100.0000 | 0.8490 |
| vdd/vdd | RivalFineHard_Exact | roof | 89.7289 | -0.1526 | 90.1757 | 99.4508 | 6.6452 |
| vdd/vdd | RivalFineHard_Exact | water | 61.0704 | -5.4821 | 64.1803 | 92.6489 | 11.9117 |
| vdd/vdd | FineRivalProjected_Exact | other | 55.0600 | 0.0000 | 80.8961 | 63.2893 | 39.9435 |
| vdd/vdd | FineRivalProjected_Exact | wall | 53.1449 | 0.0000 | 57.0418 | 88.6096 | 12.3578 |
| vdd/vdd | FineRivalProjected_Exact | road | 21.6618 | 0.0000 | 21.7082 | 99.0245 | 15.1402 |
| vdd/vdd | FineRivalProjected_Exact | vegetation | 54.2677 | 0.0000 | 94.7736 | 55.9419 | 13.5969 |
| vdd/vdd | FineRivalProjected_Exact | vehicle | 38.2597 | 0.0000 | 38.2597 | 100.0000 | 0.9360 |
| vdd/vdd | FineRivalProjected_Exact | roof | 89.8815 | 0.0000 | 90.3136 | 99.4706 | 6.6363 |
| vdd/vdd | FineRivalProjected_Exact | water | 66.5525 | 0.0000 | 68.9089 | 95.1129 | 11.3894 |
| vdd/vdd | FineBudgetOnly_Exact | other | 56.2592 | 1.1992 | 80.7432 | 64.9776 | 41.0867 |
| vdd/vdd | FineBudgetOnly_Exact | wall | 54.3088 | 1.1639 | 58.4429 | 88.4759 | 12.0433 |
| vdd/vdd | FineBudgetOnly_Exact | road | 22.3716 | 0.7098 | 22.4247 | 98.9527 | 14.6457 |
| vdd/vdd | FineBudgetOnly_Exact | vegetation | 52.7891 | -1.4786 | 95.4959 | 54.1370 | 13.0587 |
| vdd/vdd | FineBudgetOnly_Exact | vehicle | 38.4045 | 0.1448 | 38.4045 | 100.0000 | 0.9325 |
| vdd/vdd | FineBudgetOnly_Exact | roof | 90.4037 | 0.5222 | 90.8606 | 99.4468 | 6.5948 |
| vdd/vdd | FineBudgetOnly_Exact | water | 64.3964 | -2.1561 | 66.9438 | 94.4206 | 11.6384 |
| vdd/vdd | FineSalienceMass_Projected | other | 55.1410 | 0.0810 | 80.8961 | 63.3963 | 40.0110 |
| vdd/vdd | FineSalienceMass_Projected | wall | 53.2849 | 0.1400 | 57.2046 | 88.6060 | 12.3221 |
| vdd/vdd | FineSalienceMass_Projected | road | 21.6756 | 0.0138 | 21.7220 | 99.0231 | 15.1303 |
| vdd/vdd | FineSalienceMass_Projected | vegetation | 54.1551 | -0.1126 | 94.7956 | 55.8146 | 13.5628 |
| vdd/vdd | FineSalienceMass_Projected | vehicle | 38.2851 | 0.0254 | 38.2851 | 100.0000 | 0.9354 |
| vdd/vdd | FineSalienceMass_Projected | roof | 89.8912 | 0.0097 | 90.3233 | 99.4706 | 6.6356 |
| vdd/vdd | FineSalienceMass_Projected | water | 66.4384 | -0.1141 | 68.8038 | 95.0800 | 11.4028 |
| vdd/vdd | WideSalienceMass_Projected | other | 55.2315 | 0.1715 | 80.9180 | 63.5024 | 40.0671 |
| vdd/vdd | WideSalienceMass_Projected | wall | 53.3736 | 0.2287 | 57.3111 | 88.5958 | 12.2978 |
| vdd/vdd | WideSalienceMass_Projected | road | 21.7036 | 0.0418 | 21.7505 | 99.0173 | 15.1096 |
| vdd/vdd | WideSalienceMass_Projected | vegetation | 54.1797 | -0.0880 | 94.8273 | 55.8297 | 13.5619 |
| vdd/vdd | WideSalienceMass_Projected | vehicle | 38.5187 | 0.2590 | 38.5187 | 100.0000 | 0.9297 |
| vdd/vdd | WideSalienceMass_Projected | roof | 89.9212 | 0.0397 | 90.3550 | 99.4690 | 6.6332 |
| vdd/vdd | WideSalienceMass_Projected | water | 66.4431 | -0.1094 | 68.8122 | 95.0736 | 11.4007 |
| vdd/vdd | HardFixedMass_Projected | other | 37.9511 | -17.1089 | 79.4379 | 42.0854 | 27.0487 |
| vdd/vdd | HardFixedMass_Projected | wall | 29.4305 | -23.7144 | 30.5393 | 89.0184 | 23.1885 |
| vdd/vdd | HardFixedMass_Projected | road | 23.5259 | 1.8641 | 23.5433 | 99.6868 | 14.0534 |
| vdd/vdd | HardFixedMass_Projected | vegetation | 66.4379 | 12.1702 | 94.9990 | 68.8457 | 16.6935 |
| vdd/vdd | HardFixedMass_Projected | vehicle | 31.8315 | -6.4282 | 31.8315 | 100.0000 | 1.1250 |
| vdd/vdd | HardFixedMass_Projected | roof | 84.7864 | -5.0951 | 85.2346 | 99.3835 | 7.0256 |
| vdd/vdd | HardFixedMass_Projected | water | 72.4177 | 5.8652 | 73.8990 | 97.3065 | 10.8653 |
| vdd/vdd | FineSalienceMass_Projected__ClassMean | other | 55.0433 | -0.0167 | 80.8933 | 63.2689 | 39.9320 |
| vdd/vdd | FineSalienceMass_Projected__ClassMean | wall | 53.1191 | -0.0258 | 57.0115 | 88.6108 | 12.3645 |
| vdd/vdd | FineSalienceMass_Projected__ClassMean | road | 21.6584 | -0.0034 | 21.7047 | 99.0245 | 15.1426 |
| vdd/vdd | FineSalienceMass_Projected__ClassMean | vegetation | 54.2760 | 0.0083 | 94.7740 | 55.9506 | 13.5989 |
| vdd/vdd | FineSalienceMass_Projected__ClassMean | vehicle | 38.2402 | -0.0195 | 38.2402 | 100.0000 | 0.9365 |
| vdd/vdd | FineSalienceMass_Projected__ClassMean | roof | 89.8783 | -0.0032 | 90.3103 | 99.4706 | 6.6366 |
| vdd/vdd | FineSalienceMass_Projected__ClassMean | water | 66.5569 | 0.0044 | 68.9128 | 95.1147 | 11.3890 |
| vdd/vdd | FineSalienceMass_Projected__HardStrength | other | 55.1493 | 0.0893 | 80.9066 | 63.4008 | 40.0087 |
| vdd/vdd | FineSalienceMass_Projected__HardStrength | wall | 53.2708 | 0.1259 | 57.1896 | 88.6030 | 12.3249 |
| vdd/vdd | FineSalienceMass_Projected__HardStrength | road | 21.6843 | 0.0225 | 21.7309 | 99.0216 | 15.1239 |
| vdd/vdd | FineSalienceMass_Projected__HardStrength | vegetation | 54.1820 | -0.0857 | 94.8016 | 55.8411 | 13.5684 |
| vdd/vdd | FineSalienceMass_Projected__HardStrength | vehicle | 38.3183 | 0.0586 | 38.3183 | 100.0000 | 0.9346 |
| vdd/vdd | FineSalienceMass_Projected__HardStrength | roof | 89.8905 | 0.0090 | 90.3227 | 99.4706 | 6.6357 |
| vdd/vdd | FineSalienceMass_Projected__HardStrength | water | 66.4403 | -0.1122 | 68.8022 | 95.0869 | 11.4039 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle0 | other | 55.0473 | -0.0127 | 80.8945 | 63.2735 | 39.9343 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle0 | wall | 53.1500 | 0.0051 | 57.0463 | 88.6126 | 12.3572 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle0 | road | 21.6538 | -0.0080 | 21.7001 | 99.0245 | 15.1458 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle0 | vegetation | 54.2566 | -0.0111 | 94.7313 | 55.9448 | 13.6036 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle0 | vehicle | 38.2734 | 0.0137 | 38.2734 | 100.0000 | 0.9356 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle0 | roof | 89.9143 | 0.0328 | 90.3467 | 99.4706 | 6.6339 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle0 | water | 66.5504 | -0.0021 | 68.9072 | 95.1118 | 11.3895 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle1 | other | 55.0450 | -0.0150 | 80.8885 | 63.2741 | 39.9377 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle1 | wall | 53.1155 | -0.0294 | 57.0124 | 88.5988 | 12.3626 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle1 | road | 21.6615 | -0.0003 | 21.7079 | 99.0231 | 15.1401 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle1 | vegetation | 54.2669 | -0.0008 | 94.7841 | 55.9373 | 13.5942 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle1 | vehicle | 38.2130 | -0.0467 | 38.2130 | 100.0000 | 0.9371 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle1 | roof | 89.8391 | -0.0424 | 90.2708 | 99.4706 | 6.6395 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle1 | water | 66.5475 | -0.0050 | 68.9075 | 95.1054 | 11.3887 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle2 | other | 55.0350 | -0.0250 | 80.8813 | 63.2653 | 39.9356 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle2 | wall | 53.1086 | -0.0363 | 56.9994 | 88.6108 | 12.3671 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle2 | road | 21.6568 | -0.0050 | 21.7032 | 99.0231 | 15.1434 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle2 | vegetation | 54.2547 | -0.0130 | 94.7927 | 55.9214 | 13.5891 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle2 | vehicle | 38.1800 | -0.0797 | 38.1800 | 100.0000 | 0.9379 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle2 | roof | 89.8577 | -0.0238 | 90.2896 | 99.4706 | 6.6381 |
| vdd/vdd | FineSalienceMass_Projected__AliasShuffle2 | water | 66.5693 | 0.0168 | 68.9212 | 95.1239 | 11.3887 |
| vdd/vdd | WideSalienceMass_Projected__ClassMean | other | 55.0456 | -0.0144 | 80.8920 | 63.2727 | 39.9351 |
| vdd/vdd | WideSalienceMass_Projected__ClassMean | wall | 53.1197 | -0.0252 | 57.0127 | 88.6096 | 12.3641 |
| vdd/vdd | WideSalienceMass_Projected__ClassMean | road | 21.6603 | -0.0015 | 21.7067 | 99.0245 | 15.1412 |
| vdd/vdd | WideSalienceMass_Projected__ClassMean | vegetation | 54.2774 | 0.0097 | 94.7735 | 55.9522 | 13.5994 |
| vdd/vdd | WideSalienceMass_Projected__ClassMean | vehicle | 38.2578 | -0.0019 | 38.2578 | 100.0000 | 0.9360 |
| vdd/vdd | WideSalienceMass_Projected__ClassMean | roof | 89.8854 | 0.0039 | 90.3175 | 99.4706 | 6.6360 |
| vdd/vdd | WideSalienceMass_Projected__ClassMean | water | 66.5613 | 0.0088 | 68.9174 | 95.1147 | 11.3882 |
| vdd/vdd | WideSalienceMass_Projected__HardStrength | other | 55.2149 | 0.1549 | 80.9210 | 63.4787 | 40.0507 |
| vdd/vdd | WideSalienceMass_Projected__HardStrength | wall | 53.3841 | 0.2392 | 57.3216 | 88.5994 | 12.2960 |
| vdd/vdd | WideSalienceMass_Projected__HardStrength | road | 21.6944 | 0.0326 | 21.7413 | 99.0159 | 15.1157 |
| vdd/vdd | WideSalienceMass_Projected__HardStrength | vegetation | 54.2128 | -0.0549 | 94.8196 | 55.8676 | 13.5722 |
| vdd/vdd | WideSalienceMass_Projected__HardStrength | vehicle | 38.4183 | 0.1586 | 38.4183 | 100.0000 | 0.9321 |
| vdd/vdd | WideSalienceMass_Projected__HardStrength | roof | 89.9175 | 0.0360 | 90.3499 | 99.4706 | 6.6337 |
| vdd/vdd | WideSalienceMass_Projected__HardStrength | water | 66.4636 | -0.0889 | 68.8279 | 95.0858 | 11.3996 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle0 | other | 55.0419 | -0.0181 | 80.8943 | 63.2664 | 39.9299 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle0 | wall | 53.1404 | -0.0045 | 57.0356 | 88.6120 | 12.3594 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle0 | road | 21.6535 | -0.0083 | 21.6998 | 99.0245 | 15.1460 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle0 | vegetation | 54.3246 | 0.0569 | 94.7504 | 56.0104 | 13.6168 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle0 | vehicle | 38.2344 | -0.0253 | 38.2344 | 100.0000 | 0.9366 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle0 | roof | 89.8924 | 0.0109 | 90.3246 | 99.4706 | 6.6355 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle0 | water | 66.6319 | 0.0794 | 68.9931 | 95.1147 | 11.3757 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle1 | other | 55.0897 | 0.0297 | 80.8940 | 63.3298 | 39.9701 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle1 | wall | 53.1670 | 0.0221 | 57.0744 | 88.5922 | 12.3483 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle1 | road | 21.6772 | 0.0154 | 21.7237 | 99.0216 | 15.1289 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle1 | vegetation | 54.2789 | 0.0112 | 94.7930 | 55.9470 | 13.5953 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle1 | vehicle | 38.3359 | 0.0762 | 38.3359 | 100.0000 | 0.9341 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle1 | roof | 89.8654 | -0.0161 | 90.2973 | 99.4706 | 6.6375 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle1 | water | 66.5784 | 0.0259 | 68.9342 | 95.1175 | 11.3858 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle2 | other | 55.0674 | 0.0074 | 80.8834 | 63.3068 | 39.9609 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle2 | wall | 53.1465 | 0.0016 | 57.0463 | 88.6030 | 12.3559 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle2 | road | 21.6714 | 0.0096 | 21.7179 | 99.0216 | 15.1329 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle2 | vegetation | 54.2410 | -0.0267 | 94.7992 | 55.9046 | 13.5841 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle2 | vehicle | 38.2324 | -0.0273 | 38.2324 | 100.0000 | 0.9367 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle2 | roof | 89.8680 | -0.0135 | 90.2999 | 99.4706 | 6.6373 |
| vdd/vdd | WideSalienceMass_Projected__AliasShuffle2 | water | 66.5619 | 0.0094 | 68.9075 | 95.1349 | 11.3923 |
| potsdam/potsdam | Geometry | impervious surface | 48.6704 | -17.6961 | 84.3816 | 53.4889 | 25.9113 |
| potsdam/potsdam | Geometry | building | 72.8222 | -0.0441 | 79.5088 | 89.6472 | 18.1787 |
| potsdam/potsdam | Geometry | low vegetation | 26.5422 | 4.8098 | 77.4031 | 28.7716 | 6.4458 |
| potsdam/potsdam | Geometry | tree | 49.3362 | -8.7199 | 84.7436 | 54.1454 | 11.2731 |
| potsdam/potsdam | Geometry | car | 11.6958 | -13.4835 | 11.6965 | 99.9455 | 30.6388 |
| potsdam/potsdam | Geometry | clutter | 5.1543 | 2.6162 | 7.7773 | 13.2570 | 7.5523 |
| potsdam/potsdam | NoAdmission_Exact | impervious surface | 65.8977 | -0.4688 | 85.6706 | 74.0609 | 35.3370 |
| potsdam/potsdam | NoAdmission_Exact | building | 71.0529 | -1.8134 | 74.9985 | 93.1063 | 20.0156 |
| potsdam/potsdam | NoAdmission_Exact | low vegetation | 14.4124 | -7.3200 | 79.9195 | 14.9539 | 3.2447 |
| potsdam/potsdam | NoAdmission_Exact | tree | 55.5064 | -2.5497 | 91.9038 | 58.3601 | 11.2039 |
| potsdam/potsdam | NoAdmission_Exact | car | 24.5378 | -0.6415 | 24.5936 | 99.0837 | 14.4459 |
| potsdam/potsdam | NoAdmission_Exact | clutter | 2.1143 | -0.4238 | 2.6529 | 9.4321 | 15.7528 |
| potsdam/potsdam | RivalFineHard_Exact | impervious surface | 66.2446 | -0.1219 | 85.2705 | 74.8044 | 35.8593 |
| potsdam/potsdam | RivalFineHard_Exact | building | 72.5275 | -0.3388 | 76.5613 | 93.2276 | 19.6326 |
| potsdam/potsdam | RivalFineHard_Exact | low vegetation | 19.6733 | -2.0591 | 81.8153 | 20.5728 | 4.3604 |
| potsdam/potsdam | RivalFineHard_Exact | tree | 57.9390 | -0.1171 | 92.6969 | 60.7103 | 11.5554 |
| potsdam/potsdam | RivalFineHard_Exact | car | 24.9082 | -0.2711 | 24.9554 | 99.2460 | 14.2598 |
| potsdam/potsdam | RivalFineHard_Exact | clutter | 2.4111 | -0.1270 | 3.0821 | 9.9702 | 14.3326 |
| potsdam/potsdam | FineRivalProjected_Exact | impervious surface | 66.3665 | 0.0000 | 85.2000 | 75.0146 | 35.9898 |
| potsdam/potsdam | FineRivalProjected_Exact | building | 72.8663 | 0.0000 | 76.9013 | 93.2829 | 19.5574 |
| potsdam/potsdam | FineRivalProjected_Exact | low vegetation | 21.7324 | 0.0000 | 81.8728 | 22.8310 | 4.8357 |
| potsdam/potsdam | FineRivalProjected_Exact | tree | 58.0561 | 0.0000 | 92.6395 | 60.8635 | 11.5917 |
| potsdam/potsdam | FineRivalProjected_Exact | car | 25.1793 | 0.0000 | 25.2161 | 99.4242 | 14.1377 |
| potsdam/potsdam | FineRivalProjected_Exact | clutter | 2.5381 | 0.0000 | 3.2649 | 10.2339 | 13.8877 |
| potsdam/potsdam | FineBudgetOnly_Exact | impervious surface | 66.1042 | -0.2623 | 85.2298 | 74.6567 | 35.8056 |
| potsdam/potsdam | FineBudgetOnly_Exact | building | 72.9459 | 0.0796 | 77.0489 | 93.1965 | 19.5018 |
| potsdam/potsdam | FineBudgetOnly_Exact | low vegetation | 21.5223 | -0.2101 | 81.8118 | 22.6038 | 4.7911 |
| potsdam/potsdam | FineBudgetOnly_Exact | tree | 57.6287 | -0.4274 | 92.5777 | 60.4203 | 11.5150 |
| potsdam/potsdam | FineBudgetOnly_Exact | car | 24.9889 | -0.1904 | 25.0268 | 99.3976 | 14.2408 |
| potsdam/potsdam | FineBudgetOnly_Exact | clutter | 2.4949 | -0.0432 | 3.1966 | 10.2059 | 14.1457 |
| potsdam/potsdam | FineSalienceMass_Projected | impervious surface | 66.3649 | -0.0016 | 85.2025 | 75.0105 | 35.9868 |
| potsdam/potsdam | FineSalienceMass_Projected | building | 72.8376 | -0.0287 | 76.8717 | 93.2793 | 19.5642 |
| potsdam/potsdam | FineSalienceMass_Projected | low vegetation | 21.6150 | -0.1174 | 81.9042 | 22.6990 | 4.8059 |
| potsdam/potsdam | FineSalienceMass_Projected | tree | 58.0381 | -0.0180 | 92.6363 | 60.8452 | 11.5886 |
| potsdam/potsdam | FineSalienceMass_Projected | car | 25.1779 | -0.0014 | 25.2152 | 99.4162 | 14.1371 |
| potsdam/potsdam | FineSalienceMass_Projected | clutter | 2.5254 | -0.0127 | 3.2473 | 10.2005 | 13.9175 |
| potsdam/potsdam | WideSalienceMass_Projected | impervious surface | 66.3681 | 0.0016 | 85.1984 | 75.0179 | 35.9921 |
| potsdam/potsdam | WideSalienceMass_Projected | building | 72.8354 | -0.0309 | 76.8806 | 93.2628 | 19.5584 |
| potsdam/potsdam | WideSalienceMass_Projected | low vegetation | 21.5493 | -0.1831 | 81.9530 | 22.6228 | 4.7869 |
| potsdam/potsdam | WideSalienceMass_Projected | tree | 57.9935 | -0.0626 | 92.6387 | 60.7952 | 11.5788 |
| potsdam/potsdam | WideSalienceMass_Projected | car | 25.1824 | 0.0031 | 25.2199 | 99.4135 | 14.1341 |
| potsdam/potsdam | WideSalienceMass_Projected | clutter | 2.5222 | -0.0159 | 3.2415 | 10.2059 | 13.9498 |
| potsdam/potsdam | HardFixedMass_Projected | impervious surface | 65.0409 | -1.3256 | 83.9342 | 74.2895 | 36.1794 |
| potsdam/potsdam | HardFixedMass_Projected | building | 73.6407 | 0.7744 | 78.5192 | 92.2194 | 18.9360 |
| potsdam/potsdam | HardFixedMass_Projected | low vegetation | 34.5009 | 12.7685 | 79.5505 | 37.8586 | 8.2526 |
| potsdam/potsdam | HardFixedMass_Projected | tree | 61.1790 | 3.1229 | 91.8354 | 64.6980 | 12.4299 |
| potsdam/potsdam | HardFixedMass_Projected | car | 25.0333 | -0.1460 | 25.0546 | 99.6622 | 14.2629 |
| potsdam/potsdam | HardFixedMass_Projected | clutter | 4.1864 | 1.6483 | 5.8094 | 13.0321 | 9.9391 |
| potsdam/potsdam | FineSalienceMass_Projected__ClassMean | impervious surface | 66.3631 | -0.0034 | 85.1993 | 75.0108 | 35.9882 |
| potsdam/potsdam | FineSalienceMass_Projected__ClassMean | building | 72.8651 | -0.0012 | 76.9010 | 93.2814 | 19.5571 |
| potsdam/potsdam | FineSalienceMass_Projected__ClassMean | low vegetation | 21.7247 | -0.0077 | 81.8759 | 22.8222 | 4.8336 |
| potsdam/potsdam | FineSalienceMass_Projected__ClassMean | tree | 58.0535 | -0.0026 | 92.6400 | 60.8606 | 11.5911 |
| potsdam/potsdam | FineSalienceMass_Projected__ClassMean | car | 25.1810 | 0.0017 | 25.2177 | 99.4242 | 14.1368 |
| potsdam/potsdam | FineSalienceMass_Projected__ClassMean | clutter | 2.5379 | -0.0002 | 3.2643 | 10.2360 | 13.8931 |
| potsdam/potsdam | FineSalienceMass_Projected__HardStrength | impervious surface | 66.3635 | -0.0030 | 85.2081 | 75.0045 | 35.9815 |
| potsdam/potsdam | FineSalienceMass_Projected__HardStrength | building | 72.8361 | -0.0302 | 76.8689 | 93.2811 | 19.5652 |
| potsdam/potsdam | FineSalienceMass_Projected__HardStrength | low vegetation | 21.6315 | -0.1009 | 81.9094 | 22.7168 | 4.8093 |
| potsdam/potsdam | FineSalienceMass_Projected__HardStrength | tree | 58.0283 | -0.0278 | 92.6370 | 60.8341 | 11.5864 |
| potsdam/potsdam | FineSalienceMass_Projected__HardStrength | car | 25.1739 | -0.0054 | 25.2109 | 99.4202 | 14.1400 |
| potsdam/potsdam | FineSalienceMass_Projected__HardStrength | clutter | 2.5273 | -0.0108 | 3.2497 | 10.2080 | 13.9174 |
| potsdam/potsdam | FineSalienceMass_Projected__AliasShuffle0 | impervious surface | 66.3581 | -0.0084 | 85.1935 | 75.0089 | 35.9898 |
| potsdam/potsdam | FineSalienceMass_Projected__AliasShuffle0 | building | 72.8601 | -0.0062 | 76.9014 | 93.2725 | 19.5552 |
| potsdam/potsdam | FineSalienceMass_Projected__AliasShuffle0 | low vegetation | 21.6427 | -0.0897 | 81.9194 | 22.7284 | 4.8112 |
| potsdam/potsdam | FineSalienceMass_Projected__AliasShuffle0 | tree | 58.0309 | -0.0252 | 92.6388 | 60.8362 | 11.5866 |
| potsdam/potsdam | FineSalienceMass_Projected__AliasShuffle0 | car | 25.1832 | 0.0039 | 25.2203 | 99.4189 | 14.1346 |
| potsdam/potsdam | FineSalienceMass_Projected__AliasShuffle0 | clutter | 2.5323 | -0.0058 | 3.2557 | 10.2306 | 13.9226 |
| potsdam/potsdam | FineSalienceMass_Projected__AliasShuffle1 | impervious surface | 66.3572 | -0.0093 | 85.1964 | 75.0054 | 35.9869 |
| potsdam/potsdam | FineSalienceMass_Projected__AliasShuffle1 | building | 72.8434 | -0.0229 | 76.8838 | 93.2710 | 19.5593 |
| potsdam/potsdam | FineSalienceMass_Projected__AliasShuffle1 | low vegetation | 21.7018 | -0.0306 | 81.8911 | 22.7958 | 4.8271 |
| potsdam/potsdam | FineSalienceMass_Projected__AliasShuffle1 | tree | 58.0494 | -0.0067 | 92.6387 | 60.8565 | 11.5905 |
| potsdam/potsdam | FineSalienceMass_Projected__AliasShuffle1 | car | 25.1788 | -0.0005 | 25.2158 | 99.4215 | 14.1375 |
| potsdam/potsdam | FineSalienceMass_Projected__AliasShuffle1 | clutter | 2.5354 | -0.0027 | 3.2610 | 10.2296 | 13.8987 |
| potsdam/potsdam | FineSalienceMass_Projected__AliasShuffle2 | impervious surface | 66.3621 | -0.0044 | 85.1937 | 75.0138 | 35.9921 |
| potsdam/potsdam | FineSalienceMass_Projected__AliasShuffle2 | building | 72.8509 | -0.0154 | 76.8877 | 93.2776 | 19.5597 |
| potsdam/potsdam | FineSalienceMass_Projected__AliasShuffle2 | low vegetation | 21.7750 | 0.0426 | 81.8305 | 22.8813 | 4.8488 |
| potsdam/potsdam | FineSalienceMass_Projected__AliasShuffle2 | tree | 58.0461 | -0.0100 | 92.6398 | 60.8525 | 11.5896 |
| potsdam/potsdam | FineSalienceMass_Projected__AliasShuffle2 | car | 25.1839 | 0.0046 | 25.2209 | 99.4215 | 14.1346 |
| potsdam/potsdam | FineSalienceMass_Projected__AliasShuffle2 | clutter | 2.5396 | 0.0015 | 3.2675 | 10.2328 | 13.8752 |
| potsdam/potsdam | WideSalienceMass_Projected__ClassMean | impervious surface | 66.3650 | -0.0015 | 85.2012 | 75.0116 | 35.9878 |
| potsdam/potsdam | WideSalienceMass_Projected__ClassMean | building | 72.8635 | -0.0028 | 76.8985 | 93.2823 | 19.5580 |
| potsdam/potsdam | WideSalienceMass_Projected__ClassMean | low vegetation | 21.7292 | -0.0032 | 81.8621 | 22.8282 | 4.8357 |
| potsdam/potsdam | WideSalienceMass_Projected__ClassMean | tree | 58.0521 | -0.0040 | 92.6420 | 60.8581 | 11.5904 |
| potsdam/potsdam | WideSalienceMass_Projected__ClassMean | car | 25.1801 | 0.0008 | 25.2170 | 99.4228 | 14.1370 |
| potsdam/potsdam | WideSalienceMass_Projected__ClassMean | clutter | 2.5379 | -0.0002 | 3.2645 | 10.2349 | 13.8911 |
| potsdam/potsdam | WideSalienceMass_Projected__HardStrength | impervious surface | 66.3606 | -0.0059 | 85.2095 | 74.9996 | 35.9786 |
| potsdam/potsdam | WideSalienceMass_Projected__HardStrength | building | 72.8411 | -0.0252 | 76.8758 | 93.2790 | 19.5631 |
| potsdam/potsdam | WideSalienceMass_Projected__HardStrength | low vegetation | 21.6084 | -0.1240 | 81.8989 | 22.6921 | 4.8047 |
| potsdam/potsdam | WideSalienceMass_Projected__HardStrength | tree | 58.0113 | -0.0448 | 92.6375 | 60.8152 | 11.5828 |
| potsdam/potsdam | WideSalienceMass_Projected__HardStrength | car | 25.1689 | -0.0104 | 25.2061 | 99.4175 | 14.1424 |
| potsdam/potsdam | WideSalienceMass_Projected__HardStrength | clutter | 2.5246 | -0.0135 | 3.2458 | 10.2037 | 13.9286 |
| potsdam/potsdam | WideSalienceMass_Projected__AliasShuffle0 | impervious surface | 66.3712 | 0.0047 | 85.2012 | 75.0195 | 35.9916 |
| potsdam/potsdam | WideSalienceMass_Projected__AliasShuffle0 | building | 72.8430 | -0.0233 | 76.8842 | 93.2699 | 19.5590 |
| potsdam/potsdam | WideSalienceMass_Projected__AliasShuffle0 | low vegetation | 21.6493 | -0.0831 | 81.9138 | 22.7361 | 4.8131 |
| potsdam/potsdam | WideSalienceMass_Projected__AliasShuffle0 | tree | 58.0236 | -0.0325 | 92.6401 | 60.8276 | 11.5848 |
| potsdam/potsdam | WideSalienceMass_Projected__AliasShuffle0 | car | 25.1866 | 0.0073 | 25.2239 | 99.4175 | 14.1324 |
| potsdam/potsdam | WideSalienceMass_Projected__AliasShuffle0 | clutter | 2.5301 | -0.0080 | 3.2531 | 10.2199 | 13.9190 |
| potsdam/potsdam | WideSalienceMass_Projected__AliasShuffle1 | impervious surface | 66.3677 | 0.0012 | 85.2010 | 75.0153 | 35.9897 |
| potsdam/potsdam | WideSalienceMass_Projected__AliasShuffle1 | building | 72.8567 | -0.0096 | 76.8950 | 93.2764 | 19.5576 |
| potsdam/potsdam | WideSalienceMass_Projected__AliasShuffle1 | low vegetation | 21.6857 | -0.0467 | 81.8778 | 22.7790 | 4.8244 |
| potsdam/potsdam | WideSalienceMass_Projected__AliasShuffle1 | tree | 58.0386 | -0.0175 | 92.6400 | 60.8441 | 11.5880 |
| potsdam/potsdam | WideSalienceMass_Projected__AliasShuffle1 | car | 25.1860 | 0.0067 | 25.2231 | 99.4202 | 14.1332 |
| potsdam/potsdam | WideSalienceMass_Projected__AliasShuffle1 | clutter | 2.5350 | -0.0031 | 3.2600 | 10.2328 | 13.9071 |
| potsdam/potsdam | WideSalienceMass_Projected__AliasShuffle2 | impervious surface | 66.3649 | -0.0016 | 85.2061 | 75.0077 | 35.9839 |
| potsdam/potsdam | WideSalienceMass_Projected__AliasShuffle2 | building | 72.8468 | -0.0195 | 76.8826 | 93.2784 | 19.5612 |
| potsdam/potsdam | WideSalienceMass_Projected__AliasShuffle2 | low vegetation | 21.7643 | 0.0319 | 81.8339 | 22.8692 | 4.8460 |
| potsdam/potsdam | WideSalienceMass_Projected__AliasShuffle2 | tree | 58.0555 | -0.0006 | 92.6406 | 60.8625 | 11.5914 |
| potsdam/potsdam | WideSalienceMass_Projected__AliasShuffle2 | car | 25.1736 | -0.0057 | 25.2103 | 99.4242 | 14.1409 |
| potsdam/potsdam | WideSalienceMass_Projected__AliasShuffle2 | clutter | 2.5366 | -0.0015 | 3.2638 | 10.2220 | 13.8765 |
| udd5/udd5 | Geometry | vegetation | 60.2474 | -8.3139 | 89.6338 | 64.7597 | 2.2305 |
| udd5/udd5 | Geometry | building | 83.4755 | -2.4045 | 86.9865 | 95.3877 | 78.8720 |
| udd5/udd5 | Geometry | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.2018 |
| udd5/udd5 | Geometry | vehicle | 3.7107 | -2.9205 | 3.7107 | 99.9773 | 11.3160 |
| udd5/udd5 | Geometry | other | 5.0713 | -6.9433 | 23.0553 | 6.1044 | 5.3797 |
| udd5/udd5 | NoAdmission_Exact | vegetation | 46.2799 | -22.2814 | 93.8774 | 47.7203 | 1.5693 |
| udd5/udd5 | NoAdmission_Exact | building | 83.4871 | -2.3929 | 83.6594 | 99.7539 | 85.7625 |
| udd5/udd5 | NoAdmission_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2949 |
| udd5/udd5 | NoAdmission_Exact | vehicle | 5.3645 | -1.2667 | 5.3645 | 99.9773 | 7.8274 |
| udd5/udd5 | NoAdmission_Exact | other | 5.7476 | -6.2670 | 29.7284 | 6.6513 | 4.5458 |
| udd5/udd5 | RivalFineHard_Exact | vegetation | 67.6962 | -0.8651 | 91.0639 | 72.5133 | 2.4583 |
| udd5/udd5 | RivalFineHard_Exact | building | 85.5987 | -0.2813 | 85.6693 | 99.9037 | 83.8762 |
| udd5/udd5 | RivalFineHard_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.6799 |
| udd5/udd5 | RivalFineHard_Exact | vehicle | 7.0104 | 0.3792 | 7.0105 | 99.9773 | 5.9896 |
| udd5/udd5 | RivalFineHard_Exact | other | 9.8917 | -2.1229 | 39.5036 | 11.6576 | 5.9959 |
| udd5/udd5 | FineRivalProjected_Exact | vegetation | 68.5613 | 0.0000 | 90.9603 | 73.5744 | 2.4971 |
| udd5/udd5 | FineRivalProjected_Exact | building | 85.8800 | 0.0000 | 86.0701 | 99.7434 | 83.3517 |
| udd5/udd5 | FineRivalProjected_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.3867 |
| udd5/udd5 | FineRivalProjected_Exact | vehicle | 6.6312 | 0.0000 | 6.6313 | 99.9773 | 6.3321 |
| udd5/udd5 | FineRivalProjected_Exact | other | 12.0146 | 0.0000 | 44.6062 | 14.1216 | 6.4323 |
| udd5/udd5 | FineBudgetOnly_Exact | vegetation | 68.7476 | 0.1863 | 90.9648 | 73.7860 | 2.5042 |
| udd5/udd5 | FineBudgetOnly_Exact | building | 85.4999 | -0.3801 | 85.5929 | 99.8730 | 83.9252 |
| udd5/udd5 | FineBudgetOnly_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9439 |
| udd5/udd5 | FineBudgetOnly_Exact | vehicle | 6.8445 | 0.2133 | 6.8446 | 99.9773 | 6.1348 |
| udd5/udd5 | FineBudgetOnly_Exact | other | 12.3612 | 0.3466 | 45.4328 | 14.5163 | 6.4919 |
| udd5/udd5 | FineSalienceMass_Projected | vegetation | 68.5395 | -0.0218 | 90.9480 | 73.5574 | 2.4969 |
| udd5/udd5 | FineSalienceMass_Projected | building | 85.8928 | 0.0128 | 86.0847 | 99.7411 | 83.3356 |
| udd5/udd5 | FineSalienceMass_Projected | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.3960 |
| udd5/udd5 | FineSalienceMass_Projected | vehicle | 6.6190 | -0.0122 | 6.6191 | 99.9773 | 6.3438 |
| udd5/udd5 | FineSalienceMass_Projected | other | 11.9955 | -0.0191 | 44.5674 | 14.0991 | 6.4277 |
| udd5/udd5 | WideSalienceMass_Projected | vegetation | 68.2527 | -0.3086 | 91.0534 | 73.1589 | 2.4805 |
| udd5/udd5 | WideSalienceMass_Projected | building | 85.7942 | -0.0858 | 85.9626 | 99.7721 | 83.4798 |
| udd5/udd5 | WideSalienceMass_Projected | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.3016 |
| udd5/udd5 | WideSalienceMass_Projected | vehicle | 6.6362 | 0.0050 | 6.6363 | 99.9773 | 6.3274 |
| udd5/udd5 | WideSalienceMass_Projected | other | 11.8612 | -0.1534 | 44.2101 | 13.9491 | 6.4107 |
| udd5/udd5 | HardFixedMass_Projected | vegetation | 73.3235 | 4.7622 | 89.2792 | 80.4028 | 2.7803 |
| udd5/udd5 | HardFixedMass_Projected | building | 86.4035 | 0.5235 | 87.1783 | 98.9818 | 81.6637 |
| udd5/udd5 | HardFixedMass_Projected | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.9347 |
| udd5/udd5 | HardFixedMass_Projected | vehicle | 7.0260 | 0.3948 | 7.0261 | 99.9773 | 5.9763 |
| udd5/udd5 | HardFixedMass_Projected | other | 15.3357 | 3.3211 | 53.9529 | 17.6452 | 6.6450 |
| udd5/udd5 | FineSalienceMass_Projected__ClassMean | vegetation | 68.5860 | 0.0247 | 90.9519 | 73.6084 | 2.4985 |
| udd5/udd5 | FineSalienceMass_Projected__ClassMean | building | 85.8850 | 0.0050 | 86.0797 | 99.7373 | 83.3372 |
| udd5/udd5 | FineSalienceMass_Projected__ClassMean | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.3968 |
| udd5/udd5 | FineSalienceMass_Projected__ClassMean | vehicle | 6.6256 | -0.0056 | 6.6257 | 99.9773 | 6.3375 |
| udd5/udd5 | FineSalienceMass_Projected__ClassMean | other | 12.0142 | -0.0004 | 44.6169 | 14.1200 | 6.4301 |
| udd5/udd5 | FineSalienceMass_Projected__HardStrength | vegetation | 68.5433 | -0.0180 | 90.9711 | 73.5466 | 2.4959 |
| udd5/udd5 | FineSalienceMass_Projected__HardStrength | building | 85.8893 | 0.0093 | 86.0819 | 99.7402 | 83.3375 |
| udd5/udd5 | FineSalienceMass_Projected__HardStrength | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.3968 |
| udd5/udd5 | FineSalienceMass_Projected__HardStrength | vehicle | 6.6237 | -0.0075 | 6.6238 | 99.9773 | 6.3393 |
| udd5/udd5 | FineSalienceMass_Projected__HardStrength | other | 12.0059 | -0.0087 | 44.5869 | 14.1115 | 6.4305 |
| udd5/udd5 | FineSalienceMass_Projected__AliasShuffle0 | vegetation | 68.5963 | 0.0350 | 90.9370 | 73.6300 | 2.4997 |
| udd5/udd5 | FineSalienceMass_Projected__AliasShuffle0 | building | 85.8780 | -0.0020 | 86.0731 | 99.7368 | 83.3433 |
| udd5/udd5 | FineSalienceMass_Projected__AliasShuffle0 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.3901 |
| udd5/udd5 | FineSalienceMass_Projected__AliasShuffle0 | vehicle | 6.6406 | 0.0094 | 6.6407 | 99.9773 | 6.3231 |
| udd5/udd5 | FineSalienceMass_Projected__AliasShuffle0 | other | 12.0663 | 0.0517 | 44.7168 | 14.1819 | 6.4438 |
| udd5/udd5 | FineSalienceMass_Projected__AliasShuffle1 | vegetation | 68.5951 | 0.0338 | 90.9514 | 73.6192 | 2.4989 |
| udd5/udd5 | FineSalienceMass_Projected__AliasShuffle1 | building | 85.8815 | 0.0015 | 86.0775 | 99.7356 | 83.3380 |
| udd5/udd5 | FineSalienceMass_Projected__AliasShuffle1 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.3982 |
| udd5/udd5 | FineSalienceMass_Projected__AliasShuffle1 | vehicle | 6.6177 | -0.0135 | 6.6178 | 99.9773 | 6.3450 |
| udd5/udd5 | FineSalienceMass_Projected__AliasShuffle1 | other | 11.9882 | -0.0264 | 44.5846 | 14.0873 | 6.4198 |
| udd5/udd5 | FineSalienceMass_Projected__AliasShuffle2 | vegetation | 68.5847 | 0.0234 | 90.9566 | 73.6037 | 2.4982 |
| udd5/udd5 | FineSalienceMass_Projected__AliasShuffle2 | building | 85.8859 | 0.0059 | 86.0791 | 99.7393 | 83.3395 |
| udd5/udd5 | FineSalienceMass_Projected__AliasShuffle2 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.3945 |
| udd5/udd5 | FineSalienceMass_Projected__AliasShuffle2 | vehicle | 6.6290 | -0.0022 | 6.6291 | 99.9773 | 6.3342 |
| udd5/udd5 | FineSalienceMass_Projected__AliasShuffle2 | other | 12.0250 | 0.0104 | 44.6346 | 14.1331 | 6.4335 |
| udd5/udd5 | WideSalienceMass_Projected__ClassMean | vegetation | 68.5770 | 0.0157 | 90.9573 | 73.5945 | 2.4979 |
| udd5/udd5 | WideSalienceMass_Projected__ClassMean | building | 85.8800 | 0.0000 | 86.0702 | 99.7434 | 83.3516 |
| udd5/udd5 | WideSalienceMass_Projected__ClassMean | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.3874 |
| udd5/udd5 | WideSalienceMass_Projected__ClassMean | vehicle | 6.6295 | -0.0017 | 6.6296 | 99.9773 | 6.3338 |
| udd5/udd5 | WideSalienceMass_Projected__ClassMean | other | 12.0076 | -0.0070 | 44.5985 | 14.1127 | 6.4294 |
| udd5/udd5 | WideSalienceMass_Projected__HardStrength | vegetation | 68.3361 | -0.2252 | 91.0177 | 73.2778 | 2.4855 |
| udd5/udd5 | WideSalienceMass_Projected__HardStrength | building | 85.7918 | -0.0882 | 85.9641 | 99.7670 | 83.4742 |
| udd5/udd5 | WideSalienceMass_Projected__HardStrength | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.2968 |
| udd5/udd5 | WideSalienceMass_Projected__HardStrength | vehicle | 6.6259 | -0.0053 | 6.6260 | 99.9773 | 6.3372 |
| udd5/udd5 | WideSalienceMass_Projected__HardStrength | other | 11.8824 | -0.1322 | 44.3037 | 13.9691 | 6.4063 |
| udd5/udd5 | WideSalienceMass_Projected__AliasShuffle0 | vegetation | 68.7022 | 0.1409 | 90.9416 | 73.7489 | 2.5036 |
| udd5/udd5 | WideSalienceMass_Projected__AliasShuffle0 | building | 85.9104 | 0.0304 | 86.1126 | 99.7274 | 83.2971 |
| udd5/udd5 | WideSalienceMass_Projected__AliasShuffle0 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.4153 |
| udd5/udd5 | WideSalienceMass_Projected__AliasShuffle0 | vehicle | 6.6365 | 0.0053 | 6.6366 | 99.9773 | 6.3271 |
| udd5/udd5 | WideSalienceMass_Projected__AliasShuffle0 | other | 12.1121 | 0.0975 | 44.7992 | 14.2368 | 6.4569 |
| udd5/udd5 | WideSalienceMass_Projected__AliasShuffle1 | vegetation | 68.5615 | 0.0002 | 90.9441 | 73.5852 | 2.4980 |
| udd5/udd5 | WideSalienceMass_Projected__AliasShuffle1 | building | 85.9007 | 0.0207 | 86.0955 | 99.7373 | 83.3219 |
| udd5/udd5 | WideSalienceMass_Projected__AliasShuffle1 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.4002 |
| udd5/udd5 | WideSalienceMass_Projected__AliasShuffle1 | vehicle | 6.6013 | -0.0299 | 6.6014 | 99.9773 | 6.3608 |
| udd5/udd5 | WideSalienceMass_Projected__AliasShuffle1 | other | 11.9904 | -0.0242 | 44.5955 | 14.0892 | 6.4191 |
| udd5/udd5 | WideSalienceMass_Projected__AliasShuffle2 | vegetation | 68.5853 | 0.0240 | 90.9648 | 73.5991 | 2.4979 |
| udd5/udd5 | WideSalienceMass_Projected__AliasShuffle2 | building | 85.8795 | -0.0005 | 86.0679 | 99.7458 | 83.3558 |
| udd5/udd5 | WideSalienceMass_Projected__AliasShuffle2 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.3902 |
| udd5/udd5 | WideSalienceMass_Projected__AliasShuffle2 | vehicle | 6.6444 | 0.0132 | 6.6445 | 99.9773 | 6.3195 |
| udd5/udd5 | WideSalienceMass_Projected__AliasShuffle2 | other | 12.0218 | 0.0072 | 44.6073 | 14.1315 | 6.4367 |
| oem/oem | Geometry | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.9017 |
| oem/oem | Geometry | rangeland | 41.5417 | -8.3175 | 54.4153 | 63.7145 | 17.0320 |
| oem/oem | Geometry | developed space | 25.6977 | 2.2107 | 65.5485 | 29.7106 | 8.8828 |
| oem/oem | Geometry | road | 52.3958 | -3.0184 | 60.6185 | 79.4351 | 8.2388 |
| oem/oem | Geometry | tree | 65.2556 | 7.5232 | 84.8180 | 73.8857 | 23.0134 |
| oem/oem | Geometry | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0565 |
| oem/oem | Geometry | agriculture land | 71.2323 | -11.3134 | 95.7971 | 73.5302 | 16.2828 |
| oem/oem | Geometry | building | 62.4408 | 7.8917 | 64.4677 | 95.2061 | 17.5919 |
| oem/oem | NoAdmission_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.2072 |
| oem/oem | NoAdmission_Exact | rangeland | 49.5261 | -0.3331 | 70.5348 | 62.4456 | 12.8780 |
| oem/oem | NoAdmission_Exact | developed space | 23.7669 | 0.2799 | 38.9362 | 37.8898 | 19.0710 |
| oem/oem | NoAdmission_Exact | road | 54.5529 | -0.8613 | 68.0810 | 73.3007 | 6.7693 |
| oem/oem | NoAdmission_Exact | tree | 55.6605 | -2.0719 | 91.7802 | 58.5807 | 16.8622 |
| oem/oem | NoAdmission_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | NoAdmission_Exact | agriculture land | 81.6500 | -0.8957 | 82.0066 | 99.4703 | 25.7312 |
| oem/oem | NoAdmission_Exact | building | 47.0287 | -7.5204 | 68.3392 | 60.1297 | 10.4811 |
| oem/oem | RivalFineHard_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.4062 |
| oem/oem | RivalFineHard_Exact | rangeland | 49.6933 | -0.1659 | 70.6164 | 62.6472 | 12.9046 |
| oem/oem | RivalFineHard_Exact | developed space | 23.8549 | 0.3679 | 40.6619 | 36.5937 | 17.6369 |
| oem/oem | RivalFineHard_Exact | road | 54.9225 | -0.4917 | 68.1954 | 73.8350 | 6.8072 |
| oem/oem | RivalFineHard_Exact | tree | 56.9837 | -0.7487 | 91.1586 | 60.3173 | 17.4805 |
| oem/oem | RivalFineHard_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0002 |
| oem/oem | RivalFineHard_Exact | agriculture land | 82.0069 | -0.5388 | 82.3962 | 99.4271 | 25.5984 |
| oem/oem | RivalFineHard_Exact | building | 50.8636 | -3.6855 | 69.6827 | 65.3182 | 11.1660 |
| oem/oem | FineRivalProjected_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.6617 |
| oem/oem | FineRivalProjected_Exact | rangeland | 49.8592 | 0.0000 | 70.6753 | 62.8643 | 12.9385 |
| oem/oem | FineRivalProjected_Exact | developed space | 23.4870 | 0.0000 | 41.7734 | 34.9185 | 16.3817 |
| oem/oem | FineRivalProjected_Exact | road | 55.4142 | 0.0000 | 68.4843 | 74.3824 | 6.8287 |
| oem/oem | FineRivalProjected_Exact | tree | 57.7324 | 0.0000 | 90.8900 | 61.2783 | 17.8115 |
| oem/oem | FineRivalProjected_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0007 |
| oem/oem | FineRivalProjected_Exact | agriculture land | 82.5457 | 0.0000 | 82.9578 | 99.4017 | 25.4186 |
| oem/oem | FineRivalProjected_Exact | building | 54.5491 | 0.0000 | 70.4544 | 70.7286 | 11.9585 |
| oem/oem | FineBudgetOnly_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.5567 |
| oem/oem | FineBudgetOnly_Exact | rangeland | 49.8649 | 0.0057 | 70.6703 | 62.8774 | 12.9422 |
| oem/oem | FineBudgetOnly_Exact | developed space | 23.7934 | 0.3064 | 41.0958 | 36.1075 | 17.2188 |
| oem/oem | FineBudgetOnly_Exact | road | 55.3529 | -0.0613 | 68.7562 | 73.9549 | 6.7626 |
| oem/oem | FineBudgetOnly_Exact | tree | 57.7722 | 0.0398 | 90.9006 | 61.3183 | 17.8210 |
| oem/oem | FineBudgetOnly_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0008 |
| oem/oem | FineBudgetOnly_Exact | agriculture land | 82.8190 | 0.2733 | 83.2403 | 99.3925 | 25.3300 |
| oem/oem | FineBudgetOnly_Exact | building | 51.6864 | -2.8627 | 69.7801 | 66.5924 | 11.3680 |
| oem/oem | FineSalienceMass_Projected | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.6505 |
| oem/oem | FineSalienceMass_Projected | rangeland | 49.8538 | -0.0054 | 70.6735 | 62.8572 | 12.9374 |
| oem/oem | FineSalienceMass_Projected | developed space | 23.4944 | 0.0074 | 41.6825 | 34.9987 | 16.4551 |
| oem/oem | FineSalienceMass_Projected | road | 55.3975 | -0.0167 | 68.4891 | 74.3466 | 6.8249 |
| oem/oem | FineSalienceMass_Projected | tree | 57.6976 | -0.0348 | 90.9012 | 61.2340 | 17.7964 |
| oem/oem | FineSalienceMass_Projected | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0007 |
| oem/oem | FineSalienceMass_Projected | agriculture land | 82.5214 | -0.0243 | 82.9323 | 99.4031 | 25.4268 |
| oem/oem | FineSalienceMass_Projected | building | 54.3050 | -0.2441 | 70.3985 | 70.3747 | 11.9081 |
| oem/oem | WideSalienceMass_Projected | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.6447 |
| oem/oem | WideSalienceMass_Projected | rangeland | 49.8533 | -0.0059 | 70.6746 | 62.8555 | 12.9369 |
| oem/oem | WideSalienceMass_Projected | developed space | 23.5018 | 0.0148 | 41.6615 | 35.0299 | 16.4781 |
| oem/oem | WideSalienceMass_Projected | road | 55.3899 | -0.0243 | 68.4862 | 74.3365 | 6.8243 |
| oem/oem | WideSalienceMass_Projected | tree | 57.6878 | -0.0446 | 90.9067 | 61.2205 | 17.7914 |
| oem/oem | WideSalienceMass_Projected | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0007 |
| oem/oem | WideSalienceMass_Projected | agriculture land | 82.5130 | -0.0327 | 82.9236 | 99.4036 | 25.4296 |
| oem/oem | WideSalienceMass_Projected | building | 54.2244 | -0.3247 | 70.3715 | 70.2662 | 11.8943 |
| oem/oem | HardFixedMass_Projected | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9.3174 |
| oem/oem | HardFixedMass_Projected | rangeland | 49.9459 | 0.0867 | 70.5596 | 63.0945 | 13.0072 |
| oem/oem | HardFixedMass_Projected | developed space | 22.8729 | -0.6141 | 45.4059 | 31.5494 | 13.6170 |
| oem/oem | HardFixedMass_Projected | road | 56.7392 | 1.3250 | 69.4442 | 75.6176 | 6.8461 |
| oem/oem | HardFixedMass_Projected | tree | 60.3220 | 2.5896 | 89.6413 | 64.8419 | 19.1098 |
| oem/oem | HardFixedMass_Projected | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0155 |
| oem/oem | HardFixedMass_Projected | agriculture land | 85.0999 | 2.5542 | 85.7354 | 99.1365 | 24.5295 |
| oem/oem | HardFixedMass_Projected | building | 62.0243 | 7.4752 | 71.9163 | 81.8487 | 13.5573 |
| oem/oem | FineSalienceMass_Projected__ClassMean | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.6618 |
| oem/oem | FineSalienceMass_Projected__ClassMean | rangeland | 49.8605 | 0.0013 | 70.6750 | 62.8666 | 12.9391 |
| oem/oem | FineSalienceMass_Projected__ClassMean | developed space | 23.4861 | -0.0009 | 41.7739 | 34.9162 | 16.3805 |
| oem/oem | FineSalienceMass_Projected__ClassMean | road | 55.4152 | 0.0010 | 68.4865 | 74.3817 | 6.8284 |
| oem/oem | FineSalienceMass_Projected__ClassMean | tree | 57.7342 | 0.0018 | 90.8900 | 61.2803 | 17.8121 |
| oem/oem | FineSalienceMass_Projected__ClassMean | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0007 |
| oem/oem | FineSalienceMass_Projected__ClassMean | agriculture land | 82.5488 | 0.0031 | 82.9610 | 99.4017 | 25.4177 |
| oem/oem | FineSalienceMass_Projected__ClassMean | building | 54.5522 | 0.0031 | 70.4532 | 70.7352 | 11.9598 |
| oem/oem | FineSalienceMass_Projected__HardStrength | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.6535 |
| oem/oem | FineSalienceMass_Projected__HardStrength | rangeland | 49.8540 | -0.0052 | 70.6725 | 62.8582 | 12.9378 |
| oem/oem | FineSalienceMass_Projected__HardStrength | developed space | 23.4923 | 0.0053 | 41.7073 | 34.9767 | 16.4350 |
| oem/oem | FineSalienceMass_Projected__HardStrength | road | 55.3939 | -0.0203 | 68.4771 | 74.3544 | 6.8268 |
| oem/oem | FineSalienceMass_Projected__HardStrength | tree | 57.7009 | -0.0315 | 90.9012 | 61.2377 | 17.7975 |
| oem/oem | FineSalienceMass_Projected__HardStrength | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0007 |
| oem/oem | FineSalienceMass_Projected__HardStrength | agriculture land | 82.5261 | -0.0196 | 82.9371 | 99.4031 | 25.4253 |
| oem/oem | FineSalienceMass_Projected__HardStrength | building | 54.3758 | -0.1733 | 70.4129 | 70.4791 | 11.9233 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle0 | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.6673 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle0 | rangeland | 49.8605 | 0.0013 | 70.6767 | 62.8653 | 12.9385 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle0 | developed space | 23.4683 | -0.0187 | 41.8167 | 34.8470 | 16.3313 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle0 | road | 55.4175 | 0.0033 | 68.4702 | 74.4050 | 6.8322 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle0 | tree | 57.7437 | 0.0113 | 90.8871 | 61.2924 | 17.8161 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle0 | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0007 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle0 | agriculture land | 82.5480 | 0.0023 | 82.9603 | 99.4015 | 25.4178 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle0 | building | 54.7140 | 0.1649 | 70.4817 | 70.9785 | 11.9961 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle1 | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.6590 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle1 | rangeland | 49.8626 | 0.0034 | 70.6747 | 62.8703 | 12.9399 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle1 | developed space | 23.4875 | 0.0005 | 41.7382 | 34.9442 | 16.4076 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle1 | road | 55.4165 | 0.0023 | 68.4951 | 74.3739 | 6.8268 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle1 | tree | 57.7261 | -0.0063 | 90.8947 | 61.2690 | 17.8078 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle1 | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0007 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle1 | agriculture land | 82.5476 | 0.0019 | 82.9597 | 99.4017 | 25.4180 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle1 | building | 54.4614 | -0.0877 | 70.4354 | 70.6004 | 11.9400 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle2 | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.6772 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle2 | rangeland | 49.8624 | 0.0032 | 70.6741 | 62.8703 | 12.9400 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle2 | developed space | 23.4300 | -0.0570 | 41.6890 | 34.8515 | 16.3834 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle2 | road | 55.4152 | 0.0010 | 68.4825 | 74.3863 | 6.8292 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle2 | tree | 57.7317 | -0.0007 | 90.8943 | 61.2755 | 17.8098 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle2 | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0007 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle2 | agriculture land | 82.5491 | 0.0034 | 82.9615 | 99.4015 | 25.4175 |
| oem/oem | FineSalienceMass_Projected__AliasShuffle2 | building | 54.4544 | -0.0947 | 70.4234 | 70.6008 | 11.9421 |
| oem/oem | WideSalienceMass_Projected__ClassMean | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.6626 |
| oem/oem | WideSalienceMass_Projected__ClassMean | rangeland | 49.8606 | 0.0014 | 70.6749 | 62.8670 | 12.9392 |
| oem/oem | WideSalienceMass_Projected__ClassMean | developed space | 23.4858 | -0.0012 | 41.7727 | 34.9165 | 16.3811 |
| oem/oem | WideSalienceMass_Projected__ClassMean | road | 55.4155 | 0.0013 | 68.4870 | 74.3817 | 6.8284 |
| oem/oem | WideSalienceMass_Projected__ClassMean | tree | 57.7341 | 0.0017 | 90.8902 | 61.2801 | 17.8120 |
| oem/oem | WideSalienceMass_Projected__ClassMean | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0007 |
| oem/oem | WideSalienceMass_Projected__ClassMean | agriculture land | 82.5492 | 0.0035 | 82.9613 | 99.4017 | 25.4176 |
| oem/oem | WideSalienceMass_Projected__ClassMean | building | 54.5492 | 0.0001 | 70.4542 | 70.7291 | 11.9586 |
| oem/oem | WideSalienceMass_Projected__HardStrength | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.6514 |
| oem/oem | WideSalienceMass_Projected__HardStrength | rangeland | 49.8536 | -0.0056 | 70.6730 | 62.8572 | 12.9375 |
| oem/oem | WideSalienceMass_Projected__HardStrength | developed space | 23.4925 | 0.0055 | 41.6908 | 34.9887 | 16.4472 |
| oem/oem | WideSalienceMass_Projected__HardStrength | road | 55.3932 | -0.0210 | 68.4779 | 74.3521 | 6.8266 |
| oem/oem | WideSalienceMass_Projected__HardStrength | tree | 57.6924 | -0.0400 | 90.9047 | 61.2266 | 17.7935 |
| oem/oem | WideSalienceMass_Projected__HardStrength | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0007 |
| oem/oem | WideSalienceMass_Projected__HardStrength | agriculture land | 82.5219 | -0.0238 | 82.9326 | 99.4034 | 25.4268 |
| oem/oem | WideSalienceMass_Projected__HardStrength | building | 54.3417 | -0.2074 | 70.4049 | 70.4298 | 11.9163 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle0 | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.6714 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle0 | rangeland | 49.8624 | 0.0032 | 70.6775 | 62.8677 | 12.9388 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle0 | developed space | 23.4547 | -0.0323 | 41.8139 | 34.8191 | 16.3192 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle0 | road | 55.4167 | 0.0025 | 68.4645 | 74.4105 | 6.8333 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle0 | tree | 57.7486 | 0.0162 | 90.8869 | 61.2979 | 17.8178 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle0 | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0007 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle0 | agriculture land | 82.5556 | 0.0099 | 82.9679 | 99.4017 | 25.4155 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle0 | building | 54.7461 | 0.1970 | 70.4875 | 71.0266 | 12.0032 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle1 | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.6623 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle1 | rangeland | 49.8621 | 0.0029 | 70.6744 | 62.8697 | 12.9398 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle1 | developed space | 23.4711 | -0.0159 | 41.7119 | 34.9265 | 16.4096 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle1 | road | 55.4135 | -0.0007 | 68.4924 | 74.3715 | 6.8269 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle1 | tree | 57.7197 | -0.0127 | 90.8950 | 61.2618 | 17.8057 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle1 | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0007 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle1 | agriculture land | 82.5397 | -0.0060 | 82.9517 | 99.4017 | 25.4205 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle1 | building | 54.4363 | -0.1128 | 70.4307 | 70.5630 | 11.9345 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle2 | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.6844 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle2 | rangeland | 49.8661 | 0.0069 | 70.6748 | 62.8757 | 12.9410 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle2 | developed space | 23.4176 | -0.0694 | 41.6727 | 34.8355 | 16.3823 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle2 | road | 55.4153 | 0.0011 | 68.4848 | 74.3840 | 6.8288 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle2 | tree | 57.7381 | 0.0057 | 90.8941 | 61.2829 | 17.8120 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle2 | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0007 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle2 | agriculture land | 82.5634 | 0.0177 | 82.9758 | 99.4015 | 25.4131 |
| oem/oem | WideSalienceMass_Projected__AliasShuffle2 | building | 54.4377 | -0.1114 | 70.4223 | 70.5737 | 11.9377 |
| loveda/P | Geometry | building | 7.6311 | -41.2981 | 7.6311 | 100.0000 | 0.3472 |
| loveda/P | Geometry | road | 63.0522 | 2.8911 | 64.4734 | 96.6222 | 12.4720 |
| loveda/P | Geometry | water | 63.2131 | 0.2889 | 68.9718 | 88.3328 | 18.8451 |
| loveda/P | Geometry | barren | 1.9144 | 0.2420 | 34.2731 | 1.9874 | 0.6429 |
| loveda/P | Geometry | tree | 72.0056 | 15.4145 | 88.8915 | 79.1256 | 14.6560 |
| loveda/P | Geometry | farm | 89.4231 | 3.8393 | 91.1653 | 97.9076 | 53.0368 |
| loveda/P | NoAdmission_Exact | building | 60.8479 | 11.9187 | 72.1893 | 79.4788 | 0.0292 |
| loveda/P | NoAdmission_Exact | road | 53.7553 | -6.4058 | 54.7280 | 96.7995 | 14.7198 |
| loveda/P | NoAdmission_Exact | water | 62.4091 | -0.5151 | 68.1253 | 88.1486 | 19.0394 |
| loveda/P | NoAdmission_Exact | barren | 4.3215 | 2.6491 | 89.7378 | 4.3430 | 0.5366 |
| loveda/P | NoAdmission_Exact | tree | 39.9553 | -16.6358 | 99.3680 | 40.0571 | 6.6373 |
| loveda/P | NoAdmission_Exact | farm | 82.9502 | -2.6336 | 83.2671 | 99.5433 | 59.0377 |
| loveda/P | RivalFineHard_Exact | building | 53.9715 | 5.0423 | 59.0200 | 86.3192 | 0.0388 |
| loveda/P | RivalFineHard_Exact | road | 58.5446 | -1.6165 | 59.7219 | 96.7425 | 13.4810 |
| loveda/P | RivalFineHard_Exact | water | 62.5489 | -0.3753 | 67.9247 | 88.7680 | 19.2298 |
| loveda/P | RivalFineHard_Exact | barren | 2.3466 | 0.6742 | 65.0543 | 2.3766 | 0.4051 |
| loveda/P | RivalFineHard_Exact | tree | 53.2126 | -3.3785 | 97.3091 | 54.0073 | 9.1381 |
| loveda/P | RivalFineHard_Exact | farm | 84.9031 | -0.6807 | 85.2129 | 99.5737 | 57.7072 |
| loveda/P | FineRivalProjected_Exact | building | 48.9292 | 0.0000 | 49.7487 | 96.7427 | 0.0515 |
| loveda/P | FineRivalProjected_Exact | road | 60.1611 | 0.0000 | 61.4250 | 96.6927 | 13.1005 |
| loveda/P | FineRivalProjected_Exact | water | 62.9242 | 0.0000 | 68.0543 | 89.3018 | 19.3086 |
| loveda/P | FineRivalProjected_Exact | barren | 1.6724 | 0.0000 | 41.1787 | 1.7134 | 0.4613 |
| loveda/P | FineRivalProjected_Exact | tree | 56.5911 | 0.0000 | 96.7147 | 57.7003 | 9.8230 |
| loveda/P | FineRivalProjected_Exact | farm | 85.5838 | 0.0000 | 85.8926 | 99.5816 | 57.2550 |
| loveda/P | FineBudgetOnly_Exact | building | 44.3478 | -4.5814 | 44.4122 | 99.6743 | 0.0595 |
| loveda/P | FineBudgetOnly_Exact | road | 60.5514 | 0.3903 | 61.8333 | 96.6896 | 13.0136 |
| loveda/P | FineBudgetOnly_Exact | water | 63.0869 | 0.1627 | 68.1903 | 89.3950 | 19.2903 |
| loveda/P | FineBudgetOnly_Exact | barren | 1.8502 | 0.1778 | 62.0227 | 1.8714 | 0.3345 |
| loveda/P | FineBudgetOnly_Exact | tree | 57.0813 | 0.4902 | 96.4614 | 58.3021 | 9.9515 |
| loveda/P | FineBudgetOnly_Exact | farm | 85.4354 | -0.1484 | 85.7461 | 99.5778 | 57.3507 |
| loveda/P | FineSalienceMass_Projected | building | 48.2143 | -0.7149 | 49.0099 | 96.7427 | 0.0523 |
| loveda/P | FineSalienceMass_Projected | road | 60.0371 | -0.1240 | 61.2946 | 96.6958 | 13.1288 |
| loveda/P | FineSalienceMass_Projected | water | 62.9171 | -0.0071 | 68.0508 | 89.2935 | 19.3079 |
| loveda/P | FineSalienceMass_Projected | barren | 1.6815 | 0.0091 | 41.7516 | 1.7219 | 0.4573 |
| loveda/P | FineSalienceMass_Projected | tree | 56.3321 | -0.2590 | 96.7479 | 57.4193 | 9.7718 |
| loveda/P | FineSalienceMass_Projected | farm | 85.5414 | -0.0424 | 85.8510 | 99.5802 | 57.2820 |
| loveda/P | WideSalienceMass_Projected | building | 47.9675 | -0.9617 | 48.9221 | 96.0912 | 0.0520 |
| loveda/P | WideSalienceMass_Projected | road | 59.9645 | -0.1966 | 61.2189 | 96.6958 | 13.1450 |
| loveda/P | WideSalienceMass_Projected | water | 62.9142 | -0.0100 | 68.0515 | 89.2865 | 19.3061 |
| loveda/P | WideSalienceMass_Projected | barren | 1.7159 | 0.0435 | 43.7415 | 1.7546 | 0.4448 |
| loveda/P | WideSalienceMass_Projected | tree | 56.1777 | -0.4134 | 96.7644 | 57.2532 | 9.7418 |
| loveda/P | WideSalienceMass_Projected | farm | 85.4992 | -0.0846 | 85.8086 | 99.5800 | 57.3102 |
| loveda/P | HardFixedMass_Projected | building | 1.1404 | -47.7888 | 1.1404 | 100.0000 | 2.3235 |
| loveda/P | HardFixedMass_Projected | road | 66.1987 | 6.0376 | 67.7849 | 96.5859 | 11.8582 |
| loveda/P | HardFixedMass_Projected | water | 64.5042 | 1.5800 | 69.5540 | 89.8830 | 19.0153 |
| loveda/P | HardFixedMass_Projected | barren | 1.1262 | -0.5462 | 3.2283 | 1.7001 | 5.8389 |
| loveda/P | HardFixedMass_Projected | tree | 65.9827 | 9.3916 | 94.2365 | 68.7574 | 12.0132 |
| loveda/P | HardFixedMass_Projected | farm | 75.2908 | -10.2930 | 86.2843 | 85.5267 | 48.9509 |
| loveda/P | FineSalienceMass_Projected__ClassMean | building | 48.9292 | 0.0000 | 49.7487 | 96.7427 | 0.0515 |
| loveda/P | FineSalienceMass_Projected__ClassMean | road | 60.1665 | 0.0054 | 61.4307 | 96.6927 | 13.0993 |
| loveda/P | FineSalienceMass_Projected__ClassMean | water | 62.9235 | -0.0007 | 68.0528 | 89.3029 | 19.3093 |
| loveda/P | FineSalienceMass_Projected__ClassMean | barren | 1.6723 | -0.0001 | 41.0864 | 1.7134 | 0.4624 |
| loveda/P | FineSalienceMass_Projected__ClassMean | tree | 56.5958 | 0.0047 | 96.7150 | 57.7050 | 9.8237 |
| loveda/P | FineSalienceMass_Projected__ClassMean | farm | 85.5857 | 0.0019 | 85.8946 | 99.5816 | 57.2538 |
| loveda/P | FineSalienceMass_Projected__HardStrength | building | 48.2143 | -0.7149 | 49.0099 | 96.7427 | 0.0523 |
| loveda/P | FineSalienceMass_Projected__HardStrength | road | 60.0453 | -0.1158 | 61.3036 | 96.6948 | 13.1267 |
| loveda/P | FineSalienceMass_Projected__HardStrength | water | 62.9145 | -0.0097 | 68.0477 | 89.2935 | 19.3087 |
| loveda/P | FineSalienceMass_Projected__HardStrength | barren | 1.6824 | 0.0100 | 41.8020 | 1.7227 | 0.4569 |
| loveda/P | FineSalienceMass_Projected__HardStrength | tree | 56.3617 | -0.2294 | 96.7445 | 57.4513 | 9.7776 |
| loveda/P | FineSalienceMass_Projected__HardStrength | farm | 85.5475 | -0.0363 | 85.8572 | 99.5800 | 57.2777 |
| loveda/P | FineSalienceMass_Projected__AliasShuffle0 | building | 48.2143 | -0.7149 | 49.0099 | 96.7427 | 0.0523 |
| loveda/P | FineSalienceMass_Projected__AliasShuffle0 | road | 60.1216 | -0.0395 | 61.3843 | 96.6917 | 13.1090 |
| loveda/P | FineSalienceMass_Projected__AliasShuffle0 | water | 62.9383 | 0.0141 | 68.0718 | 89.3000 | 19.3033 |
| loveda/P | FineSalienceMass_Projected__AliasShuffle0 | barren | 1.6791 | 0.0067 | 40.7332 | 1.7211 | 0.4685 |
| loveda/P | FineSalienceMass_Projected__AliasShuffle0 | tree | 56.5477 | -0.0434 | 96.7131 | 57.6558 | 9.8155 |
| loveda/P | FineSalienceMass_Projected__AliasShuffle0 | farm | 85.5871 | 0.0033 | 85.8970 | 99.5802 | 57.2513 |
| loveda/P | FineSalienceMass_Projected__AliasShuffle1 | building | 48.9292 | 0.0000 | 49.7487 | 96.7427 | 0.0515 |
| loveda/P | FineSalienceMass_Projected__AliasShuffle1 | road | 60.1866 | 0.0255 | 61.4512 | 96.6937 | 13.0951 |
| loveda/P | FineSalienceMass_Projected__AliasShuffle1 | water | 62.9173 | -0.0069 | 68.0459 | 89.3023 | 19.3111 |
| loveda/P | FineSalienceMass_Projected__AliasShuffle1 | barren | 1.6772 | 0.0048 | 40.9420 | 1.7188 | 0.4655 |
| loveda/P | FineSalienceMass_Projected__AliasShuffle1 | tree | 56.6131 | 0.0220 | 96.7185 | 57.7218 | 9.8263 |
| loveda/P | FineSalienceMass_Projected__AliasShuffle1 | farm | 85.5902 | 0.0064 | 85.8992 | 99.5814 | 57.2506 |
| loveda/P | FineSalienceMass_Projected__AliasShuffle2 | building | 49.0939 | 0.1647 | 49.8328 | 97.0684 | 0.0516 |
| loveda/P | FineSalienceMass_Projected__AliasShuffle2 | road | 60.1743 | 0.0132 | 61.4388 | 96.6927 | 13.0976 |
| loveda/P | FineSalienceMass_Projected__AliasShuffle2 | water | 62.9399 | 0.0157 | 68.0737 | 89.3000 | 19.3028 |
| loveda/P | FineSalienceMass_Projected__AliasShuffle2 | barren | 1.6701 | -0.0023 | 40.6996 | 1.7118 | 0.4663 |
| loveda/P | FineSalienceMass_Projected__AliasShuffle2 | tree | 56.6392 | 0.0481 | 96.6933 | 57.7580 | 9.8350 |
| loveda/P | FineSalienceMass_Projected__AliasShuffle2 | farm | 85.5961 | 0.0123 | 85.9050 | 99.5816 | 57.2468 |
| loveda/P | WideSalienceMass_Projected__ClassMean | building | 48.9292 | 0.0000 | 49.7487 | 96.7427 | 0.0515 |
| loveda/P | WideSalienceMass_Projected__ClassMean | road | 60.1696 | 0.0085 | 61.4339 | 96.6927 | 13.0986 |
| loveda/P | WideSalienceMass_Projected__ClassMean | water | 62.9230 | -0.0012 | 68.0522 | 89.3029 | 19.3095 |
| loveda/P | WideSalienceMass_Projected__ClassMean | barren | 1.6724 | 0.0000 | 41.1325 | 1.7134 | 0.4618 |
| loveda/P | WideSalienceMass_Projected__ClassMean | tree | 56.6040 | 0.0129 | 96.7155 | 57.7134 | 9.8251 |
| loveda/P | WideSalienceMass_Projected__ClassMean | farm | 85.5862 | 0.0024 | 85.8951 | 99.5816 | 57.2534 |
| loveda/P | WideSalienceMass_Projected__HardStrength | building | 48.2143 | -0.7149 | 49.0099 | 96.7427 | 0.0523 |
| loveda/P | WideSalienceMass_Projected__HardStrength | road | 59.9873 | -0.1738 | 61.2426 | 96.6958 | 13.1399 |
| loveda/P | WideSalienceMass_Projected__HardStrength | water | 62.9118 | -0.0124 | 68.0460 | 89.2912 | 19.3087 |
| loveda/P | WideSalienceMass_Projected__HardStrength | barren | 1.7102 | 0.0378 | 43.9444 | 1.7484 | 0.4411 |
| loveda/P | WideSalienceMass_Projected__HardStrength | tree | 56.2581 | -0.3330 | 96.7588 | 57.3386 | 9.7569 |
| loveda/P | WideSalienceMass_Projected__HardStrength | farm | 85.5126 | -0.0712 | 85.8223 | 99.5799 | 57.3010 |
| loveda/P | WideSalienceMass_Projected__AliasShuffle0 | building | 48.9292 | 0.0000 | 49.7487 | 96.7427 | 0.0515 |
| loveda/P | WideSalienceMass_Projected__AliasShuffle0 | road | 60.0911 | -0.0700 | 61.3517 | 96.6937 | 13.1163 |
| loveda/P | WideSalienceMass_Projected__AliasShuffle0 | water | 62.9324 | 0.0082 | 68.0653 | 89.2994 | 19.3050 |
| loveda/P | WideSalienceMass_Projected__AliasShuffle0 | barren | 1.6828 | 0.0104 | 40.7353 | 1.7250 | 0.4695 |
| loveda/P | WideSalienceMass_Projected__AliasShuffle0 | tree | 56.4539 | -0.1372 | 96.7204 | 57.5556 | 9.7978 |
| loveda/P | WideSalienceMass_Projected__AliasShuffle0 | farm | 85.5738 | -0.0100 | 85.8839 | 99.5799 | 57.2599 |
| loveda/P | WideSalienceMass_Projected__AliasShuffle1 | building | 48.2927 | -0.6365 | 49.0909 | 96.7427 | 0.0522 |
| loveda/P | WideSalienceMass_Projected__AliasShuffle1 | road | 60.1796 | 0.0185 | 61.4439 | 96.6937 | 13.0966 |
| loveda/P | WideSalienceMass_Projected__AliasShuffle1 | water | 62.9139 | -0.0103 | 68.0426 | 89.3012 | 19.3118 |
| loveda/P | WideSalienceMass_Projected__AliasShuffle1 | barren | 1.6789 | 0.0065 | 41.0781 | 1.7204 | 0.4643 |
| loveda/P | WideSalienceMass_Projected__AliasShuffle1 | tree | 56.5623 | -0.0288 | 96.7189 | 57.6689 | 9.8172 |
| loveda/P | WideSalienceMass_Projected__AliasShuffle1 | farm | 85.5796 | -0.0042 | 85.8885 | 99.5816 | 57.2578 |
| loveda/P | WideSalienceMass_Projected__AliasShuffle2 | building | 48.9292 | 0.0000 | 49.7487 | 96.7427 | 0.0515 |
| loveda/P | WideSalienceMass_Projected__AliasShuffle2 | road | 60.1648 | 0.0037 | 61.4285 | 96.6937 | 13.0999 |
| loveda/P | WideSalienceMass_Projected__AliasShuffle2 | water | 62.9391 | 0.0149 | 68.0721 | 89.3012 | 19.3035 |
| loveda/P | WideSalienceMass_Projected__AliasShuffle2 | barren | 1.6702 | -0.0022 | 40.7298 | 1.7118 | 0.4660 |
| loveda/P | WideSalienceMass_Projected__AliasShuffle2 | tree | 56.6070 | 0.0159 | 96.7023 | 57.7213 | 9.8278 |
| loveda/P | WideSalienceMass_Projected__AliasShuffle2 | farm | 85.5893 | 0.0055 | 85.8982 | 99.5816 | 57.2513 |
| loveda/D | Geometry | background | 39.0474 | 8.3540 | 75.8250 | 44.5998 | 26.3232 |
| loveda/D | Geometry | building | 4.8538 | -26.7087 | 4.8538 | 100.0000 | 0.3016 |
| loveda/D | Geometry | road | 45.0691 | -4.2060 | 45.8485 | 96.3650 | 9.6637 |
| loveda/D | Geometry | water | 53.8049 | -1.1962 | 58.0149 | 88.1157 | 12.3473 |
| loveda/D | Geometry | barren | 0.0293 | -1.5884 | 0.8665 | 0.0304 | 0.2146 |
| loveda/D | Geometry | tree | 37.0607 | -11.4112 | 42.3596 | 74.7642 | 16.0550 |
| loveda/D | Geometry | farm | 57.2802 | 10.6858 | 64.7327 | 83.2647 | 35.0945 |
| loveda/D | NoAdmission_Exact | background | 7.3224 | -23.3710 | 75.1732 | 7.5039 | 4.4672 |
| loveda/D | NoAdmission_Exact | building | 31.2821 | -0.2804 | 34.0307 | 79.4788 | 0.0342 |
| loveda/D | NoAdmission_Exact | road | 44.1270 | -5.1481 | 44.7837 | 96.7840 | 9.9365 |
| loveda/D | NoAdmission_Exact | water | 54.4379 | -0.5632 | 58.7421 | 88.1369 | 12.1974 |
| loveda/D | NoAdmission_Exact | barren | 3.2414 | 1.6237 | 82.3610 | 3.2640 | 0.2428 |
| loveda/D | NoAdmission_Exact | tree | 35.8674 | -12.6045 | 97.1427 | 36.2498 | 3.3944 |
| loveda/D | NoAdmission_Exact | farm | 38.8741 | -7.7203 | 38.9455 | 99.5311 | 69.7275 |
| loveda/D | RivalFineHard_Exact | background | 25.9735 | -4.7199 | 85.4376 | 27.1766 | 14.2353 |
| loveda/D | RivalFineHard_Exact | building | 38.0886 | 6.5261 | 39.8551 | 89.5765 | 0.0329 |
| loveda/D | RivalFineHard_Exact | road | 48.5879 | -0.6872 | 49.4050 | 96.7083 | 9.0000 |
| loveda/D | RivalFineHard_Exact | water | 54.7088 | -0.2923 | 58.8346 | 88.6384 | 12.2475 |
| loveda/D | RivalFineHard_Exact | barren | 1.7765 | 0.1588 | 68.4211 | 1.7912 | 0.1604 |
| loveda/D | RivalFineHard_Exact | tree | 45.9895 | -2.4824 | 93.8092 | 47.4290 | 4.5990 |
| loveda/D | RivalFineHard_Exact | farm | 45.3841 | -1.2103 | 45.4771 | 99.5512 | 59.7249 |
| loveda/D | FineRivalProjected_Exact | background | 30.6934 | 0.0000 | 84.6654 | 32.5002 | 17.1790 |
| loveda/D | FineRivalProjected_Exact | building | 31.5625 | 0.0000 | 31.6946 | 98.6971 | 0.0456 |
| loveda/D | FineRivalProjected_Exact | road | 49.2751 | 0.0000 | 50.1318 | 96.6481 | 8.8640 |
| loveda/D | FineRivalProjected_Exact | water | 55.0011 | 0.0000 | 58.9493 | 89.1446 | 12.2935 |
| loveda/D | FineRivalProjected_Exact | barren | 1.6177 | 0.0000 | 66.8262 | 1.6308 | 0.1495 |
| loveda/D | FineRivalProjected_Exact | tree | 48.4719 | 0.0000 | 92.5745 | 50.4327 | 4.9555 |
| loveda/D | FineRivalProjected_Exact | farm | 46.5944 | 0.0000 | 47.1297 | 97.6203 | 56.5129 |
| loveda/D | FineBudgetOnly_Exact | background | 28.8149 | -1.8785 | 86.6920 | 30.1485 | 15.5634 |
| loveda/D | FineBudgetOnly_Exact | building | 30.0098 | -1.5527 | 30.0098 | 100.0000 | 0.0488 |
| loveda/D | FineBudgetOnly_Exact | road | 49.3559 | 0.0808 | 50.2207 | 96.6284 | 8.8465 |
| loveda/D | FineBudgetOnly_Exact | water | 55.1009 | 0.0998 | 59.0131 | 89.2607 | 12.2962 |
| loveda/D | FineBudgetOnly_Exact | barren | 1.7088 | 0.0911 | 70.4364 | 1.7211 | 0.1497 |
| loveda/D | FineBudgetOnly_Exact | tree | 49.0748 | 0.6029 | 92.1383 | 51.2196 | 5.0567 |
| loveda/D | FineBudgetOnly_Exact | farm | 46.6943 | 0.0999 | 46.7946 | 99.5431 | 58.0387 |
| loveda/D | FineSalienceMass_Projected | background | 30.5314 | -0.1620 | 84.8008 | 32.2989 | 17.0454 |
| loveda/D | FineSalienceMass_Projected | building | 31.8277 | 0.2652 | 31.9620 | 98.6971 | 0.0452 |
| loveda/D | FineSalienceMass_Projected | road | 49.2236 | -0.0515 | 50.0782 | 96.6491 | 8.8736 |
| loveda/D | FineSalienceMass_Projected | water | 54.9770 | -0.0241 | 58.9257 | 89.1352 | 12.2971 |
| loveda/D | FineSalienceMass_Projected | barren | 1.6240 | 0.0063 | 66.9745 | 1.6371 | 0.1497 |
| loveda/D | FineSalienceMass_Projected | tree | 48.2656 | -0.2063 | 92.6656 | 50.1827 | 4.9261 |
| loveda/D | FineSalienceMass_Projected | farm | 46.5823 | -0.0121 | 47.0808 | 97.7778 | 56.6629 |
| loveda/D | WideSalienceMass_Projected | background | 30.3219 | -0.3715 | 85.2478 | 32.0011 | 16.7996 |
| loveda/D | WideSalienceMass_Projected | building | 31.7895 | 0.2270 | 31.9577 | 98.3713 | 0.0451 |
| loveda/D | WideSalienceMass_Projected | road | 49.1984 | -0.0767 | 50.0516 | 96.6512 | 8.8785 |
| loveda/D | WideSalienceMass_Projected | water | 54.9685 | -0.0326 | 58.9165 | 89.1340 | 12.2989 |
| loveda/D | WideSalienceMass_Projected | barren | 1.6333 | 0.0156 | 67.2924 | 1.6464 | 0.1499 |
| loveda/D | WideSalienceMass_Projected | tree | 48.1867 | -0.2852 | 92.7055 | 50.0857 | 4.9145 |
| loveda/D | WideSalienceMass_Projected | farm | 46.6497 | 0.0553 | 47.0597 | 98.1665 | 56.9136 |
| loveda/D | HardFixedMass_Projected | background | 43.0654 | 12.3720 | 73.0855 | 51.1825 | 31.3407 |
| loveda/D | HardFixedMass_Projected | building | 21.4535 | -10.1090 | 21.4535 | 100.0000 | 0.0682 |
| loveda/D | HardFixedMass_Projected | road | 51.4749 | 2.1998 | 52.4653 | 96.4625 | 8.4535 |
| loveda/D | HardFixedMass_Projected | water | 56.8435 | 1.8424 | 60.8174 | 89.6901 | 11.9888 |
| loveda/D | HardFixedMass_Projected | barren | 1.0187 | -0.5990 | 38.8208 | 1.0353 | 0.1634 |
| loveda/D | HardFixedMass_Projected | tree | 55.2633 | 6.7914 | 82.8156 | 62.4213 | 6.8563 |
| loveda/D | HardFixedMass_Projected | farm | 43.7104 | -2.8840 | 50.5923 | 76.2660 | 41.1291 |
| loveda/D | FineSalienceMass_Projected__ClassMean | background | 30.7034 | 0.0100 | 84.6687 | 32.5109 | 17.1840 |
| loveda/D | FineSalienceMass_Projected__ClassMean | building | 31.5297 | -0.0328 | 31.6614 | 98.6971 | 0.0456 |
| loveda/D | FineSalienceMass_Projected__ClassMean | road | 49.2777 | 0.0026 | 50.1345 | 96.6481 | 8.8635 |
| loveda/D | FineSalienceMass_Projected__ClassMean | water | 55.0005 | -0.0006 | 58.9489 | 89.1440 | 12.2935 |
| loveda/D | FineSalienceMass_Projected__ClassMean | barren | 1.6177 | 0.0000 | 66.7836 | 1.6308 | 0.1496 |
| loveda/D | FineSalienceMass_Projected__ClassMean | tree | 48.4701 | -0.0018 | 92.5751 | 50.4306 | 4.9553 |
| loveda/D | FineSalienceMass_Projected__ClassMean | farm | 46.5988 | 0.0044 | 47.1339 | 97.6214 | 56.5084 |
| loveda/D | FineSalienceMass_Projected__HardStrength | background | 30.5668 | -0.1266 | 84.5905 | 32.3692 | 17.1249 |
| loveda/D | FineSalienceMass_Projected__HardStrength | building | 31.6284 | 0.0659 | 31.7610 | 98.6971 | 0.0455 |
| loveda/D | FineSalienceMass_Projected__HardStrength | road | 49.2236 | -0.0515 | 50.0782 | 96.6491 | 8.8736 |
| loveda/D | FineSalienceMass_Projected__HardStrength | water | 54.9748 | -0.0263 | 58.9232 | 89.1352 | 12.2976 |
| loveda/D | FineSalienceMass_Projected__HardStrength | barren | 1.6301 | 0.0124 | 66.9946 | 1.6433 | 0.1503 |
| loveda/D | FineSalienceMass_Projected__HardStrength | tree | 48.3063 | -0.1656 | 92.6601 | 50.2283 | 4.9309 |
| loveda/D | FineSalienceMass_Projected__HardStrength | farm | 46.5199 | -0.0745 | 47.0609 | 97.5887 | 56.5772 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle0 | background | 30.7547 | 0.0613 | 84.5056 | 32.5925 | 17.2604 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle0 | building | 31.4642 | -0.0983 | 31.5954 | 98.6971 | 0.0457 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle0 | road | 49.2915 | 0.0164 | 50.1491 | 96.6471 | 8.8609 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle0 | water | 55.0084 | 0.0073 | 58.9584 | 89.1428 | 12.2913 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle0 | barren | 1.6177 | 0.0000 | 66.7836 | 1.6308 | 0.1496 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle0 | tree | 48.4677 | -0.0042 | 92.5662 | 50.4306 | 4.9558 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle0 | farm | 46.5691 | -0.0253 | 47.1331 | 97.4950 | 56.4363 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle1 | background | 30.7042 | 0.0108 | 84.6223 | 32.5186 | 17.1975 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle1 | building | 31.4642 | -0.0983 | 31.5954 | 98.6971 | 0.0457 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle1 | road | 49.2527 | -0.0224 | 50.1089 | 96.6471 | 8.8680 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle1 | water | 54.9915 | -0.0096 | 58.9396 | 89.1416 | 12.2951 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle1 | barren | 1.6223 | 0.0046 | 66.7196 | 1.6355 | 0.1502 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle1 | tree | 48.4851 | 0.0132 | 92.5909 | 50.4422 | 4.9556 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle1 | farm | 46.5740 | -0.0204 | 47.1224 | 97.5621 | 56.4879 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle2 | background | 30.7162 | 0.0228 | 84.6604 | 32.5265 | 17.1939 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle2 | building | 31.4642 | -0.0983 | 31.5954 | 98.6971 | 0.0457 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle2 | road | 49.2902 | 0.0151 | 50.1472 | 96.6491 | 8.8614 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle2 | water | 55.0011 | 0.0000 | 58.9493 | 89.1446 | 12.2935 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle2 | barren | 1.6185 | 0.0008 | 66.8154 | 1.6316 | 0.1496 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle2 | tree | 48.4726 | 0.0007 | 92.5771 | 50.4327 | 4.9554 |
| loveda/D | FineSalienceMass_Projected__AliasShuffle2 | farm | 46.6052 | 0.0108 | 47.1406 | 97.6214 | 56.5005 |
| loveda/D | WideSalienceMass_Projected__ClassMean | background | 30.7038 | 0.0104 | 84.7024 | 32.5063 | 17.1748 |
| loveda/D | WideSalienceMass_Projected__ClassMean | building | 31.5625 | 0.0000 | 31.6946 | 98.6971 | 0.0456 |
| loveda/D | WideSalienceMass_Projected__ClassMean | road | 49.2782 | 0.0031 | 50.1353 | 96.6471 | 8.8633 |
| loveda/D | WideSalienceMass_Projected__ClassMean | water | 55.0011 | 0.0000 | 58.9493 | 89.1446 | 12.2935 |
| loveda/D | WideSalienceMass_Projected__ClassMean | barren | 1.6177 | 0.0000 | 66.7836 | 1.6308 | 0.1496 |
| loveda/D | WideSalienceMass_Projected__ClassMean | tree | 48.4689 | -0.0030 | 92.5757 | 50.4291 | 4.9551 |
| loveda/D | WideSalienceMass_Projected__ClassMean | farm | 46.6101 | 0.0157 | 47.1391 | 97.6490 | 56.5182 |
| loveda/D | WideSalienceMass_Projected__HardStrength | background | 30.3297 | -0.3637 | 85.2151 | 32.0144 | 16.8131 |
| loveda/D | WideSalienceMass_Projected__HardStrength | building | 31.8612 | 0.2987 | 31.9958 | 98.6971 | 0.0452 |
| loveda/D | WideSalienceMass_Projected__HardStrength | road | 49.2051 | -0.0700 | 50.0591 | 96.6491 | 8.8770 |
| loveda/D | WideSalienceMass_Projected__HardStrength | water | 54.9633 | -0.0378 | 58.9115 | 89.1317 | 12.2996 |
| loveda/D | WideSalienceMass_Projected__HardStrength | barren | 1.6333 | 0.0156 | 67.2710 | 1.6464 | 0.1499 |
| loveda/D | WideSalienceMass_Projected__HardStrength | tree | 48.2225 | -0.2494 | 92.6872 | 50.1297 | 4.9198 |
| loveda/D | WideSalienceMass_Projected__HardStrength | farm | 46.6488 | 0.0544 | 47.0640 | 98.1441 | 56.8955 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle0 | background | 30.6916 | -0.0018 | 84.4746 | 32.5264 | 17.2317 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle0 | building | 31.5954 | 0.0329 | 31.7277 | 98.6971 | 0.0455 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle0 | road | 49.3011 | 0.0260 | 50.1593 | 96.6460 | 8.8590 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle0 | water | 55.0130 | 0.0119 | 58.9630 | 89.1446 | 12.2906 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle0 | barren | 1.6177 | 0.0000 | 66.8262 | 1.6308 | 0.1495 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle0 | tree | 48.4768 | 0.0049 | 92.5589 | 50.4427 | 4.9573 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle0 | farm | 46.5462 | -0.0482 | 47.1091 | 97.4973 | 56.4663 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle1 | background | 30.6497 | -0.0437 | 84.6709 | 32.4504 | 17.1516 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle1 | building | 31.6946 | 0.1321 | 31.8277 | 98.6971 | 0.0454 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle1 | road | 49.2467 | -0.0284 | 50.1030 | 96.6460 | 8.8689 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle1 | water | 54.9914 | -0.0097 | 58.9384 | 89.1440 | 12.2957 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle1 | barren | 1.6224 | 0.0047 | 66.9748 | 1.6355 | 0.1496 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle1 | tree | 48.4978 | 0.0259 | 92.5824 | 50.4584 | 4.9576 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle1 | farm | 46.5739 | -0.0205 | 47.1106 | 97.6123 | 56.5312 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle2 | background | 30.8316 | 0.1382 | 84.8415 | 32.6290 | 17.2113 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle2 | building | 31.4642 | -0.0983 | 31.5954 | 98.6971 | 0.0457 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle2 | road | 49.2967 | 0.0216 | 50.1539 | 96.6491 | 8.8602 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle2 | water | 55.0070 | 0.0059 | 58.9556 | 89.1457 | 12.2923 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle2 | barren | 1.6223 | 0.0046 | 66.7196 | 1.6355 | 0.1502 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle2 | tree | 48.4649 | -0.0070 | 92.5632 | 50.4285 | 4.9557 |
| loveda/D | WideSalienceMass_Projected__AliasShuffle2 | farm | 46.6994 | 0.1050 | 47.2098 | 97.7372 | 56.4846 |
| vaihingen/vaihingen | Geometry | impervious surface | 42.0941 | -17.2825 | 77.7719 | 47.8510 | 16.8072 |
| vaihingen/vaihingen | Geometry | building | 72.7601 | 5.2927 | 73.5287 | 98.5836 | 27.6605 |
| vaihingen/vaihingen | Geometry | low vegetation | 53.0114 | 7.2372 | 92.4475 | 55.4111 | 17.7225 |
| vaihingen/vaihingen | Geometry | tree | 68.6405 | 1.0271 | 82.7688 | 80.0845 | 19.9693 |
| vaihingen/vaihingen | Geometry | car | 10.3193 | -15.9787 | 10.3220 | 99.7520 | 17.8406 |
| vaihingen/vaihingen | NoAdmission_Exact | impervious surface | 57.3052 | -2.0714 | 73.8225 | 71.9196 | 26.6125 |
| vaihingen/vaihingen | NoAdmission_Exact | building | 65.7437 | -1.7237 | 65.8760 | 99.6954 | 31.2219 |
| vaihingen/vaihingen | NoAdmission_Exact | low vegetation | 43.7153 | -2.0589 | 96.1772 | 44.4883 | 13.6772 |
| vaihingen/vaihingen | NoAdmission_Exact | tree | 67.2612 | -0.3522 | 79.0950 | 81.8036 | 21.3454 |
| vaihingen/vaihingen | NoAdmission_Exact | car | 25.1095 | -1.1885 | 25.2570 | 97.7270 | 7.1430 |
| vaihingen/vaihingen | RivalFineHard_Exact | impervious surface | 58.8103 | -0.5663 | 74.7374 | 73.4017 | 26.8285 |
| vaihingen/vaihingen | RivalFineHard_Exact | building | 67.0764 | -0.3910 | 67.2196 | 99.6834 | 30.5942 |
| vaihingen/vaihingen | RivalFineHard_Exact | low vegetation | 44.9502 | -0.8240 | 95.9989 | 45.8084 | 14.1092 |
| vaihingen/vaihingen | RivalFineHard_Exact | tree | 67.5208 | -0.0926 | 78.7110 | 82.6067 | 21.6601 |
| vaihingen/vaihingen | RivalFineHard_Exact | car | 26.3654 | 0.0674 | 26.5220 | 97.8096 | 6.8081 |
| vaihingen/vaihingen | FineRivalProjected_Exact | impervious surface | 59.3766 | 0.0000 | 75.4775 | 73.5690 | 26.6259 |
| vaihingen/vaihingen | FineRivalProjected_Exact | building | 67.4674 | 0.0000 | 67.6072 | 99.6944 | 30.4222 |
| vaihingen/vaihingen | FineRivalProjected_Exact | low vegetation | 45.7742 | 0.0000 | 95.8882 | 46.6907 | 14.3975 |
| vaihingen/vaihingen | FineRivalProjected_Exact | tree | 67.6134 | 0.0000 | 78.6701 | 82.7907 | 21.7196 |
| vaihingen/vaihingen | FineRivalProjected_Exact | car | 26.2980 | 0.0000 | 26.4463 | 97.9130 | 6.8348 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | impervious surface | 59.3423 | -0.0343 | 76.2324 | 72.8142 | 26.0918 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | building | 67.4593 | -0.0081 | 67.5948 | 99.7037 | 30.4306 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | low vegetation | 46.6048 | 0.8306 | 95.6768 | 47.6074 | 14.7126 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | tree | 67.6624 | 0.0490 | 78.6651 | 82.8697 | 21.7417 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | car | 25.6512 | -0.6468 | 25.7806 | 98.0808 | 7.0233 |
| vaihingen/vaihingen | FineSalienceMass_Projected | impervious surface | 59.3461 | -0.0305 | 75.4555 | 73.5431 | 26.6243 |
| vaihingen/vaihingen | FineSalienceMass_Projected | building | 67.4409 | -0.0265 | 67.5805 | 99.6947 | 30.4342 |
| vaihingen/vaihingen | FineSalienceMass_Projected | low vegetation | 45.7435 | -0.0307 | 95.8906 | 46.6581 | 14.3871 |
| vaihingen/vaihingen | FineSalienceMass_Projected | tree | 67.6054 | -0.0080 | 78.6731 | 82.7754 | 21.7148 |
| vaihingen/vaihingen | FineSalienceMass_Projected | car | 26.2773 | -0.0207 | 26.4259 | 97.9052 | 6.8395 |
| vaihingen/vaihingen | WideSalienceMass_Projected | impervious surface | 59.3333 | -0.0433 | 75.4451 | 73.5334 | 26.6245 |
| vaihingen/vaihingen | WideSalienceMass_Projected | building | 67.4251 | -0.0423 | 67.5646 | 99.6947 | 30.4414 |
| vaihingen/vaihingen | WideSalienceMass_Projected | low vegetation | 45.7387 | -0.0355 | 95.9004 | 46.6509 | 14.3834 |
| vaihingen/vaihingen | WideSalienceMass_Projected | tree | 67.6061 | -0.0073 | 78.6791 | 82.7699 | 21.7117 |
| vaihingen/vaihingen | WideSalienceMass_Projected | car | 26.2792 | -0.0188 | 26.4277 | 97.9052 | 6.8390 |
| vaihingen/vaihingen | HardFixedMass_Projected | impervious surface | 61.9472 | 2.5706 | 79.0381 | 74.1255 | 25.6188 |
| vaihingen/vaihingen | HardFixedMass_Projected | building | 70.9264 | 3.4590 | 71.1124 | 99.6325 | 28.9046 |
| vaihingen/vaihingen | HardFixedMass_Projected | low vegetation | 50.7946 | 5.0204 | 93.5530 | 52.6371 | 16.6363 |
| vaihingen/vaihingen | HardFixedMass_Projected | tree | 67.9433 | 0.3299 | 78.3806 | 83.6127 | 22.0163 |
| vaihingen/vaihingen | HardFixedMass_Projected | car | 26.5629 | 0.2649 | 26.6657 | 98.5690 | 6.8240 |
| vaihingen/vaihingen | FineSalienceMass_Projected__ClassMean | impervious surface | 59.3814 | 0.0048 | 75.4856 | 73.5688 | 26.6230 |
| vaihingen/vaihingen | FineSalienceMass_Projected__ClassMean | building | 67.4715 | 0.0041 | 67.6113 | 99.6944 | 30.4203 |
| vaihingen/vaihingen | FineSalienceMass_Projected__ClassMean | low vegetation | 45.7813 | 0.0071 | 95.8839 | 46.6991 | 14.4008 |
| vaihingen/vaihingen | FineSalienceMass_Projected__ClassMean | tree | 67.6127 | -0.0007 | 78.6681 | 82.7918 | 21.7205 |
| vaihingen/vaihingen | FineSalienceMass_Projected__ClassMean | car | 26.2955 | -0.0025 | 26.4437 | 97.9130 | 6.8355 |
| vaihingen/vaihingen | FineSalienceMass_Projected__HardStrength | impervious surface | 59.3494 | -0.0272 | 75.4590 | 73.5449 | 26.6237 |
| vaihingen/vaihingen | FineSalienceMass_Projected__HardStrength | building | 67.4414 | -0.0260 | 67.5809 | 99.6947 | 30.4340 |
| vaihingen/vaihingen | FineSalienceMass_Projected__HardStrength | low vegetation | 45.7511 | -0.0231 | 95.8956 | 46.6649 | 14.3885 |
| vaihingen/vaihingen | FineSalienceMass_Projected__HardStrength | tree | 67.6091 | -0.0043 | 78.6761 | 82.7775 | 21.7145 |
| vaihingen/vaihingen | FineSalienceMass_Projected__HardStrength | car | 26.2800 | -0.0180 | 26.4282 | 97.9104 | 6.8393 |
| vaihingen/vaihingen | FineSalienceMass_Projected__AliasShuffle0 | impervious surface | 59.3826 | 0.0060 | 75.4821 | 73.5739 | 26.6261 |
| vaihingen/vaihingen | FineSalienceMass_Projected__AliasShuffle0 | building | 67.4790 | 0.0116 | 67.6188 | 99.6944 | 30.4169 |
| vaihingen/vaihingen | FineSalienceMass_Projected__AliasShuffle0 | low vegetation | 45.7791 | 0.0049 | 95.8911 | 46.6951 | 14.3984 |
| vaihingen/vaihingen | FineSalienceMass_Projected__AliasShuffle0 | tree | 67.6162 | 0.0028 | 78.6645 | 82.8011 | 21.7239 |
| vaihingen/vaihingen | FineSalienceMass_Projected__AliasShuffle0 | car | 26.2984 | 0.0004 | 26.4466 | 97.9130 | 6.8347 |
| vaihingen/vaihingen | FineSalienceMass_Projected__AliasShuffle1 | impervious surface | 59.3830 | 0.0064 | 75.4850 | 73.5718 | 26.6243 |
| vaihingen/vaihingen | FineSalienceMass_Projected__AliasShuffle1 | building | 67.4795 | 0.0121 | 67.6193 | 99.6944 | 30.4167 |
| vaihingen/vaihingen | FineSalienceMass_Projected__AliasShuffle1 | low vegetation | 45.7784 | 0.0042 | 95.8845 | 46.6959 | 14.3997 |
| vaihingen/vaihingen | FineSalienceMass_Projected__AliasShuffle1 | tree | 67.6157 | 0.0023 | 78.6638 | 82.8011 | 21.7241 |
| vaihingen/vaihingen | FineSalienceMass_Projected__AliasShuffle1 | car | 26.2962 | -0.0018 | 26.4444 | 97.9130 | 6.8353 |
| vaihingen/vaihingen | FineSalienceMass_Projected__AliasShuffle2 | impervious surface | 59.3768 | 0.0002 | 75.4796 | 73.5674 | 26.6246 |
| vaihingen/vaihingen | FineSalienceMass_Projected__AliasShuffle2 | building | 67.4589 | -0.0085 | 67.5987 | 99.6944 | 30.4260 |
| vaihingen/vaihingen | FineSalienceMass_Projected__AliasShuffle2 | low vegetation | 45.7713 | -0.0029 | 95.8883 | 46.6876 | 14.3966 |
| vaihingen/vaihingen | FineSalienceMass_Projected__AliasShuffle2 | tree | 67.6150 | 0.0016 | 78.6712 | 82.7918 | 21.7196 |
| vaihingen/vaihingen | FineSalienceMass_Projected__AliasShuffle2 | car | 26.3032 | 0.0052 | 26.4516 | 97.9104 | 6.8332 |
| vaihingen/vaihingen | WideSalienceMass_Projected__ClassMean | impervious surface | 59.3808 | 0.0042 | 75.4864 | 73.5670 | 26.6221 |
| vaihingen/vaihingen | WideSalienceMass_Projected__ClassMean | building | 67.4689 | 0.0015 | 67.6087 | 99.6944 | 30.4214 |
| vaihingen/vaihingen | WideSalienceMass_Projected__ClassMean | low vegetation | 45.7806 | 0.0064 | 95.8841 | 46.6983 | 14.4005 |
| vaihingen/vaihingen | WideSalienceMass_Projected__ClassMean | tree | 67.6125 | -0.0009 | 78.6678 | 82.7918 | 21.7206 |
| vaihingen/vaihingen | WideSalienceMass_Projected__ClassMean | car | 26.2956 | -0.0024 | 26.4439 | 97.9130 | 6.8354 |
| vaihingen/vaihingen | WideSalienceMass_Projected__HardStrength | impervious surface | 59.3376 | -0.0390 | 75.4502 | 73.5351 | 26.6233 |
| vaihingen/vaihingen | WideSalienceMass_Projected__HardStrength | building | 67.4308 | -0.0366 | 67.5704 | 99.6947 | 30.4388 |
| vaihingen/vaihingen | WideSalienceMass_Projected__HardStrength | low vegetation | 45.7398 | -0.0344 | 95.8971 | 46.6528 | 14.3845 |
| vaihingen/vaihingen | WideSalienceMass_Projected__HardStrength | tree | 67.6075 | -0.0059 | 78.6790 | 82.7719 | 21.7123 |
| vaihingen/vaihingen | WideSalienceMass_Projected__HardStrength | car | 26.2729 | -0.0251 | 26.4210 | 97.9104 | 6.8411 |
| vaihingen/vaihingen | WideSalienceMass_Projected__AliasShuffle0 | impervious surface | 59.3769 | 0.0003 | 75.4796 | 73.5676 | 26.6247 |
| vaihingen/vaihingen | WideSalienceMass_Projected__AliasShuffle0 | building | 67.4676 | 0.0002 | 67.6074 | 99.6944 | 30.4221 |
| vaihingen/vaihingen | WideSalienceMass_Projected__AliasShuffle0 | low vegetation | 45.7755 | 0.0013 | 95.8902 | 46.6915 | 14.3975 |
| vaihingen/vaihingen | WideSalienceMass_Projected__AliasShuffle0 | tree | 67.6117 | -0.0017 | 78.6641 | 82.7948 | 21.7224 |
| vaihingen/vaihingen | WideSalienceMass_Projected__AliasShuffle0 | car | 26.3033 | 0.0053 | 26.4516 | 97.9130 | 6.8334 |
| vaihingen/vaihingen | WideSalienceMass_Projected__AliasShuffle1 | impervious surface | 59.3820 | 0.0054 | 75.4943 | 73.5615 | 26.6173 |
| vaihingen/vaihingen | WideSalienceMass_Projected__AliasShuffle1 | building | 67.4650 | -0.0024 | 67.6048 | 99.6944 | 30.4232 |
| vaihingen/vaihingen | WideSalienceMass_Projected__AliasShuffle1 | low vegetation | 45.7749 | 0.0007 | 95.8864 | 46.6918 | 14.3981 |
| vaihingen/vaihingen | WideSalienceMass_Projected__AliasShuffle1 | tree | 67.6131 | -0.0003 | 78.6578 | 82.8038 | 21.7265 |
| vaihingen/vaihingen | WideSalienceMass_Projected__AliasShuffle1 | car | 26.2976 | -0.0004 | 26.4459 | 97.9130 | 6.8349 |
| vaihingen/vaihingen | WideSalienceMass_Projected__AliasShuffle2 | impervious surface | 59.3701 | -0.0065 | 75.4805 | 73.5562 | 26.6202 |
| vaihingen/vaihingen | WideSalienceMass_Projected__AliasShuffle2 | building | 67.4440 | -0.0234 | 67.5836 | 99.6947 | 30.4328 |
| vaihingen/vaihingen | WideSalienceMass_Projected__AliasShuffle2 | low vegetation | 45.7632 | -0.0110 | 95.8838 | 46.6802 | 14.3950 |
| vaihingen/vaihingen | WideSalienceMass_Projected__AliasShuffle2 | tree | 67.6116 | -0.0018 | 78.6714 | 82.7865 | 21.7182 |
| vaihingen/vaihingen | WideSalienceMass_Projected__AliasShuffle2 | car | 26.3010 | 0.0030 | 26.4494 | 97.9104 | 6.8338 |
| landcoverai/landcoverai | Geometry | background | 82.6090 | -5.6280 | 96.6505 | 85.0437 | 59.6226 |
| landcoverai/landcoverai | Geometry | building | 34.9562 | -9.6375 | 35.0436 | 99.2919 | 4.2927 |
| landcoverai/landcoverai | Geometry | woodland | 78.3638 | -3.1775 | 86.4992 | 89.2841 | 22.0960 |
| landcoverai/landcoverai | Geometry | water | 93.6635 | -3.9444 | 93.6635 | 100.0000 | 8.8309 |
| landcoverai/landcoverai | Geometry | road | 14.9318 | -12.0061 | 15.6288 | 77.0019 | 5.1578 |
| landcoverai/landcoverai | NoAdmission_Exact | background | 87.4477 | -0.7893 | 93.8549 | 92.7587 | 66.9686 |
| landcoverai/landcoverai | NoAdmission_Exact | building | 44.4672 | -0.1265 | 44.8124 | 98.2973 | 3.3233 |
| landcoverai/landcoverai | NoAdmission_Exact | woodland | 78.6026 | -2.9387 | 94.9565 | 82.0272 | 18.4920 |
| landcoverai/landcoverai | NoAdmission_Exact | water | 97.6178 | 0.0099 | 97.6178 | 100.0000 | 8.4732 |
| landcoverai/landcoverai | NoAdmission_Exact | road | 26.3947 | -0.5432 | 28.8528 | 75.5990 | 2.7429 |
| landcoverai/landcoverai | RivalFineHard_Exact | background | 87.9828 | -0.2542 | 94.6196 | 92.6163 | 66.3254 |
| landcoverai/landcoverai | RivalFineHard_Exact | building | 44.5590 | -0.0347 | 44.9037 | 98.3067 | 3.3169 |
| landcoverai/landcoverai | RivalFineHard_Exact | woodland | 80.5571 | -0.9842 | 94.4300 | 84.5759 | 19.1729 |
| landcoverai/landcoverai | RivalFineHard_Exact | water | 97.6134 | 0.0055 | 97.6134 | 100.0000 | 8.4735 |
| landcoverai/landcoverai | RivalFineHard_Exact | road | 26.7045 | -0.2334 | 29.2139 | 75.6627 | 2.7113 |
| landcoverai/landcoverai | FineRivalProjected_Exact | background | 88.2370 | 0.0000 | 95.1004 | 92.4393 | 65.8639 |
| landcoverai/landcoverai | FineRivalProjected_Exact | building | 44.5937 | 0.0000 | 44.9185 | 98.4043 | 3.3191 |
| landcoverai/landcoverai | FineRivalProjected_Exact | woodland | 81.5413 | 0.0000 | 93.8335 | 86.1583 | 19.6558 |
| landcoverai/landcoverai | FineRivalProjected_Exact | water | 97.6079 | 0.0000 | 97.6079 | 100.0000 | 8.4740 |
| landcoverai/landcoverai | FineRivalProjected_Exact | road | 26.9379 | 0.0000 | 29.4886 | 75.6946 | 2.6872 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | background | 88.0617 | -0.1753 | 94.8162 | 92.5159 | 66.1161 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | building | 44.5675 | -0.0262 | 44.8926 | 98.4012 | 3.3209 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | woodland | 80.8303 | -0.7110 | 94.0020 | 85.2259 | 19.4082 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | water | 97.6173 | 0.0094 | 97.6173 | 100.0000 | 8.4732 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | road | 26.9705 | 0.0326 | 29.5339 | 75.6536 | 2.6816 |
| landcoverai/landcoverai | FineSalienceMass_Projected | background | 88.2227 | -0.0143 | 95.0758 | 92.4468 | 65.8863 |
| landcoverai/landcoverai | FineSalienceMass_Projected | building | 44.5958 | 0.0021 | 44.9226 | 98.3949 | 3.3185 |
| landcoverai/landcoverai | FineSalienceMass_Projected | woodland | 81.4856 | -0.0557 | 93.8546 | 86.0783 | 19.6332 |
| landcoverai/landcoverai | FineSalienceMass_Projected | water | 97.6079 | 0.0000 | 97.6079 | 100.0000 | 8.4740 |
| landcoverai/landcoverai | FineSalienceMass_Projected | road | 26.9296 | -0.0083 | 29.4787 | 75.6946 | 2.6881 |
| landcoverai/landcoverai | WideSalienceMass_Projected | background | 88.2204 | -0.0166 | 95.0699 | 92.4499 | 65.8926 |
| landcoverai/landcoverai | WideSalienceMass_Projected | building | 44.5990 | 0.0053 | 44.9258 | 98.3949 | 3.3182 |
| landcoverai/landcoverai | WideSalienceMass_Projected | woodland | 81.4734 | -0.0679 | 93.8617 | 86.0587 | 19.6272 |
| landcoverai/landcoverai | WideSalienceMass_Projected | water | 97.6090 | 0.0011 | 97.6090 | 100.0000 | 8.4739 |
| landcoverai/landcoverai | WideSalienceMass_Projected | road | 26.9296 | -0.0083 | 29.4787 | 75.6946 | 2.6881 |
| landcoverai/landcoverai | HardFixedMass_Projected | background | 88.8974 | 0.6604 | 96.1081 | 92.2171 | 65.0167 |
| landcoverai/landcoverai | HardFixedMass_Projected | building | 44.8945 | 0.3008 | 45.2158 | 98.4421 | 3.2985 |
| landcoverai/landcoverai | HardFixedMass_Projected | woodland | 84.5119 | 2.9706 | 92.4991 | 90.7298 | 20.9973 |
| landcoverai/landcoverai | HardFixedMass_Projected | water | 94.2712 | -3.3367 | 97.5331 | 96.5739 | 8.1900 |
| landcoverai/landcoverai | HardFixedMass_Projected | road | 28.8014 | 1.8635 | 31.7340 | 75.7083 | 2.4975 |
| landcoverai/landcoverai | FineSalienceMass_Projected__ClassMean | background | 88.2369 | -0.0001 | 95.0990 | 92.4405 | 65.8658 |
| landcoverai/landcoverai | FineSalienceMass_Projected__ClassMean | building | 44.5975 | 0.0038 | 44.9224 | 98.4043 | 3.3188 |
| landcoverai/landcoverai | FineSalienceMass_Projected__ClassMean | woodland | 81.5376 | -0.0037 | 93.8341 | 86.1536 | 19.6546 |
| landcoverai/landcoverai | FineSalienceMass_Projected__ClassMean | water | 97.6079 | 0.0000 | 97.6079 | 100.0000 | 8.4740 |
| landcoverai/landcoverai | FineSalienceMass_Projected__ClassMean | road | 26.9410 | 0.0031 | 29.4923 | 75.6946 | 2.6868 |
| landcoverai/landcoverai | FineSalienceMass_Projected__HardStrength | background | 88.2231 | -0.0139 | 95.0782 | 92.4450 | 65.8834 |
| landcoverai/landcoverai | FineSalienceMass_Projected__HardStrength | building | 44.5915 | -0.0022 | 44.9176 | 98.3980 | 3.3189 |
| landcoverai/landcoverai | FineSalienceMass_Projected__HardStrength | woodland | 81.4919 | -0.0494 | 93.8537 | 86.0861 | 19.6351 |
| landcoverai/landcoverai | FineSalienceMass_Projected__HardStrength | water | 97.6085 | 0.0006 | 97.6085 | 100.0000 | 8.4740 |
| landcoverai/landcoverai | FineSalienceMass_Projected__HardStrength | road | 26.9244 | -0.0135 | 29.4724 | 75.6946 | 2.6886 |
| landcoverai/landcoverai | FineSalienceMass_Projected__AliasShuffle0 | background | 88.2356 | -0.0014 | 95.0960 | 92.4419 | 65.8688 |
| landcoverai/landcoverai | FineSalienceMass_Projected__AliasShuffle0 | building | 44.6006 | 0.0069 | 44.9261 | 98.4012 | 3.3184 |
| landcoverai/landcoverai | FineSalienceMass_Projected__AliasShuffle0 | woodland | 81.5309 | -0.0104 | 93.8367 | 86.1440 | 19.6519 |
| landcoverai/landcoverai | FineSalienceMass_Projected__AliasShuffle0 | water | 97.6079 | 0.0000 | 97.6079 | 100.0000 | 8.4740 |
| landcoverai/landcoverai | FineSalienceMass_Projected__AliasShuffle0 | road | 26.9405 | 0.0026 | 29.4917 | 75.6946 | 2.6869 |
| landcoverai/landcoverai | FineSalienceMass_Projected__AliasShuffle1 | background | 88.2395 | 0.0025 | 95.1016 | 92.4408 | 65.8642 |
| landcoverai/landcoverai | FineSalienceMass_Projected__AliasShuffle1 | building | 44.5988 | 0.0051 | 44.9237 | 98.4043 | 3.3187 |
| landcoverai/landcoverai | FineSalienceMass_Projected__AliasShuffle1 | woodland | 81.5454 | 0.0041 | 93.8347 | 86.1619 | 19.6564 |
| landcoverai/landcoverai | FineSalienceMass_Projected__AliasShuffle1 | water | 97.6079 | 0.0000 | 97.6079 | 100.0000 | 8.4740 |
| landcoverai/landcoverai | FineSalienceMass_Projected__AliasShuffle1 | road | 26.9418 | 0.0039 | 29.4933 | 75.6946 | 2.6867 |
| landcoverai/landcoverai | FineSalienceMass_Projected__AliasShuffle2 | background | 88.2323 | -0.0047 | 95.0884 | 92.4454 | 65.8766 |
| landcoverai/landcoverai | FineSalienceMass_Projected__AliasShuffle2 | building | 44.6054 | 0.0117 | 44.9323 | 98.3949 | 3.3177 |
| landcoverai/landcoverai | FineSalienceMass_Projected__AliasShuffle2 | woodland | 81.5142 | -0.0271 | 93.8439 | 86.1193 | 19.6447 |
| landcoverai/landcoverai | FineSalienceMass_Projected__AliasShuffle2 | water | 97.6079 | 0.0000 | 97.6079 | 100.0000 | 8.4740 |
| landcoverai/landcoverai | FineSalienceMass_Projected__AliasShuffle2 | road | 26.9401 | 0.0022 | 29.4912 | 75.6946 | 2.6869 |
| landcoverai/landcoverai | WideSalienceMass_Projected__ClassMean | background | 88.2363 | -0.0007 | 95.0973 | 92.4414 | 65.8676 |
| landcoverai/landcoverai | WideSalienceMass_Projected__ClassMean | building | 44.5955 | 0.0018 | 44.9210 | 98.4012 | 3.3188 |
| landcoverai/landcoverai | WideSalienceMass_Projected__ClassMean | woodland | 81.5346 | -0.0067 | 93.8365 | 86.1483 | 19.6529 |
| landcoverai/landcoverai | WideSalienceMass_Projected__ClassMean | water | 97.6079 | 0.0000 | 97.6079 | 100.0000 | 8.4740 |
| landcoverai/landcoverai | WideSalienceMass_Projected__ClassMean | road | 26.9418 | 0.0039 | 29.4933 | 75.6946 | 2.6867 |
| landcoverai/landcoverai | WideSalienceMass_Projected__HardStrength | background | 88.2208 | -0.0162 | 95.0736 | 92.4468 | 65.8878 |
| landcoverai/landcoverai | WideSalienceMass_Projected__HardStrength | building | 44.5909 | -0.0028 | 44.9170 | 98.3980 | 3.3190 |
| landcoverai/landcoverai | WideSalienceMass_Projected__HardStrength | woodland | 81.4822 | -0.0591 | 93.8589 | 86.0710 | 19.6306 |
| landcoverai/landcoverai | WideSalienceMass_Projected__HardStrength | water | 97.6096 | 0.0017 | 97.6096 | 100.0000 | 8.4739 |
| landcoverai/landcoverai | WideSalienceMass_Projected__HardStrength | road | 26.9235 | -0.0144 | 29.4713 | 75.6946 | 2.6887 |
| landcoverai/landcoverai | WideSalienceMass_Projected__AliasShuffle0 | background | 88.2335 | -0.0035 | 95.0930 | 92.4424 | 65.8712 |
| landcoverai/landcoverai | WideSalienceMass_Projected__AliasShuffle0 | building | 44.5974 | 0.0037 | 44.9229 | 98.4012 | 3.3186 |
| landcoverai/landcoverai | WideSalienceMass_Projected__AliasShuffle0 | woodland | 81.5241 | -0.0172 | 93.8392 | 86.1342 | 19.6491 |
| landcoverai/landcoverai | WideSalienceMass_Projected__AliasShuffle0 | water | 97.6079 | 0.0000 | 97.6079 | 100.0000 | 8.4740 |
| landcoverai/landcoverai | WideSalienceMass_Projected__AliasShuffle0 | road | 26.9397 | 0.0018 | 29.4907 | 75.6946 | 2.6870 |
| landcoverai/landcoverai | WideSalienceMass_Projected__AliasShuffle1 | background | 88.2399 | 0.0029 | 95.1021 | 92.4408 | 65.8638 |
| landcoverai/landcoverai | WideSalienceMass_Projected__AliasShuffle1 | building | 44.5995 | 0.0058 | 44.9244 | 98.4043 | 3.3186 |
| landcoverai/landcoverai | WideSalienceMass_Projected__AliasShuffle1 | woodland | 81.5472 | 0.0059 | 93.8352 | 86.1634 | 19.6566 |
| landcoverai/landcoverai | WideSalienceMass_Projected__AliasShuffle1 | water | 97.6085 | 0.0006 | 97.6085 | 100.0000 | 8.4740 |
| landcoverai/landcoverai | WideSalienceMass_Projected__AliasShuffle1 | road | 26.9401 | 0.0022 | 29.4912 | 75.6946 | 2.6869 |
| landcoverai/landcoverai | WideSalienceMass_Projected__AliasShuffle2 | background | 88.2284 | -0.0086 | 95.0826 | 92.4467 | 65.8815 |
| landcoverai/landcoverai | WideSalienceMass_Projected__AliasShuffle2 | building | 44.6066 | 0.0129 | 44.9336 | 98.3949 | 3.3176 |
| landcoverai/landcoverai | WideSalienceMass_Projected__AliasShuffle2 | woodland | 81.4987 | -0.0426 | 93.8458 | 86.1004 | 19.6400 |
| landcoverai/landcoverai | WideSalienceMass_Projected__AliasShuffle2 | water | 97.6085 | 0.0006 | 97.6085 | 100.0000 | 8.4740 |
| landcoverai/landcoverai | WideSalienceMass_Projected__AliasShuffle2 | road | 26.9410 | 0.0031 | 29.4923 | 75.6946 | 2.6868 |
| flair1/flair1 | Geometry | building | 49.6979 | -7.7062 | 50.7258 | 96.0825 | 13.5991 |
| flair1/flair1 | Geometry | pervious surface | 57.0621 | 10.2549 | 92.1299 | 59.9861 | 11.1728 |
| flair1/flair1 | Geometry | impervious surface | 52.6509 | -0.2497 | 62.6245 | 76.7765 | 20.0846 |
| flair1/flair1 | Geometry | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.4411 |
| flair1/flair1 | Geometry | water | 73.2328 | 8.4491 | 77.1432 | 93.5263 | 5.3781 |
| flair1/flair1 | Geometry | coniferous | 43.0233 | 0.2968 | 61.7114 | 58.6897 | 0.5346 |
| flair1/flair1 | Geometry | deciduous | 54.0229 | -6.8819 | 78.9723 | 63.0995 | 13.6004 |
| flair1/flair1 | Geometry | brushwood | 18.6136 | 2.9830 | 23.4212 | 47.5559 | 9.9866 |
| flair1/flair1 | Geometry | vineyard | -- | -- | 0.0000 | 0.0000 | 0.0000 |
| flair1/flair1 | Geometry | herbaceous vegetation | 60.2329 | 2.6126 | 94.3169 | 62.5013 | 21.2289 |
| flair1/flair1 | Geometry | agricultural land | 18.7339 | 5.3034 | 21.6468 | 58.1977 | 0.8198 |
| flair1/flair1 | Geometry | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1540 |
| flair1/flair1 | NoAdmission_Exact | building | 56.7755 | -0.6286 | 58.0396 | 96.3057 | 11.9131 |
| flair1/flair1 | NoAdmission_Exact | pervious surface | 47.2753 | 0.4681 | 91.6998 | 49.3887 | 9.2421 |
| flair1/flair1 | NoAdmission_Exact | impervious surface | 51.9858 | -0.9148 | 59.7781 | 79.9522 | 21.9112 |
| flair1/flair1 | NoAdmission_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1475 |
| flair1/flair1 | NoAdmission_Exact | water | 61.4565 | -3.3272 | 63.2502 | 95.5889 | 6.7041 |
| flair1/flair1 | NoAdmission_Exact | coniferous | 43.5282 | 0.8017 | 65.7321 | 56.3052 | 0.4815 |
| flair1/flair1 | NoAdmission_Exact | deciduous | 59.5237 | -1.3811 | 78.8015 | 70.8722 | 15.3089 |
| flair1/flair1 | NoAdmission_Exact | brushwood | 12.3287 | -3.3019 | 26.2213 | 18.8771 | 3.5408 |
| flair1/flair1 | NoAdmission_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0016 |
| flair1/flair1 | NoAdmission_Exact | herbaceous vegetation | 57.4725 | -0.1478 | 90.8252 | 61.0148 | 21.5207 |
| flair1/flair1 | NoAdmission_Exact | agricultural land | 12.9546 | -0.4759 | 13.9145 | 65.2534 | 1.4300 |
| flair1/flair1 | NoAdmission_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7986 |
| flair1/flair1 | RivalFineHard_Exact | building | 57.1752 | -0.2289 | 58.4382 | 96.3576 | 11.8382 |
| flair1/flair1 | RivalFineHard_Exact | pervious surface | 46.6274 | -0.1798 | 91.7085 | 48.6795 | 9.1085 |
| flair1/flair1 | RivalFineHard_Exact | impervious surface | 52.5746 | -0.3260 | 60.4793 | 80.0896 | 21.6944 |
| flair1/flair1 | RivalFineHard_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1744 |
| flair1/flair1 | RivalFineHard_Exact | water | 63.6878 | -1.0959 | 65.7363 | 95.3351 | 6.4334 |
| flair1/flair1 | RivalFineHard_Exact | coniferous | 42.7771 | 0.0506 | 63.1055 | 57.0434 | 0.5081 |
| flair1/flair1 | RivalFineHard_Exact | deciduous | 60.3808 | -0.5240 | 78.8158 | 72.0787 | 15.5666 |
| flair1/flair1 | RivalFineHard_Exact | brushwood | 14.8835 | -0.7471 | 28.7712 | 23.5675 | 4.0288 |
| flair1/flair1 | RivalFineHard_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0207 |
| flair1/flair1 | RivalFineHard_Exact | herbaceous vegetation | 57.5245 | -0.0958 | 91.4034 | 60.8148 | 21.3145 |
| flair1/flair1 | RivalFineHard_Exact | agricultural land | 13.1175 | -0.3130 | 14.1254 | 64.7685 | 1.3981 |
| flair1/flair1 | RivalFineHard_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9143 |
| flair1/flair1 | FineRivalProjected_Exact | building | 57.4041 | 0.0000 | 58.6643 | 96.3928 | 11.7969 |
| flair1/flair1 | FineRivalProjected_Exact | pervious surface | 46.8072 | 0.0000 | 91.9102 | 48.8185 | 9.1145 |
| flair1/flair1 | FineRivalProjected_Exact | impervious surface | 52.9006 | 0.0000 | 60.8089 | 80.2670 | 21.6246 |
| flair1/flair1 | FineRivalProjected_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0982 |
| flair1/flair1 | FineRivalProjected_Exact | water | 64.7837 | 0.0000 | 66.9300 | 95.2835 | 6.3152 |
| flair1/flair1 | FineRivalProjected_Exact | coniferous | 42.7265 | 0.0000 | 62.4459 | 57.5017 | 0.5176 |
| flair1/flair1 | FineRivalProjected_Exact | deciduous | 60.9048 | 0.0000 | 78.7253 | 72.9040 | 15.7630 |
| flair1/flair1 | FineRivalProjected_Exact | brushwood | 15.6306 | 0.0000 | 29.7469 | 24.7769 | 4.0967 |
| flair1/flair1 | FineRivalProjected_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0203 |
| flair1/flair1 | FineRivalProjected_Exact | herbaceous vegetation | 57.6203 | 0.0000 | 91.5155 | 60.8721 | 21.3084 |
| flair1/flair1 | FineRivalProjected_Exact | agricultural land | 13.4305 | 0.0000 | 14.5039 | 64.4712 | 1.3554 |
| flair1/flair1 | FineRivalProjected_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9893 |
| flair1/flair1 | FineBudgetOnly_Exact | building | 57.3732 | -0.0309 | 58.6294 | 96.4001 | 11.8048 |
| flair1/flair1 | FineBudgetOnly_Exact | pervious surface | 47.2072 | 0.4000 | 92.1302 | 49.1908 | 9.1621 |
| flair1/flair1 | FineBudgetOnly_Exact | impervious surface | 52.9893 | 0.0887 | 60.9714 | 80.1886 | 21.5459 |
| flair1/flair1 | FineBudgetOnly_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0326 |
| flair1/flair1 | FineBudgetOnly_Exact | water | 65.2431 | 0.4594 | 67.4205 | 95.2835 | 6.2693 |
| flair1/flair1 | FineBudgetOnly_Exact | coniferous | 42.6496 | -0.0769 | 62.2418 | 57.5356 | 0.5196 |
| flair1/flair1 | FineBudgetOnly_Exact | deciduous | 60.7870 | -0.1178 | 78.7072 | 72.7507 | 15.7335 |
| flair1/flair1 | FineBudgetOnly_Exact | brushwood | 15.8816 | 0.2510 | 29.5663 | 25.5470 | 4.2498 |
| flair1/flair1 | FineBudgetOnly_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0168 |
| flair1/flair1 | FineBudgetOnly_Exact | herbaceous vegetation | 57.7631 | 0.1428 | 91.5243 | 61.0276 | 21.3608 |
| flair1/flair1 | FineBudgetOnly_Exact | agricultural land | 13.5617 | 0.1312 | 14.6595 | 64.4243 | 1.3400 |
| flair1/flair1 | FineBudgetOnly_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9648 |
| flair1/flair1 | FineSalienceMass_Projected | building | 57.3928 | -0.0113 | 58.6547 | 96.3868 | 11.7980 |
| flair1/flair1 | FineSalienceMass_Projected | pervious surface | 46.8081 | 0.0009 | 91.9075 | 48.8202 | 9.1151 |
| flair1/flair1 | FineSalienceMass_Projected | impervious surface | 52.8827 | -0.0179 | 60.7893 | 80.2600 | 21.6297 |
| flair1/flair1 | FineSalienceMass_Projected | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1012 |
| flair1/flair1 | FineSalienceMass_Projected | water | 64.7199 | -0.0638 | 66.8550 | 95.2974 | 6.3232 |
| flair1/flair1 | FineSalienceMass_Projected | coniferous | 42.7508 | 0.0243 | 62.4977 | 57.5017 | 0.5172 |
| flair1/flair1 | FineSalienceMass_Projected | deciduous | 60.8920 | -0.0128 | 78.7297 | 72.8819 | 15.7573 |
| flair1/flair1 | FineSalienceMass_Projected | brushwood | 15.5622 | -0.0684 | 29.6837 | 24.6489 | 4.0842 |
| flair1/flair1 | FineSalienceMass_Projected | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0200 |
| flair1/flair1 | FineSalienceMass_Projected | herbaceous vegetation | 57.6244 | 0.0041 | 91.5074 | 60.8803 | 21.3132 |
| flair1/flair1 | FineSalienceMass_Projected | agricultural land | 13.4337 | 0.0032 | 14.5069 | 64.4869 | 1.3554 |
| flair1/flair1 | FineSalienceMass_Projected | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9854 |
| flair1/flair1 | WideSalienceMass_Projected | building | 57.3950 | -0.0091 | 58.6568 | 96.3875 | 11.7977 |
| flair1/flair1 | WideSalienceMass_Projected | pervious surface | 46.8117 | 0.0045 | 91.9049 | 48.8249 | 9.1162 |
| flair1/flair1 | WideSalienceMass_Projected | impervious surface | 52.8766 | -0.0240 | 60.7818 | 80.2591 | 21.6321 |
| flair1/flair1 | WideSalienceMass_Projected | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1003 |
| flair1/flair1 | WideSalienceMass_Projected | water | 64.6948 | -0.0889 | 66.8283 | 95.2974 | 6.3258 |
| flair1/flair1 | WideSalienceMass_Projected | coniferous | 42.7535 | 0.0270 | 62.5035 | 57.5017 | 0.5172 |
| flair1/flair1 | WideSalienceMass_Projected | deciduous | 60.8876 | -0.0172 | 78.7279 | 72.8771 | 15.7566 |
| flair1/flair1 | WideSalienceMass_Projected | brushwood | 15.5435 | -0.0871 | 29.6679 | 24.6130 | 4.0804 |
| flair1/flair1 | WideSalienceMass_Projected | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0200 |
| flair1/flair1 | WideSalienceMass_Projected | herbaceous vegetation | 57.6202 | -0.0001 | 91.5050 | 60.8766 | 21.3124 |
| flair1/flair1 | WideSalienceMass_Projected | agricultural land | 13.4306 | 0.0001 | 14.5034 | 64.4869 | 1.3558 |
| flair1/flair1 | WideSalienceMass_Projected | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9855 |
| flair1/flair1 | HardFixedMass_Projected | building | 58.3790 | 0.9749 | 59.5933 | 96.6273 | 11.6412 |
| flair1/flair1 | HardFixedMass_Projected | pervious surface | 46.0062 | -0.8010 | 92.0247 | 47.9167 | 8.9350 |
| flair1/flair1 | HardFixedMass_Projected | impervious surface | 54.3475 | 1.4469 | 62.4904 | 80.6603 | 21.1458 |
| flair1/flair1 | HardFixedMass_Projected | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1172 |
| flair1/flair1 | HardFixedMass_Projected | water | 68.8020 | 4.0183 | 71.3958 | 94.9845 | 5.9016 |
| flair1/flair1 | HardFixedMass_Projected | coniferous | 41.0048 | -1.7217 | 58.0062 | 58.3164 | 0.5651 |
| flair1/flair1 | HardFixedMass_Projected | deciduous | 61.6430 | 0.7382 | 78.5921 | 74.0822 | 16.0449 |
| flair1/flair1 | HardFixedMass_Projected | brushwood | 19.6041 | 3.9735 | 32.6903 | 32.8736 | 4.9460 |
| flair1/flair1 | HardFixedMass_Projected | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0405 |
| flair1/flair1 | HardFixedMass_Projected | herbaceous vegetation | 57.7945 | 0.1742 | 91.9527 | 60.8735 | 21.2075 |
| flair1/flair1 | HardFixedMass_Projected | agricultural land | 14.4678 | 1.0373 | 15.7858 | 63.4074 | 1.2248 |
| flair1/flair1 | HardFixedMass_Projected | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.2304 |
| flair1/flair1 | FineSalienceMass_Projected__ClassMean | building | 57.4052 | 0.0011 | 58.6655 | 96.3928 | 11.7966 |
| flair1/flair1 | FineSalienceMass_Projected__ClassMean | pervious surface | 46.8078 | 0.0006 | 91.9103 | 48.8191 | 9.1146 |
| flair1/flair1 | FineSalienceMass_Projected__ClassMean | impervious surface | 52.9034 | 0.0028 | 60.8123 | 80.2675 | 21.6235 |
| flair1/flair1 | FineSalienceMass_Projected__ClassMean | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0980 |
| flair1/flair1 | FineSalienceMass_Projected__ClassMean | water | 64.7861 | 0.0024 | 66.9326 | 95.2835 | 6.3150 |
| flair1/flair1 | FineSalienceMass_Projected__ClassMean | coniferous | 42.7265 | 0.0000 | 62.4459 | 57.5017 | 0.5176 |
| flair1/flair1 | FineSalienceMass_Projected__ClassMean | deciduous | 60.9040 | -0.0008 | 78.7253 | 72.9029 | 15.7628 |
| flair1/flair1 | FineSalienceMass_Projected__ClassMean | brushwood | 15.6334 | 0.0028 | 29.7445 | 24.7857 | 4.0984 |
| flair1/flair1 | FineSalienceMass_Projected__ClassMean | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0203 |
| flair1/flair1 | FineSalienceMass_Projected__ClassMean | herbaceous vegetation | 57.6215 | 0.0012 | 91.5151 | 60.8736 | 21.3090 |
| flair1/flair1 | FineSalienceMass_Projected__ClassMean | agricultural land | 13.4340 | 0.0035 | 14.5080 | 64.4712 | 1.3550 |
| flair1/flair1 | FineSalienceMass_Projected__ClassMean | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9891 |
| flair1/flair1 | FineSalienceMass_Projected__HardStrength | building | 57.3950 | -0.0091 | 58.6558 | 96.3901 | 11.7982 |
| flair1/flair1 | FineSalienceMass_Projected__HardStrength | pervious surface | 46.8127 | 0.0055 | 91.9018 | 48.8269 | 9.1169 |
| flair1/flair1 | FineSalienceMass_Projected__HardStrength | impervious surface | 52.8841 | -0.0165 | 60.7906 | 80.2608 | 21.6294 |
| flair1/flair1 | FineSalienceMass_Projected__HardStrength | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0991 |
| flair1/flair1 | FineSalienceMass_Projected__HardStrength | water | 64.7311 | -0.0526 | 66.8692 | 95.2931 | 6.3216 |
| flair1/flair1 | FineSalienceMass_Projected__HardStrength | coniferous | 42.7427 | 0.0162 | 62.4804 | 57.5017 | 0.5173 |
| flair1/flair1 | FineSalienceMass_Projected__HardStrength | deciduous | 60.8882 | -0.0166 | 78.7273 | 72.8785 | 15.7571 |
| flair1/flair1 | FineSalienceMass_Projected__HardStrength | brushwood | 15.5750 | -0.0556 | 29.6924 | 24.6751 | 4.0873 |
| flair1/flair1 | FineSalienceMass_Projected__HardStrength | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0201 |
| flair1/flair1 | FineSalienceMass_Projected__HardStrength | herbaceous vegetation | 57.6191 | -0.0012 | 91.5092 | 60.8736 | 21.3104 |
| flair1/flair1 | FineSalienceMass_Projected__HardStrength | agricultural land | 13.4263 | -0.0042 | 14.4983 | 64.4869 | 1.3562 |
| flair1/flair1 | FineSalienceMass_Projected__HardStrength | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9862 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle0 | building | 57.4069 | 0.0028 | 58.6677 | 96.3915 | 11.7960 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle0 | pervious surface | 46.8108 | 0.0036 | 91.8992 | 48.8255 | 9.1169 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle0 | impervious surface | 52.9063 | 0.0057 | 60.8187 | 80.2632 | 21.6201 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle0 | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0970 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle0 | water | 64.7759 | -0.0078 | 66.9222 | 95.2824 | 6.3159 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle0 | coniferous | 42.7184 | -0.0081 | 62.4286 | 57.5017 | 0.5178 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle0 | deciduous | 60.9147 | 0.0099 | 78.7238 | 72.9194 | 15.7666 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle0 | brushwood | 15.6310 | 0.0004 | 29.7667 | 24.7643 | 4.0918 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle0 | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0205 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle0 | herbaceous vegetation | 57.6277 | 0.0074 | 91.5082 | 60.8836 | 21.3141 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle0 | agricultural land | 13.4287 | -0.0018 | 14.5019 | 64.4712 | 1.3556 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle0 | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9877 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle1 | building | 57.4084 | 0.0043 | 58.6691 | 96.3921 | 11.7958 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle1 | pervious surface | 46.7987 | -0.0085 | 91.9088 | 48.8096 | 9.1130 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle1 | impervious surface | 52.9020 | 0.0014 | 60.8097 | 80.2690 | 21.6249 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle1 | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1001 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle1 | water | 64.8053 | 0.0216 | 66.9525 | 95.2845 | 6.3132 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle1 | coniferous | 42.7319 | 0.0054 | 62.4574 | 57.5017 | 0.5175 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle1 | deciduous | 60.8880 | -0.0168 | 78.7246 | 72.8805 | 15.7580 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle1 | brushwood | 15.6460 | 0.0154 | 29.7425 | 24.8186 | 4.1041 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle1 | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0203 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle1 | herbaceous vegetation | 57.6223 | 0.0020 | 91.5144 | 60.8748 | 21.3096 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle1 | agricultural land | 13.4309 | 0.0004 | 14.5044 | 64.4712 | 1.3553 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle1 | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9881 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle2 | building | 57.4066 | 0.0025 | 58.6632 | 96.4028 | 11.7983 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle2 | pervious surface | 46.8071 | -0.0001 | 91.9124 | 48.8177 | 9.1141 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle2 | impervious surface | 52.9080 | 0.0074 | 60.8189 | 80.2667 | 21.6209 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle2 | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0985 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle2 | water | 64.7705 | -0.0132 | 66.9159 | 95.2835 | 6.3166 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle2 | coniferous | 42.7292 | 0.0027 | 62.4516 | 57.5017 | 0.5176 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle2 | deciduous | 60.8973 | -0.0075 | 78.7228 | 72.8953 | 15.7616 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle2 | brushwood | 15.6137 | -0.0169 | 29.7110 | 24.7595 | 4.0987 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle2 | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0203 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle2 | herbaceous vegetation | 57.6213 | 0.0010 | 91.5161 | 60.8730 | 21.3086 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle2 | agricultural land | 13.4326 | 0.0021 | 14.5065 | 64.4712 | 1.3552 |
| flair1/flair1 | FineSalienceMass_Projected__AliasShuffle2 | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9896 |
| flair1/flair1 | WideSalienceMass_Projected__ClassMean | building | 57.4057 | 0.0016 | 58.6659 | 96.3928 | 11.7965 |
| flair1/flair1 | WideSalienceMass_Projected__ClassMean | pervious surface | 46.8088 | 0.0016 | 91.9104 | 48.8202 | 9.1148 |
| flair1/flair1 | WideSalienceMass_Projected__ClassMean | impervious surface | 52.9032 | 0.0026 | 60.8121 | 80.2675 | 21.6236 |
| flair1/flair1 | WideSalienceMass_Projected__ClassMean | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0979 |
| flair1/flair1 | WideSalienceMass_Projected__ClassMean | water | 64.7851 | 0.0014 | 66.9316 | 95.2835 | 6.3151 |
| flair1/flair1 | WideSalienceMass_Projected__ClassMean | coniferous | 42.7265 | 0.0000 | 62.4459 | 57.5017 | 0.5176 |
| flair1/flair1 | WideSalienceMass_Projected__ClassMean | deciduous | 60.9041 | -0.0007 | 78.7257 | 72.9026 | 15.7626 |
| flair1/flair1 | WideSalienceMass_Projected__ClassMean | brushwood | 15.6327 | 0.0021 | 29.7434 | 24.7847 | 4.0984 |
| flair1/flair1 | WideSalienceMass_Projected__ClassMean | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0204 |
| flair1/flair1 | WideSalienceMass_Projected__ClassMean | herbaceous vegetation | 57.6220 | 0.0017 | 91.5154 | 60.8741 | 21.3091 |
| flair1/flair1 | WideSalienceMass_Projected__ClassMean | agricultural land | 13.4340 | 0.0035 | 14.5080 | 64.4712 | 1.3550 |
| flair1/flair1 | WideSalienceMass_Projected__ClassMean | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9889 |
| flair1/flair1 | WideSalienceMass_Projected__HardStrength | building | 57.3957 | -0.0084 | 58.6568 | 96.3895 | 11.7980 |
| flair1/flair1 | WideSalienceMass_Projected__HardStrength | pervious surface | 46.8128 | 0.0056 | 91.9009 | 48.8271 | 9.1170 |
| flair1/flair1 | WideSalienceMass_Projected__HardStrength | impervious surface | 52.8805 | -0.0201 | 60.7863 | 80.2600 | 21.6307 |
| flair1/flair1 | WideSalienceMass_Projected__HardStrength | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0995 |
| flair1/flair1 | WideSalienceMass_Projected__HardStrength | water | 64.7114 | -0.0723 | 66.8459 | 95.2974 | 6.3241 |
| flair1/flair1 | WideSalienceMass_Projected__HardStrength | coniferous | 42.7481 | 0.0216 | 62.4919 | 57.5017 | 0.5172 |
| flair1/flair1 | WideSalienceMass_Projected__HardStrength | deciduous | 60.8835 | -0.0213 | 78.7280 | 72.8712 | 15.7554 |
| flair1/flair1 | WideSalienceMass_Projected__HardStrength | brushwood | 15.5583 | -0.0723 | 29.6724 | 24.6470 | 4.0854 |
| flair1/flair1 | WideSalienceMass_Projected__HardStrength | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0201 |
| flair1/flair1 | WideSalienceMass_Projected__HardStrength | herbaceous vegetation | 57.6202 | -0.0001 | 91.5087 | 60.8750 | 21.3110 |
| flair1/flair1 | WideSalienceMass_Projected__HardStrength | agricultural land | 13.4306 | 0.0001 | 14.5034 | 64.4869 | 1.3558 |
| flair1/flair1 | WideSalienceMass_Projected__HardStrength | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9858 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle0 | building | 57.4050 | 0.0009 | 58.6655 | 96.3921 | 11.7965 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle0 | pervious surface | 46.8123 | 0.0051 | 91.9022 | 48.8263 | 9.1167 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle0 | impervious surface | 52.9040 | 0.0034 | 60.8160 | 80.2626 | 21.6209 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle0 | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0971 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle0 | water | 64.7778 | -0.0059 | 66.9242 | 95.2824 | 6.3157 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle0 | coniferous | 42.7131 | -0.0134 | 62.4171 | 57.5017 | 0.5179 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle0 | deciduous | 60.9143 | 0.0095 | 78.7245 | 72.9183 | 15.7662 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle0 | brushwood | 15.6340 | 0.0034 | 29.7759 | 24.7653 | 4.0907 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle0 | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0205 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle0 | herbaceous vegetation | 57.6261 | 0.0058 | 91.5062 | 60.8827 | 21.3143 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle0 | agricultural land | 13.4326 | 0.0021 | 14.5065 | 64.4712 | 1.3552 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle0 | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9883 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle1 | building | 57.4105 | 0.0064 | 58.6712 | 96.3921 | 11.7954 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle1 | pervious surface | 46.7983 | -0.0089 | 91.9110 | 48.8085 | 9.1125 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle1 | impervious surface | 52.8996 | -0.0010 | 60.8055 | 80.2707 | 21.6268 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle1 | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1003 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle1 | water | 64.8126 | 0.0289 | 66.9609 | 95.2835 | 6.3123 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle1 | coniferous | 42.7319 | 0.0054 | 62.4574 | 57.5017 | 0.5175 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle1 | deciduous | 60.8896 | -0.0152 | 78.7261 | 72.8816 | 15.7580 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle1 | brushwood | 15.6580 | 0.0274 | 29.7610 | 24.8361 | 4.1045 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle1 | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0204 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle1 | herbaceous vegetation | 57.6190 | -0.0013 | 91.5150 | 60.8709 | 21.3081 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle1 | agricultural land | 13.4278 | -0.0027 | 14.5009 | 64.4712 | 1.3557 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle1 | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9885 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle2 | building | 57.4073 | 0.0032 | 58.6642 | 96.4021 | 11.7980 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle2 | pervious surface | 46.8113 | 0.0041 | 91.9122 | 48.8224 | 9.1150 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle2 | impervious surface | 52.9096 | 0.0090 | 60.8211 | 80.2667 | 21.6202 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle2 | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0980 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle2 | water | 64.7752 | -0.0085 | 66.9209 | 95.2835 | 6.3161 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle2 | coniferous | 42.7292 | 0.0027 | 62.4516 | 57.5017 | 0.5176 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle2 | deciduous | 60.8967 | -0.0081 | 78.7228 | 72.8945 | 15.7614 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle2 | brushwood | 15.6177 | -0.0129 | 29.7141 | 24.7672 | 4.0996 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle2 | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0203 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle2 | herbaceous vegetation | 57.6225 | 0.0022 | 91.5162 | 60.8742 | 21.3090 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle2 | agricultural land | 13.4309 | 0.0004 | 14.5044 | 64.4712 | 1.3553 |
| flair1/flair1 | WideSalienceMass_Projected__AliasShuffle2 | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9895 |

All source/output schemes retained. Developed pilots do not establish SOTA or every-word utility.
