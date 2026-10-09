# Bounded alias / Anchored factorial: verified screen

Frozen rho4, all20 retained, original local RS and broad VIP ImageNet/profile scoring. Same96 COMPLETE images: UDD5 full40; seven other domains8 each. Not full eight-dataset results. Corrected IRRG Vaihingen; LandCover.ai substitutes for unlabeled iSAID; development data.

13 mathematical tests pass locally and remotely;11 reused Anchored tests pass locally. Actual-checkpoint mask-free smoke verifies exact unrestricted and inactive-cap local/broad fallback, unchanged frozen heads. All five original controls replay exact historical per-image confusions. Complete unique keys, input identities, confusion sums and direct factorial transition endpoints verified independently.

The responsibility ceiling constrains the derivative with respect to effective alias logits. It does not identify semantic truth or bound the entire VIP salience/encoder chain. A useful rare alias can be penalized; correlated wrong aliases can survive. Standard constrained entropy aggregation and quadratic reconstruction are not claimed novel by themselves.

The initial numerical preflight was stopped before inspecting its target metrics. This repeat only adds exact original-score/weight fallback where the cap is inactive; rho, words, sources and sample sequence are unchanged. Aborted initial outputs remain separate.

## Factorial mIoU

| Dataset/protocol | Geometry | Original fusion | Bounded fusion | Anchored | Bounded Anchored | Delta vs Anchored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 31.9462 | 56.1762 | 56.2148 | 57.0012 | 57.0317 | +0.0305 |
| potsdam/potsdam | 40.3528 | 40.8725 | 40.7937 | 41.7806 | 41.6478 | -0.1328 |
| udd5/udd5 | 50.5553 | 48.8694 | 49.1733 | 49.2117 | 49.5043 | +0.2927 |
| oem/oem | 39.3543 | 31.8024 | 31.8636 | 31.9533 | 32.0414 | +0.0881 |
| loveda/P | 62.8254 | 62.7434 | 62.7351 | 62.4863 | 62.4654 | -0.0210 |
| loveda/D | 38.4558 | 36.6573 | 36.6419 | 36.5600 | 36.5351 | -0.0249 |
| vaihingen/vaihingen | 50.2325 | 50.0312 | 49.9819 | 51.4491 | 51.3714 | -0.0777 |
| landcoverai/landcoverai | 60.9049 | 66.4263 | 66.4498 | 66.9060 | 66.9207 | +0.0146 |
| flair1/flair1 | 38.8428 | 32.3648 | 32.4371 | 33.6084 | 33.7203 | +0.1119 |
| Equal-domain mean, LoveDA D once | 43.830569 | 45.400028 | 45.444500 | 46.058780 | 46.096577 | +0.037796 |

Frozen promotion passed: False

Failed checks: focus_potsdam

## Interpretation and decision

The alias cap adds +0.037796pp to the equal-domain Anchored mean. Original Anchored adds +0.658752pp to its same-source simple fusion. These are distinct interventions; the existing spatial reconstruction has a much larger mean effect in this screen.

Local constraint activity ranges from 0.000000% to 0.027924% of window patch/class entries, versus 9.6088% to 32.1876% in the broad observer. Bounded local Geometry is nearly unchanged: its mean differs by +0.00001374pp. Most observed effects therefore concern the borrowed broad branch, not a newly grounded local alias selector.

On full UDD5, road IoU rises from39.8083 to41.6678, while vehicle falls from20.4213 to19.9795. Road gains correct coverage as well as extra false positives; this is not solely removal of wrong evidence. On the fixed8 Potsdam images, low vegetation falls from21.2306 to20.6327 (10,150 previously correct pixels lost versus514 gained), and car falls from24.2128 to24.0811 (4,920 false activations added versus215 removed). The controller caps concentration, not semantic reliability; useful concentrated evidence and competitive calibration can also be harmed.

The frozen focus_potsdam requirement fails, so no full eight-domain rollout was launched. Preserve original Geometry and Anchored results. This experiment is weak evidence for bounded influence, not a validated good/bad-word selector or a completed two-module CVPR contribution. Further designs must justify which semantic evidence is changed, rather than lower rho retrospectively or use per-dataset winners.

## Direct factorial correction counts

| Dataset/protocol | Contrast | Beneficial | Harmful | Net correct pixels |
| --- | --- | ---: | ---: | ---: |
| vdd/vdd | alias_in_fusion | 38338 | 21332 | +17006 |
| vdd/vdd | alias_in_anchored | 33813 | 18443 | +15370 |
| vdd/vdd | geometry_in_original | 1685014 | 1284936 | +400078 |
| vdd/vdd | geometry_in_bounded | 1683358 | 1284916 | +398442 |
| vdd/vdd | alias_local | 34 | 1 | +33 |
| vdd/vdd | alias_broad | 46502 | 14994 | +31508 |
| potsdam/potsdam | alias_in_fusion | 2431 | 11343 | -8912 |
| potsdam/potsdam | alias_in_anchored | 1659 | 15197 | -13538 |
| potsdam/potsdam | geometry_in_original | 260750 | 195758 | +64992 |
| potsdam/potsdam | geometry_in_bounded | 259622 | 199256 | +60366 |
| potsdam/potsdam | alias_local | 0 | 0 | +0 |
| potsdam/potsdam | alias_broad | 3575 | 18417 | -14842 |
| udd5/udd5 | alias_in_fusion | 2651482 | 1444699 | +1206783 |
| udd5/udd5 | alias_in_anchored | 2592931 | 1444238 | +1148693 |
| udd5/udd5 | geometry_in_original | 6563770 | 5373696 | +1190074 |
| udd5/udd5 | geometry_in_bounded | 6558542 | 5426558 | +1131984 |
| udd5/udd5 | alias_local | 46 | 76 | -30 |
| udd5/udd5 | alias_broad | 3378828 | 1541458 | +1837370 |
| oem/oem | alias_in_fusion | 35860 | 13248 | +22612 |
| oem/oem | alias_in_anchored | 36950 | 11243 | +25707 |
| oem/oem | geometry_in_original | 212727 | 232696 | -19969 |
| oem/oem | geometry_in_bounded | 214029 | 230903 | -16874 |
| oem/oem | alias_local | 0 | 0 | +0 |
| oem/oem | alias_broad | 44712 | 18171 | +26541 |
| loveda/P | alias_in_fusion | 133 | 692 | -559 |
| loveda/P | alias_in_anchored | 108 | 620 | -512 |
| loveda/P | geometry_in_original | 33223 | 44796 | -11573 |
| loveda/P | geometry_in_bounded | 33262 | 44788 | -11526 |
| loveda/P | alias_local | 0 | 0 | +0 |
| loveda/P | alias_broad | 538 | 1277 | -739 |
| loveda/D | alias_in_fusion | 1382 | 5822 | -4440 |
| loveda/D | alias_in_anchored | 1274 | 5015 | -3741 |
| loveda/D | geometry_in_original | 126489 | 228119 | -101630 |
| loveda/D | geometry_in_bounded | 126190 | 227121 | -100931 |
| loveda/D | alias_local | 0 | 0 | +0 |
| loveda/D | alias_broad | 670 | 2894 | -2224 |
| vaihingen/vaihingen | alias_in_fusion | 4259 | 8174 | -3915 |
| vaihingen/vaihingen | alias_in_anchored | 3338 | 8719 | -5381 |
| vaihingen/vaihingen | geometry_in_original | 236486 | 138519 | +97967 |
| vaihingen/vaihingen | geometry_in_bounded | 235124 | 138623 | +96501 |
| vaihingen/vaihingen | alias_local | 17 | 2 | +15 |
| vaihingen/vaihingen | alias_broad | 5775 | 10541 | -4766 |
| landcoverai/landcoverai | alias_in_fusion | 637 | 164 | +473 |
| landcoverai/landcoverai | alias_in_anchored | 554 | 214 | +340 |
| landcoverai/landcoverai | geometry_in_original | 29753 | 24035 | +5718 |
| landcoverai/landcoverai | geometry_in_bounded | 29491 | 23906 | +5585 |
| landcoverai/landcoverai | alias_local | 0 | 0 | +0 |
| landcoverai/landcoverai | alias_broad | 1156 | 162 | +994 |
| flair1/flair1 | alias_in_fusion | 4289 | 3387 | +902 |
| flair1/flair1 | alias_in_anchored | 6741 | 2632 | +4109 |
| flair1/flair1 | geometry_in_original | 69708 | 39377 | +30331 |
| flair1/flair1 | geometry_in_bounded | 72079 | 38541 | +33538 |
| flair1/flair1 | alias_local | 0 | 0 | +0 |
| flair1/flair1 | alias_broad | 7935 | 9265 | -1330 |

## Class-specific coverage and competition

Direct transitions isolate the alias change inside Anchored and the reconstruction change with bounded aliases. FP removal can include wrong-to-wrong changes; it is not itself a correct pixel. No target metric is used to alter the frozen readout.

| Dataset/protocol | Contrast | Class | TP gained | TP lost | FP removed | FP added |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| vdd/vdd | alias_in_anchored | other | 22450 | 1631 | 11516 | 6962 |
| vdd/vdd | alias_in_anchored | wall | 401 | 76 | 1273 | 516 |
| vdd/vdd | alias_in_anchored | road | 96 | 55 | 14927 | 26903 |
| vdd/vdd | alias_in_anchored | vegetation | 10855 | 8129 | 295 | 972 |
| vdd/vdd | alias_in_anchored | vehicle | 0 | 422 | 6040 | 49 |
| vdd/vdd | alias_in_anchored | roof | 5 | 536 | 4777 | 2939 |
| vdd/vdd | alias_in_anchored | water | 6 | 7594 | 15087 | 204 |
| vdd/vdd | geometry_in_bounded | other | 820709 | 641821 | 480768 | 584908 |
| vdd/vdd | geometry_in_bounded | wall | 69643 | 23169 | 174627 | 158430 |
| vdd/vdd | geometry_in_bounded | road | 47395 | 6193 | 496830 | 288482 |
| vdd/vdd | geometry_in_bounded | vegetation | 662210 | 579437 | 90114 | 70204 |
| vdd/vdd | geometry_in_bounded | vehicle | 2597 | 979 | 49475 | 56343 |
| vdd/vdd | geometry_in_bounded | roof | 53840 | 18145 | 629795 | 393993 |
| vdd/vdd | geometry_in_bounded | water | 26964 | 15172 | 117269 | 88076 |
| potsdam/potsdam | alias_in_anchored | impervious surface | 496 | 4780 | 6837 | 3184 |
| potsdam/potsdam | alias_in_anchored | building | 16 | 121 | 953 | 820 |
| potsdam/potsdam | alias_in_anchored | low vegetation | 514 | 10150 | 225 | 144 |
| potsdam/potsdam | alias_in_anchored | tree | 402 | 81 | 15 | 303 |
| potsdam/potsdam | alias_in_anchored | car | 132 | 0 | 215 | 4920 |
| potsdam/potsdam | alias_in_anchored | clutter | 99 | 65 | 3043 | 15455 |
| potsdam/potsdam | geometry_in_bounded | impervious surface | 130286 | 61840 | 89066 | 66287 |
| potsdam/potsdam | geometry_in_bounded | building | 5925 | 2249 | 51377 | 21065 |
| potsdam/potsdam | geometry_in_bounded | low vegetation | 32055 | 56990 | 16181 | 10201 |
| potsdam/potsdam | geometry_in_bounded | tree | 85531 | 73908 | 33191 | 20530 |
| potsdam/potsdam | geometry_in_bounded | car | 1037 | 518 | 120078 | 75882 |
| potsdam/potsdam | geometry_in_bounded | clutter | 4788 | 3751 | 118475 | 174037 |
| udd5/udd5 | alias_in_anchored | vegetation | 298707 | 204376 | 19335 | 33379 |
| udd5/udd5 | alias_in_anchored | building | 284316 | 14386 | 98763 | 762179 |
| udd5/udd5 | alias_in_anchored | road | 1739659 | 3124 | 19234 | 907258 |
| udd5/udd5 | alias_in_anchored | vehicle | 33323 | 7233 | 733698 | 1203657 |
| udd5/udd5 | alias_in_anchored | other | 236926 | 1215119 | 3354379 | 170243 |
| udd5/udd5 | geometry_in_bounded | vegetation | 3351816 | 2610479 | 392673 | 205712 |
| udd5/udd5 | geometry_in_bounded | building | 429153 | 212162 | 3057764 | 2374807 |
| udd5/udd5 | geometry_in_bounded | road | 1128880 | 1209250 | 2117221 | 1827849 |
| udd5/udd5 | geometry_in_bounded | vehicle | 87996 | 21203 | 1355719 | 1361816 |
| udd5/udd5 | geometry_in_bounded | other | 1560697 | 1373464 | 3443164 | 3464373 |
| oem/oem | alias_in_anchored | bareland | 0 | 0 | 28657 | 3179 |
| oem/oem | alias_in_anchored | rangeland | 558 | 443 | 1160 | 476 |
| oem/oem | alias_in_anchored | developed space | 33104 | 1809 | 9738 | 12904 |
| oem/oem | alias_in_anchored | road | 309 | 20 | 40 | 1841 |
| oem/oem | alias_in_anchored | tree | 2003 | 4658 | 799 | 206 |
| oem/oem | alias_in_anchored | water | 0 | 0 | 0 | 0 |
| oem/oem | alias_in_anchored | agriculture land | 152 | 0 | 6 | 4295 |
| oem/oem | alias_in_anchored | building | 824 | 4313 | 8436 | 228 |
| oem/oem | geometry_in_bounded | bareland | 0 | 0 | 105236 | 95119 |
| oem/oem | geometry_in_bounded | rangeland | 46021 | 19098 | 17886 | 33082 |
| oem/oem | geometry_in_bounded | developed space | 79147 | 85609 | 91568 | 190570 |
| oem/oem | geometry_in_bounded | road | 6798 | 9579 | 67276 | 14062 |
| oem/oem | geometry_in_bounded | tree | 44748 | 54612 | 18272 | 9877 |
| oem/oem | geometry_in_bounded | water | 0 | 0 | 0 | 0 |
| oem/oem | geometry_in_bounded | agriculture land | 6836 | 2459 | 53851 | 46314 |
| oem/oem | geometry_in_bounded | building | 30479 | 59546 | 37037 | 18976 |
| loveda/P | alias_in_anchored | building | 12 | 21 | 8 | 1 |
| loveda/P | alias_in_anchored | road | 6 | 0 | 29 | 418 |
| loveda/P | alias_in_anchored | water | 53 | 221 | 705 | 84 |
| loveda/P | alias_in_anchored | barren | 6 | 31 | 23 | 15 |
| loveda/P | alias_in_anchored | tree | 26 | 336 | 58 | 0 |
| loveda/P | alias_in_anchored | farm | 5 | 11 | 66 | 883 |
| loveda/P | geometry_in_bounded | building | 199 | 519 | 1209 | 124 |
| loveda/P | geometry_in_bounded | road | 1528 | 866 | 17578 | 16062 |
| loveda/P | geometry_in_bounded | water | 12163 | 6536 | 14732 | 9558 |
| loveda/P | geometry_in_bounded | barren | 2859 | 6009 | 8864 | 2854 |
| loveda/P | geometry_in_bounded | tree | 11480 | 28925 | 1184 | 502 |
| loveda/P | geometry_in_bounded | farm | 5033 | 1933 | 23507 | 49500 |
| loveda/D | alias_in_anchored | background | 667 | 4324 | 1407 | 590 |
| loveda/D | alias_in_anchored | building | 4 | 56 | 27 | 16 |
| loveda/D | alias_in_anchored | road | 6 | 0 | 28 | 960 |
| loveda/D | alias_in_anchored | water | 53 | 223 | 2302 | 146 |
| loveda/D | alias_in_anchored | barren | 4 | 33 | 482 | 42 |
| loveda/D | alias_in_anchored | tree | 531 | 355 | 161 | 142 |
| loveda/D | alias_in_anchored | farm | 9 | 24 | 266 | 6518 |
| loveda/D | geometry_in_bounded | background | 82631 | 178843 | 38985 | 29503 |
| loveda/D | geometry_in_bounded | building | 185 | 659 | 5717 | 2278 |
| loveda/D | geometry_in_bounded | road | 1529 | 871 | 56282 | 26399 |
| loveda/D | geometry_in_bounded | water | 16948 | 6633 | 54785 | 38314 |
| loveda/D | geometry_in_bounded | barren | 2972 | 7092 | 20497 | 9893 |
| loveda/D | geometry_in_bounded | tree | 12374 | 29523 | 26635 | 5186 |
| loveda/D | geometry_in_bounded | farm | 9551 | 3500 | 93368 | 285627 |
| vaihingen/vaihingen | alias_in_anchored | impervious surface | 1484 | 5069 | 1189 | 4574 |
| vaihingen/vaihingen | alias_in_anchored | building | 3 | 171 | 4955 | 890 |
| vaihingen/vaihingen | alias_in_anchored | low vegetation | 565 | 3435 | 200 | 65 |
| vaihingen/vaihingen | alias_in_anchored | tree | 1159 | 44 | 102 | 1145 |
| vaihingen/vaihingen | alias_in_anchored | car | 127 | 0 | 548 | 5701 |
| vaihingen/vaihingen | geometry_in_bounded | impervious surface | 108379 | 51566 | 104152 | 93725 |
| vaihingen/vaihingen | geometry_in_bounded | building | 7585 | 1463 | 107565 | 79189 |
| vaihingen/vaihingen | geometry_in_bounded | low vegetation | 45589 | 64406 | 18946 | 7646 |
| vaihingen/vaihingen | geometry_in_bounded | tree | 71116 | 20258 | 53168 | 50231 |
| vaihingen/vaihingen | geometry_in_bounded | car | 2455 | 930 | 101002 | 57541 |
| landcoverai/landcoverai | alias_in_anchored | background | 11 | 199 | 543 | 21 |
| landcoverai/landcoverai | alias_in_anchored | building | 2 | 0 | 5 | 12 |
| landcoverai/landcoverai | alias_in_anchored | woodland | 541 | 15 | 16 | 93 |
| landcoverai/landcoverai | alias_in_anchored | water | 0 | 0 | 0 | 10 |
| landcoverai/landcoverai | alias_in_anchored | road | 0 | 0 | 0 | 88 |
| landcoverai/landcoverai | geometry_in_bounded | background | 13951 | 12524 | 15565 | 11675 |
| landcoverai/landcoverai | geometry_in_bounded | building | 144 | 5 | 2829 | 2720 |
| landcoverai/landcoverai | geometry_in_bounded | woodland | 15284 | 11367 | 5316 | 5514 |
| landcoverai/landcoverai | geometry_in_bounded | water | 0 | 0 | 797 | 256 |
| landcoverai/landcoverai | geometry_in_bounded | road | 112 | 10 | 6444 | 5201 |
| flair1/flair1 | alias_in_anchored | building | 1205 | 6 | 502 | 1901 |
| flair1/flair1 | alias_in_anchored | pervious surface | 368 | 784 | 516 | 150 |
| flair1/flair1 | alias_in_anchored | impervious surface | 370 | 1249 | 2615 | 1262 |
| flair1/flair1 | alias_in_anchored | bare soil | 0 | 0 | 711 | 664 |
| flair1/flair1 | alias_in_anchored | water | 99 | 39 | 1288 | 18 |
| flair1/flair1 | alias_in_anchored | coniferous | 32 | 0 | 0 | 47 |
| flair1/flair1 | alias_in_anchored | deciduous | 1078 | 4 | 3 | 816 |
| flair1/flair1 | alias_in_anchored | brushwood | 280 | 23 | 1276 | 293 |
| flair1/flair1 | alias_in_anchored | vineyard | 0 | 0 | 7 | 0 |
| flair1/flair1 | alias_in_anchored | herbaceous vegetation | 3292 | 527 | 217 | 177 |
| flair1/flair1 | alias_in_anchored | agricultural land | 17 | 0 | 0 | 508 |
| flair1/flair1 | alias_in_anchored | plowed land | 0 | 0 | 2839 | 29 |
| flair1/flair1 | geometry_in_bounded | building | 2028 | 972 | 15398 | 7660 |
| flair1/flair1 | geometry_in_bounded | pervious surface | 8615 | 7141 | 7648 | 2234 |
| flair1/flair1 | geometry_in_bounded | impervious surface | 8962 | 4281 | 21401 | 19047 |
| flair1/flair1 | geometry_in_bounded | bare soil | 0 | 0 | 14370 | 7924 |
| flair1/flair1 | geometry_in_bounded | water | 1595 | 342 | 9296 | 7174 |
| flair1/flair1 | geometry_in_bounded | coniferous | 450 | 289 | 1908 | 455 |
| flair1/flair1 | geometry_in_bounded | deciduous | 21397 | 11406 | 9448 | 12354 |
| flair1/flair1 | geometry_in_bounded | brushwood | 3924 | 5438 | 17973 | 18601 |
| flair1/flair1 | geometry_in_bounded | vineyard | 0 | 0 | 428 | 11 |
| flair1/flair1 | geometry_in_bounded | herbaceous vegetation | 25043 | 8495 | 8684 | 8005 |
| flair1/flair1 | geometry_in_bounded | agricultural land | 65 | 177 | 7850 | 3050 |
| flair1/flair1 | geometry_in_bounded | plowed land | 0 | 0 | 7121 | 1472 |

## vdd/vdd

| Class | Geometry IoU | Bounded local | BroadVIP | Bounded broad | Anchored IoU | Primary IoU | Primary precision | Primary recall | Delta vs Anchored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| other | 25.3949 | 25.3949 | 48.3490 | 48.4451 | 53.4471 | 53.5191 | 72.8670 | 66.8392 | 0.0720 |
| wall | 11.8982 | 11.8982 | 43.4246 | 43.4325 | 57.7209 | 57.7508 | 68.4563 | 78.6910 | 0.0299 |
| road | 24.1970 | 24.1970 | 17.7913 | 17.7815 | 25.5594 | 25.4995 | 25.6938 | 97.1193 | -0.0599 |
| vegetation | 62.0257 | 62.0257 | 53.3537 | 53.3613 | 60.0299 | 60.0409 | 96.8732 | 61.2275 | 0.0110 |
| vehicle | 5.6965 | 5.6965 | 24.8750 | 25.0972 | 25.3443 | 25.4714 | 25.6796 | 96.9148 | 0.1271 |
| roof | 60.7577 | 60.7577 | 79.6473 | 79.6543 | 84.4472 | 84.4504 | 84.8330 | 99.4688 | 0.0032 |
| water | 33.6534 | 33.6533 | 90.4010 | 90.5054 | 92.4595 | 92.4898 | 94.7216 | 97.5158 | 0.0303 |

| Class | Arm | Predicted area % | Precision % | Recall % |
| --- | --- | ---: | ---: | ---: |
| other | Geometry | 29.1864 | 38.9278 | 42.2128 |
| wall | Geometry | 15.3444 | 11.9838 | 94.3324 |
| road | Geometry | 5.1104 | 24.7412 | 91.6676 |
| vegetation | Geometry | 17.8191 | 84.6841 | 69.8628 |
| vehicle | Geometry | 4.1453 | 5.6966 | 99.9563 |
| roof | Geometry | 20.9399 | 88.0153 | 66.2377 |
| water | Geometry | 7.4546 | 93.0590 | 34.5198 |
| other | BroadVIP | 24.3439 | 68.6252 | 62.0692 |
| wall | BroadVIP | 1.5969 | 67.2362 | 55.0798 |
| road | BroadVIP | 5.4096 | 18.9553 | 74.3420 |
| vegetation | BroadVIP | 12.4583 | 95.1099 | 54.8585 |
| vehicle | BroadVIP | 0.7290 | 26.3752 | 81.3897 |
| roof | BroadVIP | 34.3940 | 80.2023 | 99.1386 |
| water | BroadVIP | 21.0683 | 92.7676 | 97.2555 |
| other | Anchored_VIP | 24.6717 | 72.8291 | 66.7587 |
| wall | Anchored_VIP | 2.2412 | 68.4274 | 78.6737 |
| road | Anchored_VIP | 5.2010 | 25.7549 | 97.1162 |
| vegetation | Anchored_VIP | 13.6481 | 96.8775 | 61.2143 |
| vehicle | Anchored_VIP | 0.8983 | 25.5376 | 97.1009 |
| roof | Anchored_VIP | 32.6274 | 84.8283 | 99.4708 |
| water | Anchored_VIP | 20.7123 | 94.6528 | 97.5551 |
| other | Bounded_Anchored | 24.6887 | 72.8670 | 66.8392 |
| wall | Bounded_Anchored | 2.2408 | 68.4563 | 78.6910 |
| road | Bounded_Anchored | 5.2136 | 25.6938 | 97.1193 |
| vegetation | Bounded_Anchored | 13.6516 | 96.8732 | 61.2275 |
| vehicle | Bounded_Anchored | 0.8916 | 25.6796 | 96.9148 |
| roof | Bounded_Anchored | 32.6249 | 84.8330 | 99.4688 |
| water | Bounded_Anchored | 20.6889 | 94.7216 | 97.5158 |

Diagnostics (window/crop means, not unique-pixel estimates):
```json
{
  "tiles": 88.0,
  "observer_empty_rows": 0.0,
  "observer_rows": 1764.0,
  "broad_capped_patch_class_fraction": 0.09608843638852703,
  "broad_original_max_responsibility": 0.13704321846099837,
  "broad_bounded_max_responsibility": 0.13201331074482628,
  "broad_original_effective_aliases": 14.485313858304703,
  "broad_bounded_effective_aliases": 14.637664335114614,
  "broad_mean_absolute_score_change": 0.001287061153433403,
  "broad_inactive_max_absolute_score_change": 0.0,
  "broad_responsibility_mass_error": 1.367713723863874e-07,
  "local_capped_patch_class_fraction": 6.361131544237011e-05,
  "local_original_max_responsibility": 0.0702672080800554,
  "local_bounded_max_responsibility": 0.07026588548407225,
  "local_original_effective_aliases": 19.18104269790959,
  "local_bounded_effective_aliases": 19.181106346381174,
  "local_mean_absolute_score_change": 1.0194293847748608e-08,
  "local_inactive_max_absolute_score_change": 0.0,
  "local_responsibility_mass_error": 1.4060551857019395e-07,
  "solver_mean_absolute_innovation": 1.5585005895488637,
  "solver_changed_patch_fraction": 0.4261724298650568,
  "solver_solver_relative_residual": 5.434829262185304e-07,
  "solver_solver_iterations": 7.998579545454546,
  "solver_energy_before": 38526.1415682706,
  "solver_energy_after": 19171.381356672806
}
```

## potsdam/potsdam

| Class | Geometry IoU | Bounded local | BroadVIP | Bounded broad | Anchored IoU | Primary IoU | Primary precision | Primary recall | Delta vs Anchored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| impervious surface | 51.6875 | 51.6875 | 61.7181 | 61.6597 | 62.1577 | 62.1035 | 73.7230 | 79.7585 | -0.0542 |
| building | 78.2026 | 78.2026 | 71.1846 | 71.1738 | 75.5787 | 75.5784 | 77.0303 | 97.5668 | -0.0003 |
| low vegetation | 33.4221 | 33.4221 | 18.5600 | 17.9097 | 21.2306 | 20.6327 | 81.7194 | 21.6311 | -0.5979 |
| tree | 62.5813 | 62.5813 | 61.7470 | 61.7577 | 64.8712 | 64.8778 | 92.4792 | 68.4915 | 0.0066 |
| car | 11.6582 | 11.6582 | 27.3355 | 27.2625 | 24.2128 | 24.0811 | 24.1843 | 98.2600 | -0.1317 |
| clutter | 4.5652 | 4.5652 | 2.4836 | 2.4653 | 2.6325 | 2.6129 | 3.7404 | 7.9767 | -0.0196 |

| Class | Arm | Predicted area % | Precision % | Recall % |
| --- | --- | ---: | ---: | ---: |
| impervious surface | Geometry | 29.8644 | 75.3909 | 62.1781 |
| building | Geometry | 14.5546 | 80.6769 | 96.2262 |
| low vegetation | Geometry | 10.1329 | 72.4703 | 38.2826 |
| tree | Geometry | 17.6734 | 90.6538 | 66.8975 |
| car | Geometry | 19.8762 | 11.6716 | 99.0262 |
| clutter | Geometry | 7.8986 | 7.7447 | 10.0073 |
| impervious surface | BroadVIP | 41.0919 | 71.7944 | 81.4728 |
| building | BroadVIP | 16.2663 | 72.7786 | 97.0151 |
| low vegetation | BroadVIP | 4.6416 | 80.3488 | 19.4426 |
| tree | BroadVIP | 17.3568 | 90.8504 | 65.8414 |
| car | BroadVIP | 7.8464 | 27.8767 | 93.3682 |
| clutter | BroadVIP | 12.7970 | 3.5810 | 7.4968 |
| impervious surface | Anchored_VIP | 39.2742 | 73.6731 | 79.9064 |
| building | Anchored_VIP | 15.4589 | 77.0239 | 97.5776 |
| low vegetation | Anchored_VIP | 5.1989 | 82.1270 | 22.2591 |
| tree | Anchored_VIP | 17.7297 | 92.4963 | 68.4748 |
| car | Anchored_VIP | 9.4578 | 24.3214 | 98.1896 |
| clutter | Anchored_VIP | 12.8804 | 3.7823 | 7.9698 |
| impervious surface | Bounded_Anchored | 39.1750 | 73.7230 | 79.7585 |
| building | Bounded_Anchored | 15.4559 | 77.0303 | 97.5668 |
| low vegetation | Bounded_Anchored | 5.0774 | 81.7194 | 21.6311 |
| tree | Bounded_Anchored | 17.7374 | 92.4792 | 68.4915 |
| car | Bounded_Anchored | 9.5183 | 24.1843 | 98.2600 |
| clutter | Bounded_Anchored | 13.0360 | 3.7404 | 7.9767 |

Diagnostics (window/crop means, not unique-pixel estimates):
```json
{
  "tiles": 9.0,
  "observer_empty_rows": 0.0,
  "observer_rows": 3528.0,
  "broad_capped_patch_class_fraction": 0.17489843333532917,
  "broad_original_max_responsibility": 0.15575605394163478,
  "broad_bounded_max_responsibility": 0.14501788556420553,
  "broad_original_effective_aliases": 13.46174685160319,
  "broad_bounded_effective_aliases": 13.798998018105825,
  "broad_mean_absolute_score_change": 0.0031946868431397117,
  "broad_inactive_max_absolute_score_change": 0.0,
  "broad_responsibility_mass_error": 1.3690441846847534e-07,
  "local_capped_patch_class_fraction": 5.199291087962962e-05,
  "local_original_max_responsibility": 0.07968812392748616,
  "local_bounded_max_responsibility": 0.0796870148430268,
  "local_original_effective_aliases": 18.62515596548716,
  "local_bounded_effective_aliases": 18.625188571435434,
  "local_mean_absolute_score_change": 1.078085966178656e-08,
  "local_inactive_max_absolute_score_change": 0.0,
  "local_responsibility_mass_error": 1.343863981741446e-07,
  "solver_mean_absolute_innovation": 1.7052863472037847,
  "solver_changed_patch_fraction": 0.2276340060763889,
  "solver_solver_relative_residual": 5.248300283255958e-07,
  "solver_solver_iterations": 8.0,
  "solver_energy_before": 37661.173529730906,
  "solver_energy_after": 18719.282904730906
}
```

## udd5/udd5

| Class | Geometry IoU | Bounded local | BroadVIP | Bounded broad | Anchored IoU | Primary IoU | Primary precision | Primary recall | Delta vs Anchored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vegetation | 82.4937 | 82.4937 | 61.1183 | 61.0734 | 68.3846 | 68.4485 | 97.4162 | 69.7142 | 0.0639 |
| building | 83.9035 | 83.9035 | 80.0717 | 79.9314 | 82.5264 | 82.3920 | 83.4299 | 98.5125 | -0.1344 |
| road | 45.7856 | 45.7856 | 34.6267 | 37.0370 | 39.8083 | 41.6678 | 66.8342 | 52.5294 | 1.8595 |
| vehicle | 10.0092 | 10.0091 | 17.2973 | 17.7073 | 20.4213 | 19.9795 | 20.4469 | 89.7326 | -0.4418 |
| other | 30.5846 | 30.5846 | 33.2928 | 33.7184 | 34.9177 | 35.0339 | 49.7972 | 54.1644 | 0.1162 |

non_residual_mean_iou_percent: Geometry=55.5480, MeanLogit_VIP=52.4055, Bounded_MeanLogit=52.7472, Anchored_VIP=52.7852, Bounded_Anchored=53.1219

| Class | Arm | Predicted area % | Precision % | Recall % |
| --- | --- | ---: | ---: | ---: |
| vegetation | Geometry | 26.1079 | 96.4862 | 85.0488 |
| building | Geometry | 41.9106 | 88.3604 | 94.3293 |
| road | Geometry | 10.7242 | 70.6780 | 56.5218 |
| vehicle | Geometry | 7.7994 | 10.0323 | 97.7477 |
| other | Geometry | 13.4578 | 52.8537 | 42.0591 |
| vegetation | BroadVIP | 19.3742 | 95.9264 | 62.7469 |
| building | BroadVIP | 46.7902 | 81.7755 | 97.4639 |
| road | BroadVIP | 9.7538 | 61.0827 | 44.4283 |
| vehicle | BroadVIP | 2.5976 | 19.2910 | 62.5992 |
| other | BroadVIP | 21.4842 | 44.6387 | 56.7073 |
| vegetation | Anchored_VIP | 21.1716 | 97.4282 | 69.6418 |
| building | Anchored_VIP | 46.1437 | 83.6805 | 98.3563 |
| road | Anchored_VIP | 9.9434 | 66.8744 | 49.5860 |
| vehicle | Anchored_VIP | 3.4003 | 20.9505 | 88.9918 |
| other | Anchored_VIP | 19.3411 | 48.5109 | 55.4791 |
| vegetation | Bounded_Anchored | 21.1963 | 97.4162 | 69.7142 |
| building | Bounded_Anchored | 46.3558 | 83.4299 | 98.5125 |
| road | Bounded_Anchored | 10.5399 | 66.8342 | 52.5294 |
| vehicle | Bounded_Anchored | 3.5130 | 20.4469 | 89.7326 |
| other | Bounded_Anchored | 18.3950 | 49.7972 | 54.1644 |

Diagnostics (window/crop means, not unique-pixel estimates):
```json
{
  "tiles": 80.8,
  "observer_empty_rows": 0.0,
  "observer_rows": 1763.9999999999998,
  "broad_capped_patch_class_fraction": 0.32187642167147706,
  "broad_original_max_responsibility": 0.18595381882041692,
  "broad_bounded_max_responsibility": 0.15780350148677824,
  "broad_original_effective_aliases": 12.005129363536835,
  "broad_bounded_effective_aliases": 12.723799366950985,
  "broad_mean_absolute_score_change": 0.012482814616447142,
  "broad_inactive_max_absolute_score_change": 0.0,
  "broad_responsibility_mass_error": 1.3709068298339843e-07,
  "local_capped_patch_class_fraction": 0.00027924138849431816,
  "local_original_max_responsibility": 0.08004745155718233,
  "local_bounded_max_responsibility": 0.08004218987168069,
  "local_original_effective_aliases": 18.578234112197705,
  "local_bounded_effective_aliases": 18.578486539006228,
  "local_mean_absolute_score_change": 3.6219823007983956e-08,
  "local_inactive_max_absolute_score_change": 0.0,
  "local_responsibility_mass_error": 1.3669241558421743e-07,
  "solver_mean_absolute_innovation": 1.6315741249648004,
  "solver_changed_patch_fraction": 0.2099605121034564,
  "solver_solver_relative_residual": 5.812130979246628e-07,
  "solver_solver_iterations": 7.995170454545454,
  "solver_energy_before": 29934.970697021483,
  "solver_energy_after": 14881.26021171801
}
```

## oem/oem

| Class | Geometry IoU | Bounded local | BroadVIP | Bounded broad | Anchored IoU | Primary IoU | Primary precision | Primary recall | Delta vs Anchored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rangeland | 47.7780 | 47.7780 | 32.8318 | 32.8423 | 37.5959 | 37.6151 | 60.9038 | 49.5891 | 0.0192 |
| developed space | 28.5294 | 28.5294 | 24.3872 | 25.6024 | 24.0531 | 25.0848 | 37.4504 | 43.1726 | 1.0317 |
| road | 40.8291 | 40.8291 | 37.8035 | 37.7565 | 42.1205 | 42.0353 | 53.5458 | 66.1642 | -0.0852 |
| tree | 56.0657 | 56.0657 | 34.2920 | 34.0451 | 38.7562 | 38.6357 | 92.7933 | 39.8308 | -0.1205 |
| water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| agriculture land | 79.1353 | 79.1353 | 61.7665 | 61.4727 | 68.8612 | 68.7069 | 70.2154 | 96.9679 | -0.1543 |
| building | 62.4971 | 62.4971 | 33.8638 | 33.3836 | 44.2393 | 44.2532 | 77.9955 | 50.5664 | 0.0139 |

| Class | Arm | Predicted area % | Precision % | Recall % |
| --- | --- | ---: | ---: | ---: |
| bareland | Geometry | 10.4255 | 0.0000 | 0.0000 |
| rangeland | Geometry | 18.7786 | 65.2945 | 64.0414 |
| developed space | Geometry | 12.6303 | 61.5617 | 34.7130 |
| road | Geometry | 7.8925 | 45.7530 | 79.1399 |
| tree | Geometry | 17.7308 | 87.4475 | 60.9727 |
| water | Geometry | 0.0647 | 0.0000 | 0.0000 |
| agriculture land | Geometry | 15.9338 | 90.6413 | 86.1766 |
| building | Geometry | 16.5437 | 65.5844 | 92.9954 |
| bareland | BroadVIP | 13.0524 | 0.0000 | 0.0000 |
| rangeland | BroadVIP | 13.5183 | 59.7235 | 42.1685 |
| developed space | BroadVIP | 26.1167 | 36.4210 | 42.4656 |
| road | BroadVIP | 5.7309 | 49.2748 | 61.8880 |
| tree | BroadVIP | 9.7822 | 91.9170 | 35.3583 |
| water | BroadVIP | 0.0000 | 0.0000 | 0.0000 |
| agriculture land | BroadVIP | 25.7069 | 63.0752 | 96.7502 |
| building | BroadVIP | 6.0926 | 73.7411 | 38.5072 |
| bareland | Anchored_VIP | 11.6586 | 0.0000 | 0.0000 |
| rangeland | Anchored_VIP | 15.5965 | 60.8653 | 49.5812 |
| developed space | Anchored_VIP | 25.3728 | 36.5064 | 41.3527 |
| road | Anchored_VIP | 5.6109 | 53.7385 | 66.0817 |
| tree | Anchored_VIP | 10.9578 | 92.7506 | 39.9668 |
| water | Anchored_VIP | 0.0000 | 0.0000 | 0.0000 |
| agriculture land | Anchored_VIP | 23.0869 | 70.3828 | 96.9561 |
| building | Anchored_VIP | 7.7166 | 77.0444 | 50.9559 |
| bareland | Bounded_Anchored | 11.3267 | 0.0000 | 0.0000 |
| rangeland | Bounded_Anchored | 15.5891 | 60.9038 | 49.5891 |
| developed space | Bounded_Anchored | 25.8216 | 37.4504 | 43.1726 |
| road | Bounded_Anchored | 5.6381 | 53.5458 | 66.1642 |
| tree | Bounded_Anchored | 10.9155 | 92.7933 | 39.8308 |
| water | Bounded_Anchored | 0.0000 | 0.0000 | 0.0000 |
| agriculture land | Bounded_Anchored | 23.1447 | 70.2154 | 96.9679 |
| building | Bounded_Anchored | 7.5642 | 77.9955 | 50.5664 |

Diagnostics (window/crop means, not unique-pixel estimates):
```json
{
  "tiles": 8.375,
  "observer_empty_rows": 0.0,
  "observer_rows": 3528.0,
  "broad_capped_patch_class_fraction": 0.28923965859848977,
  "broad_original_max_responsibility": 0.1801974113623146,
  "broad_bounded_max_responsibility": 0.15961390777374618,
  "broad_original_effective_aliases": 12.026341579854488,
  "broad_bounded_effective_aliases": 12.595517568290234,
  "broad_mean_absolute_score_change": 0.006885067785211871,
  "broad_inactive_max_absolute_score_change": 0.0,
  "broad_responsibility_mass_error": 1.3294629752635956e-07,
  "local_capped_patch_class_fraction": 3.9842393663194445e-05,
  "local_original_max_responsibility": 0.0807231925945315,
  "local_bounded_max_responsibility": 0.08072266186031306,
  "local_original_effective_aliases": 18.575717478162712,
  "local_bounded_effective_aliases": 18.575742753843464,
  "local_mean_absolute_score_change": 2.4360882693801815e-09,
  "local_inactive_max_absolute_score_change": 0.0,
  "local_responsibility_mass_error": 1.3297216759787665e-07,
  "solver_mean_absolute_innovation": 1.7889477581613593,
  "solver_changed_patch_fraction": 0.25445556640625,
  "solver_solver_relative_residual": 4.5549693399809456e-07,
  "solver_solver_iterations": 8.0,
  "solver_energy_before": 55907.59183756511,
  "solver_energy_after": 27738.769029405383
}
```

## loveda/P

| Class | Geometry IoU | Bounded local | BroadVIP | Bounded broad | Anchored IoU | Primary IoU | Primary precision | Primary recall | Delta vs Anchored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| building | 64.4987 | 64.4987 | 66.1383 | 66.3676 | 79.5953 | 79.5842 | 96.0851 | 82.2513 | -0.0111 |
| road | 66.9348 | 66.9348 | 68.0178 | 67.9394 | 68.0988 | 68.0436 | 69.0724 | 97.8578 | -0.0552 |
| water | 81.1851 | 81.1851 | 75.8814 | 75.9219 | 79.3816 | 79.4111 | 85.6638 | 91.5822 | 0.0295 |
| barren | 24.3403 | 24.3403 | 22.9545 | 22.9637 | 23.8204 | 23.8140 | 74.5718 | 25.9187 | -0.0064 |
| tree | 53.7307 | 53.7307 | 22.3040 | 22.1900 | 36.7931 | 36.7364 | 99.0629 | 36.8645 | -0.0567 |
| farm | 86.2626 | 86.2626 | 82.9699 | 82.9296 | 87.2288 | 87.2029 | 87.4520 | 99.6745 | -0.0259 |

| Class | Arm | Predicted area % | Precision % | Recall % |
| --- | --- | ---: | ---: | ---: |
| building | Geometry | 0.9799 | 65.3671 | 97.9819 |
| road | Geometry | 10.3611 | 67.6522 | 98.4404 |
| water | Geometry | 22.3129 | 86.7638 | 92.6613 |
| barren | Geometry | 3.2301 | 63.8735 | 28.2261 |
| tree | Geometry | 13.0831 | 64.5184 | 76.2666 |
| farm | Geometry | 50.0330 | 95.3302 | 90.0685 |
| building | BroadVIP | 0.5597 | 86.3120 | 73.8882 |
| road | BroadVIP | 9.5989 | 70.5126 | 95.0555 |
| water | BroadVIP | 21.1773 | 85.7072 | 86.8746 |
| barren | BroadVIP | 2.9104 | 65.5566 | 26.1025 |
| tree | BroadVIP | 2.5534 | 97.2830 | 22.4438 |
| farm | BroadVIP | 63.2003 | 83.3420 | 99.4648 |
| building | Anchored_VIP | 0.5600 | 96.0602 | 82.2814 |
| road | Anchored_VIP | 10.0793 | 69.1302 | 97.8560 |
| water | Anchored_VIP | 22.3534 | 85.6142 | 91.5997 |
| barren | Anchored_VIP | 2.5412 | 74.5721 | 25.9262 |
| tree | Anchored_VIP | 4.1267 | 99.0340 | 36.9257 |
| farm | Anchored_VIP | 60.3393 | 87.4779 | 99.6747 |
| building | Bounded_Anchored | 0.5596 | 96.0851 | 82.2513 |
| road | Bounded_Anchored | 10.0880 | 69.0724 | 97.8578 |
| water | Bounded_Anchored | 22.3362 | 85.6638 | 91.5822 |
| barren | Bounded_Anchored | 2.5405 | 74.5718 | 25.9187 |
| tree | Bounded_Anchored | 4.1187 | 99.0629 | 36.8645 |
| farm | Bounded_Anchored | 60.3570 | 87.4520 | 99.6745 |

Diagnostics (window/crop means, not unique-pixel estimates):
```json
{
  "tiles": 9.0,
  "observer_empty_rows": 441.0,
  "observer_rows": 3528.0,
  "broad_capped_patch_class_fraction": 0.2140377016160831,
  "broad_original_max_responsibility": 0.16862380566696325,
  "broad_bounded_max_responsibility": 0.1600728003929059,
  "broad_original_effective_aliases": 11.992573390404385,
  "broad_bounded_effective_aliases": 12.23885464668274,
  "broad_mean_absolute_score_change": 0.001732757860224865,
  "broad_inactive_max_absolute_score_change": 0.0,
  "broad_responsibility_mass_error": 1.338000098864237e-07,
  "local_capped_patch_class_fraction": 0.0,
  "local_original_max_responsibility": 0.07710470257464934,
  "local_bounded_max_responsibility": 0.07710470257464934,
  "local_original_effective_aliases": 18.84716174337599,
  "local_bounded_effective_aliases": 18.84716174337599,
  "local_mean_absolute_score_change": 0.0,
  "local_inactive_max_absolute_score_change": 0.0,
  "local_responsibility_mass_error": 1.2914339701334635e-07,
  "solver_mean_absolute_innovation": 1.8375962343480852,
  "solver_changed_patch_fraction": 0.3457845052083333,
  "solver_solver_relative_residual": 4.436932666212417e-07,
  "solver_solver_iterations": 8.0,
  "solver_energy_before": 44075.991997612844,
  "solver_energy_after": 21874.92009819878
}
```

## loveda/D

| Class | Geometry IoU | Bounded local | BroadVIP | Bounded broad | Anchored IoU | Primary IoU | Primary precision | Primary recall | Delta vs Anchored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| background | 34.1983 | 34.1983 | 1.8793 | 1.8436 | 6.4379 | 6.3420 | 78.1094 | 6.4567 | -0.0959 |
| building | 31.1862 | 31.1862 | 33.5727 | 33.6468 | 36.3815 | 36.3090 | 39.8325 | 80.4103 | -0.0725 |
| road | 45.4818 | 45.4818 | 55.9896 | 55.8781 | 56.4727 | 56.3807 | 57.0875 | 97.8514 | -0.0920 |
| water | 61.9193 | 61.9193 | 61.1534 | 61.2483 | 62.7476 | 62.8325 | 66.7225 | 91.5091 | 0.0849 |
| barren | 14.5296 | 14.5296 | 18.9446 | 18.9507 | 20.0167 | 20.0306 | 48.4509 | 25.4555 | 0.0139 |
| tree | 26.8904 | 26.8904 | 21.5237 | 21.3964 | 31.2898 | 31.3241 | 87.2228 | 32.8306 | 0.0343 |
| farm | 54.9851 | 54.9851 | 39.9394 | 39.9102 | 42.5737 | 42.5265 | 42.6035 | 99.5767 | -0.0472 |

foreground_mean_iou_percent: Geometry=39.1654, MeanLogit_VIP=41.2662, Bounded_MeanLogit=41.2689, Anchored_VIP=41.5803, Bounded_Anchored=41.5672

| Class | Arm | Predicted area % | Precision % | Recall % |
| --- | --- | ---: | ---: | ---: |
| background | Geometry | 28.2483 | 65.7504 | 41.6109 |
| building | Geometry | 1.0641 | 31.8582 | 93.6650 |
| road | Geometry | 8.4577 | 45.8348 | 98.3352 |
| water | Geometry | 15.9605 | 65.9551 | 91.0065 |
| barren | Geometry | 1.1070 | 59.0631 | 16.1567 |
| tree | Geometry | 14.6543 | 30.0530 | 71.8726 |
| farm | Geometry | 30.5080 | 69.5721 | 72.3948 |
| background | BroadVIP | 1.1897 | 71.0509 | 1.8938 |
| building | BroadVIP | 0.7021 | 38.0921 | 73.8882 |
| road | BroadVIP | 6.4968 | 57.6729 | 95.0454 |
| water | BroadVIP | 14.6905 | 67.8265 | 86.1414 |
| barren | BroadVIP | 2.5823 | 40.8875 | 26.0906 |
| tree | BroadVIP | 1.5555 | 87.4841 | 22.2076 |
| farm | BroadVIP | 72.7832 | 40.0371 | 99.3922 |
| background | Anchored_VIP | 3.7438 | 78.1619 | 6.5558 |
| building | Anchored_VIP | 0.7314 | 39.8770 | 80.5840 |
| road | Anchored_VIP | 6.7459 | 57.1824 | 97.8495 |
| water | Anchored_VIP | 15.8922 | 66.6173 | 91.5269 |
| barren | Anchored_VIP | 2.1318 | 48.3384 | 25.4642 |
| tree | Anchored_VIP | 2.3045 | 87.2023 | 32.7959 |
| farm | Anchored_VIP | 68.4503 | 42.6508 | 99.5774 |
| background | Bounded_Anchored | 3.6897 | 78.1094 | 6.4567 |
| building | Bounded_Anchored | 0.7307 | 39.8325 | 80.4103 |
| road | Bounded_Anchored | 6.7572 | 57.0875 | 97.8514 |
| water | Bounded_Anchored | 15.8641 | 66.7225 | 91.5091 |
| barren | Bounded_Anchored | 2.1261 | 48.4509 | 25.4555 |
| tree | Bounded_Anchored | 2.3064 | 87.2228 | 32.8306 |
| farm | Bounded_Anchored | 68.5258 | 42.6035 | 99.5767 |

Diagnostics (window/crop means, not unique-pixel estimates):
```json
{
  "tiles": 9.0,
  "observer_empty_rows": 441.0,
  "observer_rows": 3528.0,
  "broad_capped_patch_class_fraction": 0.22764820546685124,
  "broad_original_max_responsibility": 0.1707162608924721,
  "broad_bounded_max_responsibility": 0.1618598652338343,
  "broad_original_effective_aliases": 11.860279270580838,
  "broad_bounded_effective_aliases": 12.110078930854797,
  "broad_mean_absolute_score_change": 0.0017581305233664224,
  "broad_inactive_max_absolute_score_change": 0.0,
  "broad_responsibility_mass_error": 1.319817134312221e-07,
  "local_capped_patch_class_fraction": 0.0,
  "local_original_max_responsibility": 0.07701630281314018,
  "local_bounded_max_responsibility": 0.07701630281314018,
  "local_original_effective_aliases": 18.8560926724994,
  "local_bounded_effective_aliases": 18.8560926724994,
  "local_mean_absolute_score_change": 0.0,
  "local_inactive_max_absolute_score_change": 0.0,
  "local_responsibility_mass_error": 1.2725118606809583e-07,
  "solver_mean_absolute_innovation": 1.863405391573906,
  "solver_changed_patch_fraction": 0.42556423611111105,
  "solver_solver_relative_residual": 4.446020859126697e-07,
  "solver_solver_iterations": 8.0,
  "solver_energy_before": 52685.462022569445,
  "solver_energy_after": 26145.425238715277
}
```

## vaihingen/vaihingen

| Class | Geometry IoU | Bounded local | BroadVIP | Bounded broad | Anchored IoU | Primary IoU | Primary precision | Primary recall | Delta vs Anchored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| impervious surface | 49.2002 | 49.2004 | 56.0846 | 55.9119 | 60.0516 | 59.8682 | 74.4513 | 75.3479 | -0.1834 |
| building | 74.6610 | 74.6610 | 64.8387 | 64.9312 | 68.2387 | 68.3432 | 68.5761 | 99.5054 | 0.1045 |
| low vegetation | 46.7115 | 46.7115 | 31.7052 | 31.5302 | 35.3912 | 35.2535 | 96.4515 | 35.7168 | -0.1377 |
| tree | 71.8329 | 71.8332 | 65.1005 | 65.1955 | 71.8529 | 71.8714 | 81.8482 | 85.4993 | 0.0185 |
| car | 8.7566 | 8.7567 | 20.4239 | 20.3240 | 21.7111 | 21.5207 | 21.7555 | 95.2253 | -0.1904 |

| Class | Arm | Predicted area % | Precision % | Recall % |
| --- | --- | ---: | ---: | ---: |
| impervious surface | Geometry | 20.4283 | 82.3971 | 54.9789 |
| building | Geometry | 28.1288 | 75.5001 | 98.5334 |
| low vegetation | Geometry | 13.4530 | 92.0732 | 48.6687 |
| tree | Geometry | 21.4401 | 82.5509 | 84.6923 |
| car | Geometry | 16.5499 | 8.7725 | 97.9748 |
| impervious surface | BroadVIP | 32.0690 | 70.2362 | 73.5698 |
| building | BroadVIP | 32.6526 | 65.2987 | 98.9252 |
| low vegetation | BroadVIP | 8.6739 | 94.7069 | 32.2771 |
| tree | BroadVIP | 20.5591 | 79.5117 | 78.2222 |
| car | BroadVIP | 6.0454 | 21.1172 | 86.1505 |
| impervious surface | Anchored_VIP | 30.9871 | 74.5909 | 75.4952 |
| building | Anchored_VIP | 31.3277 | 68.4663 | 99.5152 |
| low vegetation | Anchored_VIP | 9.4625 | 96.4478 | 35.8586 |
| tree | Anchored_VIP | 21.8030 | 81.8858 | 85.4321 |
| car | Anchored_VIP | 6.4197 | 21.9557 | 95.1174 |
| impervious surface | Bounded_Anchored | 30.9846 | 74.4513 | 75.3479 |
| building | Bounded_Anchored | 31.2744 | 68.5761 | 99.5054 |
| low vegetation | Bounded_Anchored | 9.4247 | 96.4515 | 35.7168 |
| tree | Bounded_Anchored | 21.8302 | 81.8482 | 85.4993 |
| car | Bounded_Anchored | 6.4862 | 21.7555 | 95.2253 |

Diagnostics (window/crop means, not unique-pixel estimates):
```json
{
  "tiles": 9.0,
  "observer_empty_rows": 0.0,
  "observer_rows": 3528.0,
  "broad_capped_patch_class_fraction": 0.15004251918144293,
  "broad_original_max_responsibility": 0.15192542211152613,
  "broad_bounded_max_responsibility": 0.1424178216140717,
  "broad_original_effective_aliases": 13.810723507404326,
  "broad_bounded_effective_aliases": 14.080893832445145,
  "broad_mean_absolute_score_change": 0.00322572603697778,
  "broad_inactive_max_absolute_score_change": 0.0,
  "broad_responsibility_mass_error": 1.3709068298339845e-07,
  "local_capped_patch_class_fraction": 5.967881944444445e-05,
  "local_original_max_responsibility": 0.07887870087805723,
  "local_bounded_max_responsibility": 0.07887712623924018,
  "local_original_effective_aliases": 18.73120058907403,
  "local_bounded_effective_aliases": 18.731241718928022,
  "local_mean_absolute_score_change": 2.306429046762383e-08,
  "local_inactive_max_absolute_score_change": 0.0,
  "local_responsibility_mass_error": 1.356005668640137e-07,
  "solver_mean_absolute_innovation": 1.5404627124468484,
  "solver_changed_patch_fraction": 0.1941596137152778,
  "solver_solver_relative_residual": 6.569329439937773e-07,
  "solver_solver_iterations": 8.0,
  "solver_energy_before": 25772.71923828125,
  "solver_energy_after": 12818.052517361111
}
```

## landcoverai/landcoverai

| Class | Geometry IoU | Bounded local | BroadVIP | Bounded broad | Anchored IoU | Primary IoU | Primary precision | Primary recall | Delta vs Anchored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| background | 82.6090 | 82.6090 | 86.0090 | 86.0633 | 87.4477 | 87.4655 | 93.8890 | 92.7455 | 0.0178 |
| building | 34.9562 | 34.9562 | 48.3391 | 48.3376 | 44.4672 | 44.4656 | 44.8095 | 98.3036 | -0.0016 |
| woodland | 78.3638 | 78.3638 | 71.8857 | 72.1129 | 78.6026 | 78.7020 | 94.9445 | 82.1443 | 0.0994 |
| water | 93.6635 | 93.6635 | 98.3440 | 98.3373 | 97.6178 | 97.6123 | 97.6123 | 100.0000 | -0.0055 |
| road | 14.9318 | 14.9318 | 24.7413 | 24.7269 | 26.3947 | 26.3578 | 28.8087 | 75.5990 | -0.0369 |

foreground_mean_iou_percent: Geometry=55.4788, MeanLogit_VIP=61.2530, Bounded_MeanLogit=61.2757, Anchored_VIP=61.7706, Bounded_Anchored=61.7844

non_residual_mean_iou_percent: Geometry=55.4788, MeanLogit_VIP=61.2530, Bounded_MeanLogit=61.2757, Anchored_VIP=61.7706, Bounded_Anchored=61.7844

| Class | Arm | Predicted area % | Precision % | Recall % |
| --- | --- | ---: | ---: | ---: |
| background | Geometry | 59.6226 | 96.6505 | 85.0437 |
| building | Geometry | 4.2927 | 35.0436 | 99.2919 |
| woodland | Geometry | 22.0960 | 86.4992 | 89.2841 |
| water | Geometry | 8.8309 | 93.6635 | 100.0000 |
| road | Geometry | 5.1578 | 15.6288 | 77.0019 |
| background | BroadVIP | 68.6415 | 91.8845 | 93.0799 |
| building | BroadVIP | 2.9778 | 49.1665 | 96.6355 |
| woodland | BroadVIP | 17.2348 | 93.7672 | 75.4930 |
| water | BroadVIP | 8.3994 | 98.4087 | 99.9331 |
| road | BroadVIP | 2.7464 | 27.3943 | 71.8685 |
| background | Anchored_VIP | 66.9686 | 93.8549 | 92.7587 |
| building | Anchored_VIP | 3.3233 | 44.8124 | 98.2973 |
| woodland | Anchored_VIP | 18.4920 | 94.9565 | 82.0272 |
| water | Anchored_VIP | 8.4732 | 97.6178 | 100.0000 |
| road | Anchored_VIP | 2.7429 | 28.8528 | 75.5990 |
| background | Bounded_Anchored | 66.9347 | 93.8890 | 92.7455 |
| building | Bounded_Anchored | 3.3237 | 44.8095 | 98.3036 |
| woodland | Bounded_Anchored | 18.5208 | 94.9445 | 82.1443 |
| water | Bounded_Anchored | 8.4736 | 97.6123 | 100.0000 |
| road | Bounded_Anchored | 2.7471 | 28.8087 | 75.5990 |

Diagnostics (window/crop means, not unique-pixel estimates):
```json
{
  "tiles": 1.0,
  "observer_empty_rows": 1984.5,
  "observer_rows": 3528.0,
  "broad_capped_patch_class_fraction": 0.13113662329124054,
  "broad_original_max_responsibility": 0.14453508709557356,
  "broad_bounded_max_responsibility": 0.13863800512626767,
  "broad_original_effective_aliases": 13.818177095055582,
  "broad_bounded_effective_aliases": 13.971766382455826,
  "broad_mean_absolute_score_change": 0.0016383403533142015,
  "broad_inactive_max_absolute_score_change": 0.0,
  "broad_responsibility_mass_error": 1.4118850231170654e-07,
  "local_capped_patch_class_fraction": 2.44140625e-05,
  "local_original_max_responsibility": 0.07675652131438256,
  "local_bounded_max_responsibility": 0.0767562847584486,
  "local_original_effective_aliases": 18.831919145584106,
  "local_bounded_effective_aliases": 18.83193197250366,
  "local_mean_absolute_score_change": 4.900130079477094e-10,
  "local_inactive_max_absolute_score_change": 0.0,
  "local_responsibility_mass_error": 1.3858079910278316e-07,
  "solver_mean_absolute_innovation": 1.6838625520467758,
  "solver_changed_patch_fraction": 0.08154296875,
  "solver_solver_relative_residual": 4.5208215837533317e-07,
  "solver_solver_iterations": 8.0,
  "solver_energy_before": 30940.37109375,
  "solver_energy_after": 15321.814208984375
}
```

## flair1/flair1

| Class | Geometry IoU | Bounded local | BroadVIP | Bounded broad | Anchored IoU | Primary IoU | Primary precision | Primary recall | Delta vs Anchored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| building | 49.6979 | 49.6979 | 55.6097 | 55.7799 | 56.7755 | 56.9332 | 57.9172 | 97.1024 | 0.1577 |
| pervious surface | 57.0621 | 57.0621 | 39.6383 | 38.6311 | 47.2756 | 47.2108 | 91.8559 | 49.2733 | -0.0648 |
| impervious surface | 52.6509 | 52.6509 | 50.9692 | 50.5577 | 51.9858 | 51.9525 | 59.8777 | 79.6962 | -0.0333 |
| bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| water | 73.2328 | 73.2328 | 57.7226 | 58.6240 | 61.4565 | 62.0427 | 63.8426 | 95.6534 | 0.5862 |
| coniferous | 43.0233 | 43.0233 | 33.4100 | 33.8724 | 43.5282 | 43.6037 | 65.5362 | 56.5767 | 0.0755 |
| deciduous | 54.0229 | 54.0229 | 55.8985 | 56.0855 | 59.5237 | 59.6624 | 78.6736 | 71.1732 | 0.1387 |
| brushwood | 18.6136 | 18.6136 | 7.5457 | 7.7315 | 12.3287 | 12.5698 | 26.8299 | 19.1263 | 0.2411 |
| vineyard | -- | -- | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| herbaceous vegetation | 60.2329 | 60.2329 | 49.0459 | 49.0792 | 57.4725 | 57.8636 | 90.8891 | 61.4265 | 0.3911 |
| agricultural land | 18.7339 | 18.7339 | 6.6409 | 6.2729 | 12.9546 | 12.8054 | 13.7307 | 65.5194 | -0.1492 |
| plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

| Class | Arm | Predicted area % | Precision % | Recall % |
| --- | --- | ---: | ---: | ---: |
| building | Geometry | 13.5991 | 50.7258 | 96.0825 |
| pervious surface | Geometry | 11.1728 | 92.1299 | 59.9861 |
| impervious surface | Geometry | 20.0846 | 62.6245 | 76.7765 |
| bare soil | Geometry | 3.4411 | 0.0000 | 0.0000 |
| water | Geometry | 5.3781 | 77.1432 | 93.5263 |
| coniferous | Geometry | 0.5346 | 61.7114 | 58.6897 |
| deciduous | Geometry | 13.6004 | 78.9723 | 63.0995 |
| brushwood | Geometry | 9.9866 | 23.4212 | 47.5559 |
| vineyard | Geometry | 0.0000 | 0.0000 | 0.0000 |
| herbaceous vegetation | Geometry | 21.2289 | 94.3169 | 62.5013 |
| agricultural land | Geometry | 0.8198 | 21.6468 | 58.1977 |
| plowed land | Geometry | 0.1540 | 0.0000 | 0.0000 |
| building | BroadVIP | 11.8786 | 57.3363 | 94.8633 |
| pervious surface | BroadVIP | 8.7430 | 84.1002 | 42.8494 |
| impervious surface | BroadVIP | 21.2148 | 59.8323 | 77.4815 |
| bare soil | BroadVIP | 9.2929 | 0.0000 | 0.0000 |
| water | BroadVIP | 6.9336 | 60.0121 | 93.8005 |
| coniferous | BroadVIP | 0.6272 | 47.4865 | 52.9871 |
| deciduous | BroadVIP | 14.2711 | 78.6221 | 65.9174 |
| brushwood | BroadVIP | 2.4088 | 21.3423 | 10.4526 |
| vineyard | BroadVIP | 0.3985 | 0.0000 | 0.0000 |
| herbaceous vegetation | BroadVIP | 17.9582 | 91.6078 | 51.3533 |
| agricultural land | BroadVIP | 3.8860 | 6.7159 | 85.5914 |
| plowed land | BroadVIP | 2.3873 | 0.0000 | 0.0000 |
| building | Anchored_VIP | 11.9131 | 58.0396 | 96.3057 |
| pervious surface | Anchored_VIP | 9.2422 | 91.6998 | 49.3890 |
| impervious surface | Anchored_VIP | 21.9112 | 59.7781 | 79.9522 |
| bare soil | Anchored_VIP | 7.1474 | 0.0000 | 0.0000 |
| water | Anchored_VIP | 6.7041 | 63.2502 | 95.5889 |
| coniferous | Anchored_VIP | 0.4815 | 65.7321 | 56.3052 |
| deciduous | Anchored_VIP | 15.3089 | 78.8015 | 70.8722 |
| brushwood | Anchored_VIP | 3.5408 | 26.2213 | 18.8771 |
| vineyard | Anchored_VIP | 0.0016 | 0.0000 | 0.0000 |
| herbaceous vegetation | Anchored_VIP | 21.5207 | 90.8252 | 61.0148 |
| agricultural land | Anchored_VIP | 1.4300 | 13.9145 | 65.2534 |
| plowed land | Anchored_VIP | 0.7986 | 0.0000 | 0.0000 |
| building | Bounded_Anchored | 12.0370 | 57.9172 | 97.1024 |
| pervious surface | Bounded_Anchored | 9.2049 | 91.8559 | 49.2733 |
| impervious surface | Bounded_Anchored | 21.8047 | 59.8777 | 79.6962 |
| bare soil | Bounded_Anchored | 7.1452 | 0.0000 | 0.0000 |
| water | Bounded_Anchored | 6.6463 | 63.8426 | 95.6534 |
| coniferous | Bounded_Anchored | 0.4853 | 65.5362 | 56.5767 |
| deciduous | Bounded_Anchored | 15.3989 | 78.6736 | 71.1732 |
| brushwood | Bounded_Anchored | 3.5062 | 26.8299 | 19.1263 |
| vineyard | Bounded_Anchored | 0.0013 | 0.0000 | 0.0000 |
| herbaceous vegetation | Bounded_Anchored | 21.6507 | 90.8891 | 61.4265 |
| agricultural land | Bounded_Anchored | 1.4550 | 13.7307 | 65.5194 |
| plowed land | Bounded_Anchored | 0.6646 | 0.0000 | 0.0000 |

Diagnostics (window/crop means, not unique-pixel estimates):
```json
{
  "tiles": 1.0,
  "observer_empty_rows": 441.0,
  "observer_rows": 3528.0,
  "broad_capped_patch_class_fraction": 0.2625070891372161,
  "broad_original_max_responsibility": 0.1730497460036228,
  "broad_bounded_max_responsibility": 0.15181054755036408,
  "broad_original_effective_aliases": 12.691587089250485,
  "broad_bounded_effective_aliases": 13.300929787258307,
  "broad_mean_absolute_score_change": 0.008079626368051854,
  "broad_inactive_max_absolute_score_change": 0.0,
  "broad_responsibility_mass_error": 1.4156103134155276e-07,
  "local_capped_patch_class_fraction": 2.0345052083333332e-05,
  "local_original_max_responsibility": 0.08033402667691311,
  "local_bounded_max_responsibility": 0.08033378484348458,
  "local_original_effective_aliases": 18.670701722304024,
  "local_bounded_effective_aliases": 18.67071259021759,
  "local_mean_absolute_score_change": 8.057365145456666e-10,
  "local_inactive_max_absolute_score_change": 0.0,
  "local_responsibility_mass_error": 1.3907750447591144e-07,
  "solver_mean_absolute_innovation": 1.5443993955850601,
  "solver_changed_patch_fraction": 0.1650390625,
  "solver_solver_relative_residual": 4.798062605004816e-07,
  "solver_solver_iterations": 8.0,
  "solver_energy_before": 62479.55908203125,
  "solver_energy_after": 30974.39697265625
}
```

## Combined runtime

Suite elapsed seconds: 631.3883943557739

| Dataset | Nine-arm wall seconds | Peak allocated CUDA MiB |
| --- | ---: | ---: |
| vdd | 141.5291 | 5813.1494 |
| potsdam | 10.4176 | 5520.8770 |
| udd5 | 550.0818 | 5618.2188 |
| oem | 12.4496 | 5533.7373 |
| loveda | 17.8189 | 5555.3218 |
| vaihingen | 9.7706 | 5514.8223 |
| landcoverai | 2.7449 | 5417.0361 |
| flair1 | 4.1697 | 5449.9287 |

Times cover the combined nine-arm workload, not standalone model latency. No retrospective rho adjustment, dataset winner switching or automatic full rollout. Original Geometry and Anchored_VIP remain preserved. A screen gain alone does not establish eight-domain SOTA, independent transfer or CVPR novelty.
