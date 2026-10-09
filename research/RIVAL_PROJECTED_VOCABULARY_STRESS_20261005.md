# Frozen projected screening: vocabulary-size and attachment stress

Same64 developed top-left512 windows,8/domain, full-image wide context. Not full datasets or independent validation. Geometry local20/checkpoints/operator/scores, finite VIP wide/fine, original hard risk and reconstruction frozen. Existing nested Qwen20/30/40 and historical four wrong-parent/four legitimate-paraphrase replacements only; no generation, target fitting, count switching or rule change. Construction names/slots enter diagnostics only.

All historical scenario no-admission/hard per-image confusions and clean20 projected scores/predictions match exactly. Scores and packed source flags persist before masks. All five scenarios share one Geometry and the original four fine forwards, plus unchanged full-image wide crops. Labelled centre audit does not enter prediction.

One common scored-class union across every scenario/arm in each protocol. LoveDA D once; P separate. Historical20-no-admission denominators are retained in summary.json for earlier-table interpretation; changes caused solely by denominator are not model changes. Counts and added phrase identities change together.

| Scenario | No admission | Original hard | Projected candidate | Fine-direction diagnostic | Candidate-hard pp | Candidate-clean20 pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k20 | 42.8679 | 45.0622 | 45.4187 | 45.3240 | +0.3566 | +0.0000 |
| k30 | 43.0357 | 45.1366 | 45.3651 | 45.2191 | +0.2285 | -0.0536 |
| k40 | 41.7421 | 43.9695 | 44.2044 | 44.1571 | +0.2349 | -1.2143 |
| wrong_parent | 41.9892 | 44.2848 | 44.5969 | 44.5321 | +0.3121 | -0.8218 |
| paraphrase | 43.6469 | 45.6935 | 45.9874 | 45.8657 | +0.2939 | +0.5686 |

## Predeclared Checks

```json
{
  "all_scenarios_mean_above_hard": true,
  "all_scenarios_at_least5_no_admission_wins": true,
  "all_protocols_within1pp_vs_hard": false,
  "wrong_parent_degradation_not_worse_than_hard": false,
  "this40_word_pool_harmless_vs20": false,
  "wrong_parent_candidate_change_pp": -0.8218008243166466,
  "wrong_parent_hard_change_pp": -0.7773875356730287,
  "frozen_advancement_gate": false
}
```

Passing matched no-admission/hard controls does not make arbitrary expansion harmless. Diagnostic fine direction keeps alias-derived action amplitudes and is not alias-free or a post-result selector candidate.

## Interpretation And Decision

The projected candidate improves matched original hard means in all five scenarios,
but FAILS the declared robust-advancement gate. LoveDA P loses2.7580pp against
matched hard at30 words and1.3796pp at40 words. Wrong-parent degradation is
-0.8218pp versus hard's-0.7774pp: the extra0.0444pp damage is small, but does not
meet the frozen no-worse condition. No rule/count switch or promotion follows.
Retain the established full-suite hard model and the previously stronger frozen20
transfer candidate, with its narrower evidence; do not call it expansion-safe.

Candidate30 is only0.0536pp below candidate20; candidate40 loses1.2143pp. Specific
40-word pools improve VDD and UDD5 but hurt OEM, Vaihingen and FLAIR-1. The count
and phrase identities change together: neither a universal20-word optimum nor
universal benefit/harm from adding words follows. Legitimate replacements improve
the mean0.5686pp, and wrong-parent replacements improve some individual domains,
so construction labels are not reliable final prediction oracles either.

The immutable sign rule cannot reject broad-positive/fine-nonnegative evidence.
For added40-word aliases,90.09%-94.40% of eligible query/alias/rival comparisons
remain retained across protocols. Of their broad-positive comparisons,71.33%-85.15%
are uncontradicted by fine. The post-prediction wrong-parent audit also finds
uncontradicted broad advantages, but none of these counts measures distinct error
pixels or proves which phrase causes a final error. Agreement is not correctness.

OEM building demonstrates lost coverage rather than extra building false positives:
candidate20->40 IoU54.5491->7.6778, recall70.7286->7.7172, precision70.4544->93.7719,
and predicted area11.9585%->0.9803%, with243289 target pixels unchanged. VDD wall
loses15.9226pp while vehicle gains17.3988pp under the same expansion. This is
consistent with redistributed class competition, not a diagnosis that all added
words are harmful or that suppression should simply become stronger. LoveDA P
building support is only307 pixels and should not dominate conclusions.

The next selector must demonstrate useful changes to actual conditional contributions,
not merely reshape the existing contradiction risk or globally blacklist aliases.
Prior ownership, mass and soft attenuation failures remain relevant: do not revive
them from these labels under a new name. Cache/graph execution is already verified
equivalent; removing fine observations remains a different, previously weaker rule.
Current results establish neither that every word is useful nor that soft weighting
is impossible. The strongest retained new candidate is still hard source admission
with soft rival coordination, not a continuous per-alias weight. The research goal
remains open; all candidates, outputs and vocabularies are retained.

## Per-Domain Results

| Dataset/protocol | Scenario | No admission | Original hard | Candidate | Candidate-hard pp |
| --- | --- | ---: | ---: | ---: | ---: |
| vdd/vdd | k20 | 53.7470 | 54.3441 | 54.1183 | -0.2258 |
| vdd/vdd | k30 | 56.3427 | 59.4798 | 59.7020 | +0.2222 |
| vdd/vdd | k40 | 53.8382 | 56.8961 | 57.4750 | +0.5789 |
| vdd/vdd | wrong_parent | 47.4088 | 49.8744 | 49.5236 | -0.3507 |
| vdd/vdd | paraphrase | 54.5307 | 53.1031 | 53.5458 | +0.4427 |
| potsdam/potsdam | k20 | 38.9203 | 40.6173 | 41.1231 | +0.5058 |
| potsdam/potsdam | k30 | 40.4884 | 42.0377 | 42.3003 | +0.2626 |
| potsdam/potsdam | k40 | 37.9945 | 39.6864 | 40.1319 | +0.4455 |
| potsdam/potsdam | wrong_parent | 36.0706 | 37.4898 | 38.0572 | +0.5674 |
| potsdam/potsdam | paraphrase | 41.6633 | 43.5511 | 44.0639 | +0.5128 |
| udd5/udd5 | k20 | 28.1758 | 34.0394 | 34.6174 | +0.5780 |
| udd5/udd5 | k30 | 33.1717 | 38.5768 | 38.4878 | -0.0890 |
| udd5/udd5 | k40 | 31.0774 | 36.5909 | 36.5185 | -0.0724 |
| udd5/udd5 | wrong_parent | 27.6166 | 31.6763 | 31.7720 | +0.0957 |
| udd5/udd5 | paraphrase | 31.1693 | 36.2215 | 36.4067 | +0.1853 |
| oem/oem | k20 | 39.0232 | 39.7906 | 40.4484 | +0.6578 |
| oem/oem | k30 | 32.5577 | 33.3079 | 33.7154 | +0.4074 |
| oem/oem | k40 | 31.6448 | 32.4874 | 32.8811 | +0.3937 |
| oem/oem | wrong_parent | 40.0533 | 41.3962 | 41.6142 | +0.2180 |
| oem/oem | paraphrase | 39.6194 | 40.4645 | 40.9347 | +0.4702 |
| loveda/P | k20 | 50.7066 | 52.5879 | 52.6436 | +0.0557 |
| loveda/D | k20 | 30.7360 | 37.2156 | 37.6023 | +0.3867 |
| loveda/P | k30 | 53.7666 | 56.8817 | 54.1237 | -2.7580 |
| loveda/D | k30 | 33.6186 | 37.1572 | 37.5578 | +0.4006 |
| loveda/P | k40 | 53.4619 | 57.4164 | 56.0367 | -1.3796 |
| loveda/D | k40 | 32.8865 | 37.4571 | 37.6099 | +0.1527 |
| loveda/P | wrong_parent | 49.9897 | 53.1129 | 53.8062 | +0.6933 |
| loveda/D | wrong_parent | 30.3335 | 36.0609 | 37.2366 | +1.1757 |
| loveda/P | paraphrase | 48.7357 | 50.5932 | 50.2645 | -0.3286 |
| loveda/D | paraphrase | 29.5140 | 37.0266 | 36.9648 | -0.0618 |
| vaihingen/vaihingen | k20 | 51.8270 | 52.9446 | 53.3059 | +0.3613 |
| vaihingen/vaihingen | k30 | 48.6944 | 50.2587 | 50.5371 | +0.2784 |
| vaihingen/vaihingen | k40 | 47.7031 | 49.2751 | 49.4814 | +0.2063 |
| vaihingen/vaihingen | wrong_parent | 53.0602 | 54.5650 | 55.0127 | +0.4477 |
| vaihingen/vaihingen | paraphrase | 52.5185 | 54.1384 | 54.3697 | +0.2312 |
| landcoverai/landcoverai | k20 | 66.9060 | 67.4834 | 67.7836 | +0.3002 |
| landcoverai/landcoverai | k30 | 67.1040 | 67.4379 | 67.5423 | +0.1043 |
| landcoverai/landcoverai | k40 | 67.0446 | 67.2510 | 67.3024 | +0.0515 |
| landcoverai/landcoverai | wrong_parent | 68.1235 | 69.3668 | 69.3972 | +0.0305 |
| landcoverai/landcoverai | paraphrase | 65.6296 | 65.9775 | 66.2833 | +0.3058 |
| flair1/flair1 | k20 | 33.6084 | 34.0624 | 34.3507 | +0.2883 |
| flair1/flair1 | k30 | 32.3083 | 32.8371 | 33.0785 | +0.2414 |
| flair1/flair1 | k40 | 31.7477 | 32.1119 | 32.2348 | +0.1230 |
| flair1/flair1 | wrong_parent | 33.2469 | 33.8489 | 34.1619 | +0.3129 |
| flair1/flair1 | paraphrase | 34.5301 | 35.0649 | 35.3300 | +0.2651 |

## Per-Class Expansion And Attachment Effects

Exact confusion-derived metrics; changes compare the SAME projected candidate across pools, not different readers. Precision/recall/area for every arm/scenario are in summary.json. Classes absent from targets can have false positives; zero target support does not establish recall.

| Dataset/protocol | Class | Target pixels | Candidate20 IoU | Candidate20 precision | Candidate20 recall | Candidate30-20 IoU pp | Candidate40-20 IoU pp | Wrong-parent-20 IoU pp | Paraphrase-20 IoU pp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | other | 1070714 | 55.0600 | 80.8961 | 63.2893 | +7.4178 | +3.8170 | -4.0628 | -1.4182 |
| vdd/vdd | wall | 166833 | 53.1449 | 57.0418 | 88.6096 | +2.0614 | -15.9226 | -11.6989 | -5.3907 |
| vdd/vdd | road | 69605 | 21.6618 | 21.7082 | 99.0245 | +1.5292 | +10.0774 | +5.8423 | +0.7416 |
| vdd/vdd | vegetation | 483080 | 54.2677 | 94.7736 | 55.9419 | +14.1361 | +10.7758 | -4.8655 | +7.0549 |
| vdd/vdd | vehicle | 7510 | 38.2597 | 38.2597 | 100.0000 | +14.1551 | +17.3988 | -13.2664 | -5.5492 |
| vdd/vdd | roof | 126362 | 89.8815 | 90.3136 | 99.4706 | +0.0745 | +0.5550 | +2.5415 | +1.6844 |
| vdd/vdd | water | 173048 | 66.5525 | 68.9089 | 95.1129 | -0.2884 | -3.2046 | -6.6530 | -1.1303 |
| potsdam/potsdam | impervious surface | 857241 | 66.3665 | 85.2000 | 75.0146 | -5.5475 | -8.4655 | +3.0789 | -3.3327 |
| potsdam/potsdam | building | 338121 | 72.8663 | 76.9013 | 93.2829 | +4.9106 | +6.3824 | +3.3540 | +0.9354 |
| potsdam/potsdam | low vegetation | 363664 | 21.7324 | 81.8728 | 22.8310 | +12.4768 | +4.3633 | -6.9790 | +18.7386 |
| potsdam/potsdam | tree | 370013 | 58.0561 | 92.6395 | 60.8635 | -0.2518 | -2.2862 | -22.5668 | +3.1642 |
| potsdam/potsdam | car | 75196 | 25.1793 | 25.2161 | 99.4242 | -5.2004 | -6.5432 | +3.1899 | -3.2777 |
| potsdam/potsdam | clutter | 92917 | 2.5381 | 3.2649 | 10.2339 | +0.6752 | +0.6019 | +1.5277 | +1.4167 |
| udd5/udd5 | vegetation | 64744 | 68.5613 | 90.9603 | 73.5744 | +2.8707 | +3.2062 | -0.0426 | +2.8671 |
| udd5/udd5 | building | 1508385 | 85.8800 | 86.0701 | 99.7434 | -0.1240 | -1.3790 | +1.8428 | -0.1010 |
| udd5/udd5 | road | 89116 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| udd5/udd5 | vehicle | 8808 | 6.6312 | 6.6313 | 99.9773 | +4.0622 | +2.7473 | -4.0723 | +0.7441 |
| udd5/udd5 | other | 426099 | 12.0146 | 44.6062 | 14.1216 | +12.5430 | +4.9307 | -11.9550 | +5.4365 |
| oem/oem | bareland | 0 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| oem/oem | rangeland | 297086 | 49.8592 | 70.6753 | 62.8643 | -0.7447 | -2.8282 | +0.2133 | +1.1231 |
| oem/oem | developed space | 400255 | 23.4870 | 41.7734 | 34.9185 | -1.0149 | -3.4514 | +8.2227 | -5.3289 |
| oem/oem | road | 128408 | 55.4142 | 68.4843 | 74.3824 | -1.5158 | +0.9009 | +0.9190 | +0.5461 |
| oem/oem | tree | 539563 | 57.7324 | 90.8900 | 61.2783 | -3.4208 | -4.0521 | -1.6010 | +1.3688 |
| oem/oem | water | 503 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| oem/oem | agriculture land | 433260 | 82.5457 | 82.9578 | 99.4017 | -2.0779 | -4.2366 | +0.0467 | +1.1901 |
| oem/oem | building | 243289 | 54.5491 | 70.4544 | 70.7286 | -45.0905 | -46.8713 | +1.5256 | +4.9912 |
| loveda/P | building | 307 | 48.9292 | 49.7487 | 96.7427 | +8.7776 | +17.6652 | -0.4443 | -10.2642 |
| loveda/P | road | 96423 | 60.1611 | 61.4250 | 96.6927 | -2.2684 | -1.7954 | +0.2438 | -3.0116 |
| loveda/P | water | 170486 | 62.9242 | 68.0543 | 89.3018 | -0.7053 | +0.5904 | -2.0627 | +0.5698 |
| loveda/P | barren | 128461 | 1.6724 | 41.1787 | 1.7134 | +0.8114 | +3.8162 | -0.1686 | -0.0289 |
| loveda/P | tree | 190765 | 56.5911 | 96.7147 | 57.7003 | +0.5695 | -1.5801 | +6.5216 | -1.7408 |
| loveda/P | farm | 572179 | 85.5838 | 85.8926 | 99.5816 | +1.6959 | +1.6623 | +2.8858 | +0.2010 |
| loveda/D | background | 938531 | 30.6934 | 84.6654 | 32.5002 | +6.7647 | +5.9404 | -6.4407 | -2.9483 |
| loveda/D | building | 307 | 31.5625 | 31.6946 | 98.6971 | +4.3828 | -2.9911 | +8.9233 | +0.3502 |
| loveda/D | road | 96423 | 49.2751 | 50.1318 | 96.6481 | -0.5092 | -1.2554 | -0.6371 | -3.3106 |
| loveda/D | water | 170486 | 55.0011 | 58.9493 | 89.1446 | +1.5822 | +2.5457 | -3.2314 | +0.5024 |
| loveda/D | barren | 128461 | 1.6177 | 66.8262 | 1.6308 | +0.0193 | +0.5248 | -0.7005 | -0.0737 |
| loveda/D | tree | 190765 | 48.4719 | 92.5745 | 50.4327 | -13.1728 | -5.2919 | +0.0198 | +2.3856 |
| loveda/D | farm | 572179 | 46.5944 | 47.1297 | 97.6203 | +0.6217 | +0.5805 | -0.4936 | -1.3682 |
| vaihingen/vaihingen | impervious surface | 572872 | 59.3766 | 75.4775 | 73.5690 | -10.1670 | -12.2986 | +2.1064 | +0.6009 |
| vaihingen/vaihingen | building | 432655 | 67.4674 | 67.6072 | 99.6944 | +3.7550 | +2.8429 | +6.7871 | +0.9699 |
| vaihingen/vaihingen | low vegetation | 620087 | 45.7742 | 95.8882 | 46.6907 | +3.4516 | +1.9290 | +5.8442 | +4.6500 |
| vaihingen/vaihingen | tree | 432823 | 67.6134 | 78.6701 | 82.7907 | -0.1450 | -0.1786 | -0.4955 | +0.0121 |
| vaihingen/vaihingen | car | 38715 | 26.2980 | 26.4463 | 97.9130 | -10.7389 | -11.4174 | -5.7084 | -0.9142 |
| landcoverai/landcoverai | background | 1421030 | 88.2370 | 95.1004 | 92.4393 | -0.0412 | -0.3243 | +0.9789 | -0.9584 |
| landcoverai/landcoverai | building | 31773 | 44.5937 | 44.9185 | 98.4043 | -0.7754 | -0.1164 | +6.2822 | -5.2056 |
| landcoverai/landcoverai | woodland | 448933 | 81.5413 | 93.8335 | 86.1583 | +3.1786 | +4.1953 | +1.8965 | +0.2195 |
| landcoverai/landcoverai | water | 173462 | 97.6079 | 97.6079 | 100.0000 | +0.0456 | -0.5440 | -0.1185 | -0.1475 |
| landcoverai/landcoverai | road | 21954 | 26.9379 | 29.4886 | 75.6946 | -3.6141 | -5.6163 | -0.9709 | -1.4096 |
| flair1/flair1 | building | 150504 | 57.4041 | 58.6643 | 96.3928 | -0.6239 | +1.3862 | -3.5856 | -1.3941 |
| flair1/flair1 | pervious surface | 359720 | 46.8072 | 91.9102 | 48.8185 | -2.5283 | -6.8430 | +0.2025 | +2.6284 |
| flair1/flair1 | impervious surface | 343424 | 52.9006 | 60.8089 | 80.2670 | +2.8662 | +5.5579 | +0.0069 | +0.4135 |
| flair1/flair1 | bare soil | 0 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| flair1/flair1 | water | 92992 | 64.7837 | 66.9300 | 95.2835 | -1.1757 | +2.9474 | +4.4355 | +4.1994 |
| flair1/flair1 | coniferous | 11784 | 42.7265 | 62.4459 | 57.5017 | -1.4030 | +0.7917 | -9.8484 | +0.1459 |
| flair1/flair1 | deciduous | 356824 | 60.9048 | 78.7253 | 72.9040 | -2.1172 | -3.0248 | -3.9532 | +1.1229 |
| flair1/flair1 | brushwood | 103104 | 15.6306 | 29.7469 | 24.7769 | -0.3259 | -0.8877 | +5.2278 | +2.3766 |
| flair1/flair1 | vineyard | 0 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| flair1/flair1 | herbaceous vegetation | 671552 | 57.6203 | 91.5155 | 60.8721 | -6.6579 | -19.3207 | +4.1525 | +1.7489 |
| flair1/flair1 | agricultural land | 6392 | 13.4305 | 14.5039 | 64.4712 | -3.3008 | -5.9970 | +1.0959 | +0.5100 |
| flair1/flair1 | plowed land | 0 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Uncontradicted Broad-Advantage Diagnostics

Noncanonical query/alias/rival comparisons only; own-parent rivals and invalid queries excluded. broad>0 and fine>=0 is retained by construction, regardless of whether its parent is correct. These are source comparisons, not distinct pixels or evidence that each causes a final error.

| Dataset/protocol | Scenario | Subset | Broad advantages uncontradicted % | Retained % | Audit wrong-parent uncontradicted % | Wrong-parent broad advantages uncontradicted % |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| vdd/vdd | k20 | all | 74.9771 | 89.5065 | 2.7741 | 29.2071 |
| vdd/vdd | k20 | old20 | 74.9771 | 89.5065 | 2.7741 | 29.2071 |
| vdd/vdd | k30 | all | 74.4159 | 89.7749 | 2.6956 | 28.2745 |
| vdd/vdd | k30 | old20 | 75.5558 | 89.6844 | 3.0518 | 30.2737 |
| vdd/vdd | k30 | added | 71.8573 | 89.9470 | 2.0188 | 23.7665 |
| vdd/vdd | k40 | all | 74.6397 | 90.1691 | 3.4461 | 32.0028 |
| vdd/vdd | k40 | old20 | 76.1301 | 89.8390 | 3.6607 | 32.5926 |
| vdd/vdd | k40 | added | 72.9251 | 90.4826 | 3.2422 | 31.3936 |
| vdd/vdd | wrong_parent | all | 72.4145 | 90.2789 | 5.6271 | 39.6551 |
| vdd/vdd | wrong_parent | old20 | 72.4145 | 90.2789 | 5.6271 | 39.6551 |
| vdd/vdd | wrong_parent | replacement_slots | 72.0867 | 86.9081 | 14.9078 | 51.3067 |
| vdd/vdd | paraphrase | all | 72.9316 | 89.0827 | 2.8279 | 30.6864 |
| vdd/vdd | paraphrase | old20 | 72.9316 | 89.0827 | 2.8279 | 30.6864 |
| vdd/vdd | paraphrase | replacement_slots | 61.1063 | 76.9671 | 5.3619 | 26.0033 |
| potsdam/potsdam | k20 | all | 83.4407 | 93.0751 | 6.5769 | 51.3149 |
| potsdam/potsdam | k20 | old20 | 83.4407 | 93.0751 | 6.5769 | 51.3149 |
| potsdam/potsdam | k30 | all | 82.8813 | 93.3744 | 6.8240 | 53.7046 |
| potsdam/potsdam | k30 | old20 | 83.7369 | 93.1932 | 7.2747 | 53.2727 |
| potsdam/potsdam | k30 | added | 80.8015 | 93.7188 | 5.9675 | 54.7325 |
| potsdam/potsdam | k40 | all | 82.6017 | 93.4073 | 7.3914 | 53.6410 |
| potsdam/potsdam | k40 | old20 | 83.8356 | 93.0371 | 8.6342 | 54.5728 |
| potsdam/potsdam | k40 | added | 81.0702 | 93.7589 | 6.2107 | 52.4580 |
| potsdam/potsdam | wrong_parent | all | 81.9642 | 93.0982 | 7.9605 | 55.4782 |
| potsdam/potsdam | wrong_parent | old20 | 81.9642 | 93.0982 | 7.9605 | 55.4782 |
| potsdam/potsdam | wrong_parent | replacement_slots | 78.3003 | 89.5388 | 13.8367 | 58.7884 |
| potsdam/potsdam | paraphrase | all | 82.8614 | 93.0997 | 6.2207 | 50.4029 |
| potsdam/potsdam | paraphrase | old20 | 82.8614 | 93.0997 | 6.2207 | 50.4029 |
| potsdam/potsdam | paraphrase | replacement_slots | 74.4973 | 85.5388 | 9.6167 | 42.2912 |
| udd5/udd5 | k20 | all | 77.9532 | 91.4107 | 3.9250 | 51.4333 |
| udd5/udd5 | k20 | old20 | 77.9532 | 91.4107 | 3.9250 | 51.4333 |
| udd5/udd5 | k30 | all | 78.7511 | 92.4804 | 3.8543 | 54.9610 |
| udd5/udd5 | k30 | old20 | 78.3884 | 91.2026 | 4.3036 | 55.2522 |
| udd5/udd5 | k30 | added | 79.8609 | 94.9083 | 3.0005 | 54.1827 |
| udd5/udd5 | k40 | all | 78.6074 | 92.6671 | 4.1449 | 55.5736 |
| udd5/udd5 | k40 | old20 | 78.2342 | 91.1348 | 4.6804 | 56.4116 |
| udd5/udd5 | k40 | added | 79.1203 | 94.1227 | 3.6362 | 54.5821 |
| udd5/udd5 | wrong_parent | all | 73.2390 | 92.0619 | 8.5062 | 60.8081 |
| udd5/udd5 | wrong_parent | old20 | 73.2390 | 92.0619 | 8.5062 | 60.8081 |
| udd5/udd5 | wrong_parent | replacement_slots | 77.1287 | 90.6517 | 24.6094 | 69.8545 |
| udd5/udd5 | paraphrase | all | 71.1530 | 88.7789 | 3.5873 | 45.3289 |
| udd5/udd5 | paraphrase | old20 | 71.1530 | 88.7789 | 3.5873 | 45.3289 |
| udd5/udd5 | paraphrase | replacement_slots | 39.1714 | 68.7436 | 4.3900 | 28.2710 |
| oem/oem | k20 | all | 85.8912 | 94.2361 | 4.6888 | 46.1320 |
| oem/oem | k20 | old20 | 85.8912 | 94.2361 | 4.6888 | 46.1320 |
| oem/oem | k30 | all | 85.2771 | 94.0910 | 5.4040 | 45.9780 |
| oem/oem | k30 | old20 | 85.8880 | 94.2353 | 5.2831 | 46.4818 |
| oem/oem | k30 | added | 84.0543 | 93.8168 | 5.6337 | 45.1068 |
| oem/oem | k40 | all | 85.5485 | 94.2968 | 5.9498 | 47.7126 |
| oem/oem | k40 | old20 | 86.3735 | 94.1911 | 6.2278 | 48.6455 |
| oem/oem | k40 | added | 84.6321 | 94.3972 | 5.6857 | 46.7791 |
| oem/oem | wrong_parent | all | 83.0521 | 94.0469 | 7.2002 | 51.2188 |
| oem/oem | wrong_parent | old20 | 83.0521 | 94.0469 | 7.2002 | 51.2188 |
| oem/oem | wrong_parent | replacement_slots | 83.1684 | 92.6964 | 14.6152 | 60.7687 |
| oem/oem | paraphrase | all | 85.5442 | 94.1552 | 4.6189 | 47.8258 |
| oem/oem | paraphrase | old20 | 85.5442 | 94.1552 | 4.6189 | 47.8258 |
| oem/oem | paraphrase | replacement_slots | 83.3817 | 91.8068 | 6.8725 | 50.8193 |
| loveda/P | k20 | all | 71.4933 | 88.6012 | 4.5055 | 44.5624 |
| loveda/P | k20 | old20 | 71.4933 | 88.6012 | 4.5055 | 44.5624 |
| loveda/D | k20 | all | 72.9872 | 89.1484 | 6.7517 | 46.0604 |
| loveda/D | k20 | old20 | 72.9872 | 89.1484 | 6.7517 | 46.0604 |
| loveda/P | k30 | all | 72.1052 | 88.9798 | 4.3263 | 44.7020 |
| loveda/P | k30 | old20 | 73.2753 | 88.7813 | 4.8653 | 44.3689 |
| loveda/P | k30 | added | 69.4239 | 89.3569 | 3.3023 | 45.6615 |
| loveda/D | k30 | all | 74.3118 | 89.7091 | 6.3698 | 45.9399 |
| loveda/D | k30 | old20 | 74.6043 | 89.3884 | 7.0655 | 45.9147 |
| loveda/D | k30 | added | 73.6807 | 90.3185 | 5.0480 | 46.0069 |
| loveda/P | k40 | all | 73.4521 | 89.6635 | 4.2239 | 43.9613 |
| loveda/P | k40 | old20 | 75.2287 | 89.2190 | 4.9591 | 43.6144 |
| loveda/P | k40 | added | 71.3276 | 90.0858 | 3.5254 | 44.4336 |
| loveda/D | k40 | all | 75.0581 | 90.2203 | 6.3078 | 44.5556 |
| loveda/D | k40 | old20 | 76.3713 | 89.7803 | 7.5387 | 46.2362 |
| loveda/D | k40 | added | 73.5327 | 90.6383 | 5.1384 | 42.4073 |
| loveda/P | wrong_parent | all | 70.9341 | 89.8479 | 6.8744 | 52.7510 |
| loveda/P | wrong_parent | old20 | 70.9341 | 89.8479 | 6.8744 | 52.7510 |
| loveda/P | wrong_parent | replacement_slots | 68.3014 | 87.6627 | 13.9078 | 61.5738 |
| loveda/D | wrong_parent | all | 71.8766 | 89.9641 | 8.3935 | 47.8474 |
| loveda/D | wrong_parent | old20 | 71.8766 | 89.9641 | 8.3935 | 47.8474 |
| loveda/D | wrong_parent | replacement_slots | 69.7949 | 88.2642 | 11.1287 | 48.1038 |
| loveda/P | paraphrase | all | 71.2039 | 88.6653 | 4.5622 | 44.2746 |
| loveda/P | paraphrase | old20 | 71.2039 | 88.6653 | 4.5622 | 44.2746 |
| loveda/P | paraphrase | replacement_slots | 70.2955 | 84.8462 | 6.5137 | 45.4889 |
| loveda/D | paraphrase | all | 72.2445 | 88.9945 | 6.9605 | 45.5401 |
| loveda/D | paraphrase | old20 | 72.2445 | 88.9945 | 6.9605 | 45.5401 |
| loveda/D | paraphrase | replacement_slots | 70.1338 | 84.8010 | 9.9701 | 50.0408 |
| vaihingen/vaihingen | k20 | all | 80.3058 | 91.7737 | 6.3252 | 47.2755 |
| vaihingen/vaihingen | k20 | old20 | 80.3058 | 91.7737 | 6.3252 | 47.2755 |
| vaihingen/vaihingen | k30 | all | 79.7872 | 92.3512 | 6.3548 | 49.1351 |
| vaihingen/vaihingen | k30 | old20 | 80.0406 | 91.7499 | 6.5393 | 47.1335 |
| vaihingen/vaihingen | k30 | added | 79.1495 | 93.4937 | 6.0043 | 53.8687 |
| vaihingen/vaihingen | k40 | all | 79.4979 | 92.6581 | 7.1066 | 50.3940 |
| vaihingen/vaihingen | k40 | old20 | 80.1275 | 91.4918 | 8.2860 | 49.6750 |
| vaihingen/vaihingen | k40 | added | 78.6198 | 93.7662 | 5.9862 | 51.3717 |
| vaihingen/vaihingen | wrong_parent | all | 79.8635 | 92.7263 | 7.3221 | 49.4189 |
| vaihingen/vaihingen | wrong_parent | old20 | 79.8635 | 92.7263 | 7.3221 | 49.4189 |
| vaihingen/vaihingen | wrong_parent | replacement_slots | 80.9764 | 89.8915 | 18.2388 | 55.9925 |
| vaihingen/vaihingen | paraphrase | all | 80.0716 | 92.1620 | 5.9035 | 45.5825 |
| vaihingen/vaihingen | paraphrase | old20 | 80.0716 | 92.1620 | 5.9035 | 45.5825 |
| vaihingen/vaihingen | paraphrase | replacement_slots | 76.9667 | 86.3451 | 11.0229 | 41.7923 |
| landcoverai/landcoverai | k20 | all | 85.9246 | 93.8749 | 1.4435 | 45.6054 |
| landcoverai/landcoverai | k20 | old20 | 85.9246 | 93.8749 | 1.4435 | 45.6054 |
| landcoverai/landcoverai | k30 | all | 85.7919 | 94.1496 | 1.9588 | 52.7055 |
| landcoverai/landcoverai | k30 | old20 | 86.6847 | 94.2254 | 1.9228 | 49.5837 |
| landcoverai/landcoverai | k30 | added | 83.8045 | 94.0055 | 2.0273 | 59.4505 |
| landcoverai/landcoverai | k40 | all | 84.6126 | 93.9923 | 2.3783 | 54.4521 |
| landcoverai/landcoverai | k40 | old20 | 86.3046 | 94.0497 | 2.3865 | 51.0742 |
| landcoverai/landcoverai | k40 | added | 82.6092 | 93.9378 | 2.3705 | 58.1291 |
| landcoverai/landcoverai | wrong_parent | all | 84.1947 | 94.1737 | 3.3108 | 54.9050 |
| landcoverai/landcoverai | wrong_parent | old20 | 84.1947 | 94.1737 | 3.3108 | 54.9050 |
| landcoverai/landcoverai | wrong_parent | replacement_slots | 83.3268 | 92.5415 | 10.3149 | 63.5398 |
| landcoverai/landcoverai | paraphrase | all | 86.0046 | 94.2033 | 1.9404 | 48.0453 |
| landcoverai/landcoverai | paraphrase | old20 | 86.0046 | 94.2033 | 1.9404 | 48.0453 |
| landcoverai/landcoverai | paraphrase | replacement_slots | 84.0028 | 90.5840 | 5.0148 | 47.4345 |
| flair1/flair1 | k20 | all | 85.8162 | 94.2625 | 6.5250 | 62.5685 |
| flair1/flair1 | k20 | old20 | 85.8162 | 94.2625 | 6.5250 | 62.5685 |
| flair1/flair1 | k30 | all | 85.2523 | 94.3123 | 8.0040 | 62.8305 |
| flair1/flair1 | k30 | old20 | 85.1627 | 94.4731 | 7.1409 | 62.8800 |
| flair1/flair1 | k30 | added | 85.4067 | 94.0067 | 9.6438 | 62.7610 |
| flair1/flair1 | k40 | all | 84.8801 | 94.3141 | 9.5967 | 65.0998 |
| flair1/flair1 | k40 | old20 | 84.5675 | 94.5275 | 8.0477 | 65.3718 |
| flair1/flair1 | k40 | added | 85.1457 | 94.1113 | 11.0683 | 64.9133 |
| flair1/flair1 | wrong_parent | all | 85.2495 | 94.2648 | 6.6343 | 63.1407 |
| flair1/flair1 | wrong_parent | old20 | 85.2495 | 94.2648 | 6.6343 | 63.1407 |
| flair1/flair1 | wrong_parent | replacement_slots | 87.1137 | 93.1430 | 11.7447 | 64.4966 |
| flair1/flair1 | paraphrase | all | 85.6940 | 94.3268 | 6.1406 | 61.5505 |
| flair1/flair1 | paraphrase | old20 | 85.6940 | 94.3268 | 6.1406 | 61.5505 |
| flair1/flair1 | paraphrase | replacement_slots | 86.8334 | 92.9519 | 11.6282 | 62.1890 |

Targets are read only for this post-prediction audit. A true-parent mismatch at a centre does not certify a phrase is semantically misattached: a legitimate alias may also fire falsely, and Geometry/class competition can change the final outcome. Do not tune a new rule from these targets within this study.

## Shared Evaluation Cost

| Domain | Five-scenario/four-arm wall s | Peak allocated MiB |
| --- | ---: | ---: |
| vdd | 9.883 | 5851.875 |
| potsdam | 7.815 | 5748.650 |
| udd5 | 7.501 | 5681.804 |
| oem | 9.483 | 6009.466 |
| loveda | 13.582 | 6062.671 |
| vaihingen | 6.930 | 5681.804 |
| landcoverai | 6.972 | 5681.804 |
| flair1 | 13.726 | 6692.016 |

Shared costs exclude loading/text encoding and are not independent per-model inference latency. Packed masks/scores remain at /data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/rival_projected_vocabulary_stress_20261005/DATASET/source_cache/; collected results/confusions/logs and summary are under research/rival_projected_vocabulary_stress_20261005/.

All old models, vocabulary pools and every scheme remain preserved. No full20092 rollout or post-result promotion.
