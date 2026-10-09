# Controlled relation between taxonomy and readout gain

## Interpretation of the controlled evidence

At fixed strength2, natural rank-odds quantization selects the labelled-development best gain in6/7 natural protocols: VOC20/21 g2; PC59/60 g1; ADE150 and COCO-Stuff171 g.5. COCO-Object81 is the counterexample: rank chooses g1, but g2 scores41.5929 versus39.3094 (+2.2835pp). VOC protocols and PC protocols share source images, so these are not7 independent domains. The gain menu/rule also has prior development provenance; this is descriptive evidence, not independent validation or a discovered law.

The same rank rule does not transfer directly to the four remote-sensing banks. It chooses g2 for all, but original ImageNet VDD/Potsdam prefer g.5; focused VDD prefers g1, focused Potsdam prefers g.5 at fixed strength2. Thus rank/class-count alone cannot determine universal coupling. The two VDD banks share images/taxonomy yet prefer different gains, directly showing representation/word-bank interaction without changing dataset identity.

Read-strength interaction matters: focused Potsdam prefers g.5 at strength1/2/3 but g2 on the original read operator; PC60 prefers g1 at strength1/2/3 but g2 on the original operator. Strength and coupling cannot be interpreted as independent settings. Within natural strength2→3, best-gain gains are small: VOC21 +.1194pp, PC60 +.0334pp, PC59 +.0051pp, VOC20 +.0105pp. These numbers alone do not justify an unverified full-model upgrade.

ADE development comes from96 training images; the other development cohorts are fixed labelled validation subsets. Full-test results have already informed earlier work, so do not call these untouched holdouts. The PC60 grid here predates the extra401-concept residual/protected route; its optimum is not verified for that deployed information budget. Alias banks/templates remain fixed within each row, but differ across protocols. Effective rank and class count are confounded across this small task set; neither establishes that fine-grained labels require a particular gain.

Decision: retain the already verified family-conditioned selector, with RS conservative g.5 and natural rank menu, rather than importing the natural rank rule into RS. No parameter or vocabulary changes are made by this analysis. A stronger generality claim needs frozen-rule evaluation on a genuinely excluded task, with representation and residual ontology accounted for; fitting a new rule to these full-test winners is not that evidence.

Existing labelled development only; frozen bank, T.07/tau1/tem1, zero bias and no rejection. No independent generalization claim or new parameter selection.

Strength2 is held fixed in this table to avoid confusing read-strength and gain. Each bank is held fixed. The best gain is a labelled-development oracle, not a deployable adaptive decision. The three gains and previous rank rule are development-derived.

| Protocol / bank | Classes | Dev images | Canonical rank fraction | g=.5 | g=1 | g=2 | Dev best g | Rank g | Rank regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/original_imagenet | 7 | 16 | 0.8042 | 51.7598 | 50.9039 | 48.5532 | 0.5 | 2.0 | 3.2066 |
| vdd/focused20 | 7 | 16 | 0.7735 | 46.8279 | 50.0866 | 49.5126 | 1.0 | 2.0 | 0.5741 |
| potsdam/original_imagenet | 6 | 64 | 0.7676 | 47.8009 | 46.1352 | 43.4602 | 0.5 | 2.0 | 4.3407 |
| potsdam/focused20 | 6 | 64 | 0.7463 | 50.0980 | 48.7388 | 46.1986 | 0.5 | 2.0 | 3.8994 |
| voc21/semantic_segmentation | 21 | 64 | 0.6243 | 48.1061 | 61.1775 | 70.0255 | 2.0 | 2.0 | 0.0000 |
| context60/semantic_segmentation | 60 | 64 | 0.5225 | 37.6942 | 38.9987 | 38.1652 | 1.0 | 1.0 | 0.0000 |
| coco_object81/semantic_segmentation | 81 | 64 | 0.4719 | 34.7548 | 39.3094 | 41.5929 | 2.0 | 1.0 | 2.2835 |
| ade150/semantic_segmentation | 150 | 96 | 0.3416 | 29.2834 | 27.9600 | 24.3769 | 0.5 | 0.5 | 0.0000 |
| context59/semantic_segmentation | 59 | 64 | 0.5297 | 40.4564 | 41.3745 | 40.4022 | 1.0 | 1.0 | 0.0000 |
| voc20/semantic_segmentation | 20 | 64 | 0.6300 | 90.0077 | 92.7477 | 92.9958 | 2.0 | 2.0 | 0.0000 |
| coco_stuff171/semantic_segmentation | 171 | 64 | 0.3005 | 32.0210 | 31.0684 | 29.1279 | 0.5 | 0.5 | 0.0000 |

## Read-strength / gain interaction

| Protocol / bank | Strength | g=.5 | g=1 | g=2 | Dev best g |
| --- | --- | ---: | ---: | ---: | ---: |
| vdd/original_imagenet | 1.0 | 52.5654 | 51.6824 | 48.8055 | 0.5 |
| vdd/original_imagenet | 2.0 | 51.7598 | 50.9039 | 48.5532 | 0.5 |
| vdd/original_imagenet | 3.0 | 51.2820 | 50.5073 | 48.3750 | 0.5 |
| vdd/original_imagenet | original | 51.6645 | 50.7855 | 48.5407 | 0.5 |
| vdd/focused20 | 1.0 | 47.2762 | 50.5637 | 49.7488 | 1.0 |
| vdd/focused20 | 2.0 | 46.8279 | 50.0866 | 49.5126 | 1.0 |
| vdd/focused20 | 3.0 | 47.0105 | 49.9146 | 49.3472 | 1.0 |
| vdd/focused20 | original | 50.2743 | 51.9608 | 49.5030 | 1.0 |
| potsdam/original_imagenet | 1.0 | 47.0528 | 45.6519 | 43.4934 | 0.5 |
| potsdam/original_imagenet | 2.0 | 47.8009 | 46.1352 | 43.4602 | 0.5 |
| potsdam/original_imagenet | 3.0 | 48.1616 | 46.2960 | 43.4191 | 0.5 |
| potsdam/original_imagenet | original | 46.4781 | 45.3149 | 43.3892 | 0.5 |
| potsdam/focused20 | 1.0 | 47.3008 | 47.2950 | 46.1947 | 0.5 |
| potsdam/focused20 | 2.0 | 50.0980 | 48.7388 | 46.1986 | 0.5 |
| potsdam/focused20 | 3.0 | 51.5147 | 49.3749 | 46.1760 | 0.5 |
| potsdam/focused20 | original | 45.2773 | 46.0227 | 46.0525 | 2.0 |
| voc21/semantic_segmentation | 1.0 | 43.1312 | 58.4853 | 69.7223 | 2.0 |
| voc21/semantic_segmentation | 2.0 | 48.1061 | 61.1775 | 70.0255 | 2.0 |
| voc21/semantic_segmentation | 3.0 | 51.1561 | 62.7851 | 70.1449 | 2.0 |
| voc21/semantic_segmentation | original | 42.6178 | 57.7840 | 69.3966 | 2.0 |
| context60/semantic_segmentation | 1.0 | 37.0667 | 38.6566 | 38.0857 | 1.0 |
| context60/semantic_segmentation | 2.0 | 37.6942 | 38.9987 | 38.1652 | 1.0 |
| context60/semantic_segmentation | 3.0 | 37.8831 | 39.0321 | 38.1741 | 1.0 |
| context60/semantic_segmentation | original | 35.2597 | 37.7441 | 38.0151 | 2.0 |
| coco_object81/semantic_segmentation | 1.0 | 33.8481 | 38.7152 | 41.4138 | 2.0 |
| coco_object81/semantic_segmentation | 2.0 | 34.7548 | 39.3094 | 41.5929 | 2.0 |
| coco_object81/semantic_segmentation | 3.0 | 35.0166 | 40.1409 | 41.7255 | 2.0 |
| coco_object81/semantic_segmentation | original | 34.3635 | 39.0730 | 41.4323 | 2.0 |
| ade150/semantic_segmentation | 1.0 | 29.9941 | 28.6644 | 24.3944 | 0.5 |
| ade150/semantic_segmentation | 2.0 | 29.2834 | 27.9600 | 24.3769 | 0.5 |
| ade150/semantic_segmentation | 3.0 | 28.6336 | 27.4240 | 24.3283 | 0.5 |
| ade150/semantic_segmentation | original | 30.8089 | 29.0408 | 24.2959 | 0.5 |
| context59/semantic_segmentation | 1.0 | 39.5659 | 41.0224 | 40.3112 | 1.0 |
| context59/semantic_segmentation | 2.0 | 40.4564 | 41.3745 | 40.4022 | 1.0 |
| context59/semantic_segmentation | 3.0 | 40.6789 | 41.3796 | 40.4159 | 1.0 |
| context59/semantic_segmentation | original | 37.4248 | 39.5053 | 40.2039 | 2.0 |
| voc20/semantic_segmentation | 1.0 | 89.8281 | 92.4674 | 92.9687 | 2.0 |
| voc20/semantic_segmentation | 2.0 | 90.0077 | 92.7477 | 92.9958 | 2.0 |
| voc20/semantic_segmentation | 3.0 | 90.1013 | 92.8527 | 93.0063 | 2.0 |
| voc20/semantic_segmentation | original | 89.1797 | 92.3620 | 92.9590 | 2.0 |
| coco_stuff171/semantic_segmentation | 1.0 | 32.1863 | 31.4695 | 29.1174 | 0.5 |
| coco_stuff171/semantic_segmentation | 2.0 | 32.0210 | 31.0684 | 29.1279 | 0.5 |
| coco_stuff171/semantic_segmentation | 3.0 | 31.5309 | 30.8459 | 29.1230 | 0.5 |
| coco_stuff171/semantic_segmentation | original | 31.6941 | 31.2792 | 29.1862 | 0.5 |
