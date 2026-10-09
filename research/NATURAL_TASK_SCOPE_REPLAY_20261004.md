# Task-scope replay and calibration stability

This CPU-only experiment reproduces the existing VOC20/VOC21 scope replay from
the two frozen64-image selections. Ordered image keys, witness coordinates,
foreground class order and checkpoints match. No GT, new image inference,
vocabulary changes or deployed selection changes are involved.

The rule keeps a VOC20 witness only when the paired VOC21 image-derived witness
is trusted foreground of the same semantic class. Its new weight is the
geometric mean of the two original quality weights. The original profile rule
is unchanged: retain default unless the upper95 image-block error difference
favors the frozen profile.

## Reproduced result

| Quantity | Value |
| --- | ---: |
| Original trusted coordinates | 22937 |
| Retained same-class foreground coordinates | 12810 |
| Trusted background challenges | 4845 |
| Trusted foreground-class disagreements | 0 |
| Original witnesses without trusted open-world support | 5282 |
| Original frozen-minus-default pseudo error (pp) | -6.0965 |
| Original upper95 difference (pp) | -2.6725 |
| Scoped frozen-minus-default pseudo error (pp) | -0.4996 |
| Scoped upper95 difference (pp) | +1.2656 |

These values exactly reproduce the saved earlier snapshot. The full-image
original rule selects frozen; the scoped rule retains default. Their already
completed, matched1449-image outputs score89.6655 and91.8121 mIoU, respectively.
That2.1466pp difference is reuse of existing evaluated arms, not a newly run
benchmark or a word-selection gain.

## New Out-of-Fold Check

Images retain their frozen order. Eight folds hold out indices modulo8; each
profile decision uses only the other56 images. Both rules are then scored on
the same scoped witnesses in the eight held-out images. No parameters or fold
count are optimized from these outcomes.

| Fold | Original-rule choice | Scoped-rule choice | Original-rule scoped error (%) | Scoped-rule scoped error (%) |
| --- | --- | --- | ---: | ---: |
| 0 | frozen | default | 0.7790 | 0.5928 |
| 1 | frozen | default | 5.2893 | 1.9673 |
| 2 | frozen | default | 6.8625 | 6.7751 |
| 3 | frozen | default | 0.4102 | 0.2410 |
| 4 | frozen | default | 0.9306 | 1.5757 |
| 5 | frozen | default | 0.2795 | 0.3179 |
| 6 | frozen | default | 0.0541 | 7.1900 |
| 7 | frozen | default | 0.0921 | 0.0343 |
| All64 held-out images | frozen in8/8 folds | default in8/8 folds | 1.8372 | 2.3368 |

The scoped rule is stable across training folds, but its held-out pseudo error
is0.4996pp worse. For image2007_003137, scoped default/frozen errors are50.1508%
and0.4327%; this is influential pseudo-label evidence, not proof of a true
semantic error or a harmful alias. The direction of the pseudo-error comparison
does not match the full GT mIoU comparison.

Scope correction removes a demonstrated calibration contamination, but it does
not validate the pseudo-label objective as an IoU surrogate. The full mIoU
advantage here is consistent with conservative retention of default settings;
it cannot establish that a learned lexical ranking or more aggressive confidence
fitting will transfer. Do not promote this diagnostic to an alias-selection
module or change the active frozen queues. The fixed-setting qualified-word
comparison remains the next test of actual lexical input changes.

This hypothesis was motivated by a prior labeled development audit. The replay
itself reads no GT; these are exploratory developed-set diagnostics, not untouched
independent validation or CVPR/SOTA evidence. Context59/60 replay is supported by
the script but has not been run because the paired selections are not ready.

## Reproduction

```powershell
& F:/APP/codetool/anaconda3/python.exe tools/task_scope_witness_replay.py --pair voc --collect
& F:/APP/codetool/anaconda3/python.exe -m unittest discover -s DINOtool/tests -p test_task_scope_witness_replay.py -v
```

Omit `--collect` to reuse the downloaded mask-free inputs. Exact per-image
errors, fold decisions and metadata are in
`research/natural_task_scope_replay_20261004/voc_results.json`.
Five tests cover alignment, frozen mask-free inputs, input preservation, missing
scope support and exclusion of held-out image blocks from profile selection.
