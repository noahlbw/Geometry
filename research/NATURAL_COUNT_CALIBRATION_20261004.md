# Natural-image calibration: verified diagnosis and isolated next test

## Verified full results

The existing natural-text suite completed both VOC protocols with 1,449 unique
images each. Its merged files verify global coverage, checkpoint/text identity,
frozen head weights and per-image confusion sums. Other domains remain running.

| Protocol | Uncalibrated semantic pool | Adaptive, no rejection | Adaptive, with rejection | VIP official finite |
| --- | ---: | ---: | ---: | ---: |
| VOC20 | 91.8146 | 89.6651 | 89.6651 | 92.5051 |
| VOC21 | 57.3617 | 48.8813 | 63.6504 | 73.2591 |

The VOC20 pool and adaptive arm use the same aliases and seg templates; no alias
was deleted and no background threshold is applied. The -2.1495pp loss therefore
comes from the tau/tem calibration change, not vocabulary deletion. On VOC21,
the same parameter change loses 8.4804pp without rejection. Adding the frozen
image-only threshold recovers 14.7691pp, for a net +6.2887pp over the pool.
This does not establish that alias screening improved segmentation.

The pinned finite-row VIP comparator scores are consistent with the supplied
VOC table to rounding, but this does not certify all other protocols. The repair
is Self-Value only on all-masked proxy rows. These are local reproduction results,
not newly reported published benchmark measurements.

## Where the remaining VOC21 gap lies

| Metric | Pool | Adaptive, no rejection | Adaptive, with rejection | VIP |
| --- | ---: | ---: | ---: | ---: |
| Foreground-only mIoU | 56.3328 | 47.8526 | 62.4487 | 72.3575 |
| Background IoU | 77.9412 | 69.4546 | 87.6851 | 91.2911 |
| True foreground sent to background (%) | 2.9412 | 1.2566 | 27.4514 | 10.3416 |
| True background sent to foreground (%) | 21.2245 | 30.2278 | 3.5550 | 5.2731 |
| Predicted background area (%) | 58.5414 | 51.4909 | 78.0361 | 72.2112 |

Ground-truth background occupies 73.3180% of scored pixels. The calibrated arm
is too aggressive at rejecting foreground despite its higher overall mIoU.
Person recall is 55.9680% versus VIP 91.3107%; motorbike recall is 46.0076%
versus 94.5322%; chair recall is 36.8350% versus 66.5543%. A higher global
rejection threshold is not the supported next action.

All five mask-free NLL searches selected seg_template, tau=5 and tem=0.3, at the
same grid boundaries. NLL is computed on a no-admission coupling proxy, whereas
deployment includes the full fine-rival admission potential. Confidence ranking
on that proxy has now failed a full same-vocabulary control on VOC20.

## Count bias in variable-size vocabularies

The inherited wide score contains log-sum-exp rather than log-mean-exp. If the
profiled alias evidences are identical, a class with n aliases receives an
additional log(n)/tau compared with a singleton. Salience normalization does
not remove this aggregation offset. For the current selected vocabularies:

| Protocol | Smallest/largest alias count | Maximum pure-count gap at tau=1 | At tau=5 |
| --- | ---: | ---: | ---: |
| VOC20 | 3 / 23 | 2.036882 | 0.407376 |
| VOC21 | 3 / 56 | 2.926739 | 0.585348 |
| ADE150 | 1 / 24 | 3.178054 | 0.635611 |
| COCO Stuff171 | 1 / 17 | 2.833213 | 0.566643 |
| COCO Object81 | 3 / 32 | 2.367124 | 0.473425 |

These are analytic logit offsets under identical profiled evidence, not measured
IoU losses or estimates of the entire real vocabulary's effect.

## Implemented diagnostic

The main model and existing runs are unchanged. A separate calibration interface
subtracts a strength-scaled broad count prior before the original reconstruction.
For original scores S = L + H(B - L + P), its exact scalar-prior replay is:

```text
S_alpha = S - alpha * (H @ ones)[:, None] * log(n)[None, :] / tau
alpha in {0, 0.5, 1}
```

Geometry L, reconstruction H and the existing fine-rival potential P are retained.
This is an exact counterfactual of this specified score-offset operation on a
central tile, not an exact counterfactual of deleting an alias everywhere. The
equal-count case leaves per-tile class probabilities unchanged. A test of
duplicate-only wide evidence verifies removal of its pure size advantage; it
does not establish invariance of all full-model admission decisions to arbitrary
vocabulary duplication.

The calibration driver reuses the original fixed, at-most-64 image keys, includes
the actual fine-rival output and never loads masks. Candidate ranking uses
class-balanced pseudo-witness disagreement rather than NLL. It retains alpha=0
unless at least eight image blocks show a negative upper-95 paired error margin.
That normal-approximation margin is a selection heuristic, not proof of improved
ground-truth IoU. No new full-image rejection threshold is fitted in this test.

Files:

- `DINOtool/dinotool/natural_count_calibration.py`
- `DINOtool/scripts/calibrate_natural_count_prior.py`
- `DINOtool/tests/test_natural_count_calibration.py`
- `tools/natural_count_calibration_experiment.py`

Sixteen math and input-contract tests now pass locally and on A800. A separate
paired evaluator tests external count calibration and taxonomy-only word
curation; no running model source was replaced. The manager checks GPU idleness,
refuses existing outputs and requires completed image smoke before selection.

New CUDA launches are restricted to physical GPUs 4-7, with the existing
idle-GPU and no-overwrite checks retained. Existing full evaluations are not
restarted or interrupted.

Two CPU-only smoke attempts failed and their separate logs were preserved:
the BF16 attempt raised `RuntimeError: Unexpected floating ScalarType in
at::autocast::prioritize`; the FP16 attempt raised
`ValueError: Nonfinite observer features after empty-row fallback.` Neither
produced a selection. CPU compatibility work is stopped; the next smoke uses
the original CUDA observer on an idle GPU 4-7, without changing the main model.

## Verified CUDA and paired-pilot outcome

VOC20 and VOC21 CUDA image-only smokes passed on GPUs4 and5. Both full64-image
selections completed without masks or head changes; both retained strength0.
VOC20 half normalization reduced mean pseudo-error by0.5904pp, but its upper95
paired delta was+0.0774pp, so the frozen rule rejected it. On VOC21, half/full
normalization increased mean pseudo-error by2.5740/6.0637pp.

Full-image source-reader equivalence smokes and16-image paired pilots completed
for both protocols. All new control confusion matrices exactly match the
existing pilot, with identical global image-key SHA and source-selection SHA.
Each shard also checked exact source prediction equality on its first image.
With the same class-support mask shared by all old/new pilot arms:

| Protocol | Source | Image-selected count calibration | Full norm diagnostic | Diagnostic delta |
| --- | ---: | ---: | ---: | ---: |
| VOC20 | 59.8371 | 59.8371 | 58.5861 | -1.2510pp |
| VOC21 | 40.5066 | 40.5066 | 40.0506 | -0.4560pp |

These are16-image diagnostic scores, not full-validation benchmark results.
There is no gain from the retained count candidate; its zero fallback avoids
the normalization loss. Neither case justifies launching another identical
full evaluation or selecting the full-normalization arm from target labels.
This rules out explicit scalar multiplicity removal as an improvement on these
pilots, not every possible form of vocabulary calibration or sense screening.
The separate ADE/COCO-Stuff word-only curation tests remain in progress.

The explicit log(n)/tau offset is only one count effect: within-class salience
profiling also uses n*softmax(salience). This diagnostic deliberately leaves
that profiling and the actual fine-rival potential unchanged.

See `research/NATURAL_ALIAS_CALIBRATION_PAIRED_20261004.md` and its downloaded
selection/paired-confusion artifacts for exact frozen choices and verification.

The interpretation remains exploratory: this next test was motivated by the
completed VOC development analysis. Image-only calibration is transductive, not
untouched independent validation or evidence of a CVPR-ready contribution.

## Sources

- `research/natural_text_adaptation_20261003/voc20/full_merged.json`
- `research/natural_text_adaptation_20261003/voc21/full_merged.json`
- `research/natural_text_adaptation_20261003/*/selection.json`
- `research/NATURAL_TEXT_ADAPTATION_20261003.md`
