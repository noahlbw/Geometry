# Cross-Support Conditional Alias Evidence

Frozen Geometry patch2 + wide coupling, original20 aliases and same96 developed complete RS images. This tests information/action identification, not a new deployed model or mIoU improvement.

Class-only predictions are frozen before a train-only residual alias extension. Left/right supports are separated by four patch columns. Original W, not signed H, supplies support. Native canonical contrast is a pseudo-reference; spatially disjoint folds still share visual encoders.

| Domain/protocol | Proxy MSE reduction % | Above spatial null % | True Brier gain | Brier-gain 95% image interval | Tested / eligible folds |
| --- | ---: | ---: | ---: | --- | ---: |
| vdd/vdd | -10.99246 | -10.51605 | 0.00019 | [-0.00088, 0.00107] | 8740/13502 |
| potsdam/potsdam | -5.52094 | -4.75717 | -0.00035 | [-0.00096, 0.00026] | 10610/15214 |
| udd5/udd5 | -9.69719 | -8.91086 | 0.00031 | [-0.00009, 0.00068] | 40466/56494 |
| oem/oem | -0.62594 | 0.05620 | 0.00026 | [0.00010, 0.00046] | 11530/15320 |
| loveda/P | -9.64835 | -8.82425 | 0.00052 | [-0.00041, 0.00155] | 10926/13646 |
| loveda/D | -8.10544 | -7.31458 | -0.00016 | [-0.00047, 0.00019] | 13586/17742 |
| vaihingen/vaihingen | -7.07082 | -6.11874 | 0.00012 | [-0.00016, 0.00040] | 13810/17710 |
| landcoverai/landcoverai | -3.28943 | -3.04869 | 0.00013 | [-0.00051, 0.00097] | 4546/4936 |
| flair1/flair1 | -8.17963 | -7.57850 | -0.00098 | [-0.00268, 0.00024] | 16634/23250 |

Positive keep-value correlation means that larger proxy increment predicts a more harmful deletion. Scores/actual effects are ranked and centered within each image/class; a class-constant adjustment has zero identifiable word advantage.

| Exact deletion action | Within-image/class keep-value rank correlation | Alias-identity shuffle95 |
| --- | ---: | --- |
| Local_FixedSlots | 0.00167 | [-0.02021, 0.02016] |
| Local_SurvivorNormalized | 0.00648 | [-0.02324, 0.02392] |
| Wide_FixedSlots | 0.00416 | [-0.02398, 0.02503] |
| Wide_SurvivorNormalized | 0.01291 | [-0.02226, 0.02152] |
| Joint_FixedSlots | -0.00465 | [-0.01916, 0.02151] |
| Joint_SurvivorNormalized | 0.00998 | [-0.02050, 0.02147] |
| DirectWide_FixedSlots | -0.00105 | [-0.02303, 0.02099] |
| DirectWide_SurvivorNormalized | 0.00595 | [-0.02110, 0.01542] |

Equal-domain summaries (LoveDA D once): {"proxy_relative_reduction": -0.0668523051034029, "proxy_above_spatial_null": -0.060235492630396445, "brier_increment": -5.980757492374033e-05, "brier_above_spatial_null": -2.086219305738505e-05}.

The 95% intervals resample complete images1000 times; correlated folds, class pairs and aliases are not independent sample units. Alias-identity controls use199 within-image/class permutations. They diagnose identity correspondence, not independent validation or a deployable selector.

Source statistics were persisted before each mask. No audit label fitted coefficients or any weight/deletion decision. Near-canonical aliases were excluded only from this information test. Complete-image baseline confusions and ordered source/checkpoint/vocabulary identities replay exactly. Joined actions preserve exact TP loss, FP reduction and mIoU outcomes for all archived paths/pools.

Detailed protocol/status counts, proxy/null statistics, semantic audit and per-domain action association: summary.json. Per-image/class/word exact action joins: alias_action_joins.json. Runtime is diagnostic execution including fitting and source collection, not inference latency. Variable-vocabulary experiments and any soft-weight rule remain deferred.

## Interpretation And Decision

The current estimator fails the practical gate. The alias extension increases
held-out proxy MSE by6.6852% in the equal-domain mean; it is6.0235% worse than the
spatial-null extensions relative to class-only error. All eight primary domain
proxy means are negative. This is an error-prediction metric, NOT an mIoU delta.

There are some reproducible proxy increments: of59961 eligible bidirectional
tile/pair/alias tests (LoveDA D once),15086 improve both directions and11502 beat
all three spatial controls in both directions. Equal-domain both-positive rate
is25.2331%, versus19.8511% for spatial-null fits. These correlated, numerous tests
suggest that word-specific structure is not entirely absent; they do not certify
which alias action will improve segmentation or justify selecting these cases
after seeing the held-out outcomes. Their positive subset cannot erase the
negative all-case average.

True owner/rival balanced Brier improvement averages-0.00005981 across domains:
essentially no useful aggregate gain. OEM has a small positive image-bootstrap
interval; the other primary domains' intervals cross zero. OEM's wide deletion
association is nevertheless negative. Conversely, FLAIR-1 has an exploratory
positive wide-action correlation0.0906 (uncorrected identity permutation p.005),
but negative average Brier gain; this does not pass the joint semantic/action gate.

For8319 informative noncanonical image/class/alias entries, the primary wide
fixed-slot keep-value rank correlation is0.00416, inside the alias-identity null
interval[-0.02398,0.02503] (one-sided p.41). Local and joint fixed-slot correlations
are0.00167 and-0.00465. These scores cannot reliably identify real complete-image
retention value beyond class-level calibration on this panel.

Two boundaries matter. First, native canonical contrast is neither ground truth
nor an independent visual measurement: correctly predicting it may reproduce its
semantic errors. Left/right extrapolation also tests a stationary conditional
relation across different image regions, which need not hold. The diagnostic
class inputs pool aligned/interpolated alias fields; they are not the exact VIP
crop-pool-before-interpolation logits. Actual retained coupled predictions were
not changed and replayed exactly. Second, the archived action join removes one
alias throughout an image, whereas a future conditional action would affect only
specified rival/support locations. The near-zero association rejects this score
as a reliable image-level deletion predictor; it cannot rule out every localized
competitive policy.

Do not choose sigmoid/softmax weights, a deletion threshold or a variable word
budget from these outcomes. Preserve the retained model. The next missing
evidence is a mask-free estimate that predicts the SIGN of a matched, localized
competitive action, including its actual coupled writeback, rather than merely
predicting a native-head canonical reference. Define and test that quantity before
promoting a weighting mechanism. This diagnostic changes no final model.

## Verification

All96 complete inputs finished on idle GPUs0-7. Original per-image baseline
confusions, keys, vocabulary and checkpoints match the archived audit. Persisted
pre-mask statistics are unchanged in the final labeled records for every image.
Eight predictor tests, three identity-join tests and nine existing path-audit
tests passed. Existing outputs and unrelated tmux sessions were preserved.
