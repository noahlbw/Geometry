# Cross-Support Conditional Alias Evidence: Frozen Diagnostic

This is a diagnostic, not a new selector, soft weighting model or parameter search.
The retained Geometry patch-only strength2 + wide coupled model and its original20
alias banks are unchanged. Variable vocabularies follow only if this hypothesis
survives. Sources use at most4 local512 and4 wide336 encodings, with0 fine encodings.

Use the previous developed complete-image panel: all40 UDD5 images and8 evenly
spaced images in each other RS domain (96 unique inputs). Original per-image
confusions, vocabulary, checkpoints and ordered keys must match the old bounded
alias-path audit. Exact old single-alias removals are reused, not reevaluated.

For every unordered frozen coupled top2 class pair in a tile, form a support
distribution from the mean original Geometry W rows at its competing queries.
Remove diagonal self-support and padded tokens; do not substitute signed H.
Fit on columns0..13 and evaluate on columns18..31 of the32x32 patch grid, then
swap. The four intervening columns are omitted. Require32 positions and Kish
effective support32 in BOTH folds, otherwise report insufficient support.

The pseudo-reference is the native-head canonical owner-minus-rival text margin.
It is not supplied to the baseline predictor. The baseline predicts this margin
from intercept plus four class-only log-mean-exp scores (local owner/rival,
wide owner/rival). The extension adds one profiled, interpolated wide alias logit.
Remove the tested owner alias and owner near duplicates (encoded text cosine>=.95)
from BOTH baseline pools. Candidates near the owner's canonical text are excluded
from this test, not deleted from inference. Require at least2 remaining aliases.

Weighted train-only standardization and ridge.01 are fixed a priori.
The class-only fit is frozen; project the candidate alias on its training
class inputs by weighted least squares, and fit a scalar ridge residual increment.
An exactly redundant feature therefore cannot gain by splitting a ridge penalty.
Report held-out MSE reduction and three controls that independently permute the alias's
responses inside training/testing folds while keeping class inputs unchanged.
This breaks spatial correspondence; renaming or permuting feature columns would
not. No weights, thresholds, routes or deletion policy are selected by this test.

Persist all source statistics before loading the current image's target mask.
Afterward audit native canonical prediction and its alias extension using
balanced owner/rival Brier on held-out donor positions (at least4 positions of
each class). Join the per-image/class/alias statistics to existing complete-image
Local/Wide/Joint fixed-slot and survivor-normalized deletion confusions. Report
own TP coverage lost, FP reduction, and full-image mIoU deletion effect. Test
word-identity specificity by permuting scores within each image/class; compare
within-group centered evidence with centered action benefit so class calibration
alone cannot establish an alias mechanism. LoveDA P/D separate, D once in means.

Interpretation: held-out pseudo-reference prediction gain is necessary evidence
for this particular estimator, not proof of semantic correctness. Spatial folds
share backbone attention, Geometry and wide crops, and are not independent visual
measurements. Label auditing is not a deployable training-free result. A useful
alias need not have a harmful deletion, and a redundant alias need not be harmful.
The panel is developed/exploratory; no claim of untouched validation is justified.

Decision: report separately (1) reproducible proxy increments beyond spatial
nulls, (2) true owner/rival discrimination, and (3) association with real frozen
competitive actions. If only(1) is present, do not derive soft weights from it.
If(2)/(3) lack support or disagree across domains, report uncertainty/failure
instead of promoting another final model. Variable-vocabulary testing is deferred.
