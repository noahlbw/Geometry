# Frozen Contextual Alias Excess Rejection

Signature: geometry-canonical-excess-alias-rejection-v1-20261003. Primary: ExcessReject_Coupled.

The previous calibrated group-action screen rejected its own advancement; its results remain unchanged. This successor implements the proposed suppression-only writer and replaces alias-dependent pseudo-reference selection with canonical descriptors on the existing Geometry relation. The rule is frozen before target metrics. It is not an Astra priority claim or proof of novelty.

## Source And Writer

Canonical descriptors are the original public class names encoded with the same RS templates as aliases. A tested noncanonical alias's semantic contradiction against rival d is positive cosine-to-d minus cosine-to-parent, divided by its across-class cosine range. Zero range and all nonpositive differences are unknown. Canonical aliases are protected.

Original Geometry relation G is masked to valid donors outside the query's64px cell, then row-normalized. Canonical class softmax at temperature0.07 supplies spatial support, not all20 alias predictions. Fewer than4 effective donors or zero mass is unknown. These constants are inherited from prior geometry/alias protocols, not fitted here. Shared encoder mistakes remain possible.

Rival leakage is max((supported_prob_d-supported_prob_parent)/(supported_prob_d+supported_prob_parent),0). Contextual gain is clamp((wide_alias_cosine-local_alias_cosine)/0.07,0,1). Both cosines use the SAME RS text bank; the original wide prediction keeps its original ImageNet templates and salience. Risk=max_d(text_contradiction*leakage)*gain. Retention=1-risk. Unknown retains1. Every rival is considered, not only the original top2.

For original fixed salience-profiled logit r and q=softmax(beta*r):

```text
delta = log(1 - sum_a (1-rho_a)*max(q_a-1/K,0)) / beta
b_corrected = b_original + composed_crop_interpolation(delta)
```

No new class-mass redistribution or reliability-scaled exponent. rho=1 is exact replay; equal profiled evidence is neutral; lowering retention cannot raise that class score; the log argument is>=1/K. This writer cannot fix equally wrong responses from every alias. It controls the broad evidence only, not the unchanged local anchor.

Final inference uses unchanged anchored reconstruction: min_z .5||z-g||^2+.5||G(z-b_corrected)||^2. No new supervised loss, encoder, crop, dataset parameter, background threshold, or solver.

## Matched Controls And Data

Nine arms: Geometry, BroadVIP, Anchored_VIP, ExcessReject_Coupled, TextOnlyReject_Coupled, HardDelete_Coupled, and three shuffled rejection controls. Shuffles preserve exact per-query/class retention spectra. TextOnly uses the same source and writer without image conditioning. HardDelete actually removes every alias with positive primary risk, normalizes its remaining count, and leaves at least the canonical alias; it does not re-profile original salience. It is not the official VIP filter or the earlier fixed15 selector.

Clean run:96 fixed complete images, UDD5 full40 and seven other datasets8 each, exact previous manifests. Stress runs: first8 of each dataset,64 images/scenario. LoveDA P/D share images and means count D once. Corrected IRRG Vaihingen, LandCover.ai substitution. These are development data, not untouched full-dataset validation.

Stress changes broad vocabulary ONLY; local original20 anchor is fixed. Each class keeps20 entries and its canonical phrase. Replace the last4 noncanonical entries using a cyclic rival from public class ordering (wrong-parent) or an overhead-view paraphrase of the original entries (appearance/control). Exact replacements are released. Paraphrases are a controlled meaning-preserving intent, not a guarantee of equal model embeddings. Cyclic ontology neighbors can share meaning. These are constructed perturbations, NOT fresh LLM outputs, and cannot alone substantiate an arbitrary-LLM robustness claim.

Masks load only after all predictions for that image. Inspect mIoU, class IoU/area/precision/recall, direct beneficial/harmful corrections, nonzero rejection, and shuffled controls. Verify exact three historical controls, fixed clean vocab/checkpoints/profile, unique complete coverage, stress Geometry invariance, and per-image confusion/transition reconstruction.

## Prospective Scheduling Gate

Two routes are declared before smoke or labeled scores. Neither is a statistical significance test:

- Clean-gain route: mean>=+0.1pp vs original coupling, >=5/8 wins, worst protocol loss<=1pp including LoveDA P, mean above text-only and each shuffle.
- Robustness route: clean mean loss<=0.1pp and worst<=0.5pp; wrong-parent mean gain>=0.5pp and >=5/8 wins; wrong-parent mean above text-only, hard deletion, and each shuffle; paraphrase mean loss<=0.1pp and worst<=0.5pp.

Passing either justifies a next real raw-vocabulary test and standalone cost assessment; it does not automatically launch20,092-image validation or establish a final CVPR model. Failing both preserves original Geometry/coupling and rejects this fixed source/operator. Do not select per-domain arms or tune the constants from outcomes.

Authorized A800 GPUs0-7, occupancy checks before smoke and every worker. Existing outputs and unrelated processes/sessions are never overwritten or stopped. Primary inference/model code is isolated from original implementations.
