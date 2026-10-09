# Complete Model Planning After The Native-Query Experiment

## Latest Follow-Up

The subsequently frozen vocabulary stress study is complete: clean mean
42.867949 -> 43.241696 and wrong-parent mean41.989160 -> 42.251500,
but its gate FAILS against hard deletion and matched random/alias controls.
No selector or control is promoted. Current complete-model planning is in
`GEOMETRY_COMPLETE_MODEL_POST_NOISE_PLAN_20261003.md`; detailed verified results
are in `NATIVE_ALIAS_NOISE_SCREEN_20261003.md`. Earlier failed gates below stand.

## Decision And Evidence

The requested next experiment is complete. The retained executable model stays
original Geometry plus the attributed wide observer and anchored reconstruction.
Do NOT promote the new selector, a winning control, or a per-dataset combination
to the final paper model. The full useful-controller/CVPR objective is unachieved.

Same64 developed IDs, eight/domain, one top-left512 window/image, original whole
image broad observation. Frozen weights, exactly20 aliases/class, fixed equations
and config. Masks only evaluate already-persisted predictions, not inference.
All eight domains informed research; no independent-validation claim. Corrected
IRRG; LandCover.ai replaces unlabeled iSAID. LoveDA P/D share images and the
equal-domain mean counts D once. These are NOT full eight-dataset metrics.

| Dataset/protocol | Retained coupling | Native query plus Geometry witness | Delta pp | Native query only | Native observation anchored |
| --- | ---: | ---: | ---: | ---: | ---: |
| VDD |53.746954|54.108731|+0.361777|54.157343|42.463741|
| Potsdam |38.920258|38.978671|+0.058413|39.008602|40.186786|
| UDD5 |28.175811|28.884694|+0.708883|28.982212|37.613470|
| OEM |39.023152|39.107565|+0.084412|39.144152|42.324986|
| LoveDA P |50.706552|51.118473|+0.411921|52.053326|51.274060|
| LoveDA D |30.736030|31.572239|+0.836208|31.959082|35.547668|
| Vaihingen |51.826962|52.090467|+0.263505|52.149506|56.001372|
| LandCover.ai |66.906020|66.915048|+0.009028|66.893754|66.613584|
| FLAIR-1 |33.608404|33.628472|+0.020068|33.638912|33.609461|
| Equal-domain mean, D once |42.867949|43.160736|+0.292787|43.241696|44.295134|

Primary beats new shared support43.014270, shuffled support42.919055, all three
alias shuffles43.093171/43.130909/43.106226 and directional matched controls
42.946484-42.972402 on mean. This is limited useful-allocation evidence beyond
those controls, not significance or proof of semantic correctness. The mean
improvement over the strongest alias shuffle is only0.029826pp.

The predeclared gate FAILS: query-only and direct-native controls are stronger,
as are prior collapsed/rival/matched sources43.548346/43.448829/43.300974. Keep
failed gates intact. Eight positive domain deltas do not prove an all-eight
full-dataset winner. Selection rule and thresholds were not changed after results.

## What The Experiment Actually Establishes

1. A physically fine unmasked observation is a useful new hypothesis. Replacing
   the broad source by native336 observations and applying the same old anchored
   writer gives44.295134, +1.427185pp over retained coupling and +0.777347pp over
   native mean-logit fusion. However VDD loses11.283213pp and LandCover.ai0.292436pp.
   It is a control, not a selected final model or a universal fine-view solution.
2. Query-specific Geometry support is better than cell-average or shuffled
   support on the mean, but requiring its agreement as an extra veto suppresses
   useful actions too. Query-only beats primary by0.080960pp and in7/8 domains,
   except LandCover.ai. The equation makes primary risk pointwise no greater
   than query-only risk. These controls establish a tradeoff, not its unique cause.
3. The current alias controller only weakly repairs precision. Potsdam car
   IoU24.5378->24.5646, TP+13/FP-279, recall99.1010%, precision24.6195%.
   VDD vehicle50.3084->48.0825, TP+6/FP+703, recall100%.
   UDD5 vehicle5.3645->5.8714, TP0/FP-14174. These windows do not demonstrate
   general small-object recovery. UDD5 road has89116 target pixels but0 TP in
   both retained/primary outputs; do not describe it as an absent true class.
4. Native anchoring can also worsen small objects: Potsdam car21.6833 and VDD
   vehicle10.1448, despite useful average gains. Higher physical resolution is
   not calibrated class recognition. Aliases and competing classes remain coupled.
5. Therefore neither increasing rejection nor replacing all wide views by native
   views is supported as a universal next step. Broad-only attenuation also
   cannot repair every error in the unchanged local anchor or canonical response.

## Retained Model And The Final Architecture Target

Use TWO mechanisms, with alias handling INSIDE mechanism2:

```text
image + class names + candidate aliases
                 |
       frozen DINOv3 / DINO.text
                 |
      +----------+--------------------------------+
      |                                           |
1. Geometry local reader                 contextual observations
native-mass-preserving head              original broad source
local descriptor Y, scores g             optional physically fine witness
query relation G                         attributed frozen observer/templates
      |                                           |
      +------> 2. competitive contextual admission <+
               per query x alias x rival
               retain useful/unknown evidence
               attenuate only supported excess contribution
                         |
               one coherent admitted class field b*
                         |
               Geometry-anchored reconstruction
               z = g + H_G(b*-g)
                         |
               unchanged dense assembly/prediction
```

Only mechanism1 and mechanism2 are candidate contributions. Frozen encoders,
LLM candidate generation, the VIP observer, template ensemble, interpolation
and classical solvers are attributed components, not five more inventions.
The native witness does NOT replace Geometry's local anchor. The currently
retained model has no promoted alias admission; b*=b.

### Mechanism1: Geometry Remains Fixed

Preserve `A_special V_special + m_patch G V_patch` and the frozen projection,
residual, MLP and normalization in both head blocks. Geometry's distinct claim
is native conditional allocation/mass preservation, not inventing DINO affinity.
Preserve the original local normalized alias log-mean-exp at0.07. Its pixelwise
responsibilities are dynamic evidence influence, NOT semantic quality estimates.
Do not refit Geometry on these repeatedly inspected target labels.

### Mechanism2: Narrow Its Responsibility

The intended interface is gamma(i,a,d) in[0,1], not a global good-word list.
A phrase may be ambiguous against B yet useful against C. No forced survivor
count; unknown remains neutral. The tested native source compares actual
profiled alias contribution with actual rival-class log-mean-exp before exact
interpolation. All observations use matched templates/profiling; this is
not a semantic oracle.

Keep the accountable writer unless a separate experiment changes it:

```text
q(c,a) = softmax(beta*r(c,a)), K20/beta1
d(c|d) = log(1-sum_a gamma(c,a,d)*max(q(c,a)-1/K,0))/beta
e(c,d) = d(c|d)-d(d|c)
v_c = mean_d e(c,d)
b* = b+v
H_G = (I+G^T G)^(-1)G^T G
z = g+H_G(b*-g)
```

These last consistency/reconstruction equations are classical least squares.
Zero gamma recovers the retained model exactly. Retained useful aliases are not
newly boosted; their original contributions remain. Positive v is relative
multiclass competition, not a positive quality label for an alias. A/B and A/C
final margins cannot be independently prescribed with one multiclass score field.

This experiment argues AGAINST making G an obligatory extra semantic veto.
The minimal next source candidate is the already-declared native-query-only
witness, leaving Geometry responsible for the visual reading and writeback.
It is a candidate for a new frozen comparison, NOT a promoted winner selected
after seeing this screen. No new strength, canonical exception or dataset
routing is justified by these results. Its clean-vocabulary gain is insufficient
to claim a complete universally useful selector.

## Next Work In Order

1. Do not launch full eight datasets from this failed gate. Freeze the architecture
   above as the research target, while keeping original coupling executable.
2. Judge contextual admission under a predeclared independent vocabulary regime:
   public raw-LLM candidates plus fixed legitimate shared/wrong-parent distractors.
   Distinguish word-quality changes from count changes. Labels audit actions only;
   no words, thresholds, temperatures or seeds are chosen by target mIoU. Use
   all-retained, hard deletion, count-matched random, text-only, alias/action
   shuffles and matched-source fusion. Test whether a visual witness adds value
   beyond a text attachment check; correctly attached but false car responses
   must not be conflated with roof incorrectly attached to wall.
3. Any future successor needs a prospective criterion that separates source
   relevance, word/location allocation and reconstruction advantage. Preserve
   this failed gate, rather than changing its threshold to promote query-only.
   Compare complete predictions, class precision/recall/area and changed-class
   transitions; one stronger ranking metric is not a useful final reader.
4. After a new mechanism actually passes, verify COMPLETE images on fixed
   development IDs, then freeze once for full eight-domain evaluations and an
   independent region/vocabulary. Keep local and broad admission claims separate.
   If adding native semantic scores themselves, it is a new source/writer
   experiment with matched information/cost, not a free extension of this source.
5. Report full official VIP fairly, attributed matched20 adaptations separately,
   Geometry's nearest-method boundary, image/scene uncertainty, size/boundary
   errors, and standalone whole-predictor cost. General OVSS claims also need
   natural-image evaluation with shared general templates and no domain routing.

The paper story is NOT that publication requires deleting some of20 words.
It is that semantic evidence has a bounded, query- and rival-dependent role
inside Geometry-grounded reading. Wrong parent attachments, true shared language,
and visual false activations are different failure types. Only demonstrated
reliability on those regimes can support the proposed second contribution.

If realistic-noise experiments still cannot beat simple/text-only and matched
allocation controls, remove the unsupported visual selector claim. Write the
focused Geometry/coupling paper with a clearly attributed observer and proven
nearest-method distinction, or acquire genuinely different semantic information.
Do not manufacture a third module, reinterpret a control as a prospective win,
or promise CVPR acceptance from current scores.

## Verification And Artifacts

Corrected r2 covers64 unique IDs with matching vocabularies/checkpoints/config,
all six retained numerical/per-image endpoints, native16px query coverage,
protected bounded actions and dense confusion/transition sums. All fields
precede target masks. The first run's overlap/stride API error produced nine
crops and failed verification; its outputs remain preserved, and no metrics
were used to choose the repair. A sampler regression was added after completion
without changing the corrected observation grid or rerunning predictions.

Suite wall136.8777s; peak allocated5561.8828-5749.0425MiB. Four extra forwards
per window, native observation about0.104-0.114s/window in this shared suite.
Worker16.3-42.9s includes20-arm decoding/cache persistence, not standalone latency.
All workers/controller terminal; GPUs0-7 idle; unrelated tmux sessions intact.

Protocol/results: NATIVE_QUERY_ALIAS_PROTOCOL_20261003.md,
NATIVE_QUERY_ALIAS_SCREEN_20261003.md, native_query_alias_screen_r2_20261003/.
Implementation: DINOtool/dinotool/native_query_alias.py; evaluator/runner:
DINOtool/scripts/eval_native_query_alias.py and run_native_query_alias.py;
deployment/independent verification: tools/native_query_alias_experiment.py.
