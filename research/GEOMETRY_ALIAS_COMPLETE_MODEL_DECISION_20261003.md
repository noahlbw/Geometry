# Geometry: Complete-Model Decision After The Class-Relative And Action-Capacity Experiments

## Decision

Latest writer execution is COMPLETE: rival-preserving43.448829 versus original42.867949 (+0.580881pp),5/8 wins and above all new alias/directional controls, but below prior collapsed43.548346 and shuffled-support43.737596. Frozen gate FAILS. No full rollout or tuning. All40 tests, mask-free smoke, exact18 historical numerical/per-image controls, full risk/coverage/transition/capacity replay pass. RIVAL_PRESERVING_COMPLETE_MODEL_DECISION_20261003.md records the latest complete-architecture decision and unresolved semantic-source requirement; RIVAL_PRESERVING_ALIAS_SCREEN_20261003.md contains exact results. The previously proposed writer is implemented but not promoted. The overall useful-alias/CVPR goal remains unachieved.

Previous source continuation: the frozen contrastive-reversal source improves the64-window mean by0.680398pp (42.867949 ->43.548346), but its prospective gate fails on4/8 wins and a superior alias-risk shuffle43.577281. Potsdam loses0.327383pp. No full rollout or final selector promotion. Report: CONTRASTIVE_REVERSAL_ALIAS_SCREEN_20261003.md. Its post-experiment plan GEOMETRY_POST_REVERSAL_COMPLETE_MODEL_PLAN_20261003.md proposed the rival-preserving writer, now executed separately as recorded above. The original anchored reference remains retained. All28 source tests, mask-free smoke, exact coverage/replay/control/transition/capacity checks passed and workers were terminal. The useful universal second-reader/CVPR objective remains incomplete.

Keep original Geometry plus the original attributed VIP wide observation and anchored reconstruction as the retained coupled reference. Do not insert ClassRelativeReject_Coupled into the final deployed model. No universal useful alias controller is yet established. The CVPR research objective remains open, not achieved by this implementation.

This round implements the bounded class-relative successor described in the prior plan, not a claim that Astra endorsed this exact formula or that its mathematics is novel. One changed hypothesis, not another encoder/solver stack. The earlier candidates and strongest historical models remain intact.

## Verified Experiment

Frozen weights and fixed20 aliases/class. Clean96 COMPLETE images: UDD5 full40 and seven domains8 each. Each stress scenario uses64 first8 images/domain; only the broad vocabulary changes, local original20 is fixed. LoveDA P/D share images; means count D once. Corrected IRRG Vaihingen. LandCover.ai replaces unavailable labeled iSAID. All domains informed development; these are not eight untouched full-dataset results.

Seventeen tests and a mask-free real-checkpoint smoke pass. Four historical controls replay exactly per image on clean and both stresses. Unique complete coverage, configuration/vocabulary/checkpoint identity, confusion sums, transition endpoints and matched shuffled spectra are verified remotely and after download. No dataset-specific numerical fitting or winner routing.

| Dataset/protocol | Original coupling | Class-relative | Delta pp |
| --- | ---: | ---: | ---: |
| UDD5, full40 |49.211686|49.233522|+0.021836|
| VDD,8 |57.001172|57.004842|+0.003670|
| Potsdam,8 |41.780594|41.744121|-0.036474|
| OEM,8 |31.953271|32.095133|+0.141863|
| LoveDA P,8 |62.486347|62.485847|-0.000500|
| LoveDA D,8 |36.559979|36.553670|-0.006308|
| Vaihingen,8 |51.449095|51.228930|-0.220165|
| LandCover.ai,8 |66.906020|66.922266|+0.016246|
| FLAIR-1,8 |33.608426|33.476562|-0.131864|
| Eight-domain mean, D once |46.058780|46.032381|-0.026400|

| Scenario | Original | Old gain-conditioned | New class-relative | Text-only | Hard deletion | New shuffled range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean96 |46.058780|46.057795|46.032381|45.884504|44.376221|46.021428-46.037623|
| Wrong-parent64 |45.315359|45.318669|45.410067|45.482304|43.615551|45.300182-45.333361|
| Paraphrase64 |45.982989|45.981510|45.952508|44.985799|44.051824|45.942745-45.968927|

Clean wins4/8, wrong-parent5/8, paraphrase3/8. Both predeclared advancement routes fail. Soft rejection is much less damaging than this hard-deletion operator, but that alone is not useful allocation. On constructed wrong-parent words it beats all shuffles, with only+0.094708pp and below text-only; on clean/paraphrase it is within the shuffled range. Do not turn those different findings into either universal success or proof that all screening is impossible.

Wrong-parent/paraphrase replacements are precise semantic constructions, NOT fresh raw-LLM outputs. They remove four original entries too. A wrong-parent assignment can improve actual predictions by changing salience/competition, so its provenance is not a guaranteed harmful-action label.

Shared suite wall1346.5701s, about22.44min; maximum allocated6187.8667MiB, about6.04GiB. These include all ten arms/three vocabularies and are not standalone primary latency. All workers/controllers are terminal; physical GPUs0-7 are idle and unrelated sessions remain.

## What Failed And What Did Not

The original V1 risk multiplied canonical text contradiction, Geometry-supported rival leakage, and absolute wide-minus-local alias gain. New risk removes the last factor only. Suppression writer, original observation and reconstruction stay fixed.

The new source is genuinely more active. Clean Potsdam mean rejection goes0.00044266 ->0.02374721;21.9439% of alias/query pairs have positive rejection. This rules out merely saying this successor did nothing. Its semantic allocation is still insufficient.

Four distinctions matter:

1. A misplaced alias such as roof attached to wall is an attachment problem. A valid car alias firing on a roof is a visual semantic error. Parent/rival text cosine can flag some of the former, but does not certify the latter.
2. Canonical descriptor cosine dominance is an embedding comparison, not a logical contradiction. A phrase can describe both an object and its legitimate scene. Geometry witnesses from the same frozen semantic path can agree on a wrong class.
3. Local canonical support is not an independent teacher. Rejecting wide evidence because local support favors a rival can erase precisely the new information that would repair a locally missed object. Removing the gain gate does not repair that circularity.
4. The reader changes broad evidence only. It cannot claim to clean all aliases in the untouched local anchor. Canonical aliases are protected, so canonical false activation can remain completely outside its action space.

Potsdam shows the actual competitive tradeoff. Low vegetation IoU21.2306 ->22.6976, but car24.2128 ->23.3065. Car recall remains98.2627%, precision is23.4029%, predicted area9.8363%: this is not primarily missed-car recovery. There are55,928 beneficial and33,165 harmful clean corrections, yet macro mIoU falls. Correcting more pixels of large classes can coexist with worse small-class IoU.

UDD5 road39.8083 ->39.9398 while vehicle20.4213 ->20.4070. These changes do not establish that road's own aliases were improved: changing any competing class can change road's winning pixels. The complete confusion/transition tables, not isolated response magnitude, determine whether the competition improves.

## The Retained Complete Architecture

```text
Image + candidate class names/aliases
                 |
       Frozen DINOv3 / DINO.text
                 |
    +------------+-------------------+
    |                                |
Local512 Geometry               Wide-view observation
raw DINO relation G             inherited VIP visual reader
native patch/special mass        original ImageNet queries/salience
preserved in both head blocks    short-side448, crop336/stride224
    |                                |
descriptors Y -> local logits g       broad logits b
    +----------------+---------------+
                     |
        Geometry-anchored innovation reconstruction
        z = g + (I+G^T G)^(-1)G^T G(b-g)
                     |
          original interpolation / overlap assembly
                     |
                  prediction
```

There are TWO main mechanisms, not seven modules: Geometry reading and Geometry-anchored contextual reconstruction. Frozen encoders, text expansion, the inherited wide observer, interpolation and the standard linear solver are components, not separate claimed inventions. An alias controller would live inside the second mechanism only after demonstrating useful evidence allocation.

Geometry's actual attention operation is `A_special V_special + m_patch G V_patch`, with the original frozen projection, residual, MLP and normalization. Its defensible distinction is conditional native-mass preservation, not inventing DINO affinity, two-block reading or residual/MLP retention. The matched publication audit already establishes those boundaries.

Local all20 aggregation is normalized log-mean-exp over cosine responses at0.07. Candidate entries have no learned quality weights, but their score responsibilities vary by pixel; high-response words contribute more. That is dynamic evidence influence, NOT verified semantic reliability. The broad observer additionally uses inherited crop-dependent salience. Keeping all candidates is not equivalent to claiming every word is equally useful.

The complete model has no supervised training loss. Its frozen inference objective is:

```text
min_z 0.5*||z-g||^2 + 0.5*||G(z-b)||^2
(I+G^T G)*delta = G^T G*(b-g)
z = g+delta
```

The existing implementation uses up to32 conjugate-gradient iterations. If `G=U diag(s) V^T`, the exact solution admits innovations in V modes with gain `s^2/(1+s^2)`, not one scalar fusion weight. A mode in the nullspace of G is unchanged. These are algebraic fidelity properties, not a guarantee of correct labels, protected small objects or new solver priority.

This gives a coherent scientific story: Geometry supplies an explicit relation without replacing the frozen head's native pathway allocation; contextual information changes only the supported innovation while preserving a local anchor. The VIP observer must be credited. Classical reconstruction mathematics alone does not prove a new contribution.

Historical FULL eight-domain results are in geometry_publication_20261001/NOVELTY_AND_MECHANISM_AUDIT.md and GEOMETRY_SEMANTIC_INNOVATION_20261001.md. The retained coupling improves VDD/Potsdam/Vaihingen/LandCover.ai and loses LoveDA D/UDD5/OEM/FLAIR-1 against Geometry. It is a useful reference, not an all-eight-domain winner. BroadVIP here is a matched all20 observer, not the complete official VIP system.

## Final-Model Target And The Next Decision

Keep the target narrow: Geometry plus ONE competitive contextual evidence reader. The desired internal retention is query/class/rival-dependent, leaves useful and unknown evidence intact, and suppresses only supported harmful contribution without forced counts or mass redistribution. This architecture target remains valid; the tested source of that retention does not meet its requirement.

Do not prescribe another rejection multiplier or another solver from the current scores. The following diagnostic is now COMPLETE; it separates two possible bottlenecks using the same fixed views/readout:

1. Action-capacity upper bound: audit the available per-alias/group suppression interventions on labeled development samples, including protected-canonical errors. Determine whether ideal choices can materially repair the actual lost competitions under this writer and anchored reconstruction. This is an audit-only label-assisted ceiling, never an inference arm, selector-fitting set or independent validation claim.
2. Source correctness: compare the already frozen, image-only retention ranking with the audit's useful/harmful/unknown actions, including per-class true-coverage loss and rival false-activation reduction. Report class-pair tradeoffs and scene dependence, not AUC alone. Do not fit a threshold or deploy the oracle's choices.

The verified action diagnostic uses64 fixed images, eight/domain, ONE top-left512 window/image, with original whole-image wide observations. It is not another full-image/full-dataset screen. All20 words, weights, views, salience and the original reconstruction objective are fixed; observations/actions are saved before masks load. Labels define audit outcomes and the explicitly privileged feasible policies only, not a fitted selector.

The paired metric now fixes the Anchored_Exact union-positive class set per dataset/protocol/scenario, using IoU0 if a scored class subsequently has zero union. Raw standard metrics are preserved. Removing FLAIR vineyard's last34 false predictions appeared+3.054150pp under the changing standard denominator but is-0.001063pp under this fixed set. Predictions were not changed or rerun to correct reporting.

| Window scenario | Original exact coupling | Current label-free source | Protected label policy | Unrestricted label policy |
| --- | ---: | ---: | ---: | ---: |
| Clean |42.867949|42.865702|51.417508|51.643322|
| Constructed wrong-parent |42.366965|42.428434|52.174181|52.460068|
| Constructed paraphrase |43.646857|43.619378|52.543982|52.653233|

Protected label policy improves all eight clean domains, a+8.549559pp window mean gain. This is ONE realizable label-assisted choice, not an optimum, upper bound, deployable model or expected label-free gain. Canonical-unprotected policy adds only+0.225814pp on clean. Pointwise attainable envelopes also leave many errors impossible in the bounded action space: for example VDD1544/2556 wrong centres and Potsdam2076/3074. Envelopes are grouped by true class; zero vehicle misses does not imply zero false vehicle predictions. Different attainable locations can require incompatible donor decisions.

The source, rather than only action strength, is a demonstrated bottleneck on these windows. Single attenuation of UDD5 vehicle / cars seen from above gives+2.261230pp with TP-1/FP-77653, and Potsdam car / traffic vehicle gives+0.640937pp with TP-27/FP-20702; both have frozen risk0. Conversely Potsdam impervious surface / road pavement loses0.772698pp and26864 TP under attenuation despite risk0.103774. These are labeled scene-dependent actions, not a blacklist. All eight clean global-action source AUCs are below0.5 after the fixed-denominator correction; this alone is not a querywise or universal impossibility theorem.

Therefore do not increase the same canonical-cosine risk, remove canonical protection, deploy the audit's helpful words as bans or change the solver. The different Geometry-guided target/context source is now implemented and tested separately below. Its gate fails despite a positive pilot gain; no source is promoted to the final model. Concrete architecture and revised next decision are in GEOMETRY_ALIAS_POST_AUDIT_MODEL_PLAN_20261003.md.

### Target/Context Source Execution

Signature geometry-target-context-alias-source-v1-20261003. Same64 clean development windows and cached original exact-system relation, source inputs and three per-image controls. Sixteen class-agnostic Geometry supports/window, two fixed target-kept/removed RGB fills, original ImageNet-template raw cosines, canonical protected, all competitors considered, original bounded writer. No label-fitted threshold, changed salience, new encoder or new reconstruction objective. The initial mask-free attempt failed from CPU/CUDA shuffle index placement before labels; r2 changes only device placement and tests, with the failed root preserved.

Original window mean42.867949 -> primary43.119647 (+0.251698pp), wins6/8. Mean ContextOnly42.603621; three exact-spectrum alias shuffles43.089248/43.089369/43.090702; shuffled spatial support43.145589. Same-source simple mean fusion42.180843 ->42.383200 (+0.202357pp). The prospective gate fails because the spatial-support control beats the primary. Do not interpret the small0.028945pp advantage over the strongest alias shuffle as significance or Geometry-specific grounding.

VDD+0.708867pp, UDD5+0.701239, LoveDA D+0.710492, OEM+0.053208, Potsdam+0.029680, LandCover.ai+0.023109; Vaihingen-0.071265, FLAIR-1-0.141748; LoveDA P-0.024594. These are window-only outcomes, not full-image/full-dataset results. Potsdam car24.5378 ->24.1387, area14.4459 ->14.6983%, while impervious surface65.8977 ->65.2643. Marginally suppressing a valid harmful alias does not guarantee improvement after every class is suppressed and reconstructed jointly.

New frozen risk for UDD5 cars seen from above is0.144588 instead of0; Potsdam traffic vehicle0.046410 instead of0. Yet pavement risk remains nonzero and global-action AUC is below0.5 in five domains and LoveDA P. The source is materially different and active, but universal useful allocation is not established. Shuffled supports preserve MASK spectra, not the resulting retention/action strength; aggregate class calibration and OOD masking are unresolved explanations.

No full rollout or post-result parameter search. Next bounded evidence test: reuse these already frozen fields to compare actual spatial suppression with per-class exact-spectrum spatial action permutations and constant mean suppression. Preserve each class's aggregate suppression to separate allocation from calibration. This must be a preregistered diagnostic, not a retrospectively promoted model; no re-encoding or oracle-word selection is needed.

Nine mathematical tests (including remote GPU regression), mask-free real-checkpoint smoke, exact historical per-image control/coverage/confusion/transition verification pass. Shared suite wall166.1478s; peaks5553.6367-5579.9956MiB. The true intervention branch adds64-150.5 forwards/window on average and1.4079-3.3610s in this shared pilot; standalone/amortized inference cost is unestablished. All sessions terminal, GPUs0-7 idle, unrelated sessions intact. Report/protocol/data: TARGET_CONTEXT_ALIAS_SOURCE_20261003.md, TARGET_CONTEXT_ALIAS_SOURCE_PROTOCOL_20261003.md, target_context_alias_source_r2_20261003/.

The planned budget-matched spatial-action diagnostic is now COMPLETE. Primary43.119647 versus CapacityMean42.996564 and exact-spectrum feasible spatial shuffles42.999234/42.978309/42.978078 gives0.120412-0.141569pp limited positional evidence beyond class-total suppression. It wins this comparison mainly on VDD/UDD5/LoveDA D; Potsdam still prefers the controls and car loses. RawMean equals CapacityMean without violations; primary dense replay and all numerical/coverage/transition checks pass. Cached wall16.0097s with zero new encoder forwards is not deployment latency. The prior source gate remains FAILED. Full report: ALIAS_SPATIAL_ALLOCATION_AUDIT_20261003.md.

Post-experiment complete-model planning is in GEOMETRY_COMPETITIVE_READER_FINAL_PLAN_20261003.md. Its pair-specific bounded joint reader is now implemented and tested on the same64 windows:42.892916 versus original42.867949, old independent43.119647 and shuffled-support42.942351. Gate FAILS on gain<0.1pp and below shuffled support despite5/8 wins. Shrinkage leaves19-33% of old suppression; cap saturation is zero. Eleven tests, mask-free actual-checkpoint smoke and matched source/control/coverage/transition/KKT checks pass. This is a failed universal-selector candidate, not a completed final module or evidence that every pair-conditioned method is impossible. PAIR_CONTEXT_READER_SCREEN_20261003.md contains exact results. No extra encoder, label tuning, audit-word blacklist or full rollout. All workers/controller terminal; physical GPUs0-7 idle, unrelated sessions preserved. The next requirement is useful competing-class margin evidence; no new witness is implemented by this update.

Only after a frozen useful source beats all20, text-only, same-strength shuffled allocation and same-source simple fusion should it become the second reader's core. Test released real raw-LLM vocabularies, legitimate/shared appearance phrases and separate candidate-count changes before claiming unrestricted expansion robustness. Then run one fixed rule across the complete20,092 available images, without per-domain routing.

## CVPR Completion Requirements

- Demonstrated useful allocation, not an almost-neutral extra selector or improvements over a badly damaged hard-deletion control.
- A matched coupling benefit over simple fusion using exactly the same semantic source and information budget.
- Fair complete official VIP comparisons, separately from matched operator adaptations; disclose query/templates, field of view, transductive selection and background rules.
- Per-class coverage/false-activation and object-size/boundary evidence, with paired scene-level uncertainty and standalone full-predictor cost.
- An independent domain/region/vocabulary evaluation after freezing the final method. All present eight domains are development data.
- If claiming general OVSS beyond remote sensing, evaluate natural images with a declared common general template protocol; do not silently carry only RS-specific templates into that claim.
- A focused verified nearest-method novelty comparison. The existing Geometry audit does not prove universal priority or current SOTA.

The strongest available paper plan is the verified two-mechanism model and its real boundaries. An ineffective screening module does not make that story stronger. Current implementation is complete; the useful universal second reader and publication claims are not yet established.

## Reproduction

Exact all-arm, all-class metrics/precision/recall/area/transition diagnostics: CLASS_RELATIVE_ALIAS_REJECTION_SCREEN_20261003.md. Frozen rule/gate: CLASS_RELATIVE_ALIAS_REJECTION_PROTOCOL_20261003.md. Downloaded merged results, per-image matrices, selection manifests and logs: class_relative_alias_rejection_screen_20261003/.

Model: DINOtool/dinotool/class_relative_alias_rejection.py. Tests: DINOtool/tests/test_class_relative_alias_rejection.py. Shared evaluator/runner retain original default behavior; new wrappers: eval_class_relative_alias_rejection.py and run_class_relative_alias_suite.py. Deployment/collection/report: tools/class_relative_alias_experiment.py. No full evaluation was launched after the failed gate.

Action audit: ALIAS_ACTION_CAPACITY_PROTOCOL_20261003.md and ALIAS_ACTION_CAPACITY_AUDIT_20261003.md; verified local data in alias_action_capacity_audit_20261003/. Signature geometry-alias-action-capacity-audit-v1-20261003. Ten mathematical tests and mask-free GPU smoke passed; four fixed-class reporting regression tests passed. Original32-step CG and exact-system FP64 audit have maximum score discrepancy<=1.55e-5 and zero changed patch-centre controls; this is not a proposed solver change. Shared audit wall146.2614s, peaks5608-5721MiB, not standalone inference cost. Final terminal check finds physical GPUs0-7 idle at1MiB each and only unrelated tmux sessions controlled_index and qlift_smoke_matched_20260921 remaining.
