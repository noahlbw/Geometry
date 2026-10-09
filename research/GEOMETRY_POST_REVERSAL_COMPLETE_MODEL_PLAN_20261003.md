# Geometry: Complete-Model Planning After Contrastive Reversal

## Decision And Status

### Execution Update: Rival-Preserving Writer

The previously proposed writer below is now IMPLEMENTED and its64-window screen is COMPLETE, signature geometry-rival-preserving-alias-reader-v1-20261003. Mean43.448829 versus original42.867949 (+0.580881pp),5/8 wins, but below the previous collapsed writer43.548346 and new shuffled-support control43.737596. The frozen gate FAILS. It beats all three new alias-risk and directional-budget/spatial controls on mean; this is limited descriptive evidence, not an established universal selector. Potsdam car still loses0.3833pp with4793 extra false positives. All40 tests, mask-free smoke, exact18 historical numerical/per-image and full risk replay, coverage/transition/capacity checks pass. No full rollout or tuning. Latest decision/remaining requirements: RIVAL_PRESERVING_COMPLETE_MODEL_DECISION_20261003.md; exact report: RIVAL_PRESERVING_ALIAS_SCREEN_20261003.md. The architecture below is now a historical tested candidate, NOT the next unimplemented successor or promoted final model.

The frozen contrastive-reversal experiment is complete. Preserve original Geometry, the attributed original anchored wide-view model, the useful-but-unpromoted sources and all failed outputs. Do NOT put this source into a claimed final model or launch full datasets from its failed gate.

The two-module research target remains: Geometry local reading plus ONE competitive contextual observation/reconstruction mechanism. Alias admission belongs inside that second mechanism, not a third independently tuned prompt selector. The historical design below was a prospective hypothesis; it is now implemented and fails promotion as recorded above. It is not established as uniquely novel or a guaranteed CVPR model. This is an evidence-driven continuation of the requested research direction, not a claim that Astra specified these exact formulas.

## Verified New Evidence

Signature: geometry-contrastive-reversal-alias-source-v1-20261003. Physical A800 GPUs0-7. Frozen weights, unchanged writer/reconstruction, exactly20 aliases/class. Same64 fixed development image IDs, eight/domain, ONE top-left512 window/image; original whole-image wide observation. LoveDA P/D share images and the mean counts D once. Corrected IRRG; LandCover.ai substitutes for unlabeled iSAID. These are not complete-image, full-dataset or untouched validation results.

| Dataset/protocol | Original exact coupling | Contrast reversal | Delta pp |
| --- | ---: | ---: | ---: |
| VDD |53.746954|55.619109|+1.872155|
| Potsdam |38.920258|38.592874|-0.327383|
| UDD5 |28.175811|30.178447|+2.002637|
| OEM |39.023152|39.232807|+0.209654|
| LoveDA P |50.706552|53.152193|+2.445641|
| LoveDA D |30.736030|32.743830|+2.007799|
| Vaihingen |51.826962|51.608847|-0.218115|
| LandCover.ai |66.906020|66.865626|-0.040395|
| FLAIR-1 |33.608404|33.545232|-0.063172|
| Equal-domain mean, D once |42.867949|43.548346|+0.680398|

| Matched comparison | Mean mIoU % |
| --- | ---: |
| Previous independent target/context source |43.119647|
| New primary |43.548346|
| Same-source simple logit fusion |42.846909|
| Relative dependence without sign requirement |42.708620|
| Text-only source |42.562614|
| Shuffled visual supports |43.466052|
| Three alias-risk shuffles |43.446617 /43.577281 /43.396436|
| Same class-total suppression, CapacityMean |43.341059|
| Three exact-spectrum spatial action shuffles |43.302973 /43.302383 /43.316138|

The prospective gate FAILS: only4/8 domain wins, below the required5, and alias shuffle1 exceeds the primary by0.028935pp. Other gates pass, including gain>=0.1pp and no protocol loss>1pp. Do not weaken the declared criteria, choose a favorable fill/seed, route by dataset or call this successful universal alias selection.

There is genuine progress: sign reversal produces a larger paired gain than the prior source; the primary exceeds its capacity/spatial controls on mean by0.207288-0.245964pp and its same-source fusion by0.701438pp. Yet only UDD5, OEM and LoveDA D beat CapacityMean among the eight counted domains. Class-total calibration and useful spatial allocation are not established as independent additive contributions. Beating most shuffled seeds is not a significance result or proof of word-specific reliability.

All28 source/reader/allocation mathematical tests, mask-free real-checkpoint smoke, exact four retained per-image controls,64 unique coverage, actual vocabulary/checkpoint/config identities, source/Geometry replay, confusion sums, transitions and bounds/budgets/spectra pass. Suite wall121.2118s; peaks5561.8047-5588.6323MiB. These reuse the prior64-150.5 extra masked forwards/window and are NOT standalone inference costs. All new fields precede target-mask loading. All workers/controller terminal and GPUs0-7 idle; unrelated sessions retained.

## What Needs Repair

### Useful Words And Useful Joint Actions Are Different

The source tests alias/rival matched-template margins:

```text
m_full = s_full(alias)-s_full(rival canonical)
m_keep = s_keep(alias)-s_keep(rival canonical)
m_remove = s_remove(alias)-s_remove(rival canonical)

eligible: m_full>0, m_keep<0, m_remove>0
gamma(c,alias,d) = min_over_two_fills clamp(-m_keep/(m_remove-m_keep),0,1)
```

It rejects a background-surviving advantage that reverses on retained target content. It is not a correctness test: support can mix objects, masking changes the input distribution, canonical predictions can be wrong, and false target-supported semantics need not reverse. Unsupported/ambiguous evidence remains retained; canonical risk0.

Current execution collapses rival indices by maximum risk and suppresses each class independently. Consequently a word suspect against ONE rival loses contribution against ALL rivals at that query. Simultaneously changing competitors can outweigh useful attenuation of the tested class. This is an actual information loss in the implementation; it is not yet established as the sole cause of the observed errors.

Concrete frozen endpoints:

| Dataset/class | IoU delta pp | TP change | FP change |
| --- | ---: | ---: | ---: |
| VDD vehicle |+4.8802|-351|-1955|
| VDD water |+4.1029|-464|-22297|
| UDD5 vehicle |+1.6721|0|-39009|
| Potsdam car |-0.9034|+20|+11691|
| Potsdam impervious surface |-1.2099|-12169|-791|

Thus VDD vehicle improves precision with some recall loss; Potsdam car remains an overprediction problem, not mainly missed cars. Suppressing pavement can hand its pixels to car even while car aliases themselves are attenuated. Larger-class pixel correction counts do not guarantee macro mIoU improvement.

The audit-only source correspondence is also incomplete. Capacity-weighted new risk is0.507358 for UDD5 cars seen from above, with prior single-alias attenuation utility+2.261230pp. Potsdam traffic vehicle has risk0.076978 and utility+0.640937pp, but road pavement has slightly greater risk0.082254 despite utility-0.772698pp. No threshold should be fitted between those values: these are global scene-dependent, nonadditive interventions, not querywise labels. Potsdam ranking AUC0.679924 coexists with worse final mIoU.

## Retained Executable Model

```text
Image + class names/candidate aliases
                 |
      frozen DINOv3 / DINO.text
                 |
      +----------+----------------------+
      |                                 |
Module 1: Geometry                 Original wide observer
local descriptors Y, relation G    attributed VIP head/views
native attention-mass preserved   broad alias fields -> logits b
local all-alias logits g                 |
      +----------------+----------------+
                       |
Module 2: anchored contextual reconstruction
z_original = g + H_G(b-g)
H_G = (I+G^T G)^(-1)G^T G
                       |
           original dense assembly
```

Geometry's attention reading preserves original special-token terms and native patch mass: A_special V_special + m_patch G V_patch, with frozen projection/residual/MLP/normalization. DINO affinity, those head components and classical reconstruction have prior art and require attribution. The VIP wide observer remains borrowed; renaming it does not make it original.

Local normalized log-mean-exp gives high-response aliases larger pixelwise responsibilities; this is NOT semantic quality weighting. Existing alias controllers change only the broad branch. Do not claim that local aliases or every phrase in the whole predictor are screened.

The model has no supervised training loss. Its reconstruction is a frozen inference objective: min_z .5||z-g||^2+.5||G(z-b)||^2. The effective operator can have signed entries; a negative donor update is not a guarantee of negative receiving scores or improved semantics.

## Historical Candidate: Preserve Rival-Conditioned Evidence

Do not add another encoder, global TopK, per-dataset forbidden words, output threshold or gain multiplier. One next hypothesis is to keep the existing sign-reversal source's rival dimension through observation and use ONE consistent class-score correction. This changes the writer, not the source's validity. A failed source cannot become reliable merely through a new equation.

At an original profiled wide crop location define, with exactly the original beta/K/salience:

```text
q_a = softmax(beta*r)_a
d(c|d) = log(1-sum_a gamma(c,a,d)*positive(q_a-1/K))/beta
e(c,d) = d(c|d)-d(d|c), e(d,c)=-e(c,d)
```

Carry e through the original crop stencil and overlap without max-over-rivals. Good/unknown aliases remain in the unmodified observation; no survivor renormalization or new positive alias boost. The per-rival observation can keep a phrase useful against C while attenuating its ambiguous contribution against B.

A complete multiclass predictor cannot realize arbitrary inconsistent pair margins. A minimal standard, coefficient-free consistency read is:

```text
v = argmin_(sum_c v_c=0) sum_(c<d) ((v_c-v_d)-e(c,d))^2
v_c = (1/C)*sum_d e(c,d)
z_candidate = z_original + H_G v
```

This is classical complete-graph least squares, NOT a novel solver. The potential/cycle decomposition and Geometry writeback make the implementation explicit and small. Zero risk gives v0 and EXACT original scores. There is no second multiplication by risk or tuned amplification. Unlike the failed previous box projection, it does not constrain all class corrections to be negative or add the previous risk-weighted shrinkage objective. Some class corrections can be positive because they represent relative competition, not positive promotion of an alias.

Importantly, preserving rival-specific observation does NOT mean the final A/C margin stays unchanged when A/B changes: shared class scores necessarily couple margins. The projection discards inconsistent cycles. It may dilute a sparse requested change and may still fail; report realized margins, endpoint errors and classes. Do not promise exact independent pair decisions, semantic correctness or a solution to false car activation.

This candidate was proposed under the prospective protocol below and is now a tested, failed-promotion candidate. It must not be called an established innovation or scheduled for full data. The observer still has mixed-support/OOD/canonical limitations and expensive masks. Its broad-only admission claim must remain narrow. Extending admission to the local branch would require a separate matched local-path measurement, not copying broad ImageNet-template risk into RS-template Geometry logits without validation.

## Order Of Remaining Work

1. Freeze this one rival-preserving complete candidate and its invariants before results; retain the new source and original readout/config, not another source search simultaneously.
2. Use current cached64 windows for the first matched decision. Compare current max-collapsed writer, proposed rival-preserving writer, original coupling, same-source simple fusion, alias-risk shuffles and class-budget/spatial controls. Require gain>=0.1pp,>=5/8 wins, no loss>1pp including LoveDA P and superiority to every declared matched control. Do not weaken the earlier source gate retroactively; this would be a distinct writer experiment.
3. If that gate passes, validate COMPLETE images on the already fixed development IDs, not only top-left windows; many object distributions are absent in those windows. UDD5 road has no true coverage in the current window screen, so do not extrapolate road behavior to full40. Add genuine released raw-LLM candidates and legitimate shared phrases under a predeclared common rule. Keep candidate-count perturbations separate from word-quality perturbations. No labels choose words or parameters.
4. Establish source cost and only then test a separately frozen shared-feature/amortized approximation. It must reproduce useful competitive actions and per-class results, not only tensor similarity. No current claim of an efficient deployment-ready selector.
5. Only after those gates, freeze one rule for full LoveDA1669, UDD540, OEM384, VDD80, Potsdam504, Vaihingen113, LandCover.ai1602 and FLAIR-115700, plus a held-out region/vocabulary/domain. All present eight datasets informed development. Report fair complete official VIP separately from matched20 BroadVIP adaptations.

The desired contribution is geometry-conditioned contextual admission with competitive writeback, not a sequence of seven modules. Accuracy against hard deletion alone, better AUC, numerical convergence or more active suppression is insufficient. If the next paired mechanism still cannot beat word/spatial controls or source-matched simple fusion, stop promoting this observer/writer as useful alias control. Retain the executable Geometry-plus-attributed-coupling paper and narrow its claims; do not keep strengthening the same failed source to satisfy a narrative.

## CVPR Publication Boundary

Current evidence does not establish a universal final alias module, full eight-dataset SOTA or novelty priority. Geometry and attributed anchored reconstruction can form a focused paper, but its claimed contributions must survive nearest-method comparison. No requirement that a paper discard aliases is justified by publication alone; useful reliability handling must be demonstrated.

Needed: meaningful word/location/competition allocation beyond matched controls; a coupling advantage at matched source/information budget; realistic vocabulary robustness; full official VIP fairness; size/boundary/precision-recall mechanisms; scene-level uncertainty; standalone full-predictor runtime/memory; and independent validation after freezing. Natural-image evaluation with common general templates is needed for a general OVSS claim, not silently inferred from remote-sensing performance. No guarantee of CVPR acceptance follows from this plan.

Source protocol/results/code: CONTRASTIVE_REVERSAL_ALIAS_PROTOCOL_20261003.md; CONTRASTIVE_REVERSAL_ALIAS_SCREEN_20261003.md; contrastive_reversal_alias_screen_20261003/; DINOtool/dinotool/contrastive_reversal_alias.py; DINOtool/scripts/eval_contrastive_reversal_alias.py and run_contrastive_reversal_alias.py; tools/contrastive_reversal_alias_experiment.py. The rival-preserving writer is now implemented separately in DINOtool/dinotool/rival_preserving_alias.py with its evaluator/runner/tests and tools/rival_preserving_alias_experiment.py. Its gate fails; it is not a promoted final model. Refer to the execution update above, not older prospective tense, for current status.
