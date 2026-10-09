# Frozen Vocabulary-Time Semantic Membership Pilot

One independent semantic source, not a weighting-formula sweep. Preserve the
full20092-image Geometry_PatchOnly2Coupled reference and every earlier experiment.
The previous analysis-only goal turn made no implementation progress.

## Source Frozen Before Target Masks

Use public cross-encoder/nli-MiniLM2-L6-H768, exact upstream revision
b95119ce93d3e065de6214e38cd4a97b0f2f2c6d. Its model/config/card explicitly map
0 contradiction,1 entailment,2 neutral. The frozen pretrained NLI model adds
external SNLI/MultiNLI semantic supervision, not target-data training. Attribute
it; adding NLI is not itself a novel architecture. Trust no remote code.

Premise and hypothesis both use exactly `The object or surface is {phrase}.`
Premise fills the original alias; hypothesis fills a declared canonical class.
No definitions, handwritten word exceptions, prompt sweep, target image, target
mask or labeled alias marginal enters the vocabulary-time cache. All eight
unchanged20 vocabularies use this same source/rule. Cache all three posterior
components and source identity before loading evaluation masks; no per-image
NLI execution or extra visual/head/text-embedding observations.

For each original bounded baseline query's top2 classes c/r, let o be the NLI
entailment posterior alias->c and r be alias->r. Define conservative unknown
mass u=(1-o)(1-r), then w=(o+u)/(o+r+u), with numerical floor1e-6. This is an
explicit heuristic membership allocation, not a calibrated target posterior
or a new probability theorem. Own-exclusive evidence retains1; rival-exclusive
evidence tends0; equally strong shared evidence tends1/2; neither-supported
evidence stays neutral. Canonical slots, self-competition and padding keep1.
The same word may be attenuated versus one class and retained versus another.
It cannot identify a correctly named car firing on road solely from text.

## Frozen Reader And Action

Signature geometry-bounded896-semantic-membership-fixed-slot-v1-20261006.
Geometry_PatchOnly2 local896 and independent VIP wide448, max4 actual crops
each; zero fine observations. Original20 scoring/text templates/salience stay.
Apply weights outside the exponential to original fixed-slot wide alias mass,
without survivor redistribution. Original H writes the antisymmetric top2
potential; unchanged whole-image probability assembly restores original size.
No extra solver, class threshold, strength or domain-specific route.

## Paired Developed Panel And Controls

UDD5 all40 and eight evenly spaced complete images/domain elsewhere:96 unique
images. LoveDA P/D share inputs; D enters the eight-domain mean once. All prior
original Geometry/no-admission/patch2 per-image endpoints must replay exactly.
Masks load only after predictions. These domains were developed, not independent.

Controls use identical images, encoder outputs and source cache: NLI posteriors
pooled across noncanonical words BEFORE the weight formula; class-uniform mean
weights; three fixed word-identity shuffles; NLI rival collapsed across other
declared classes; original H correspondence shuffle; direct writing. Match
word-only post-H field norms for attribution as well as retaining raw deployable
class controls. Matched nulls are diagnostics, not deployable substitutes.

Freeze source advancement criteria now: mean gain>=0.1pp over same-information
baseline, at least5/8 main-domain gains, worst protocol loss<=1pp; gain>=0.05pp
and paired conditional95% lower bound>0 over raw pooled source, matched pooled,
matched class mean and mean3 matched identity nulls. Report rival and writing
attribution without claiming those operators are new mathematics. No post-result
prompt, seed, posterior formula, strength, class exception or favorable-control
promotion. Failure rejects this source and stops its timing/stress/full rollout.

After a source pass, actual serial complete-image timing must measure baseline,
primary and repaired VIP20 on three fixed inputs/domain, three warm synchronized
singleton repetitions. Report vocabulary-time NLI load/forward/memory separately.
New engineering budget: mean paired overhead<=1.5x baseline and all domain-average
latencies<=1000ms, reflecting the user's tolerance for moderate extra cost;
speed still matters equally, and source validity remains mandatory. Neither
budget nor advancement thresholds are CVPR acceptance rules. Shared-resident
memory is not standalone deployment memory.

Only a passed source/cost pilot justifies vocabulary-pollution/paraphrase tests
and unchanged full20092 evaluation. No retained model or paused automation changes.
