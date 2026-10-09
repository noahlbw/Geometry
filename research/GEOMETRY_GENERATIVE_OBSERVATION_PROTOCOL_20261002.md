# Geometry-supported generative observation: prospective bounded diagnostic

2026-10-02. This is a new, unverified observer hypothesis after the complete
last-four-block re-acquisition trial failed its eight-domain goal. It is not a
successful final model or an originality claim. Preserve original Geometry,
historical best models and all earlier negative results.

## Why change the observation

The completed replay changes descriptors but improves only two primary domains.
Potsdam car FP decreases while VDD vehicle FP increases. The evidence therefore
does not support using structure consistency as a semantic correctness test.
Instead of another gate on correlated semantic scores, test a different frozen
pretraining objective: conditional denoising compatibility with actual image
content. This is also uncertain under aerial imagery and particularly IRRG.

## Public precedent and source boundary

The author implementation of Li et al., ICCV 2023, *Your Diffusion Model is
Secretly a Zero-Shot Classifier*, is pinned to
`diffusion-classifier/diffusion-classifier@e9f772d78b75976112d88a1002c496f6ef0cb27e`.
Its `eval_prob_adaptive.py` compares text-conditioned epsilon-prediction errors
with the same noise/timesteps across competing prompts and uses the VAE latent
mean. Its README normally starts with50 samples and then500 for surviving
prompts. Our fixed32-sample spatial/all20 diagnostic is not that complete system.

The research-lookup Search backend could not run because CLI device authorization
was missing. Direct public author code/README was read instead; no unpublished
model details or study data were sent to a literature provider. Public precedent
does not establish the novelty of any future Geometry coupling.

Frozen source: `CompVis/stable-diffusion-v1-4` at
`133a221b8aa7292a167afc5127cb63fb5005638b`, trained epsilon-prediction schedule.
Download only VAE, U-Net, text encoder/tokenizer and configuration, not a new
image-generation pipeline or safety-checker bypass. No images are generated.
Respect CreativeML OpenRAIL-M and retain the model README. The pinned diffusers
0.35.2 dependency resides in a project-only vendor path, leaving the original
server environment unchanged.

## Fixed observation

Use the exact original sixteen512 windows from each of VDD and Potsdam (two
label-independently cached windows per each of eight fixed images). Original
Geometry scores, image dimensions, sample keys, vocabulary and relation G are
unchanged. No masks select windows, support, aliases or noise.

Encode actual RGB into pretrained latent mean `z=scale*VAE_mean(2*RGB-1)`.
For32 midpoint timesteps over the checkpoint's1000-step training schedule,
sample one epsilon and share `z_t=sqrt(alpha_t)*z+sqrt(1-alpha_t)*epsilon` across
every candidate prompt. All20 aliases/class use the same template
`an aerial photograph of {label}.`; no word deletion, mean text embedding or
best-alias selection occurs. Canonical names are a separate observation control.

At each latent location, compute channel-mean squared epsilon prediction error,
average equally over timesteps and aliases, and average2x2 latent cells to the
original32x32 patch grid. Compare:

- Geometry: exact original saved class scores.
- DenoiseLocal20: negative local all20 mean epsilon error.
- DenoiseGeometry20: negative error pooled by original G on image-valid donors.
- DenoiseGlobal20: the common full-valid-window mean error broadcast to patches.
- DenoiseGeometryCanonical: same support with canonical names only.

The output is a compatibility observation, not a class posterior. Uniform
alias-mean error is not a marginalized generative likelihood. Spatial losses
retain full-window U-Net context and do not prove isolated regional likelihood.
No partially noised/inpainted distribution is invented. Record alternating
half-schedule class agreement to expose Monte Carlo/step sensitivity; it is not
a correctness gate. Retain complete per-prompt spatial errors for later audits.

## Execution and prospective decision

Physical A800 GPUs4-7 only. One smoke snapshot first verifies actual pretrained
execution. Then shard original sixteen windows into two disjoint eight-window
shards per dataset. Predictions and raw observations are saved before target
centers are accessed; labels enter only endpoint metrics and old/new/truth
transitions. No target-fitted coefficients, prompts or timesteps.

Report overlapping patch-center metrics as such, not full-image mIoU. Verify
the exact16 unique snapshot identities, original baseline and additive confusion
matrices; report car/vehicle TP/FP/FN and water/low-vegetation recall. Do not infer
success from ranking AUC or net correct pixels alone.

Do not promote the observer to a coupled full-model trial unless the fixed
Geometry-supported observer improves diagnostic mIoU over original Geometry on
both datasets, beneficial corrections exceed harmful corrections on both, and
car/vehicle FP falls with at least99% of original TP retained. These are labeled
development acceptance checks, not target-label inference. Even a pass would
require a prespecified complete coupling, same-observation controls and full
eight-domain validation before claiming the user objective achieved.

If it fails, preserve the result and reject this specific regional epsilon-loss
proxy rather than changing prompts/noise or adding a gate from the same labels.
This would not reject all generative features or prove a training-free ceiling.

## Completed outcome

All four shards completed on physical A800 GPUs4-7 without failures. Each dataset
has exactly16 unique original snapshot identities, matching source signatures,
all20 counts and unchanged Geometry confusion. Per-prompt raw observations and
old/new/truth transitions are retained locally.

| Patch-center development diagnostic | Original Geometry | Local epsilon20 | Geometry-supported epsilon20 | Global epsilon20 | Canonical supported |
| --- | ---: | ---: | ---: | ---: | ---: |
| VDD | 44.3340 | 11.5002 | 22.1666 | 31.4587 | 21.0308 |
| Potsdam | 40.0511 | 11.4330 | 19.9017 | 11.3467 | 12.7543 |

Both datasets fail all four prospective acceptance checks. VDD beneficial/harmful
counts are869/4510; Potsdam1648/5752. Vehicle/car FP increases and TP decreases.
Alternating half-schedule agreement is47.36%/44.67%. Geometry pooling improves
the weak local observation but does not produce a better semantic source than
original Geometry. This candidate is rejected for full eight-domain correction.
No parameters, prompts, timestep counts or noise were changed from these results.

Saved all20 local scores reconstruct from raw errors to5.96e-8. Separately batched
identical canonical prompts differ by up to3.57e-4(VDD) and4.19e-4(Potsdam) in
FP16 error maps; the canonical control is therefore not purely a content-only
comparison. Low half-schedule agreement, numerical sensitivity, limited32-sample
budget and aerial-domain mismatch are limitations, not separately identified
causes. Do not infer that the complete original50/500-sample Diffusion Classifier
or all generative observations must fail from this test.

The result report is `GEOMETRY_GENERATIVE_OBSERVATION_RESULTS_20261002.md`; raw
arrays and merged diagnostic results are under
`research/geometry_generative_diagnostic_20261002`. All newly launched jobs are
terminal; GPUs0-3 were untouched. The user research objective remains unachieved:
there is still no verified complete new coupled model superior across eight
datasets. Original Geometry and historical best models remain unchanged.
