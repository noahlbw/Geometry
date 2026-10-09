# Geometry-supported strong semantic observer: prospective protocol

Primary: `Geometry_RegionStrong`. Signature: `geometry-region-so400m-footprint-joint-v1-20261002`.
Only authorized A800 physical GPUs4-7. Preserve original Geometry, base256 regional observer, SAT transport, and historical best-model results.

## Execution status

Implemented and complete at `results/geometry_region_strong_screen_20261002`, former controller `grs02_suite`, workers `grs02_screen_DATASET_s0`. All eight fixed-domain evaluations are terminal; outputs are merged, downloaded and checked against the original per-image Geometry/SCLIP/VIPProxy matrices. The new13 unit tests pass locally and remotely; the original nine regional tests and seven SAT-controller regression tests also pass locally.

Actual-checkpoint smoke on physical GPU4 loads no masks: original Geometry score error0, native cache error0, original trained pooling replay error0, all weights frozen,729 real observer tokens. Pretrained semantic temperature is0.0090996521. Weight SHA256 is `9f4f4a49f908ef0c979bce8ff5a5c0e88882dc6c5dc4304387cbbd152558e2c2` (4,544,143,072 bytes).

Window medians from three repetitions are Geometry0.025123s and primary1.461670s (58.18x). Both encoders are resident during both measurements; these are not independent Geometry-only memory or full-image timings.

The final verified screen fails the frozen promotion gate. Full40 UDD5 is50.5553->48.0467; each other domain uses eight fixed images, not a full-set benchmark. Potsdam40.3528->38.3952, OEM39.3543->32.3310, FLAIR-138.8428->31.2528, LandCover.ai60.9049->57.8141 and LoveDA P62.8254->54.9680/D38.4558->35.3908 also decline. Only VDD improves31.9462->32.9216. The equal-domain mean is40.246801 versus original Geometry43.830569. The same-information joint inference beats RegionFusion on all eight primary entries, but beats original Geometry on only1/8. No full rollout was launched and no parameters changed after outcomes. Do not promote this candidate or reinterpret its larger borrowed observer as an original contribution.

Verified downloads: `research/geometry_region_strong_screen_20261002/`; report generator: `tools/report_geometry_region_strong.py`; final report: `research/GEOMETRY_REGION_STRONG_SCREEN_20261002.md`. Controller duration was5181.978 seconds, including loading and concurrent allocation, not independent deployed latency. After completion all physical GPUs4-7 were idle; unrelated tmux sessions were untouched. The overall research goal remains unachieved.

## Frozen complete architecture

The original DINOv3/DINO.text Geometry defines all connected fine64/coarse16 supports from native image descriptors. Its restricted relation determines within-support pooling weights. A pinned independent SigLIP2 SO400M encoder reads actual detail and context RGB through its pretrained probe, attention projections, residual and MLP with those support priors. Both encoders use unchanged20 aliases and six RS templates; no word selection is introduced.

Source: `google/siglip2-so400m-patch14-384`, revision `e8e487298228002f3d8a82e0cd5c8ea9c567f57f`. Author config:384 input,14 stride convolution,27x27 tokens,1152 hidden width and27 vision blocks. Pooling weights are integrated over the actual14x14 footprints; the final six input pixels are not stretched into an invented token. The source's fixed pretrained logit scale determines its scoring temperature. Geometry retains its0.07 temperature and original512/128 sliding-window/Hann probability assembly.

There is one complete joint primary, not a per-dataset choice. Continuous entropy/view agreement limits regional influence; it is not a correctness certificate. Joint pixel/regional inference minimizes the existing convex KL energy:

`sum_i KL(p_i||g_i) + sum_r n_r KL(q_r||z_r) + sum_ri W_ri KL(p_i||z_r)`.

The32-step numerical solver, supports, aliases and rule are fixed before outcomes. No labels enter inference. This is an observer-quality test of the previous complete regional architecture, not a claim that upgrading a checkpoint or using a standard KL solver constitutes an original contribution.

## Controls and promotion

Record Geometry, matched SCLIP_Two/VIPProxy_Two, CropGlobalFusion, RegionSemantic, equal-information RegionFusion, and the joint primary. Use the existing SAT screen's fixed image-only sequence: full40 UDD5 and eight samples for each other domain,96 images total. Verify unique complete coverage and exact per-image matched baseline replay against the authoritative full references.

Promotion requires the equal-domain primary mean to exceed Geometry, matched VIPProxy_Two and RegionFusion; no protocol may lose more than1pp against Geometry; VDD and Potsdam must exceed both Geometry and same-information fusion. LoveDA D enters the domain mean once; P is reported separately. A passed screen triggers full eight-domain evaluation with the same rule, not target-label tuning. A failed screen rejects this candidate without launching expensive full runs. The screen is exploratory development, not an eight-domain full-set SOTA benchmark.

## Source fairness and novelty boundary

The SigLIP2 card states WebLI pretraining. That is not an independently verified absence of evaluation-image overlap. This candidate adds a separate encoder and must report extra compute; a comparison to single-backbone VIP is not an equal-information comparison.

RemoteCLIP was not adopted. Its pinned public annotation-to-caption SEG-4 training filenames include LoveDA4187, Potsdam5421 and Vaihingen742 entries. Local provenance matching finds1667/1669 LoveDA numeric filenames and parent-scene matches for504/504 Potsdam tiles and113/113 Vaihingen tiles. Parent-scene filename overlap does not prove identical crop pixels, but prohibits silently claiming clean unseen-dataset transfer. No RemoteCLIP weights or target captions are used.

Matched VIPProxy_Two is not official full VIP. The historical official VIP VDD/Potsdam results use different protocols and must remain separately identified. LandCover.ai substitutes for unlabeled iSAID. All eight current domains are development data. CVPR readiness still requires closest-method comparisons, meaningful coupling/compute evidence and genuinely independent validation.
