# Shared local observation: frozen lightweight alias candidate

Retain OneSide_Stream/OneSide_Soft and all previous results as the accuracy
reference. The new candidate reuses the SAME512 Geometry backbone tokens and
adds one attributed matched VIPProxy_Two head per tile, at most4. No extra RGB
encoder, quadrant crop or resized feature pretending to be physical8 detail.
Geometry/W/H, bounded896/448 protocol, fixed20 vocabularies, source checkpoints
and all target scoring stay unchanged. VIPProxy_Two is inherited methodology,
not a new operator innovation. Its local context/resolution differs from the
prior four cropped native heads and from16 independent physical8 observations.

Preaverage the existing ImageNet template embeddings before the FP32 linear
dot product. Retain inherited40 logit scale, tem1 salience, tau1 class evidence.
Use the established one-sided margin-contradiction risk and fixed-slot soft
attenuation; use half positive local observation and half wide action through
the unchanged H. No target-label parameter tuning or per-domain routing.

Controls: original Geometry, no admission, patch2 coupling, same-local-source
without attenuation, independently deployable pooled-class calibration, and
fixed within-class noncanonical alias-risk shuffle. Same96 developed COMPLETE
images (UDD540 and8/other domain), LoveDA P/D shared; D counted once. These are
not untouched independent validation or full eight-dataset results.

Mask-free real-checkpoint smokes must replay all three original endpoints and
verify singleton/joint execution. Then panel and paired single-GPU full-image
timing: current retained OneSide_Stream, new SharedLocal_Soft, same-local mean,
bounded baseline and VIP_All20;3 fixed images/domain,5 rotated warmed repeats.
Extra head cost is included. Preserve old outputs. Check target-GPU occupancy;
other GPU jobs may run and their state is recorded, without interference.

Prospective engineering target: domain-mean latency ratio<=3x VIP, report
every domain ratio, mean accuracy loss<=1pp relative to retained50.2181 and
no protocol loss>2pp. These tolerance choices operationalize 'not too much'
for this first candidate, not a CVPR standard or new user requirement. Report
all failures; no automatic control promotion. Word-specific value additionally
requires beating same-source mean and pooled-class/shuffled controls. No
automatic full20092 rollout or replacing the retained model.
