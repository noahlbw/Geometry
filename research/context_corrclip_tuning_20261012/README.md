# PC59/PC60: alias and scalar development versus CorrCLIP

This trial changes only text aliases/templates and existing scalar parameters.
The Geometry relation and depth, local/wide readout equations, coupled writeback,
visual weights/dtypes, cap4+4/zero-fine image views, and image resizing are fixed.
PC60 retains its existing `residual_max` rule and the identical401 residual concepts.

| Protocol | Retained full mIoU | CorrCLIP full mIoU | Improvement required |
| --- | ---: | ---: | ---: |
| PC59 | 46.3075 | 47.7036 | >1.3961 pp |
| PC60 | 44.7613 | 45.1164 | >0.3551 pp |

The finite candidate menu and source vocabularies are in `protocol.json`.
Candidates include existing official/semantic words, the other protocol's best
groups, literal natural-language anchors, and explicit sense-disambiguated words.
The search compares banks and readout scalars, permits one class-group swap sweep,
then confirms shortlisted configurations on256 deterministic development images.
Each accepted class swap must improve whole-development mIoU, including competing
classes. No visual weight fitting or new inference module is added.

The development-selected choice is frozen before full5105-image inference per
protocol. Unique coverage, per-image confusion sums, scored target counts and
exact development replay are checked when merging. Seven fixed whole images
receive independent synchronized timing/memory measurements with two warmups and
seven repeats. Search caches and allocations are excluded from online costs.

These are exploratory labeled development experiments on previously viewed
validation data. Variable word counts also alter the inherited wide aggregation
mass. CorrCLIP is the verified local native MetaCLIP+online-SAM2 implementation;
its scores are full-method references, not a matched-backbone ablation or new
published benchmark numbers. Existing configurations and raw results remain intact.

Local manager: `python tools/context_corrclip_tuning_experiment.py status` or
`collect` after completion. Controller `gpcc12_controller` queues only idle
GPUs4–7, then runs cost jobs alone within this suite. It preserves existing outputs.

Both full protocols and standalone costs are complete. See
[CONTEXT_CORRCLIP_TUNING_20261012.md](../CONTEXT_CORRCLIP_TUNING_20261012.md)
for the scores, parameters, per-class gains/losses and measured costs.

Portable final-model inference uses `configs/context59.yaml` or
`configs/context60.yaml` and the corresponding `vocabularies/` file:

```bash
python DINOtool/scripts/infer_frozen_geometry.py \
  --config research/context_corrclip_tuning_20261012/configs/context59.yaml \
  --checkpoint-dir /path/to/checkpoints --dinov3-repo DINOtool/dinov3_hub \
  --upstream-root third_party/VIP --text-cache /path/to/cache/pc59.pt \
  --input /path/to/image.jpg --output /path/to/new_prediction.png --efficient
```

`evaluate_frozen_geometry.py` accepts the same model options plus `--data-root`
and a new JSON `--output`. It does not require historical experiment folders.
The development controller/manager additionally uses the previously collected
experimental artifacts. Reported full scores use the exact cached selected text
vectors; fresh encoding on other software/hardware may differ at floating-point
roundoff level and is not asserted to be bit-exact.
