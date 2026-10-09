# Natural category-sense vocabulary candidate

This candidate changes words and protected text representatives only, not the model or source tau/tem/threshold. No images, masks or class-frequency metadata select it. It is not an evaluated or promoted final model. It is separate from the running four-word taxonomy repair and scalar-count experiments.

| Protocol | Source pool aliases | Candidate aliases | Min/max per class | Changed representatives | Explicit sense removals |
| --- | ---: | ---: | --- | ---: | ---: |
| voc20 | 116 | 116 | 3-23 | 3 | 0 |
| voc21 | 172 | 172 | 3-56 | 3 | 0 |
| ade150 | 382 | 458 | 1-24 | 19 | 7 |
| coco_stuff171 | 639 | 644 | 1-18 | 33 | 0 |
| coco_object81 | 325 | 325 | 3-32 | 4 | 0 |

ADE step retains stair-step semantics and drops pedestal/plinth/footstall. Base gains pedestal; apparel gains clothing; missing ordinary names such as washing machine, toy and drinking glass are added. Subtype words such as dog/cat/horse under ADE animal remain; they are not rejected merely for being narrower.

COCO-Stuff compound category strings gain natural noun-phrase representatives. The split textile query fragments are removed and replaced with complete bed-sheet/tablecloth descriptions. The v1 manifests remain preserved; none of the running inputs is replaced.

Non-ADE protocols resolve obvious object names/compound labels and share conventional visual synonyms where the category meaning is unchanged. Residual background descriptions remain source-defined. There is no forced alias count or cosine-only rejection.

The source calibration was developed on natural-image windows, so subsequent evaluation is exploratory. A better-sounding vocabulary is not evidence of higher IoU; paired full-image evaluation is required.
