# Fixed expansion class-field attribution

Reuse all64 developed windows and persisted20/40 scores from the completed frozen
vocabulary stress study. Three fixed endpoints: no admission, original hard and
projected candidate. No image/text network forward, selector, target fitting, new
word generation or automatic model choice. Geometry, observers, admission and
coupling outputs remain unchanged. This is diagnosis, not independent validation.

Because class counts change20->40 simultaneously, the theoretical broad log K
offset maps through unchanged H to one common class-logit shift. Remove this shift
and require identical predictions. It does NOT isolate salience-profile changes,
word identities or their effects on admission/posterior: those are not pure count.

For each endpoint, fix the class-logit gauge by subtracting the query class mean.
Validate centered20/40 predictions and per-image confusions against prior outputs.
For every class c, produce both factorial interventions: own20/rivals40 and
own40/rivals20. These replace centered FINAL score columns, equivalently fixed
admitted-field columns under the same linear H/local anchor. They do not rerun
the selector on a hybrid vocabulary and do not identify individual harmful words.

Persist all intervention scores before masks. Report each class's IoU, precision,
recall and predicted area under all20/all40/own20-rivals40/own40-rivals20, same target
counts and per-image coverage. Also report mIoU changes on the source stress class
supports; LoveDA D once, P separate. The two factors can interact through argmax
and IoU, so their improvements are not additive attribution percentages.

Do not select a class-wise winning arm, tune alias weights from these targets or
promote an oracle restoration as a model. Preserve every prior scheme and output.
